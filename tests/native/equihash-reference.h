// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

// Host reference model of the Equihash packed-record arenas: an unbucketed level-zero record,
// checked provenance recovery, and scalar collision/scatter rounds. The production solvers use the
// bucketed records in sycl/equihash-core.h; tests compare those kernels against this model.

#include "../../sycl/equihash-core.h"

namespace mom_equihash {

// Packed records are intentionally word-aligned: six 24-bit fields plus the 25-bit leaf index
// occupy 169 bits, while a first collision's five fields plus two parent references occupy 172
// bits.  Both therefore fit in six words, versus seven words for the expanded level-0 Row.
template <typename Spec>
struct alignas(8) PackedLevel0Record {
  std::uint32_t words[Spec::level0_words];
};

template <typename Spec>
inline void clear_packed(PackedLevel0Record<Spec>& record) {
  for (unsigned i = 0; i < Spec::level0_words; ++i) {
    record.words[i] = 0;
  }
}

template <typename Spec>
// Pack the generated collision fields and original leaf index into a zeroed level-zero record.
inline void store_level0(PackedLevel0Record<Spec>& record,
                         const std::uint32_t fields[Spec::rounds], const std::uint32_t leaf) {
  clear_packed<Spec>(record);
  for (unsigned i = 0; i < Spec::rounds; ++i)
    packed_store(record.words, Spec::level0_words, i * Spec::collision_bits,
                 Spec::collision_bits, fields[i]);
  packed_store(record.words, Spec::level0_words, Spec::rounds * Spec::collision_bits,
               Spec::index_bits, leaf);
}

template <typename Spec>
inline std::uint32_t load_level0_field(const PackedLevel0Record<Spec>& record, const unsigned field) {
  return packed_load(record.words, Spec::level0_words, field * Spec::collision_bits,
                     Spec::collision_bits);
}

template <typename Spec>
inline std::uint32_t load_level0_leaf(const PackedLevel0Record<Spec>& record) {
  return packed_load(record.words, Spec::level0_words, Spec::rounds * Spec::collision_bits,
                     Spec::index_bits);
}

// Parent references are deliberately kept compact in device records.  This view supplies the
// round-specific arenas needed to walk those references back to level zero on the host (or in a
// later checked solver pass).  ZHash has five collision rounds, so round five is the root record
// itself and only rounds one through four need to be retained as child arenas.
template <typename Spec>
struct ProvenanceView {
  static_assert(Spec::rounds == 6, "the checked provenance view currently targets ZHash144_5");
  using Level0 = PackedLevel0Record<Spec>;
  using Round1 = CollisionRoundRecord<Spec, 1>;
  using Round2 = CollisionRoundRecord<Spec, 2>;
  using Round3 = CollisionRoundRecord<Spec, 3>;
  using Round4 = CollisionRoundRecord<Spec, 4>;

  const Level0* level0 = nullptr;
  std::uint32_t level0_capacity = 0;
  const Round1* round1 = nullptr;
  std::uint32_t round1_capacity = 0;
  const Round2* round2 = nullptr;
  std::uint32_t round2_capacity = 0;
  const Round3* round3 = nullptr;
  std::uint32_t round3_capacity = 0;
  const Round4* round4 = nullptr;
  std::uint32_t round4_capacity = 0;
};

template <typename Spec>
struct LeafSet {
  std::uint32_t values[Spec::proof_indices];
  std::uint32_t count;
};

template <typename Spec>
// Reject repeated leaves when joining reconstructed provenance subtrees.
inline bool leaf_sets_disjoint(const LeafSet<Spec>& left, const LeafSet<Spec>& right) {
  for (std::uint32_t i = 0; i < left.count; ++i)
    for (std::uint32_t j = 0; j < right.count; ++j)
      if (left.values[i] == right.values[j]) return false;
  return true;
}

template <typename Spec>
// Join distinct child leaf sets only when their combined size fits the complete proof.
inline bool leaf_set_append(const LeafSet<Spec>& left, const LeafSet<Spec>& right,
                            LeafSet<Spec>& out) {
  if (left.count + right.count > Spec::proof_indices || !leaf_sets_disjoint(left, right))
    return false;
  out.count = 0;
  for (std::uint32_t i = 0; i < left.count; ++i) {
    out.values[out.count++] = left.values[i];
  }
  for (std::uint32_t i = 0; i < right.count; ++i) {
    out.values[out.count++] = right.values[i];
  }
  return true;
}

template <typename Spec>
inline bool recover_node(const ProvenanceView<Spec>& view, const unsigned round,
                         const std::uint32_t ref, LeafSet<Spec>& out);

template <typename Spec>
// Read one retained provenance field only after checking round, slot, and field bounds.
inline bool load_node_field(const ProvenanceView<Spec>& view, const unsigned round,
                            const std::uint32_t ref, const unsigned field,
                            std::uint32_t& value) {
  if (round == 0) {
    if (view.level0 == nullptr || ref >= view.level0_capacity || field >= Spec::rounds)
      return false;
    value = load_level0_field<Spec>(view.level0[ref], field);
    return true;
  }
  if (round == 1) {
    if (view.round1 == nullptr || ref >= view.round1_capacity || field >= Spec::rounds - 1)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 1>(view.round1[ref], field);
    return true;
  }
  if (round == 2) {
    if (view.round2 == nullptr || ref >= view.round2_capacity || field >= Spec::rounds - 2)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 2>(view.round2[ref], field);
    return true;
  }
  if (round == 3) {
    if (view.round3 == nullptr || ref >= view.round3_capacity || field >= Spec::rounds - 3)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 3>(view.round3[ref], field);
    return true;
  }
  if (round == 4) {
    if (view.round4 == nullptr || ref >= view.round4_capacity || field >= Spec::rounds - 4)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 4>(view.round4[ref], field);
    return true;
  }
  return false;
}

template <typename Spec, unsigned FieldCount>
// Validate child collision fields, canonical order, and disjoint leaves while reconstructing provenance.
inline bool recover_collision_record(const ProvenanceView<Spec>& view,
                                     const PackedCollisionRecord<Spec, FieldCount>& record,
                                     const unsigned parent_round, LeafSet<Spec>& out) {
  static_assert(FieldCount > 0, "a provenance record must retain a parent pair");
  const std::uint32_t left_ref = load_collision_left<Spec, FieldCount>(record);
  const std::uint32_t right_ref = load_collision_right<Spec, FieldCount>(record);
  if (left_ref == right_ref) return false;

  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, parent_round, left_ref, left) ||
      !recover_node<Spec>(view, parent_round, right_ref, right))
    return false;
  std::uint32_t left_field = 0, right_field = 0;
  if (!load_node_field<Spec>(view, parent_round, left_ref, 0, left_field) ||
      !load_node_field<Spec>(view, parent_round, right_ref, 0, right_field) ||
      left_field != right_field)
    return false;
  for (unsigned field = 0; field < FieldCount; ++field) {
    if (!load_node_field<Spec>(view, parent_round, left_ref, field + 1, left_field) ||
        !load_node_field<Spec>(view, parent_round, right_ref, field + 1, right_field) ||
        load_collision_field<Spec, FieldCount>(record, field) != (left_field ^ right_field))
      return false;
  }
  // Every record is stored with its lower minimum-leaf child first.  Checking this while walking
  // the tree makes the encoded solution deterministic and catches malformed parent references.
  if (left.count == 0 || right.count == 0 || left.values[0] >= right.values[0]) return false;
  return leaf_set_append<Spec>(left, right, out);
}

// Dispatch bounded provenance recovery to the retained level, rejecting unsupported rounds.
template <typename Spec>
inline bool recover_node(const ProvenanceView<Spec>& view, const unsigned round,
                         const std::uint32_t ref, LeafSet<Spec>& out) {
  out.count = 0;
  if (round == 0) {
    if (view.level0 == nullptr || ref >= view.level0_capacity) return false;
    out.values[0] = load_level0_leaf<Spec>(view.level0[ref]);
    out.count = 1;
    return true;
  }
  if (round == 1) {
    if (view.round1 == nullptr || ref >= view.round1_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 1>(view, view.round1[ref], 0, out);
  }
  if (round == 2) {
    if (view.round2 == nullptr || ref >= view.round2_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 2>(view, view.round2[ref], 1, out);
  }
  if (round == 3) {
    if (view.round3 == nullptr || ref >= view.round3_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 3>(view, view.round3[ref], 2, out);
  }
  if (round == 4) {
    if (view.round4 == nullptr || ref >= view.round4_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 4>(view, view.round4[ref], 3, out);
  }
  return false;
}

template <typename Spec>
inline bool is_zero_root(const ZHashRootRecord<Spec>& root) {
  static_assert(Spec::rounds == 6, "the ZHash root has one final active field");
  return load_collision_field<Spec, 1>(root, 0) == 0;
}

template <typename Spec>
// Expand a zero-XOR root only if its checked provenance contains exactly one complete proof.
inline bool recover_root(const ProvenanceView<Spec>& view, const ZHashRootRecord<Spec>& root,
                         std::uint32_t leaves[Spec::proof_indices]) {
  if (!is_zero_root<Spec>(root)) return false;
  LeafSet<Spec> recovered{};
  if (!recover_collision_record<Spec, 1>(view, root, Spec::rounds - 2, recovered) ||
      recovered.count != Spec::proof_indices)
    return false;
  for (unsigned i = 0; i < Spec::proof_indices; ++i) {
    leaves[i] = recovered.values[i];
  }
  return true;
}

template <typename Spec>
// Encode only a root whose retained provenance passes complete recovery checks.
inline bool recover_solution(const ProvenanceView<Spec>& view, const ZHashRootRecord<Spec>& root,
                             std::uint32_t leaves[Spec::proof_indices],
                             std::uint8_t solution[Spec::solution_length]) {
  if (!recover_root<Spec>(view, root, leaves)) return false;
  encode_indices<Spec>(leaves, solution);
  return true;
}

template <typename Spec>
// Build the first collision record only for matching fields and canonical distinct leaf order.
inline bool make_collision(const PackedLevel0Record<Spec>& a, const PackedLevel0Record<Spec>& b,
                           const std::uint32_t a_ref, const std::uint32_t b_ref,
                           PackedCollisionRecord<Spec, Spec::rounds - 1>& out) {
  if (load_level0_field<Spec>(a, 0) != load_level0_field<Spec>(b, 0)) return false;
  const std::uint32_t a_leaf = load_level0_leaf<Spec>(a);
  const std::uint32_t b_leaf = load_level0_leaf<Spec>(b);
  if (a_leaf == b_leaf) return false;
  std::uint32_t fields[Spec::rounds - 1];
  for (unsigned i = 0; i < Spec::rounds - 1; ++i)
    fields[i] = load_level0_field<Spec>(a, i + 1) ^ load_level0_field<Spec>(b, i + 1);
  if (a_leaf < b_leaf) {
    store_collision<Spec>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec>(out, fields, b_ref, a_ref);
  }
  return true;
}

// A later collision consumes a compact record from the preceding round.  `a_order` and `b_order`
// are the minimum original leaf indices of those records; keeping that small parallel array lets
// the compact record retain only two parent references while still imposing canonical child order.
template <typename Spec, unsigned FieldCount>
inline bool make_collision(
    const PackedCollisionRecord<Spec, FieldCount>& a,
    const PackedCollisionRecord<Spec, FieldCount>& b, const std::uint32_t a_ref,
    const std::uint32_t b_ref, const std::uint32_t a_order, const std::uint32_t b_order,
    PackedCollisionRecord<Spec, FieldCount - 1>& out) {
  static_assert(FieldCount > 0, "a collision must retain an active field");
  if (load_collision_field<Spec, FieldCount>(a, 0) !=
      load_collision_field<Spec, FieldCount>(b, 0)) return false;
  if (a_order == b_order) return false;
  const std::uint32_t a_left = load_collision_left<Spec, FieldCount>(a);
  const std::uint32_t a_right = load_collision_right<Spec, FieldCount>(a);
  const std::uint32_t b_left = load_collision_left<Spec, FieldCount>(b);
  const std::uint32_t b_right = load_collision_right<Spec, FieldCount>(b);
  if (a_left == b_left || a_left == b_right || a_right == b_left || a_right == b_right)
    return false;
  std::uint32_t fields[FieldCount - 1];
  for (unsigned i = 0; i < FieldCount - 1; ++i)
    fields[i] = load_collision_field<Spec, FieldCount>(a, i + 1) ^
                load_collision_field<Spec, FieldCount>(b, i + 1);
  if (a_order < b_order) {
    store_collision<Spec, FieldCount - 1>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec, FieldCount - 1>(out, fields, b_ref, a_ref);
  }
  return true;
}

// Checked variants use the recursive arena view instead of trusting only the parallel minimum
// index.  They are the host/reference contract for rejecting a pair whose subtrees share any leaf;
// the compact device kernels retain the same parent references for a later checked pass.
template <typename Spec>
inline bool make_collision_checked(
    const PackedLevel0Record<Spec>& a, const PackedLevel0Record<Spec>& b,
    const std::uint32_t a_ref, const std::uint32_t b_ref, const ProvenanceView<Spec>& view,
    CollisionRoundRecord<Spec, 1>& out) {
  if (load_level0_field<Spec>(a, 0) != load_level0_field<Spec>(b, 0)) return false;
  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, 0, a_ref, left) || !recover_node<Spec>(view, 0, b_ref, right) ||
      left.count != 1 || right.count != 1 ||
      left.values[0] != load_level0_leaf<Spec>(a) ||
      right.values[0] != load_level0_leaf<Spec>(b) ||
      !leaf_sets_disjoint<Spec>(left, right))
    return false;
  std::uint32_t fields[Spec::rounds - 1];
  for (unsigned i = 0; i < Spec::rounds - 1; ++i)
    fields[i] = load_level0_field<Spec>(a, i + 1) ^ load_level0_field<Spec>(b, i + 1);
  if (left.values[0] < right.values[0]) {
    store_collision<Spec, Spec::rounds - 1>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec, Spec::rounds - 1>(out, fields, b_ref, a_ref);
  }
  return true;
}

template <typename Spec, unsigned FieldCount>
inline bool make_collision_checked(
    const PackedCollisionRecord<Spec, FieldCount>& a,
    const PackedCollisionRecord<Spec, FieldCount>& b, const std::uint32_t a_ref,
    const std::uint32_t b_ref, const ProvenanceView<Spec>& view,
    PackedCollisionRecord<Spec, FieldCount - 1>& out) {
  static_assert(FieldCount > 1, "the five-round ZHash solver stops at one root field");
  const unsigned parent_round = Spec::rounds - 1 - FieldCount;
  std::uint32_t left_field = 0;
  std::uint32_t right_field = 0;
  if (!load_node_field<Spec>(view, parent_round, a_ref, 0, left_field) ||
      !load_node_field<Spec>(view, parent_round, b_ref, 0, right_field) ||
      left_field != right_field)
    return false;

  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, parent_round, a_ref, left) ||
      !recover_node<Spec>(view, parent_round, b_ref, right) ||
      !leaf_sets_disjoint<Spec>(left, right) || left.values[0] == right.values[0])
    return false;
  std::uint32_t fields[FieldCount - 1];
  for (unsigned i = 0; i < FieldCount - 1; ++i)
    fields[i] = load_collision_field<Spec, FieldCount>(a, i + 1) ^
                load_collision_field<Spec, FieldCount>(b, i + 1);
  if (left.values[0] < right.values[0])
    store_collision<Spec, FieldCount - 1>(out, fields, a_ref, b_ref);
  else
    store_collision<Spec, FieldCount - 1>(out, fields, b_ref, a_ref);
  return true;
}

template <typename Spec, typename Layout>
// Provide the host first-round bucket oracle with bounded output and explicit truncation accounting.
inline std::uint32_t bucket_round_reference(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec>* const output, const std::uint32_t output_capacity) {
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto key = load_level0_field<Spec>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(key);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      const auto& left = input[base + a];
      unsigned link = bin_head[Layout::local_bin(load_level0_field<Spec>(left, 0))];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a) continue;
        const auto& right = input[base + b];
        PackedCollisionRecord<Spec> collision{};
        if (!make_collision<Spec>(left, right,
                                  static_cast<std::uint32_t>(base + a),
                                  static_cast<std::uint32_t>(base + b),
                                  collision)) continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

template <typename Spec, typename Layout>
// Run the first-round host oracle with checked provenance rather than trusting parent references.
inline std::uint32_t bucket_round_reference_checked(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec>* const output, std::uint32_t* const output_min_indices,
    const std::uint32_t output_capacity, const ProvenanceView<Spec>& view) {
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      const unsigned bin = Layout::local_bin(load_level0_field<Spec>(input[base + i], 0));
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(load_level0_field<Spec>(input[base + a], 0))];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a) continue;
        PackedCollisionRecord<Spec> collision{};
        if (!make_collision_checked<Spec>(input[base + a], input[base + b],
                                          static_cast<std::uint32_t>(base + a),
                                          static_cast<std::uint32_t>(base + b), view, collision))
          continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          const std::uint32_t a_leaf = load_level0_leaf<Spec>(input[base + a]);
          const std::uint32_t b_leaf = load_level0_leaf<Spec>(input[base + b]);
          output_min_indices[emitted] = a_leaf < b_leaf ? a_leaf : b_leaf;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

// Scatter preserves bucket-major storage and counts every reservation, including one that misses
// the bounded arena.  Thus a caller can clamp each bucket to its capacity for the next round while
// the separate overflow counter still makes an incomplete solve visible to its scheduler.
template <typename Spec, typename Layout>
inline std::uint32_t scatter_level0_reference(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t input_count,
    PackedLevel0Record<Spec>* const output, std::uint32_t* const bucket_counts,
    std::uint32_t* const overflow_count) {
  const std::uint32_t before = *overflow_count;
  for (std::uint32_t i = 0; i < input_count; ++i) {
    const unsigned bucket = Layout::bucket_for(load_level0_field<Spec>(input[i], 0));
    const std::uint32_t slot = bucket_counts[bucket]++;
    if (slot < Layout::bucket_slot_capacity)
      output[static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity + slot] = input[i];
    else
      ++*overflow_count;
  }
  return input_count - (*overflow_count - before);
}

template <typename Spec, unsigned FieldCount, typename Layout>
// Scatter reference collision records into bounded buckets while recording dropped reservations.
inline std::uint32_t scatter_collision_reference(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t input_count,
    PackedCollisionRecord<Spec, FieldCount>* const output,
    std::uint32_t* const output_min_indices, std::uint32_t* const bucket_counts,
    std::uint32_t* const overflow_count) {
  const std::uint32_t before = *overflow_count;
  for (std::uint32_t i = 0; i < input_count; ++i) {
    const unsigned bucket = Layout::bucket_for(load_collision_field<Spec, FieldCount>(input[i], 0));
    const std::uint32_t slot = bucket_counts[bucket]++;
    if (slot < Layout::bucket_slot_capacity) {
      const std::size_t output_slot =
          static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity + slot;
      output[output_slot] = input[i];
      output_min_indices[output_slot] = input_min_indices[i];
    } else {
      ++*overflow_count;
    }
  }
  return input_count - (*overflow_count - before);
}

template <typename Spec, unsigned FieldCount, typename Layout>
// Provide the host later-round collision oracle, preserving bounded output counters.
inline std::uint32_t collision_round_reference(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec, FieldCount - 1>* const output,
    std::uint32_t* const output_min_indices, const std::uint32_t output_capacity) {
  static_assert(FieldCount > 0, "a collision round needs an active field");
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  std::uint32_t keys[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      keys[i] = load_collision_field<Spec, FieldCount>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(keys[i]);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(keys[a])];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a || keys[b] != keys[a]) continue;
        PackedCollisionRecord<Spec, FieldCount - 1> collision{};
        if (!make_collision<Spec, FieldCount>(
                input[base + a], input[base + b], static_cast<std::uint32_t>(base + a),
                static_cast<std::uint32_t>(base + b), input_min_indices[base + a],
                input_min_indices[base + b], collision)) continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          output_min_indices[emitted] = input_min_indices[base + a] < input_min_indices[base + b]
                                             ? input_min_indices[base + a]
                                             : input_min_indices[base + b];
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

template <typename Spec, unsigned FieldCount, typename Layout>
// Run the later-round host oracle with provenance checks before accepting collision output.
inline std::uint32_t collision_round_reference_checked(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec, FieldCount - 1>* const output,
    std::uint32_t* const output_min_indices, const std::uint32_t output_capacity,
    const ProvenanceView<Spec>& view) {
  static_assert(FieldCount > 1, "the five-round ZHash solver stops at one root field");
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  std::uint32_t keys[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      keys[i] = load_collision_field<Spec, FieldCount>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(keys[i]);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(keys[a])];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a || keys[b] != keys[a]) continue;
        PackedCollisionRecord<Spec, FieldCount - 1> collision{};
        if (!make_collision_checked<Spec, FieldCount>(
                input[base + a], input[base + b], static_cast<std::uint32_t>(base + a),
                static_cast<std::uint32_t>(base + b), view, collision))
          continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          const std::uint32_t a_min = input_min_indices[base + a];
          const std::uint32_t b_min = input_min_indices[base + b];
          output_min_indices[emitted] = a_min < b_min ? a_min : b_min;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

}  // namespace mom_equihash
