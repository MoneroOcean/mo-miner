template <bool NAIVE_TWIST> class ZelHashGenTestKernel;

template <bool NAIVE_TWIST>
static sycl::event submit_gen_test(
  sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint8_t* d_rows /* EQUIHASH_TEST_ROWS * HASH_LENGTH */
) {
  if constexpr (NAIVE_TWIST) {
    constexpr unsigned WG = 64;
    const size_t global = (EQUIHASH_TEST_ROWS + WG - 1) / WG * WG;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      h.parallel_for<ZelHashGenTestKernel<true>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const uint32_t e = (uint32_t)it.get_global_id(0);
        if (e >= EQUIHASH_TEST_ROWS) return;
        uint64_t base_h[8];
        for (unsigned i = 0; i < 8; ++i) base_h[i] = d_base_h[i];
        uint8_t pending[12];
        for (unsigned i = 0; i < 12; ++i) pending[i] = d_base_pending[i];
        uint8_t row[HASH_LENGTH];
        entry_row_naive(base_h, pending, e, row);
        for (unsigned i = 0; i < HASH_LENGTH; ++i) d_rows[(size_t)e * HASH_LENGTH + i] = row[i];
      });
    });
  } else {
    // Optimized: one sub-group of 16 lanes per 16-aligned block. Each lane g2 computes its own stock
    // hash; inclusive_scan_over_group (lane-wise u32 add) gives the twist prefix-sum for every entry
    // in the block in one pass. EQUIHASH_TEST_ROWS entries span EQUIHASH_TEST_ROWS/4 hash indices, but
    // each hash index e/4 belongs to a 16-block; we run one work-group of 16 per block touched.
    // The largest hash index we need is (EQUIHASH_TEST_ROWS-1)/4; round up to whole 16-blocks.
    constexpr unsigned MAX_HASH_IDX = (EQUIHASH_TEST_ROWS - 1) / INDICES_PER_HASH; // inclusive
    constexpr unsigned NUM_BLOCKS   = (MAX_HASH_IDX / 16) + 1;
    constexpr unsigned SG = 16;
    const size_t global = (size_t)NUM_BLOCKS * SG;
    return q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      h.parallel_for<ZelHashGenTestKernel<false>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(SG)),
        [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
        const unsigned lane = (unsigned)it.get_local_id(0);
#else
        const sycl::sub_group sg = it.get_sub_group();
        const unsigned lane = sg.get_local_linear_id();          // 0..15 within the block
#endif
        const uint32_t block = (uint32_t)it.get_group(0);        // which 16-aligned hash-index block
        const uint32_t hash_idx = block * 16u + lane;            // this lane's hash index g

        uint64_t base_h[8];
        for (unsigned i = 0; i < 8; ++i) base_h[i] = d_base_h[i];
        uint8_t pending[12];
        for (unsigned i = 0; i < 12; ++i) pending[i] = d_base_pending[i];

        // This lane's own stock hash (16 u32 lanes), then an inclusive prefix-sum across the 16 lanes.
        uint32_t mine[16];
        stock_hash_words(base_h, pending, hash_idx, mine);
        uint32_t acc[16];
        for (unsigned i = 0; i < 16; ++i) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
          // Portable OpenCL may split this 16-item work-group into smaller sub-groups (the
          // pocl CPU runtime uses eight lanes under QEMU). The ZelHash twist is defined over
          // the whole 16-hash block, so use the work-group collective in the portable image.
          acc[i] = sycl::inclusive_scan_over_group(
            it.get_group(), mine[i], sycl::plus<uint32_t>());
#else
          acc[i] = sycl::inclusive_scan_over_group(sg, mine[i], sycl::plus<uint32_t>());
#endif
        }

        // acc now holds the twist accumulator for hash index `hash_idx` (sum over its 16-block prefix
        // up to and including this lane). Mask bytes 15/31/47/63, then emit the 4 sub-element rows for
        // the (up to) 4 entry indices e = hash_idx*4 + s that fall inside EQUIHASH_TEST_ROWS.
        acc[3]  &= 0xF8FFFFFFu; acc[7]  &= 0xF8FFFFFFu;
        acc[11] &= 0xF8FFFFFFu; acc[15] &= 0xF8FFFFFFu;
        for (unsigned s = 0; s < INDICES_PER_HASH; ++s) {
          const uint32_t e = hash_idx * INDICES_PER_HASH + s;
          if (e >= EQUIHASH_TEST_ROWS) continue;
          uint8_t sub[SUB_ELEMENT_BYTES];
          const unsigned base_word = s * (SUB_ELEMENT_BYTES / 4);
          for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
            const uint32_t v = acc[base_word + w];
            sub[4 * w]     = (uint8_t)(v & 0xFF);
            sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
            sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
            sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
          }
          uint8_t row[HASH_LENGTH];
          expand_array_25(sub, row);
          for (unsigned i = 0; i < HASH_LENGTH; ++i) d_rows[(size_t)e * HASH_LENGTH + i] = row[i];
        }
      });
    });
  }
}

// ===========================================================================================
// M2/M3 solver kernels: gen-fill (round-0 bucketing) + collision rounds + candidate collection.
// ===========================================================================================
//
// A level-L slot lives at level[L] + ((bucket * slot_capacity + slot) * level_u32(L, gpu_compact)).
// At L>=1, w0 carries active_low plus the parent bucket. Parent slots are reconstructed from that
// bucket after the final filter. Level 0 has no parent metadata; its dense record carries the leaf index.
// nslots[L*NBUCKETS + bucket] counts occupancy.

using dev_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device, sycl::access::address_space::global_space>;
using slm_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                        sycl::memory_scope::work_group, sycl::access::address_space::local_space>;

inline uint32_t zelhash_scan16(const sycl::sub_group& sg, uint32_t x) {
#if defined(__NVPTX__)
  // SYCL's generic segmented scan expands into a much larger NVPTX shuffle network. ZelHash always
  // resets at a 16-hash boundary, so use PTX's native 16-lane clamp and let one warp carry two
  // independent prefixes. This is the CUDA analogue of Intel's native 16-wide sub-group scan.
  const unsigned lane = sg.get_local_linear_id() & 15u;
  uint32_t prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 1, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 1u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 2, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 2u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 4, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 4u) x += prior;
  asm volatile("shfl.sync.up.b32 %0, %1, 8, 0x1000, 0xffffffff;" : "=r"(prior) : "r"(x));
  if (lane >= 8u) x += prior;
  return x;
#elif defined(MOM_SYCL_HAS_HIP)
  // AMD wavefronts are wider than each independent 16-hash ZelHash prefix. Use four native
  // subgroup permutes with an explicit 16-lane boundary so a 256-thread work-group can keep whole
  // wavefronts occupied instead of launching one half-wave work-group per hash block.
  const unsigned lane = (unsigned)sg.get_local_linear_id();
  const unsigned team_lane = lane & 15u;
  for (unsigned delta = 1u; delta <= 8u; delta <<= 1u) {
    const unsigned source = team_lane >= delta ? lane - delta : lane;
    const uint32_t prior = mom_select_from_group(sg, x, source);
    if (team_lane >= delta) x += prior;
  }
  return x;
#else
  return sycl::inclusive_scan_over_group(sg, x, sycl::plus<uint32_t>());
#endif
}

#if defined(__NVPTX__)
inline void zelhash_scan16_words(const sycl::sub_group& sg, const uint32_t mine[16], uint32_t acc[16]) {
  const unsigned lane = sg.get_local_linear_id() & 15u;
  uint32_t prior[16];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 1, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(mine[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) acc[i] = mine[i] + (lane >= 1u ? prior[i] : 0u);
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 2, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 2u) acc[i] += prior[i];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 4, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 4u) acc[i] += prior[i];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i)
    asm volatile("shfl.sync.up.b32 %0, %1, 8, 0x1000, 0xffffffff;" : "=r"(prior[i]) : "r"(acc[i]));
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) if (lane >= 8u) acc[i] += prior[i];
}
#endif

// ---- Gen-fill: produce all 2^26 entries and scatter each into its round-0 bucket. One sub-group of
// 16 lanes per 16-aligned hash-index block (the validated M1 scan), then each lane emits its 4
// sub-element entries. field_bucket() maps field-0 to the physical bucket; the recovery index is
// packed into each common record or written to CUDA's separate cold stream.
template <bool GPU_COMPACT> class ZelHashGenFillKernelT;
template <bool GPU_COMPACT = false>
static sycl::event submit_gen_fill(
  sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_base_h, const uint8_t* d_base_pending,
  uint32_t* d_level0, uint32_t* d_l0_index, uint32_t* d_nslots, unsigned slot_capacity
) {
  constexpr uint32_t NUM_HASHES = (uint32_t)(NUM_ENTRIES / INDICES_PER_HASH);   // 2^24
#if defined(MOM_SYCL_HAS_HIP)
  constexpr unsigned WG = 256u;
#elif defined(MOM_SYCL_HAS_CUDA)
  // The CUDA-selected template uses full blocks for occupancy; Intel launches the false template
  // with its measured 16-wide work-group. AdaptiveCpp/CUDA retains 16 because its SSCP path cannot
  // carry the inline PTX specialization above.
  constexpr unsigned WG = GPU_COMPACT ? 256u : 16u;
#else
  constexpr unsigned WG = 16u;
#endif
  const size_t global = NUM_HASHES;
  // The 76-byte Blake base is identical for every work-item. Capture it as plain kernel arguments
  // so all SYCL backends can use their constant/uniform path instead of issuing the same USM loads
  // from every lane. This is a small, measured cross-vendor win and keeps the generator common.
  BaseState base_capture{};
  std::memcpy(base_capture.h, d_base_h, sizeof(base_capture.h));
  std::memcpy(base_capture.pending, d_base_pending, sizeof(base_capture.pending));
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashGenFillKernelT<GPU_COMPACT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) MOM_REQD_SG_16 {
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
      const sycl::sub_group sg = it.get_sub_group();
#endif
      const uint32_t hash_idx = (uint32_t)it.get_global_linear_id();

      // Pair-form base state + the two non-zero message words (block = pending[0..11] || htole32(g)).
      // These are loop-invariant across the 16-block; only g (m1.hi) varies per lane.
      B2bPair base_hp[8];
      for (unsigned i = 0; i < 8; ++i)
        base_hp[i] = B2bPair{ (uint32_t)(base_capture.h[i] & 0xffffffffull),
                              (uint32_t)(base_capture.h[i] >> 32) };
      const B2bPair m0{ b2b_load32le(base_capture.pending + 0),
                        b2b_load32le(base_capture.pending + 4) };
      const B2bPair m1{ b2b_load32le(base_capture.pending + 8), 0u };
      // m1.hi = g is set inside stock_hash_words_pair.

      uint32_t mine[16];
      stock_hash_words_pair(base_hp, m0, m1, hash_idx, mine);
      uint32_t acc[16];
#if defined(__NVPTX__)
      zelhash_scan16_words(sg, mine, acc);
#elif defined(MOM_SYCL_PORTABLE_OPENCL)
      // Unlike the tuned GPU images, a portable OpenCL device is not required to expose
      // 16-lane sub-groups. WG is exactly 16 here, making the work-group scan both portable
      // and equivalent to ZelHash's required 16-hash prefix.
      for (unsigned i = 0; i < 16; ++i)
        acc[i] = sycl::inclusive_scan_over_group(
          it.get_group(), mine[i], sycl::plus<uint32_t>());
#else
      for (unsigned i = 0; i < 16; ++i)
        acc[i] = zelhash_scan16(sg, mine[i]);
#endif
      acc[3] &= 0xF8FFFFFFu; acc[7] &= 0xF8FFFFFFu; acc[11] &= 0xF8FFFFFFu; acc[15] &= 0xF8FFFFFFu;

#if defined(MOM_EQ_FORWARD_MAP)
      uint32_t l0_forward[INDICES_PER_HASH] = {
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu
      };
#endif
      for (unsigned s = 0; s < INDICES_PER_HASH; ++s) {
        const uint32_t e = hash_idx * INDICES_PER_HASH + s;
        const unsigned base_word = s * (SUB_ELEMENT_BYTES / 4);
        uint32_t f[NLEVELS];
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
        // AdaptiveCpp lowers this byte expansion faster than its fixed shift network. DPC++/NVPTX
        // instead uses the direct four-word slicer below; all later solver stages remain common.
        uint8_t sub[SUB_ELEMENT_BYTES];
        for (unsigned w = 0; w < SUB_ELEMENT_BYTES / 4; ++w) {
          const uint32_t v = acc[base_word + w];
          sub[4 * w]     = (uint8_t)(v & 0xFF);
          sub[4 * w + 1] = (uint8_t)((v >> 8) & 0xFF);
          sub[4 * w + 2] = (uint8_t)((v >> 16) & 0xFF);
          sub[4 * w + 3] = (uint8_t)((v >> 24) & 0xFF);
        }
        uint8_t row[HASH_LENGTH];
        expand_array_25(sub, row);
        row_to_fields(row, f);
#else
        words_to_fields_25(acc + base_word, f);
#endif

        const uint32_t bucket = field_bucket(f[0]);
        const uint32_t slot = dev_atomic_u32(d_nslots[bucket]).fetch_add(1u);
#if defined(MOM_EQ_FORWARD_MAP)
        if constexpr (GPU_COMPACT)
          l0_forward[s] = slot < slot_capacity ? (bucket << 14) | slot : 0xffffffffu;
#endif
        if (slot >= slot_capacity) continue;                         // drop on overflow
        size_t pos = (size_t)bucket * slot_capacity + slot;
#if defined(MOM_SYCL_HAS_CUDA)
        if constexpr (GPU_COMPACT) pos = cuda_record_pos(bucket, slot);
#endif
        uint32_t* rec = d_level0 + pos * level_u32(0, GPU_COMPACT);
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
        if constexpr (GPU_COMPACT) {
          dense_l0_store_compact(rec, f);
#if !defined(MOM_EQ_FORWARD_MAP)
          d_l0_index[pos] = e;
#endif
        } else
#endif
        {
          dense_l0_store(rec, f, e);
        }
      }
#if defined(MOM_EQ_FORWARD_MAP)
      if constexpr (GPU_COMPACT)
        store_aligned_u32x4(d_l0_index + (size_t)hash_idx * INDICES_PER_HASH,
          l0_forward[0], l0_forward[1], l0_forward[2], l0_forward[3]);
#endif
    });
  });
}

// ---- Collision round R: read level R, collide field R across (bucket, RESTBITS) pairs, XOR the
// carried fields -> write level R+1. One work-group per bucket; SLM rest-bin table indexes by the low
// RESTBITS of field R. For each colliding pair we XOR fields R+1..4, write a survivor at level R+1,
// map its new active field through field_bucket(), and retain its parent bucket for recovery.
// The final round does not materialize level 4: only pairs whose last field XORs to zero can be
// solutions, so it records their two level-3 nodes directly for the rare recovery pass.
