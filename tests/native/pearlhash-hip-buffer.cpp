#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define DEV inline
#define PEARLHASH_ER 12
#define PEARLHASH_EC 4
#ifndef PEARLHASH_K
#define PEARLHASH_K 2048
#endif
using v2i = std::array<uint32_t, 2>;
struct __amdgpu_buffer_rsrc_t {
  const uint8_t* base;
  uint64_t bytes;
  short stride;
  uint32_t flags;
};

static int uniform_reads;
static int __builtin_amdgcn_readfirstlane(int tile) { ++uniform_reads; return tile; }
static __amdgpu_buffer_rsrc_t __builtin_amdgcn_make_buffer_rsrc(
    void* base, short stride, uint64_t bytes, uint32_t flags) {
  return {static_cast<const uint8_t*>(base), bytes, stride, flags};
}
static v2i __builtin_amdgcn_raw_buffer_load_b64(
    __amdgpu_buffer_rsrc_t resource, int offset, int scalar_offset, int cache_flags) {
  if (resource.stride != 0 || resource.flags != 0x31004000u || cache_flags != 0 ||
      offset < 0 || scalar_offset < 0)
    throw std::runtime_error("invalid gfx12 resource or load bounds");
  if (uint64_t(offset) + uint64_t(scalar_offset) + 8 > resource.bytes)
    return {}; // gfx12's descriptor bounds mask an incomplete final wave group.
  v2i out;
  std::memcpy(out.data(), resource.base + offset + scalar_offset, sizeof(out));
  return out;
}

#include "input-resource.inc"
#include "pack-offset.inc"

static v2i load_a(__amdgpu_buffer_rsrc_t a_resource, int kk, int lane, int i) {
  const int a_tiles = static_cast<int>(a_resource.bytes / (uint64_t(PEARLHASH_K) * 16));
  v2i a[PEARLHASH_ER];
#include "load-a.inc"
  return a[i];
}
static v2i load_b(__amdgpu_buffer_rsrc_t b_resource, int kk, int lane, int j) {
  const int b_tiles = static_cast<int>(b_resource.bytes / (uint64_t(PEARLHASH_K) * 16));
  v2i b[PEARLHASH_EC];
#include "load-b.inc"
  return b[j];
}

static uint8_t canonical_byte(const uint64_t row, const uint64_t depth) {
  return static_cast<uint8_t>((row * 37u + depth * 101u + (depth >> 8) * 13u +
                               (row >> 4) * 7u) & 255u);
}

static uint32_t canonical_word(const uint64_t row, const uint64_t depth) {
  uint32_t word = 0;
  for (int byte = 0; byte < 4; ++byte)
    word |= static_cast<uint32_t>(canonical_byte(row, depth + static_cast<uint64_t>(byte))) <<
        (byte * 8);
  return word;
}

static void check_layout(const int k) {
  const size_t tile_bytes = static_cast<size_t>(k) * 16;
  std::vector<uint8_t> seen(tile_bytes, 0);
  size_t covered = 0;
  for (int row = 0; row < 16; ++row) {
    for (int depth = 0; depth < k; ++depth) {
      const size_t offset = amd_wmma_offset(static_cast<size_t>(row),
                                            static_cast<size_t>(depth), k);
      if (offset >= tile_bytes || seen[offset])
        throw std::runtime_error("WMMA pack offset is out of range or aliases a byte");
      seen[offset] = 1;
      ++covered;
      if ((depth & 3) == 0 && (offset & 3) != 0)
        throw std::runtime_error("four-byte A producer word is not aligned");
    }
  }
  if (covered != tile_bytes)
    throw std::runtime_error("WMMA pack offset does not cover one complete tile");
  for (const uint8_t byte : seen)
    if (!byte)
      throw std::runtime_error("WMMA pack offset left an uncovered byte");

  // Check the exact two-phase lane expression independently of the packed writes. This catches
  // a helper that is bijective but does not match the wave64 b64 load's lane/phase decomposition.
  for (int kk = 0; kk < k; kk += 32) {
    for (int lane = 0; lane < 64; ++lane) {
      for (int phase = 0; phase < 2; ++phase) {
        for (int byte = 0; byte < 4; ++byte) {
          const int depth = kk + (lane >> 5) * 8 + ((lane >> 4) & 1) * 4 + phase * 16 + byte;
          const size_t helper_offset = amd_wmma_offset(static_cast<size_t>(lane & 15),
                                                       static_cast<size_t>(depth), k);
          const size_t load_offset = static_cast<size_t>(kk / 32) * 512 +
              static_cast<size_t>(lane) * 8 + static_cast<size_t>(phase) * 4 + byte;
          if (helper_offset != load_offset)
            throw std::runtime_error("WMMA helper disagrees with wave64 two-phase load mapping");
        }
      }
    }
  }
}

static std::vector<int8_t> pack_input(const int k, const int tiles) {
  const size_t tile_bytes = static_cast<size_t>(k) * 16;
  std::vector<int8_t> input(tile_bytes * static_cast<size_t>(tiles));
  for (int tile = 0; tile < tiles; ++tile) {
    for (int row = 0; row < 16; ++row) {
      const uint64_t global_row = static_cast<uint64_t>(tile * 16 + row);
      for (int depth = 0; depth < k; depth += 4) {
        const uint32_t word = canonical_word(global_row, static_cast<uint64_t>(depth));
        const size_t offset = static_cast<size_t>(tile) * tile_bytes +
            amd_wmma_offset(static_cast<size_t>(row), static_cast<size_t>(depth), k);
        if ((offset & 3) != 0 ||
            offset + sizeof(word) > static_cast<size_t>(tile + 1) * tile_bytes)
          throw std::runtime_error("packed A word is not contained in its tile");
        std::memcpy(input.data() + offset, &word, sizeof(word));
      }
    }
  }
  return input;
}

static v2i expected_load(const int first, const int group, const int kk, const int lane) {
  const uint64_t row = static_cast<uint64_t>(first + group) * 16 +
      static_cast<uint64_t>(lane & 15);
  v2i expected{};
  for (int phase = 0; phase < 2; ++phase) {
    const int depth = kk + (lane >> 5) * 8 + ((lane >> 4) & 1) * 4 + phase * 16;
    expected[phase] = canonical_word(row, static_cast<uint64_t>(depth));
  }
  return expected;
}

int main() try {
  constexpr uint64_t tile_bytes = uint64_t(PEARLHASH_K) * 16;
  constexpr int input_tiles = 19;
  check_layout(PEARLHASH_K);
  const std::vector<int8_t> input = pack_input(PEARLHASH_K, input_tiles);
  for (const int first : {3, 7}) {
    const int before = uniform_reads;
    const auto a_resource = pearlhash_input_resource<12>(input.data(), first);
    const auto b_resource = pearlhash_input_resource<4>(input.data(), first);
    if (uniform_reads != before + 4)
      throw std::runtime_error("resource base and extent must both be explicitly wave-uniform");
    const uint8_t* const expected_base = reinterpret_cast<const uint8_t*>(input.data()) +
        first * tile_bytes;
    if (a_resource.base != expected_base || a_resource.bytes != 12 * tile_bytes ||
        b_resource.base != expected_base || b_resource.bytes != 4 * tile_bytes)
      throw std::runtime_error("resource changed its wide base or group extent");
    for (int group = 0; group < 12; ++group) {
      for (int kk = 0; kk < PEARLHASH_K; kk += 32) {
        for (int lane = 0; lane < 64; ++lane) {
          const v2i expected = expected_load(first, group, kk, lane);
          if (load_a(a_resource, kk, lane, group) != expected)
            throw std::runtime_error("raw input differs from the independently reconstructed row/depth load");
        }
      }
    }
    for (int group = 0; group < 4; ++group) {
      for (int kk = 0; kk < PEARLHASH_K; kk += 32) {
        for (int lane = 0; lane < 64; ++lane) {
          if (load_b(b_resource, kk, lane, group) != expected_load(first, group, kk, lane))
            throw std::runtime_error("raw input differs from the independently reconstructed row/depth load");
        }
      }
    }
  }
  for (int available = 1; available <= 12; ++available) {
    const auto resource = pearlhash_input_resource<12>(input.data(), 3, available);
    if (resource.bytes != uint64_t(available) * tile_bytes)
      throw std::runtime_error("A resource extent includes masked tail rows");
    for (int group = available; group < 12; ++group) {
      for (const int kk : {0, PEARLHASH_K - 32}) {
        for (const int lane : {0, 17, 48, 63}) {
          if (load_a(resource, kk, lane, group) != load_a(resource, kk, lane, 0))
            throw std::runtime_error("masked A tail did not reuse an in-bounds input");
        }
      }
    }
  }
  for (int available = 1; available <= 4; ++available) {
    const auto resource = pearlhash_input_resource<4>(input.data(), 3, available);
    if (resource.bytes != uint64_t(available) * tile_bytes)
      throw std::runtime_error("B resource extent includes masked tail rows");
    for (int group = available; group < 4; ++group) {
      for (const int kk : {0, PEARLHASH_K - 32}) {
        for (const int lane : {0, 17, 48, 63}) {
          if (load_b(resource, kk, lane, group) != load_b(resource, kk, lane, 0))
            throw std::runtime_error("masked B tail did not reuse an in-bounds input");
        }
      }
    }
  }
  std::puts("PASS actual-source HIP resource flags, A12/B4 bases, extents, K32 wave64 pack offsets, and independent row/depth loads");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "FAIL %s\n", error.what());
  return 1;
}
