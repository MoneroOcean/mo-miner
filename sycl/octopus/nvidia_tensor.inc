namespace {

#if defined(MOM_SYCL_HAS_HIP)
constexpr unsigned NativeGroups = 4096;
constexpr unsigned NativeParallel = 1;
#elif defined(OCTOPUS_ESIMD)
constexpr unsigned NativeGroups = 65536;
constexpr unsigned NativeParallel = 2;
#else
// A larger batch exposes enough independent GEMM tiles without adding split-K reduction state.
constexpr unsigned NativeGroups = 1024;
#endif
constexpr unsigned NativeM = NativeGroups;
constexpr unsigned NativeN = 1024;
constexpr unsigned NativeK = 1024;
constexpr unsigned NativeProducts = 6;
using NativeProduct = int32_t;
#if defined(MOM_SYCL_HAS_HIP) || defined(OCTOPUS_ESIMD)
// Two queues leave more gfx12 execution capacity for the dominant finish stage; three and four
// slots both lost throughput despite exposing more GEMM/finish overlap.
// B580 retains its three-slot throughput with two slots while using one-third less scratch.
constexpr unsigned NativeSlots = 2;
#else
constexpr unsigned NativeSlots = 3;
#endif
constexpr unsigned NativeSubgroup = 32;
constexpr unsigned NativeWorkgroup = 128;
constexpr unsigned NativeFinishSubgroup = NativeSubgroup;
constexpr unsigned NativeFinishPolynomial = POLYNOMIAL_SIZE * NativeFinishSubgroup / NativeSubgroup;
constexpr unsigned NativeFinishSubgroups = NativeWorkgroup / NativeFinishSubgroup;
#if defined(MOM_SYCL_HAS_HIP) || defined(OCTOPUS_ESIMD)
constexpr unsigned NativeTile = 64;
constexpr unsigned NativeBlock = 128;
#else
constexpr unsigned NativeTile = 32;
constexpr unsigned NativeBlock = 64;
#endif
#if defined(OCTOPUS_ESIMD)
namespace octopus_esimd = sycl::ext::intel::esimd;
#endif

class OctopusSyclNativeMakeA;
class OctopusSyclNativeMakeB;
class OctopusSyclNativeGemm;
#if defined(OCTOPUS_ESIMD)
template <unsigned Width> class OctopusSyclNativeGemmIntel;
#endif
class OctopusSyclNativeFinish;

inline void native_split_field(const uint32_t value, int8_t out[NativeProducts]) {
  int8_t limb[3];
  uint32_t remaining = value;
#pragma unroll
  for (unsigned i = 0; i < 3; ++i) {
    int digit = static_cast<int>(remaining & 127U);
    if (digit >= 64)
      digit -= 128;
    limb[i] = static_cast<int8_t>(digit);
    remaining = static_cast<uint32_t>((static_cast<int>(remaining) - digit) / 128);
  }
  out[0] = limb[0];
  out[1] = limb[1];
  out[2] = limb[2];
  out[3] = static_cast<int8_t>(limb[0] + limb[1]);
  out[4] = static_cast<int8_t>(limb[0] + limb[2]);
  out[5] = static_cast<int8_t>(limb[1] + limb[2]);
}

inline int64_t native_weight(const unsigned product) {
  constexpr int64_t b = 128, b2 = b * b, b3 = b2 * b, b4 = b2 * b2;
  switch (product) {
  case 0:
    return 1 - b - b2;
  case 1:
    return -b + b2 - b3;
  case 2:
    return -b2 - b3 + b4;
  case 3:
    return b;
  case 4:
    return b2;
  default:
    return b3;
  }
}

#if defined(OCTOPUS_ESIMD)
constexpr unsigned NativeModBits = 20;
constexpr unsigned NativeModFoldShift = 14;
constexpr uint64_t NativeModMask = (uint64_t{1} << NativeModBits) - 1;
constexpr uint64_t NativeModFoldFactor = (uint64_t{1} << NativeModFoldShift) - 1;
static_assert(MOD == (uint64_t{1} << NativeModBits) - NativeModFoldFactor);
constexpr uint64_t NativeMaxTotal =
    static_cast<uint64_t>(NativeK) * (static_cast<uint64_t>(MOD) - 1) *
    (static_cast<uint64_t>(MOD) - 1);
static_assert(NativeMaxTotal < (uint64_t{1} << 50));

constexpr uint64_t native_max_after_fold(const uint64_t max) {
  const uint64_t quotient = max >> NativeModBits;
  const uint64_t remainder = max & NativeModMask;
  const uint64_t last_full =
      quotient == 0 ? remainder : (quotient - 1) * NativeModFoldFactor + NativeModMask;
  const uint64_t tail = quotient * NativeModFoldFactor + remainder;
  return last_full > tail ? last_full : tail;
}

constexpr uint64_t NativeFold1Bound = native_max_after_fold(NativeMaxTotal);
constexpr uint64_t NativeFold2Bound = native_max_after_fold(NativeFold1Bound);
constexpr uint64_t NativeFold3Bound = native_max_after_fold(NativeFold2Bound);
constexpr uint64_t NativeFold4Bound = native_max_after_fold(NativeFold3Bound);
constexpr uint64_t NativeFold5Bound = native_max_after_fold(NativeFold4Bound);
static_assert(NativeFold3Bound < (uint64_t{1} << 32));
static_assert(NativeFold5Bound < static_cast<uint64_t>(MOD) * 2);

// MOD = 2^20 - (2^14 - 1). Five shift/subtract folds normalize every legal
// 1024-term dot product without vector division; switching to 32-bit after the
// third fold also halves Intel's GEMM-to-finish scratch traffic.
inline octopus_esimd::simd<int32_t, 128> native_reduce(
    const octopus_esimd::simd<int64_t, 128> total) {
  octopus_esimd::simd<uint64_t, 128> value = octopus_esimd::convert<uint64_t>(total);
#pragma unroll
  for (unsigned fold = 0; fold < 3; ++fold) {
    const octopus_esimd::simd<uint64_t, 128> high = value >> NativeModBits;
    value = (value & NativeModMask) + ((high << NativeModFoldShift) - high);
  }
  octopus_esimd::simd<uint32_t, 128> reduced = octopus_esimd::convert<uint32_t>(value);
#pragma unroll
  for (unsigned fold = 0; fold < 2; ++fold) {
    const octopus_esimd::simd<uint32_t, 128> high = reduced >> NativeModBits;
    reduced = (reduced & static_cast<uint32_t>(NativeModMask)) +
              ((high << NativeModFoldShift) - high);
  }
  const octopus_esimd::simd<uint32_t, 128> mod = MOD;
  reduced.merge(reduced - mod, reduced >= mod);
  return octopus_esimd::convert<int32_t>(reduced);
}
#endif

inline uint32_t native_field_value(const NativeProduct* const products, const unsigned group,
                                   const unsigned column) {
  const size_t index = static_cast<size_t>(group) * NativeN + column;
  return static_cast<uint32_t>(products[index]);
}

static sycl::event native_make_a(sycl::queue& queue, const uint8_t* const header,
                                 const uint64_t start, int8_t* const digits) {
  const size_t global = static_cast<size_t>(NativeGroups / 4) * NativeWorkgroup;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusSyclNativeMakeA>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(NativeWorkgroup)),
        [header, start, digits](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned local = static_cast<unsigned>(item.get_local_id(0));
          const unsigned lane = local & (NativeSubgroup - 1);
          const unsigned group =
              static_cast<unsigned>(item.get_group(0)) * 4 + local / NativeSubgroup;
          SipHash sip{load64(header), load64(header + 8), load64(header + 16), load64(header + 24)};
          sip.hash24((start & ~31ULL) + static_cast<uint64_t>(group) * 32ULL + lane);
          for (unsigned row = 0; row < 32; ++row) {
            sip.round();
            int8_t value[NativeProducts];
            native_split_field(static_cast<uint32_t>(sip.value()) % MOD, value);
#if !defined(OCTOPUS_ESIMD)
            const size_t index = static_cast<size_t>(group) * NativeK + row * NativeSubgroup + lane;
#endif
#pragma unroll
            for (unsigned product = 0; product < NativeProducts; ++product) {
#if defined(OCTOPUS_ESIMD)
              const size_t tiled = static_cast<size_t>(group / 8) * (NativeK / 32) * 256 +
                                   static_cast<size_t>(row) * 256 + (group & 7) * 32 + lane;
              digits[static_cast<size_t>(product) * NativeGroups * NativeK + tiled] =
                  value[product];
#else
              digits[static_cast<size_t>(product) * NativeGroups * NativeK + index] =
                  value[product];
#endif
            }
          }
        });
  });
}

static sycl::event native_make_b(sycl::queue& queue, const uint32_t* const points,
                                 int8_t* const digits) {
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusSyclNativeMakeB>(
        sycl::nd_range<1>(sycl::range<1>(NativeN), sycl::range<1>(NativeWorkgroup)),
        [points, digits](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned column = static_cast<unsigned>(item.get_global_id(0));
          uint64_t power = 1;
          const uint32_t x = points[column];
          for (unsigned k = 0; k < NativeK; ++k) {
            int8_t value[NativeProducts];
            native_split_field(static_cast<uint32_t>(power), value);
#if !defined(OCTOPUS_ESIMD)
            const size_t index = static_cast<size_t>(column) * NativeK + k;
#endif
#pragma unroll
            for (unsigned product = 0; product < NativeProducts; ++product) {
#if defined(OCTOPUS_ESIMD)
              const size_t tiled = static_cast<size_t>(column / 16) * NativeK * 16 +
                                   static_cast<size_t>(k / 4) * 64 + (column & 15) * 4 + (k & 3);
              digits[static_cast<size_t>(product) * NativeK * NativeN + tiled] = value[product];
#else
              digits[static_cast<size_t>(product) * NativeK * NativeN + index] = value[product];
#endif
            }
            power = power * x % MOD;
          }
        });
  });
}

struct NativeGemmContext {
#if defined(OCTOPUS_ESIMD)
  unsigned dpas_width = 0;
#endif
#if defined(MOM_SYCL_HAS_HIP)
  OctopusAmdWmma wmma;
  bool logged = false;

  bool ensure(sycl::queue& queue) {
    return wmma.ensure(queue, NativeM);
  }
#elif defined(MOM_SYCL_HAS_CUDA)
  bool checked = false;
  bool supported = false;

  bool ensure(sycl::queue& queue) {
    if (!mom_is_cuda(queue.get_device()))
      return true;
    if (!checked) {
      supported = mom::nvidia::has_int8_async_matrix(
          mom::nvidia::compute_capability(queue.get_device()));
      checked = true;
    }
    return supported;
  }
#else
  bool ensure(sycl::queue&) {
    return true;
  }
#endif
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
inline void octopus_cuda_mma(int32_t accumulators[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(accumulators[0]), "+r"(accumulators[1]), "+r"(accumulators[2]),
                 "+r"(accumulators[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

inline uint32_t octopus_cuda_shared_address(const void* const pointer) {
  uint32_t address;
  asm("{ .reg .u64 shared; cvta.to.shared.u64 shared, %1; cvt.u32.u64 %0, shared; }"
      : "=r"(address)
      : "l"(pointer));
  return address;
}

inline void octopus_cuda_copy16(const uint32_t shared, const int8_t* const global) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" : : "r"(shared), "l"(global));
}

inline void octopus_cuda_copy_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <unsigned Pending> inline void octopus_cuda_copy_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(Pending));
}

inline void octopus_cuda_load_matrix_a(uint32_t values[4], const uint32_t shared) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
               : "r"(shared));
}

inline void octopus_cuda_load_matrix_b(uint32_t values[2], const uint32_t shared) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(values[0]), "=r"(values[1])
               : "r"(shared));
}

inline void octopus_cuda_load4(uint32_t values[4], const uint32_t* const source) {
  asm volatile("ld.global.L1::evict_first.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
               : "l"(source)
               : "memory");
}

inline void octopus_cuda_store2(int32_t* const destination, const int32_t first,
                                const int32_t second) {
  asm volatile("st.global.v2.s32 [%0], {%1,%2};\n"
               :
               : "l"(destination), "r"(first), "r"(second)
               : "memory");
}

inline unsigned octopus_cuda_swizzle(const unsigned offset) {
  return offset ^ ((offset & 0x380U) >> 3);
}
#endif

#if defined(OCTOPUS_ESIMD)
template <unsigned Width> struct NativeIntelGemm {
  const int8_t* a;
  const int8_t* b;
  NativeProduct* products;
#if defined(MOM_OCTOPUS_INTEL_LARGE_GRF)
  // 32-byte-register Intel width-8 tiles use two row fragments; 64-byte-register tiles retain
  // four.
  static constexpr unsigned Fragments = Width == 8 ? 2 : 4;
#else
  static constexpr unsigned Fragments = 2;
#endif
  static constexpr unsigned Rows = 8 * Fragments;

  MOM_SYCL_KERNEL_ARGS_RESTRICT
  SYCL_ESIMD_KERNEL void operator()(sycl::id<1> id) const {
    constexpr size_t tiles_n = NativeN / 16;
    const size_t tile = id[0];
    const unsigned row = static_cast<unsigned>(tile / tiles_n) * Rows;
    const unsigned column = static_cast<unsigned>(tile % tiles_n) * 16;
    octopus_esimd::simd<int64_t, 128> totals[Fragments];
#pragma unroll
    for (unsigned fragment = 0; fragment < Fragments; ++fragment)
      totals[fragment] = 0;
    for (unsigned product = 0; product < NativeProducts; ++product) {
      octopus_esimd::simd<int32_t, 128> accumulators[Fragments];
#pragma unroll
      for (unsigned fragment = 0; fragment < Fragments; ++fragment)
        accumulators[fragment] = 0;
      const int8_t* const ap = a + product * NativeM * NativeK;
      const int8_t* const bp = b + product * NativeK * NativeN;
      const size_t a_origin = static_cast<size_t>(row / 8) * (NativeK / 32) * 256;
      const size_t b_origin = static_cast<size_t>(column / 16) * NativeK * 16;
      octopus_esimd::simd<int8_t, 256> current_a[Fragments];
#pragma unroll
      for (unsigned fragment = 0; fragment < Fragments; ++fragment)
        current_a[fragment] = octopus_esimd::block_load<int8_t, 256>(
            ap + a_origin + static_cast<size_t>(fragment) * (NativeK / 32) * 256);
      octopus_esimd::simd<int8_t, 512> current_b =
          octopus_esimd::block_load<int8_t, 512>(bp + b_origin);
      // Load the next bounded K tile before using this tile's matrix accumulators.
      // The final tile is consumed after the loop, with no speculative out-of-range read.
      for (unsigned k = 32; k < NativeK; k += 32) {
        const octopus_esimd::simd<int8_t, 512> next_b = octopus_esimd::block_load<int8_t, 512>(
            bp + b_origin + static_cast<size_t>(k / 4) * 64);
        octopus_esimd::simd<int8_t, 256> next_a[Fragments];
#pragma unroll
        for (unsigned fragment = 0; fragment < Fragments; ++fragment)
          next_a[fragment] = octopus_esimd::block_load<int8_t, 256>(
              ap + a_origin + static_cast<size_t>(fragment) * (NativeK / 32) * 256 +
              static_cast<size_t>(k / 32) * 256);
#pragma unroll
        for (unsigned fragment = 0; fragment < Fragments; ++fragment)
          accumulators[fragment] = mom::intel_matrix::dpas_8x16x32<Width>(
              accumulators[fragment], current_b, current_a[fragment]);
        current_b = next_b;
#pragma unroll
        for (unsigned fragment = 0; fragment < Fragments; ++fragment)
          current_a[fragment] = next_a[fragment];
      }
#pragma unroll
      for (unsigned fragment = 0; fragment < Fragments; ++fragment)
        accumulators[fragment] = mom::intel_matrix::dpas_8x16x32<Width>(
            accumulators[fragment], current_b, current_a[fragment]);
#pragma unroll
      for (unsigned fragment = 0; fragment < Fragments; ++fragment)
        totals[fragment] +=
            octopus_esimd::convert<int64_t>(accumulators[fragment]) * native_weight(product);
    }
    int32_t* const out = products + static_cast<size_t>(row) * NativeN + column;
#pragma unroll
    for (unsigned fragment = 0; fragment < Fragments; ++fragment) {
      octopus_esimd::simd<int32_t, 128> output = native_reduce(totals[fragment]);
#pragma unroll
      for (unsigned r = 0; r < 8; ++r)
        octopus_esimd::block_store<int32_t, 16>(
            out + static_cast<size_t>(fragment * 8 + r) * NativeN,
            output.template select<16, 1>(r * 16));
    }
  }

#if defined(MOM_OCTOPUS_INTEL_LARGE_GRF)
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>};
  }
#endif
};

template <unsigned Width>
static sycl::event native_gemm_intel(sycl::queue& queue, const int8_t* a, const int8_t* b,
                                    NativeProduct* products) {
  constexpr size_t tiles_m = NativeM / NativeIntelGemm<Width>::Rows, tiles_n = NativeN / 16;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusSyclNativeGemmIntel<Width>>(
        sycl::range<1>(tiles_m * tiles_n), NativeIntelGemm<Width>{a, b, products});
  });
}
#endif

static sycl::event native_gemm(sycl::queue& queue, NativeGemmContext& context,
                               const int8_t* const a, const int8_t* const b,
                               NativeProduct* const products) {
#if defined(MOM_SYCL_HAS_HIP)
  return octopus_amd_gemm(queue, context.wmma, a, b, products, context.logged);
#elif defined(OCTOPUS_ESIMD)
  context.dpas_width = mom::intel_matrix::dpas_width(queue.get_device());
  if (context.dpas_width == 8)
    return native_gemm_intel<8>(queue, a, b, products);
  return native_gemm_intel<16>(queue, a, b, products);
#else
  (void)context;
  constexpr unsigned stage_k = 64;
  constexpr unsigned stages = 3;
  constexpr unsigned chunks = NativeK / stage_k;
  constexpr size_t stage_elements = NativeBlock * stage_k;
  constexpr unsigned stage_copies = stage_elements / 16;
  constexpr size_t blocks_m = NativeM / NativeBlock;
  constexpr size_t blocks_n = NativeN / NativeBlock;
  const size_t global = blocks_m * blocks_n * NativeWorkgroup;
  // A 64x64 tile doubles operand reads but keeps all six weighted products in registers,
  // replacing six product buffers and their weighted rereads with one normalized buffer.
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<int8_t, 1> a_stages(sycl::range<1>(stages * stage_elements), handler);
    sycl::local_accessor<int8_t, 1> b_stages(sycl::range<1>(stages * stage_elements), handler);
    handler.parallel_for<OctopusSyclNativeGemm>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(NativeWorkgroup)),
        [a, b, products, a_stages, b_stages](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(32)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
              const unsigned local = static_cast<unsigned>(item.get_local_id(0));
              const unsigned lane = local & (NativeSubgroup - 1);
              const unsigned subgroup = local / NativeSubgroup;
              const unsigned block = static_cast<unsigned>(item.get_group(0));
              const unsigned block_row = block / (NativeN / NativeBlock);
              const unsigned block_column = block % (NativeN / NativeBlock);
              const unsigned row_base = block_row * NativeBlock;
              const unsigned column_base = block_column * NativeBlock;
              const unsigned warp_row = (subgroup / 2) * NativeTile;
              const unsigned warp_column = (subgroup & 1) * NativeTile;
              int8_t* const a_shared = a_stages.get_multi_ptr<sycl::access::decorated::no>().get();
              int8_t* const b_shared = b_stages.get_multi_ptr<sycl::access::decorated::no>().get();

              auto load_stage = [&](const unsigned buffer, const unsigned chunk,
                                    const int8_t* const a_source, const int8_t* const b_source) {
                int8_t* const a_destination = a_shared + buffer * stage_elements;
                int8_t* const b_destination = b_shared + buffer * stage_elements;
                const unsigned chunk_offset = chunk * stage_k;
                for (unsigned copy = local; copy < stage_copies; copy += NativeWorkgroup) {
                  const unsigned byte = copy * 16;
                  const unsigned line = byte / stage_k;
                  const unsigned k = byte % stage_k;
                  const unsigned shared = octopus_cuda_swizzle(line * stage_k + k);
                  octopus_cuda_copy16(octopus_cuda_shared_address(a_destination + shared),
                                      a_source + static_cast<size_t>(row_base + line) * NativeK +
                                          chunk_offset + k);
                  octopus_cuda_copy16(octopus_cuda_shared_address(b_destination + shared),
                                      b_source + static_cast<size_t>(column_base + line) * NativeK +
                                          chunk_offset + k);
                }
                octopus_cuda_copy_commit();
              };

              int64_t totals[2][4][4]{};
#pragma unroll 1
              for (unsigned product = 0; product < NativeProducts; ++product) {
                const int8_t* const a_source = a + static_cast<size_t>(product) * NativeM * NativeK;
                const int8_t* const b_source = b + static_cast<size_t>(product) * NativeK * NativeN;
                int32_t accumulators[2][4][4]{};
                load_stage(0, 0, a_source, b_source);
                load_stage(1, 1, a_source, b_source);
#pragma unroll 1
                for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                  const bool has_next = chunk + 2 < chunks;
                  if (has_next)
                    load_stage((chunk + 2) % stages, chunk + 2, a_source, b_source);
                  if (has_next)
                    octopus_cuda_copy_wait<2>();
                  else if (chunk + 1 < chunks)
                    octopus_cuda_copy_wait<1>();
                  else
                    octopus_cuda_copy_wait<0>();
                  sycl::group_barrier(item.get_group());

                  const int8_t* const a_stage = a_shared + (chunk % stages) * stage_elements;
                  const int8_t* const b_stage = b_shared + (chunk % stages) * stage_elements;

#pragma unroll
                  for (unsigned k = 0; k < stage_k; k += 32) {
                    uint32_t a_fragments[2][4];
                    uint32_t b_fragments[4][2];
#pragma unroll
                    for (unsigned row = 0; row < 2; ++row) {
                      // ldmatrix.x4 takes one row address from each eight-lane matrix group. The four
                      // matrices are A's top-left, bottom-left, top-right, and bottom-right quadrants.
                      const unsigned matrix = lane >> 3;
                      const unsigned logical =
                          (warp_row + row * 16 + (lane & 7U) + (matrix & 1U) * 8) * stage_k + k +
                          (matrix >> 1) * 16;
                      octopus_cuda_load_matrix_a(
                          a_fragments[row],
                          octopus_cuda_shared_address(a_stage + octopus_cuda_swizzle(logical)));
                    }
#pragma unroll
                    for (unsigned column = 0; column < 4; ++column) {
                      // x2 consumes rows from lanes 0..15; mirror those addresses into the upper lanes
                      // so every supported PTX target sees valid pointers.
                      const unsigned address_lane = lane & 15U;
                      const unsigned logical =
                          (warp_column + column * 8 + (address_lane & 7U)) * stage_k + k +
                          (address_lane >> 3) * 16;
                      octopus_cuda_load_matrix_b(
                          b_fragments[column],
                          octopus_cuda_shared_address(b_stage + octopus_cuda_swizzle(logical)));
                    }
#pragma unroll
                    for (unsigned column = 0; column < 4; ++column) {
#pragma unroll
                      for (unsigned row = 0; row < 2; ++row)
                        octopus_cuda_mma(accumulators[row][column], a_fragments[row],
                                         b_fragments[column]);
                    }
                  }
                  sycl::group_barrier(item.get_group());
                }

                const int64_t weight = native_weight(product);
#pragma unroll
                for (unsigned row = 0; row < 2; ++row) {
#pragma unroll
                  for (unsigned column = 0; column < 4; ++column) {
#pragma unroll
                    for (unsigned value = 0; value < 4; ++value)
                      totals[row][column][value] +=
                          static_cast<int64_t>(accumulators[row][column][value]) * weight;
                  }
                }
                sycl::group_barrier(item.get_group());
              }

#pragma unroll
              for (unsigned row = 0; row < 2; ++row) {
#pragma unroll
                for (unsigned column = 0; column < 4; ++column) {
                  int32_t normalized[4];
#pragma unroll
                  for (unsigned value = 0; value < 4; ++value) {
                    int64_t field_value = totals[row][column][value] % static_cast<int64_t>(MOD);
                    if (field_value < 0)
                      field_value += MOD;
                    normalized[value] = static_cast<int32_t>(field_value);
                  }
                  int32_t* const destination =
                      products +
                      static_cast<size_t>(row_base + warp_row + row * 16 + (lane >> 2)) * NativeN +
                      column_base + warp_column + column * 8 + (lane & 3U) * 2;
                  octopus_cuda_store2(destination, normalized[0], normalized[1]);
                  octopus_cuda_store2(destination + 8 * NativeN, normalized[2], normalized[3]);
                }
              }
#else
              (void)item;
              (void)a;
              (void)b;
              (void)products;
              (void)a_stages;
              (void)b_stages;
#endif
            });
  });
#endif
}

#if !defined(OCTOPUS_ESIMD)
static sycl::event native_finish(sycl::queue& queue, const uint8_t* const header,
                                 const uint32_t* const dag, const FastModData page_mod,
                                 const NativeProduct* const products, const uint64_t start,
                                 const uint32_t count, const uint8_t* const target,
                                 Result* const result, const bool is_test) {
  const size_t global = static_cast<size_t>(NativeGroups / 4) * NativeWorkgroup;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> polynomial(
        sycl::range<1>(NativeFinishSubgroups * NativeFinishPolynomial), handler);
    const sycl::nd_range<1> range{sycl::range<1>(global), sycl::range<1>(NativeWorkgroup)};
    auto finish_kernel = [header, dag, page_mod, products, start, count, target, result, is_test,
                          polynomial](sycl::nd_item<1> item)
        [[sycl::reqd_sub_group_size(NativeFinishSubgroup)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned local = static_cast<unsigned>(item.get_local_id(0));
          const unsigned lane = local & (NativeFinishSubgroup - 1);
          const unsigned subgroup_index = local / NativeFinishSubgroup;
          const unsigned nonce_index =
              static_cast<unsigned>(item.get_group(0)) * NativeWorkgroup + local;
          const unsigned group = nonce_index / NativeSubgroup;
          const unsigned group_lane = nonce_index & (NativeSubgroup - 1);
          const uint64_t nonce = (start & ~31ULL) + nonce_index;
          const size_t polynomial_base =
              static_cast<size_t>(subgroup_index) * NativeFinishPolynomial;
          uint64_t polynomial_hash = 0;
#pragma unroll
          for (unsigned point = 0; point < NativeSubgroup; ++point) {
            const uint32_t value =
                native_field_value(products, group, group_lane + point * NativeSubgroup);
            polynomial[polynomial_base + point * NativeFinishSubgroup + lane] = value;
            polynomial_hash = fnv64(polynomial_hash, value);
          }
          const sycl::sub_group subgroup = item.get_sub_group();
          sycl::group_barrier(subgroup);

          uint32_t seed[CACHE_NODE_WORDS];
          keccak512_header_seed(header, polynomial_hash, seed);
          sycl::group_barrier(subgroup);

#if defined(OCTOPUS_ESIMD) || defined(MOM_SYCL_HAS_HIP)
          const unsigned team_base = lane & ~15U;
          const unsigned team_lane = lane & 15U;
          const unsigned mix_part = team_lane & 3U;
          uint32_t final_mix[8]{};
          for (unsigned first = 0; first < 16; first += NativeParallel) {
            uint32_t mix[NativeParallel][4];
            uint32_t initial[NativeParallel];
            for (unsigned parallel = 0; parallel < NativeParallel; ++parallel) {
              const unsigned source = team_base + first + parallel;
              initial[parallel] = mom_select_from_group(subgroup, seed[0], source);
#pragma unroll
              for (unsigned part = 0; part < 4; ++part) {
#pragma unroll
                for (unsigned word = 0; word < 4; ++word) {
                  const uint32_t selected =
                      mom_select_from_group(subgroup, seed[part * 4 + word], source);
                  if (mix_part == part)
                    mix[parallel][word] = selected;
                }
              }
            }

            for (unsigned access = 0; access < ACCESSES; access += 4) {
              const unsigned address_lane = team_base + access / 4;
              for (unsigned step = 0; step < 4; ++step) {
#pragma unroll
                for (unsigned parallel = 0; parallel < NativeParallel; ++parallel) {
                  const unsigned hash_lane = team_base + first + parallel;
                  const uint32_t polynomial_value =
                      polynomial[polynomial_base + (access + step) * NativeFinishSubgroup +
                                 hash_lane];
                  uint32_t page =
                      fast_mod_dev(fnv32(initial[parallel] ^ (access + step) ^ polynomial_value,
                                         mix[parallel][step]),
                                   page_mod);
                  page = mom_select_from_group(subgroup, page, address_lane);
                  const size_t dag_offset =
                      static_cast<size_t>(page) * PAGE_WORDS + static_cast<size_t>(team_lane) * 4;
#pragma unroll
                  for (unsigned word = 0; word < 4; ++word)
                    mix[parallel][word] = fnv32(mix[parallel][word], dag[dag_offset + word]);
                }
              }
            }

#pragma unroll
            for (unsigned parallel = 0; parallel < NativeParallel; ++parallel) {
              uint32_t reduced = fnv32(mix[parallel][0], mix[parallel][1]);
              reduced = fnv32(reduced, mix[parallel][2]);
              reduced = fnv32(reduced, mix[parallel][3]);
              const unsigned hash_lane = team_base + first + parallel;
#pragma unroll
              for (unsigned word = 0; word < 8; ++word) {
                const uint32_t compressed =
                    fnv32(mom_select_from_group(subgroup, reduced, team_base + word),
                          mom_select_from_group(subgroup, reduced, team_base + 8 + word));
                if (lane == hash_lane)
                  final_mix[word] = compressed;
              }
            }
          }
#else
          // NVIDIA's four-lane teams keep eight independent DAG pages in flight per warp while
          // only one lane computes each page index. The old 16-lane layout repeated that arithmetic
          // for four hashes on every lane. Reuse the polynomial scratch for the compressed mix so
          // the larger per-lane page slice does not extend another array across the DAG loop.
          constexpr unsigned DagTeam = 4;
          constexpr unsigned HashesPerBatch = NativeFinishSubgroup / DagTeam;
          constexpr unsigned MixWordsPerLane = PAGE_WORDS / DagTeam;
          const unsigned team_base = lane & ~(DagTeam - 1);
          const unsigned team_lane = lane & (DagTeam - 1);
          const unsigned team_index = lane / DagTeam;
          for (unsigned first = 0; first < NativeFinishSubgroup; first += HashesPerBatch) {
            const unsigned source = first + team_index;
            const uint32_t initial = mom_select_from_group(subgroup, seed[0], source);
            uint32_t mix[MixWordsPerLane];
#pragma unroll
            for (unsigned word = 0; word < MixWordsPerLane; ++word)
              mix[word] = mom_select_from_group(subgroup, seed[word], source);

            for (unsigned access = 0; access < ACCESSES; ++access) {
              const uint32_t address_mix = mom_select_from_group(
                  subgroup, mix[access & (MixWordsPerLane - 1)],
                  team_base + access / MixWordsPerLane);
              uint32_t page = 0;
              if (team_lane == 0) {
                const uint32_t polynomial_value =
                    polynomial[polynomial_base + access * NativeFinishSubgroup + source];
                page = fast_mod_dev(fnv32(initial ^ access ^ polynomial_value, address_mix),
                                    page_mod);
              }
              page = mom_select_from_group(subgroup, page, team_base);
              const size_t dag_offset = static_cast<size_t>(page) * PAGE_WORDS +
                                        static_cast<size_t>(team_lane) * MixWordsPerLane;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
              // DAG offsets advance in 16-byte units. Vector loads avoid scalar lowering, and the
              // one-pass random pages are marked evict-first instead of displacing reusable L1 data.
#pragma unroll
              for (unsigned first_word = 0; first_word < MixWordsPerLane; first_word += 4) {
                uint32_t values[4];
                octopus_cuda_load4(values, dag + dag_offset + first_word);
#pragma unroll
                for (unsigned word = 0; word < 4; ++word)
                  mix[first_word + word] = fnv32(mix[first_word + word], values[word]);
              }
#else
#pragma unroll
              for (unsigned word = 0; word < MixWordsPerLane; ++word)
                mix[word] = fnv32(mix[word], dag[dag_offset + word]);
#endif
            }

            uint32_t reduced[MixWordsPerLane / 4];
#pragma unroll
            for (unsigned word = 0; word < MixWordsPerLane / 4; ++word)
              reduced[word] = fnv32(fnv32(fnv32(mix[word * 4], mix[word * 4 + 1]),
                                          mix[word * 4 + 2]),
                                    mix[word * 4 + 3]);
#pragma unroll
            for (unsigned word = 0; word < MixWordsPerLane / 4; ++word) {
              const unsigned half = team_lane & 1U;
              const uint32_t left =
                  mom_select_from_group(subgroup, reduced[word], team_base + half);
              const uint32_t right =
                  mom_select_from_group(subgroup, reduced[word], team_base + half + 2);
              if (team_lane < 2) {
                const unsigned compressed_word = half * (MixWordsPerLane / 4) + word;
                polynomial[polynomial_base + compressed_word * NativeFinishSubgroup + source] =
                    fnv32(left, right);
              }
            }
          }
          sycl::group_barrier(subgroup);
          uint32_t final_mix[8];
#pragma unroll
          for (unsigned word = 0; word < 8; ++word)
            final_mix[word] =
                polynomial[polynomial_base + word * NativeFinishSubgroup + lane];
#endif

          uint8_t output[HASH_LEN];
          keccak256_seed_mix(seed, final_mix, output);
          const bool active = nonce >= start && nonce - start < count;
          if (active && (is_test ? nonce == start : meets_target(output, target)))
            save_result(result, nonce, output);
        };
    handler.parallel_for<OctopusSyclNativeFinish>(range, finish_kernel);
  });
}
#endif

#include "esimd_finish.inc"

} // namespace

class OctopusSyclNativeSearch {
  struct Slot {
    int8_t* a = nullptr;
    NativeProduct* products = nullptr;
    std::unique_ptr<sycl::queue> queue;
  };

  int8_t* b_ = nullptr;
  std::array<Slot, NativeSlots> slots_{};
  NativeGemmContext gemm_;
  sycl::queue* base_queue_ = nullptr;
  uint64_t points_version_ = ~uint64_t(0);
  bool ready_ = false;
  bool native_failed_ = false;
#if defined(OCTOPUS_ESIMD)
  const bool profile_ = [] {
    const char* value = std::getenv("MOM_OCTOPUS_PROFILE");
    return value && *value && std::strcmp(value, "0") != 0;
  }();
  unsigned profile_reports_ = 0;
#endif

  bool ensure(sycl::queue& base) {
    if (ready_)
      return true;
    base_queue_ = &base;
    try {
      if (!gemm_.ensure(base))
        return false;
      for (Slot& slot : slots_) {
#if defined(OCTOPUS_ESIMD)
        if (profile_) {
          slot.queue = std::make_unique<sycl::queue>(
              base.get_context(), base.get_device(),
              sycl::property_list{sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{}});
          continue;
        }
#endif
        slot.queue =
            std::make_unique<sycl::queue>(base.get_context(), base.get_device(),
                                          sycl::property_list{sycl::property::queue::in_order{}});
      }
      b_ = sycl::malloc_device<int8_t>(static_cast<size_t>(NativeProducts) * NativeK * NativeN,
                                       base);
      if (!b_)
        return false;
      for (Slot& slot : slots_) {
        slot.a = sycl::malloc_device<int8_t>(
            static_cast<size_t>(NativeProducts) * NativeM * NativeK, base);
        slot.products = sycl::aligned_alloc_device<NativeProduct>(
            8, static_cast<size_t>(NativeM) * NativeN, base);
        if (!slot.a || !slot.products)
          return false;
      }
      ready_ = true;
      return true;
    } catch (...) {
      return false;
    }
  }

  bool prepare_b(sycl::queue& base, const uint32_t* const points, const uint64_t points_version) {
    if (points_version_ == points_version)
      return true;
    sycl_wait_and_throw(native_make_b(base, points, b_), base.get_device());
    points_version_ = points_version;
    return true;
  }

public:
  ~OctopusSyclNativeSearch() {
    release();
  }

  void release() noexcept {
    for (Slot& slot : slots_)
      if (slot.queue)
        try {
          slot.queue->wait_and_throw();
        } catch (...) {
        }
    if (base_queue_)
      try {
        base_queue_->wait_and_throw();
      } catch (...) {
      }

    if (base_queue_) {
      for (Slot& slot : slots_) {
        if (slot.products)
          try {
            sycl::free(slot.products, *base_queue_);
          } catch (...) {
          }
        if (slot.a)
          try {
            sycl::free(slot.a, *base_queue_);
          } catch (...) {
          }
        slot.products = nullptr;
        slot.a = nullptr;
        slot.queue.reset();
      }
      if (b_)
        try {
          sycl::free(b_, *base_queue_);
        } catch (...) {
        }
    } else {
      for (Slot& slot : slots_) {
        slot.products = nullptr;
        slot.a = nullptr;
        slot.queue.reset();
      }
    }
    b_ = nullptr;
    base_queue_ = nullptr;
    points_version_ = ~uint64_t(0);
    ready_ = false;
  }

  bool search(sycl::queue& base, const uint8_t* const header, const uint32_t* const dag,
              const FastModData page_mod, const uint32_t* const points,
              const uint64_t points_version, const uint64_t start, const uint32_t count,
              const uint8_t* const target, Result* const result, const bool is_test) {
    if (native_failed_)
      return false;
    try {
      if (!ensure(base)) {
        release();
        return false;
      }
      if (!prepare_b(base, points, points_version)) {
        release();
        return false;
      }

      const unsigned search_count = is_test ? 1u : count;
      unsigned offset = 0;
      unsigned chunk_index = 0;
      std::array<sycl::event, NativeSlots> completed{};
#if defined(OCTOPUS_ESIMD)
      const bool collect_profile = profile_ && profile_reports_ < 8;
      std::vector<std::array<sycl::event, 3>> profile_events;
#endif
      while (offset < search_count) {
        const unsigned slot_index = chunk_index % NativeSlots;
        Slot& slot = slots_[slot_index];
        const uint64_t batch_start = start + offset;
        const unsigned capacity =
            NativeGroups * NativeSubgroup - static_cast<unsigned>(batch_start & 31ULL);
        const unsigned chunk = std::min(search_count - offset, capacity);
#if defined(OCTOPUS_ESIMD)
        const sycl::event make_a = native_make_a(*slot.queue, header, batch_start, slot.a);
        const sycl::event gemm = native_gemm(*slot.queue, gemm_, slot.a, b_, slot.products);
        const auto finish = gemm_.dpas_width == 8 ? native_finish_esimd<2> : native_finish_esimd<4>;
        completed[slot_index] =
            finish(*slot.queue, header, dag, page_mod, slot.products, batch_start, chunk,
                   target, result, is_test);
        if (collect_profile)
          profile_events.push_back({make_a, gemm, completed[slot_index]});
#else
        native_make_a(*slot.queue, header, batch_start, slot.a);
        native_gemm(*slot.queue, gemm_, slot.a, b_, slot.products);
        completed[slot_index] =
            native_finish(*slot.queue, header, dag, page_mod, slot.products, batch_start, chunk,
                          target, result, is_test);
#endif
        offset += chunk;
        ++chunk_index;
      }
      const unsigned active_slots = std::min<unsigned>(chunk_index, NativeSlots);
      for (unsigned slot = 0; slot < active_slots; ++slot)
        sycl_wait_and_throw(completed[slot], base.get_device());
#if defined(OCTOPUS_ESIMD)
      if (collect_profile) {
        ++profile_reports_;
        try {
          std::array<double, 3> elapsed{};
          for (const auto& batch : profile_events) {
            for (unsigned stage = 0; stage < elapsed.size(); ++stage) {
              const uint64_t begin =
                  batch[stage].get_profiling_info<sycl::info::event_profiling::command_start>();
              const uint64_t end =
                  batch[stage].get_profiling_info<sycl::info::event_profiling::command_end>();
              elapsed[stage] += static_cast<double>(end - begin) / 1.0e6;
            }
          }
          std::fprintf(stderr,
                       "Octopus ESIMD fused profile make_a_ms=%.3f gemm_ms=%.3f "
                       "finish_ms=%.3f batches=%zu\n",
                       elapsed[0], elapsed[1], elapsed[2], profile_events.size());
        } catch (...) {
          std::fprintf(stderr, "Octopus ESIMD profile unavailable\n");
        }
      }
#endif
      return true;
    } catch (const std::exception& error) {
      native_failed_ = true;
      if (std::getenv("MOM_OCTOPUS_SYCL_NATIVE_LOG"))
        std::fprintf(stderr, "Octopus SYCL-native unavailable: %s\n", error.what());
      release();
      return false;
    } catch (const std::string& error) {
      native_failed_ = true;
      if (std::getenv("MOM_OCTOPUS_SYCL_NATIVE_LOG"))
        std::fprintf(stderr, "Octopus SYCL-native unavailable: %s\n", error.c_str());
      release();
      return false;
    } catch (...) {
      native_failed_ = true;
      if (std::getenv("MOM_OCTOPUS_SYCL_NATIVE_LOG"))
        std::fprintf(stderr, "Octopus SYCL-native unavailable: unknown error\n");
      release();
      return false;
    }
  }
};
