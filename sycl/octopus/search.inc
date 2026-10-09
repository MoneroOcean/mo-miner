// DAG construction and search submissions, including capability-gated matrix backends.
class OctopusDagKernel;
template <bool FULL_DAG> class OctopusScalarSearchKernel;
template <bool FULL_DAG> class OctopusBatchedSearchKernel;
class OctopusCooperativeSearchKernel;

// Materialize the full DAG from the epoch cache; padded work-items stop before the dataset bound.
static sycl::event build_dag(sycl::queue& queue, const uint32_t* const cache,
                             const uint32_t cache_nodes,
                             uint32_t* const dag, const uint32_t start, const uint32_t count) {
  constexpr unsigned WORKGROUP = 128;
  const FastModData cache_mod = make_fast_mod_data(cache_nodes);
  const size_t global = (static_cast<size_t>(count) + WORKGROUP - 1) / WORKGROUP * WORKGROUP;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusDagKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WORKGROUP)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t local = static_cast<uint32_t>(item.get_global_id(0));
          if (local >= count)
            return;
          const uint32_t index = start + local;
          uint32_t node[CACHE_NODE_WORDS];
          const uint32_t cache_index = fast_mod_dev(index, cache_mod);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            node[word] = cache[cache_index * CACHE_NODE_WORDS + word];
          node[0] ^= index;
          keccak512_words(node);
          for (uint32_t i = 0; i < DATASET_PARENTS; ++i) {
            const uint32_t parent = fast_mod_dev(fnv32(index ^ i, node[i & 15]), cache_mod);
            for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
              node[word] = fnv32(node[word], cache[parent * CACHE_NODE_WORDS + word]);
          }
          keccak512_words(node);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            dag[static_cast<size_t>(index) * CACHE_NODE_WORDS + word] = node[word];
        });
  });
}

template <bool FULL_DAG>
static sycl::event
search_scalar(sycl::queue& queue, const uint8_t* const header, const uint32_t* const cache,
              const FastModData cache_mod, const uint32_t* const dag, const FastModData page_mod,
              const uint64_t start, const uint32_t count, const uint8_t* const target,
              Result* const result, const bool is_test) {
  const size_t local = POLYNOMIAL_LANES;
  const size_t global = static_cast<size_t>(count) * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(sycl::range<1>(POLYNOMIAL_SIZE), handler);
    sycl::local_accessor<uint32_t, 1> polynomial(sycl::range<1>(POLYNOMIAL_LANES), handler);
    sycl::local_accessor<uint32_t, 1> parameters(sycl::range<1>(5), handler);
    sycl::local_accessor<uint32_t, 1> seed(sycl::range<1>(CACHE_NODE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> mix(sycl::range<1>(PAGE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> dataset_words(sycl::range<1>(PAGE_WORDS), handler);
    sycl::local_accessor<uint32_t, 1> page(sycl::range<1>(1), handler);
    sycl::local_accessor<uint32_t, 1> compressed(sycl::range<1>(8), handler);
    handler.parallel_for<OctopusScalarSearchKernel<FULL_DAG>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t index = static_cast<uint32_t>(item.get_group(0));
          octopus_hash_scalar<FULL_DAG>(
              item, header, start + index, cache, cache_mod, dag, page_mod,
              index < count && (!is_test || index == 0), is_test, target, result, coefficients,
              polynomial, parameters, seed, mix, dataset_words, page, compressed);
        });
  });
}

template <bool FULL_DAG>
static sycl::event
search_batched(sycl::queue& queue, const uint8_t* const header, const uint32_t* const cache,
               const FastModData cache_mod, const uint32_t* const dag, const FastModData page_mod,
               const uint32_t* const points, const uint64_t start, const uint32_t count,
               const uint8_t* const target, Result* const result, const bool is_test) {
  constexpr size_t local = POLYNOMIAL_LANES * 4;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(POLYNOMIAL_LANES) - 1);
  const uint64_t covered = (start - aligned_start) + count;
  const size_t groups = static_cast<size_t>((covered + local - 1) / local);
  const size_t global = groups * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(sycl::range<1>(POLYNOMIAL_SIZE * 4), handler);
    handler.parallel_for<OctopusBatchedSearchKernel<FULL_DAG>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          octopus_hash_batched<FULL_DAG>(item, header, start, cache, cache_mod, dag, page_mod,
                                         count, target, result, is_test, coefficients, points);
        });
  });
}

#if !defined(MOM_SYCL_PORTABLE_OPENCL)
// Submit the full-DAG subgroup route; portable OpenCL excludes its subgroup-specific operations.
static sycl::event search_cooperative(sycl::queue& queue, const uint8_t* const header,
                                      const uint32_t* const dag,
                                      const FastModData page_mod, const uint32_t* const points,
                                      const uint64_t start, const uint32_t count,
                                      const uint8_t* const target, Result* const result,
                                      const bool is_test) {
  constexpr size_t local = POLYNOMIAL_LANES * COOPERATIVE_WARPS;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(POLYNOMIAL_LANES) - 1);
  const uint64_t covered = (start - aligned_start) + count;
  const size_t global = static_cast<size_t>((covered + local - 1) / local) * local;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<uint32_t, 1> coefficients(
        sycl::range<1>(POLYNOMIAL_SIZE * COOPERATIVE_WARPS), handler);
    handler.parallel_for<OctopusCooperativeSearchKernel>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
          octopus_hash_cooperative(item, header, start, dag, page_mod, count, target, result,
                                   is_test, coefficients, points);
        });
  });
}
#endif

#if defined(MOM_OCTOPUS_HAS_SYCL_NATIVE)
#if defined(MOM_SYCL_HAS_HIP)
#include "amd_wmma.inc"
#endif
#include "nvidia_tensor.inc"
#endif
