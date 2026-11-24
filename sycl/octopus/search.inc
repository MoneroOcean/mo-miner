// DAG construction and search submissions, including capability-gated matrix backends.
template <bool FULL_DAG> class OctopusScalarSearchKernel;
template <bool FULL_DAG> class OctopusBatchedSearchKernel;
class OctopusCooperativeSearchKernel;

// Keep Octopus's existing chunk/wait boundary while sharing the consensus node transform.
static sycl::event build_dag(sycl::queue& queue, const uint32_t* const cache,
                             const uint32_t cache_nodes,
                             uint32_t* const dag, const uint32_t start, const uint32_t count) {
  const FastModData cache_mod = make_fast_mod_data(cache_nodes);
  return submit_ethash_dag_chunk<DATASET_PARENTS, size_t>(queue, cache, cache_mod, dag, start,
                                                          count, 128);
}

// Submit the one-nonce-per-work-group reference kernel (MOM_OCTOPUS_SCALAR correctness oracle).
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

// Submit the portable batched kernel over the lane-aligned range covering [start, start + count).
template <bool FULL_DAG>
static sycl::event
search_batched(sycl::queue& queue, const uint8_t* const header, const uint32_t* const cache,
               const FastModData cache_mod, const uint32_t* const dag, const FastModData page_mod,
               const uint32_t* const points, const uint64_t start, const uint32_t count,
               const uint8_t* const target, Result* const result, const bool is_test) {
  constexpr size_t local = POLYNOMIAL_LANES * 4;
  const size_t global = mom_round_up_size(start % POLYNOMIAL_LANES + count, local);
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
  const size_t global = mom_round_up_size(start % POLYNOMIAL_LANES + count, local);
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
#include "amd-wmma.inc"
#endif
#include "matrix-search.inc"
#endif
