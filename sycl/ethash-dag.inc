// Ethash light-cache and DAG generation shared by Etchash and the ProgPoW family. Include it inside
// the algorithm's namespace after its device code: the DAG kernel uses that algorithm's fnv_dev and
// keccak_512_words_dev (legacy-padded Keccak-512 over one 16-word node).

constexpr uint32_t ETHASH_DAG_NODE_WORDS = 16;

// Fill `cache` with the light cache derived from `seed_epoch`'s seed hash; `failure` names the
// algorithm in the error raised when libethash rejects the size.
inline void compute_light_cache(std::vector<uint32_t>& cache, const uint64_t cache_bytes,
                                const uint32_t seed_epoch, const char* const failure) {
  cache.assign(cache_bytes / sizeof(uint32_t), 0);
  const ethash_h256_t seed = ethash_get_seedhash(seed_epoch);
  if (!ethash_compute_cache_nodes(cache.data(), cache_bytes, &seed))
    throw std::string(failure);
}

// DAG generation workgroup: an `env` override must be a supported size that fits the device.
inline unsigned ethash_dag_workgroup(const sycl::device& dev, const char* const env) {
  const unsigned fallback =
      sycl_default_workgroup(dev, {32, 64, 128, 256, 512}, dev.is_cpu() ? 128 : 64);
  unsigned long parsed = 0;
  if (!mom_parse_env_ulong(env, parsed))
    return fallback;
  switch (parsed) {
  case 32:
  case 64:
  case 128:
  case 256:
  case 512:
    if (sycl_workgroup_fits(dev, static_cast<unsigned>(parsed)))
      return static_cast<unsigned>(parsed);
    break;
  }
  return fallback;
}

// Generate DAG nodes [0, total) from the uploaded light cache, mixing `Parents` cache parents into
// each node. chunk_nodes == 0 submits one dispatch; otherwise each chunk is submitted and retired
// before the next is recorded, so a desktop or WDDM driver never sees one watchdog-length batch.
// Returns once every node is written; submitted faults propagate.
template <uint32_t Parents>
void generate_ethash_dag(sycl::queue& queue, const sycl::device& device,
                         const uint32_t* const d_light, const uint32_t light_nodes,
                         uint32_t* const d_dag, const uint32_t total, const unsigned dag_workgroup,
                         const uint32_t chunk_nodes) {
  const FastModData light_mod = make_fast_mod_data(light_nodes);
  sycl::event dag_event;
  for (uint32_t start_node = 0; start_node < total;) {
    const uint32_t current_nodes = chunk_nodes ? std::min(chunk_nodes, total - start_node) : total;
    const uint32_t chunk_start = start_node;
    dag_event = queue.submit([&](sycl::handler& h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(mom_round_up_size(current_nodes, dag_workgroup)),
                            sycl::range<1>(dag_workgroup)),
          [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
            const uint32_t local_node = item.get_global_id(0);
            if (local_node >= current_nodes)
              return;
            const uint32_t node_index = chunk_start + local_node;

            uint32_t dag_node[ETHASH_DAG_NODE_WORDS];
            const uint32_t init = fast_mod_dev(node_index, light_mod);
#pragma unroll
            for (unsigned w = 0; w < ETHASH_DAG_NODE_WORDS; ++w)
              dag_node[w] = d_light[init * ETHASH_DAG_NODE_WORDS + w];
            dag_node[0] ^= node_index;
            keccak_512_words_dev(dag_node);

            for (uint32_t i = 0; i < Parents; ++i) {
              const uint32_t parent_index = fast_mod_dev(
                  fnv_dev(node_index ^ i, dag_node[i & (ETHASH_DAG_NODE_WORDS - 1)]), light_mod);
              const uint32_t parent_base = parent_index * ETHASH_DAG_NODE_WORDS;
#pragma unroll
              for (unsigned w = 0; w < ETHASH_DAG_NODE_WORDS; ++w)
                dag_node[w] = fnv_dev(dag_node[w], d_light[parent_base + w]);
            }
            keccak_512_words_dev(dag_node);

            const uint32_t base = node_index * ETHASH_DAG_NODE_WORDS;
#pragma unroll
            for (unsigned w = 0; w < ETHASH_DAG_NODE_WORDS; ++w)
              d_dag[base + w] = dag_node[w];
          });
    });
    if (!chunk_nodes)
      break;
    sycl_wait_and_throw(dag_event, device);
    start_node += current_nodes;
  }
  if (!chunk_nodes)
    sycl_wait_and_throw(dag_event, device);
}
