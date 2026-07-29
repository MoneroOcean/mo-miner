// The environment cleanup hook calls kawpow_cleanup_states() at the correct runtime-lifetime boundary.
static KawpowStateMap& kawpow_states() {
  static KawpowStateMap* const states = new KawpowStateMap;
  return *states;
}

static std::mutex& kawpow_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static KawpowState& kawpow_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(kawpow_states_mutex());
  auto& state = kawpow_states()[dev_str];
  if (!state) state = std::make_unique<KawpowState>(dev_str);
  return *state;
}

void kawpow_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(kawpow_states_mutex());
    kawpow_states().clear();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "kawpow: ordered SYCL cleanup failed: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "kawpow: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_kawpow

using namespace mom_kawpow;

// Per-work-item value-captured copy of the seal mode + 15 magic words (the device lambda cannot
// dereference the host `KawpowVariant::magic` constexpr pointer). FIRO ignores the magic words.
struct SealParams {
  SealMode seal;
  uint32_t magic[15];
};

static int kawpow_core(
  const KawpowVariant& variant,
  const uint32_t block_height, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad kawpow input length");

  const bool loop_stats = kawpow_loop_stats();
  const uint64_t t_enter = loop_stats ? kawpow_now_us() : 0;

  SealParams seal_params{variant.seal, {}};
  for (unsigned i = 0; i < 15; ++i) seal_params.magic[i] = variant.magic[i];

  KawpowState& state = kawpow_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  // Publish this call's seal into the state (under the mutex) so build_period_bundle bakes it into the
  // runtime-JIT source and the bundle cache key reflects it. A device shared across algos can switch seal.
  state.seal = seal_params.seal;
  for (unsigned i = 0; i < 15; ++i) state.magic[i] = seal_params.magic[i];
  // Publish this call's inner-loop shape (32/11/18 for KawPoW/FiroPow/EvrProgPow; 16/6/9 for MeowPow)
  // so make_program(), the bundle cache key, the source-JIT shape bake, and the Intel kernel's runtime
  // template dispatch below all see the active variant's shape.
  state.regs = variant.regs;
  state.cnt_cache = variant.cnt_cache;
  state.cnt_math = variant.cnt_math;
  state.ensure_input(input_size);

  const uint32_t epoch = block_height / variant.epoch_length;
  const uint64_t period = block_height / variant.period_length;
  // FiroPow's full dataset uses chfast/EIP-1057 sizing (1.5 GiB init); KawPoW/EvrProgPow use the
  // classic Ethereum dataset table. The cache sizing is the classic table EXCEPT for MeowPow.
  if (epoch >= 2048) throw std::string("Bad kawpow epoch");
  // MeowPow epoch-110 "dag change" (Meowcoin core create_epoch_context): for epoch >= dagchange_epoch
  // the dataset AND cache SIZES use a scaled epoch (epoch*4), while the keccak SEED keeps the real
  // epoch. size_epoch == epoch for the other variants and for MeowPow below the fork.
  const uint32_t size_epoch =
    (variant.dagchange_epoch && epoch >= variant.dagchange_epoch) ? epoch * 4 : epoch;
  if (size_epoch >= 2048) throw std::string("Bad kawpow epoch");
  const uint64_t dag_bytes = variant.chfast_init_items
    ? chfast_dag_bytes(size_epoch, variant.chfast_init_items, variant.chfast_growth_items)
    : dag_sizes[epoch];
  // Light cache: MeowPow uses chfast cache sizing (scales with the dag change); everyone else uses the
  // classic cache_sizes[] table. The SEED always uses the real epoch (passed separately below).
  const uint64_t cache_bytes = variant.chfast_cache_init_items
    ? chfast_cache_bytes(size_epoch, variant.chfast_cache_init_items, variant.chfast_cache_growth_items)
    : cache_sizes[epoch];
  const uint64_t t_state = loop_stats ? kawpow_now_us() : 0;
  state.ensure_epoch(epoch, cache_bytes, dag_bytes, epoch, !is_benchmark);
  const uint64_t t_epoch = loop_stats ? kawpow_now_us() : 0;
  state.ensure_period(period);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 32, sizeof(start_nonce));
  uint32_t search_intensity = intensity;
  uint32_t global_size = round_up(std::max(search_intensity, state.workgroup), state.workgroup);
  const uint32_t dag_elements = static_cast<uint32_t>(dag_bytes / 256);
  const FastModData dag_mod = make_fast_mod_data(dag_elements);

  if (state.shared_io) std::memcpy(state.input, input, input_size);
  else state.queue.memcpy(state.input, input, input_size);
  std::memset(state.result, 0, sizeof(KawpowResult));

  sycl::queue& q = state.queue;
  const unsigned local_size = state.workgroup;
  uint8_t* const d_input = state.input;
  const DagLoad* const d_dag_load = reinterpret_cast<const DagLoad*>(state.dag);
  KawpowResult* const d_result = state.result;

  // GPU devices use the subgroup (warp-shuffle) exchange; the CPU SYCL device
  // (is_gpu()==false) and an explicit MOM_KAWPOW_EXCHANGE=slm use the
  // barrier/local-memory exchange.
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const char* const exchange_env = std::getenv("MOM_KAWPOW_EXCHANGE");
  bool use_sg = state.default_subgroup_exchange;
  if (exchange_env && std::strcmp(exchange_env, "sg") == 0) use_sg = state.device.is_gpu();
  else if (exchange_env && std::strcmp(exchange_env, "slm") == 0) use_sg = false;
#endif
  // The Intel (spec-const) kernels can't bake REGS/CNT_* at JIT time the way the CUDA path does, so they
  // pick the active inner-loop shape at runtime. There are exactly two ProgPoW shapes: MeowPow's 16/6/9
  // and everyone else's 32/11/18. (variant.regs is published into state.regs above.)
  const bool meowpow_shape = (variant.regs == MEOWPOW_REGS);

#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  if (use_sg) {
    // Sub-group (warp-shuffle) exchange path. KawpowSgExchange is sub-group-size-agnostic (its
    // `& ~(LANES-1)` base runs two 16-lane ProgPoW groups inside a 32-wide warp), so the same kernel
    // runs at sub-group=16 on Intel and natively at warp=32 on CUDA (reqd_sub_group_size dropped here).
    // The per-period ProgPoW program is folded to constants: Intel via SYCL-2020 specialization
    // constants + kernel_bundle; CUDA via a runtime source-compiled kernel (jit.inc).
    const uint64_t t_io = loop_stats ? kawpow_now_us() : 0;
    state.ensure_period_bundle(period, epoch, dag_mod);
    const uint64_t t_bundle = loop_stats ? kawpow_now_us() : 0;
    sycl::event search_event;
#if defined(MOM_SYCL_HAS_CUDA)
    // Must match build_period_bundle's choice: launch the JIT kernel iff that path was selected.
    if (mom_is_cuda(state.device) && state.cuda_use_jit.load() == 1) {
    // Launch the source-JIT'd kernel: program/dag-mod are baked into the bundle, so the only kernel
    // args are the per-call buffers/scalars; work-group local memory comes from work_group_static.
    const std::string jit_kernel_name = kawpow_jit_kernel_name(
      period, dag_mod, state.seal, state.magic,
      state.regs, state.cnt_cache, state.cnt_math);
    sycl::kernel jit_kernel = state.period_bundle->ext_oneapi_get_kernel(jit_kernel_name);
    search_event = q.submit([&](sycl::handler& h) {
      h.set_args(d_input, d_dag_load, d_result,
                 search_intensity, start_nonce, target, static_cast<int>(is_test));
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)), jit_kernel);
    });
    } else
#endif
    {
    search_event = q.submit([&](sycl::handler& h) {
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
      h.use_kernel_bundle(*state.period_bundle);
#endif
#if defined(MOM_SYCL_ADAPTIVECPP)
      // AdaptiveCpp's generic SSCP image stays architecture-neutral, but its JIT can still treat
      // selected kernel captures as compile-time constants. The ProgPoW instruction stream changes
      // only once per period, so specialize it (and the matching fast modulus/shape) to let the
      // AMDGPU backend erase the interpreter exactly as a source-generated kernel would. The
      // specialized value participates in AdaptiveCpp's persistent kernel-cache key.
      const sycl::specialized<KawpowProgram> adaptive_program{state.program};
      const sycl::specialized<FastModData> adaptive_dag_mod{dag_mod};
      const sycl::specialized<bool> adaptive_meowpow_shape{meowpow_shape};
      h.parallel_for<KawpowSgKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) MOM_REQD_SG_16 {
          const KawpowProgram program = adaptive_program;
          const FastModData dag_mod = adaptive_dag_mod;
          const bool specialized_meowpow_shape = adaptive_meowpow_shape;
#elif defined(MOM_SYCL_PORTABLE_OPENCL)
      const KawpowProgram adaptive_program = state.program;
      const FastModData adaptive_dag_mod = dag_mod;
      h.parallel_for<KawpowSgKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) MOM_REQD_SG_16 {
          const KawpowProgram program = adaptive_program;
          const FastModData dag_mod = adaptive_dag_mod;
          const bool specialized_meowpow_shape = meowpow_shape;
#else
      h.parallel_for<KawpowSgKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) MOM_REQD_SG_16 {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const bool specialized_meowpow_shape = meowpow_shape;
#endif
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < search_intensity;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const auto sg = item.get_sub_group();
          const uint32_t lane_id = static_cast<uint32_t>(sg.get_local_id()[0]) & (KAWPOW_LANES - 1);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32),
                             seal_params.seal, seal_params.magic, state2);

          uint32_t digest[8];
          const KawpowSgExchange ex{sg,
            static_cast<uint32_t>(sg.get_local_id()[0]) & ~(KAWPOW_LANES - 1u)};
          // Select the inner-loop shape at runtime (the only two ProgPoW shapes here): MeowPow runs the
          // shorter 16/6/9 sequence, the others the 32/11/18 sequence. The spec-const program already
          // carries the matching active-prefix layout (make_program(period, regs, cnt_cache, cnt_math)).
          if (specialized_meowpow_shape) {
            kawpow_search_dev<KawpowSgExchange, sycl::local_accessor<uint32_t, 1>,
              MEOWPOW_REGS, MEOWPOW_CNT_CACHE, MEOWPOW_CNT_MATH>(
              ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);
          } else {
            kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);
          }

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, seal_params.seal, seal_params.magic, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    });
    }
    if (loop_stats) {
      const uint64_t t_submit = kawpow_now_us();
      const uint64_t pre_us = t_submit - t_enter;
      if (pre_us > 50000) fprintf(stderr,
        "KAWSTALL t=%llu pre=%.1fms (state=%.1f epoch=%.1f period+io=%.1f bundle=%.1f submit=%.1f)\n",
        static_cast<unsigned long long>(time(nullptr)),
        pre_us / 1e3, (t_state - t_enter) / 1e3, (t_epoch - t_state) / 1e3,
        (t_io - t_epoch) / 1e3, (t_bundle - t_io) / 1e3, (t_submit - t_bundle) / 1e3);
    }
    sycl_wait_and_throw(search_event, state.device);
  } else
#endif
  {
    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      const auto share = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size), h);
      const auto offsets = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(local_size / KAWPOW_LANES), h);
      const auto c_dag = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(KAWPOW_CACHE_WORDS), h);
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
      const KawpowProgram adaptive_program = state.program;
      const FastModData adaptive_dag_mod = dag_mod;
      const bool adaptive_cpu_offset_barrier = state.device.is_cpu();
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const KawpowProgram program = adaptive_program;
          const FastModData dag_mod = adaptive_dag_mod;
          const bool cpu_offset_barrier = adaptive_cpu_offset_barrier;
#else
      h.set_specialization_constant<kawpow_program_id>(state.program);
      h.set_specialization_constant<kawpow_dag_mod_id>(dag_mod);
      h.set_specialization_constant<kawpow_cpu_offset_barrier_id>(state.device.is_cpu());
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item, sycl::kernel_handler kh) {
          const KawpowProgram program = kh.get_specialization_constant<kawpow_program_id>();
          const FastModData dag_mod = kh.get_specialization_constant<kawpow_dag_mod_id>();
          const bool cpu_offset_barrier = kh.get_specialization_constant<kawpow_cpu_offset_barrier_id>();
#endif
          const uint32_t lid = item.get_local_id(0);
          const uint32_t gid = item.get_global_id(0);
          const bool active = gid < search_intensity;

          const uint32_t lane_id = lid & (KAWPOW_LANES - 1);
          const uint32_t group_id = lid / KAWPOW_LANES;
          const uint64_t full_nonce = start_nonce + gid;

          for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
               word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
            const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
#pragma unroll
            for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
          }
          item.barrier(sycl::access::fence_space::local_space);

          uint32_t state2[8];
          kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                             static_cast<uint32_t>(full_nonce >> 32),
                             seal_params.seal, seal_params.magic, state2);

          uint32_t digest[8];
          const KawpowSlmExchange ex{
            item, &share[group_id * KAWPOW_LANES], &offsets[group_id], lane_id, cpu_offset_barrier
          };
          // Runtime shape select (see the SG path): MeowPow 16/6/9, others 32/11/18. The spec-const
          // program already carries the matching active-prefix layout.
          if (meowpow_shape) {
            kawpow_search_dev<KawpowSlmExchange, sycl::local_accessor<uint32_t, 1>,
              MEOWPOW_REGS, MEOWPOW_CNT_CACHE, MEOWPOW_CNT_MATH>(
              ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);
          } else {
            kawpow_search_dev(ex, lane_id, state2, c_dag, d_dag_load, program, dag_mod, digest);
          }

          uint32_t final_state[25];
          kawpow_final_dev(state2, digest, seal_params.seal, seal_params.magic, final_state);
          if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
            store_kawpow_result(d_result, full_nonce, final_state, digest);
          }
        }
      );
    }), state.device);
  }

  const uint32_t result_count = std::min(state.result->count, MAX_KAWPOW_OUTPUTS);
  for (uint32_t index = 0; index < result_count; ++index) {
    if (!is_test && !kawpow_meets_target_words(state.result->output[index], target)) continue;

    std::memcpy(output, state.result->output[index], HASH_LEN);
    std::memcpy(mix_hash, state.result->mix_hash[index], HASH_LEN);
    *pnonce = state.result->nonce[index];
    return 1;
  }

  return 0;
}

// ABI entrypoints (gpu_kawpow_hash_fun). The fn-pointer signature does not carry the algo name, so
// each ProgPoW variant is a DISTINCT exported function that bakes its own epoch/period/seal/magic.
int kawpow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(KAWPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}

int firopow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(FIROPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}

int evrprogpow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(EVRPROGPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}

int meowpow(
  const unsigned job_id, const uint32_t block_height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint8_t* const mix_hash, uint64_t* const pnonce, const uint64_t target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  return kawpow_core(MEOWPOW_VARIANT, block_height, input, input_size, output, mix_hash, pnonce,
                     target, intensity, is_test, is_benchmark, dev_str);
}
