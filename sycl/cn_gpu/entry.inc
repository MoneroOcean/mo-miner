// cn_gpu owns the full submit chain so the hot path is visible in one function.
void cn_gpu(const uint8_t* const inputs, const unsigned input_size, uint8_t* const output,
            const unsigned batch, const std::string& dev_str, const std::string& backend) {
  const unsigned batch_eff = cn_gpu_padded_batch(batch);
  if (!inputs || !output || !input_size || !batch_eff || input_size >= HASH_DATA_AREA)
    throw std::string("cn/gpu requires a padded input below 136 bytes and a valid batch");
  CnGpuState& state = cn_gpu_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  // The packed recurrence and AES stages need an even batch. Initialize the extra odd-batch
  // scratchpad below, but never copy an input or read an output for that padding hash.
  state.ensure_capacity(batch_eff, input_size);
  sycl::queue& q = state.queue;
  uint8_t* const d_inputs = state.inputs;
  uint8_t* const d_outputs = state.outputs;
  const size_t input_bytes = static_cast<size_t>(input_size) * batch;
  const size_t output_bytes = static_cast<size_t>(HASH_LEN) * batch;
  const bool use_native_width8 = mom_intel_eu_simd_width(state.device) == 8;

  if constexpr (mom_sycl_portable_opencl) {
    state.buffered_inputs.write(q, inputs, input_bytes);
  } else {
    if (state.shared_io) {
      std::memcpy(d_inputs, inputs, input_bytes);
    } else {
      q.memcpy(d_inputs, inputs, input_bytes);
    }
  }

  // Initial Keccak pads each input into its 200-byte state.
  [[maybe_unused]] const sycl::event initial_keccak = q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    h.parallel_for(
        sycl::nd_range(sycl::range(round_up_size(batch, WORKGROUP_SIZE)),
                       sycl::range(WORKGROUP_SIZE)),
        [=](sycl::nd_item<1> nd) {
          const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4, d_lpads16] =
              cn_views.pointers();
          const auto t = nd.get_global_id(0);
          if (t >= batch) {
            // Odd batches add one recurrence lane. Initialize that lane deterministically; rounded
            // work-items beyond it have no backing allocation.
            if (t < batch_eff) {
              for (unsigned i = 0; i < 25; ++i) {
                d_spads[25 * t + i] = 0;
              }
            }
            return;
          }

          uint8_t spad[200];
          std::memcpy(spad, &d_inputs[input_size * t], input_size);
          spad[input_size] = 1;
          std::memset(spad + input_size + 1, 0, 200 - input_size - 1);
          spad[HASH_DATA_AREA - 1] |= 0x80;
          keccak(reinterpret_cast<uint64_t*>(spad));
          std::memcpy(&d_spads[25 * t], spad, 200);
        });
  });

  // Expand Keccak state into each 2 MiB scratchpad (one thread per 512-byte stripe). Windows
  // NVIDIA uses the source-JIT override only when the policy/user selected native; every other
  // device and an unavailable/failed NVRTC path executes this complete SYCL implementation.
  bool native_expansion = false;
  if constexpr (!mom_sycl_portable_opencl) {
    const bool native_requested = backend == "auto" || backend == "native";
    if (native_requested && mom_is_cuda(state.device)) {
      std::string reason;
      if (state.cuda_expand.ensure(q, &reason)) {
        try {
          sycl_wait_and_throw(initial_keccak, state.device);
          state.cuda_expand.launch(state.spads, state.lpads, batch_eff);
          state.cuda_expand.wait();
          native_expansion = true;
        } catch (const std::string& error) {
          state.cuda_expand.fail(error);
        } catch (const std::exception& error) {
          state.cuda_expand.fail(error.what());
        }
      } else if (!state.native_notice) {
        state.native_notice = true;
        std::fprintf(stderr, "cn/gpu native expansion unavailable, using SYCL: %s\n",
                     reason.c_str());
      }
    }
  }
  if (!native_expansion) {
    const auto submit_expansion = [&]<bool IntelNative64>() {
      q.submit([&](sycl::handler& h) {
        CnGpuActiveViews cn_views(state, h);
        const unsigned stripes = CN_MEMORY8 / 64; // 512-byte stripes per scratchpad
        // Expand the odd-batch padding hash too: its recurrence needs initialized scratch memory.
        const size_t total_threads = static_cast<size_t>(batch_eff) * stripes;
        unsigned expand_wg = WORKGROUP_SIZE;
#if defined(_WIN32)
        // Eight-thread blocks improve Windows CUDA by 7.9%; other paths retain sixteen.
        if (mom_is_cuda(state.device))
          expand_wg = 8;
#endif
        const unsigned wg_size = std::min(expand_wg, stripes);
        h.parallel_for(sycl::nd_range(sycl::range(round_up_size(total_threads, wg_size)),
                                     sycl::range(wg_size)),
                       [=](sycl::nd_item<1> nd) {
                         const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4,
                                     d_lpads16] = cn_views.pointers();
                         const auto t = nd.get_global_id(0);
                         if (t >= total_threads)
                           return;

                         const unsigned stripe = t % stripes;
                         const size_t hash_idx = t / stripes;
                         const uint64_t* const spad = &d_spads[hash_idx * 25];
                         uint64_t* const lpad = &d_lpads[hash_idx * CN_MEMORY8];
                         generate512<IntelNative64>(stripe, spad, &lpad[stripe * 64]);
                       });
      });
    };
    // Width-8 Intel spills the pair32 expansion. Keep the existing Xe2/CUDA/HIP choice elsewhere.
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
    if (use_native_width8)
      submit_expansion.template operator()<true>();
    else
#endif
      submit_expansion.template operator()<false>();
  }

  // Keep the long floating-point recurrence in the common SYCL pipeline on every backend.
  // The Intel OpenCL CPU runtime accepts the requested 16-wide subgroup but does not provide the
  // local-memory ordering this recurrence needs from subgroup-only barriers. Keep the measured
  // narrow barrier path on GPUs; the archive/generic-OpenCL CPU validation path uses the original
  // workgroup barrier. This is a uniform host-captured branch and does not alter GPU arithmetic.
  // A 16-wide requirement makes the width-8 Intel compiler spill this recurrence. Let it use its
  // native width and synchronize the whole workgroup; width-16 Intel and other proven paths keep
  // the narrower subgroup barrier.
  const bool use_narrow_recurrence_barriers =
      !mom_sycl_adaptivecpp && !mom_sycl_portable_opencl && state.device.is_gpu() &&
      !use_native_width8;
  const unsigned recurrence_workgroup =
      cn_gpu_recurrence_group_size(use_native_width8);
  const unsigned hashes_per_group = recurrence_workgroup / WORKGROUP_SIZE;
  q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    // The recurrence uses 16 threads/hash. Native-width-8 Intel uses one hash per workgroup;
    // wider devices pack two independent hashes so a 32-lane warp/wave is full. Every local-memory
    // dependency stays inside its 16-lane slice. The measured DPC++ paths use subgroup
    // barriers instead of paying four workgroup barriers in each of 49,152 rounds (+2.9% NVIDIA,
    // +5.1% AMD and +3.2% Intel in counterbalanced tests). AdaptiveCpp deliberately retains the
    // workgroup barriers: its generic/SSCP lowering was neutral on Windows AMD and 0.6-0.9% slower
    // on NVIDIA, so keeping the proven fallback avoids a compiler-specific regression.
    const auto vi0 = sycl::local_accessor<sycl::int4, 1>(sycl::range(recurrence_workgroup), h);
    const auto vi1 = sycl::local_accessor<sycl::int4, 1>(sycl::range(recurrence_workgroup), h);
    const auto vf0 = sycl::local_accessor<sycl::float4, 1>(sycl::range(recurrence_workgroup), h);
    const auto recurrence_range =
        sycl::nd_range(sycl::range(static_cast<size_t>(batch_eff) * WORKGROUP_SIZE),
                       sycl::range(recurrence_workgroup));
    const auto recurrence_kernel = [=](sycl::nd_item<1> nd) {
      const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4, d_lpads16] =
          cn_views.pointers();
      const unsigned lid = nd.get_local_id();
      const unsigned sub = lid >> 4;     // independent hash within the workgroup
      const unsigned l = lid & 15U;      // 0..15 lane role within the hash
      const unsigned base16 = sub * 16U; // int4 slice base for this hash
      const unsigned base32 = sub * 64U; // int32/float slice base for this hash
      const unsigned ld = l / 4;
      const unsigned lm = l % 4;
      const size_t group_hash = nd.get_group().get_group_id() * hashes_per_group + sub;
      const uint32_t* const spad = &d_spads4[group_hash * (25 * 2)];
      int32_t* const lpad = &d_lpads4[cn_gpu_scratchpad_word_offset(group_hash)];
      uint32_t s = *spad >> 8;
      sycl::float4 sf(0.0f);
      sycl::int4* const vi = &vi0[base16];
      int32_t* const vi4 = reinterpret_cast<int32_t*>(&vi0[0]) + base32;
      sycl::int4* const vo = &vi1[base16];
      int32_t* const vo4 = reinterpret_cast<int32_t*>(&vi1[0]) + base32;
      sycl::float4* const vf = &vf0[base16];
      float* const vf4 = reinterpret_cast<float*>(&vf0[0]) + base32;
      for (unsigned i = 0; i < CN_GPU_ITER; ++i) {
        const int32_t xi = lpad_ptr(s, ld, lpad)[lm];
        vi4[l] = xi;

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        single_comupte_wrap(vi[cn_lane_select(l, 0)], vi[cn_lane_select(l, 1)],
                            vi[cn_lane_select(l, 2)], vi[cn_lane_select(l, 3)], lm,
                            cn_lane_count(l), sf, vf + l, vo + l);

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        // The seed reads finished at the preceding barrier. Reuse that buffer for reductions
        // instead of overwriting vo/vf values another subgroup may not have read yet. This keeps
        // the existing allocation and four barriers; subgroup width no longer supplies ordering.
        const int32_t xo = cn_gpu_reduce_int(vo4, l);
        lpad_ptr(s, ld, lpad)[lm] = xo ^ xi;
        vi4[l] = xo;
        vi4[16 + l] = std::bit_cast<int32_t>(cn_gpu_reduce_float(vf4, l));

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        const float xf = sycl::fabs(cn_gpu_reduce_staged_float(vf4, vi4, l));
        vo4[l] = cn_gpu_reduce_staged_int(vo4, vi4, l) ^
                 static_cast<int32_t>(xf * 16777216.0f);
        vf4[l] = xf * 0.015625f;

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        sf = vf[0];
        s = vo[0][0] ^ vo[0][1] ^ vo[0][2] ^ vo[0][3];
      }
    };
    if (use_native_width8)
      h.parallel_for(recurrence_range, [=](sycl::nd_item<1> nd) { recurrence_kernel(nd); });
    else
      h.parallel_for(recurrence_range,
                     [=](sycl::nd_item<1> nd) MOM_REQD_SG_16 { recurrence_kernel(nd); });
  });

  // AES uses eight lanes per hash; width-8 Intel need not emulate a wider subgroup. Odd batches
  // already have one padding hash. Use two rows when that even count is not divisible by four,
  // so every backend receives a uniform nd_range without another allocation.
  const unsigned aes_wg = cn_gpu_aes_group_rows(batch_eff);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const bool use_gpu_aes = !mom_sycl_portable_opencl && state.device.is_gpu();
  if (use_gpu_aes) { // GPU path replaces the AES local-memory ring with subgroup shuffles.
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const AesTable aes0(sycl::range(256), h), aes1(sycl::range(256), h),
          aes2(sycl::range(256), h), aes3(sycl::range(256), h);
      const auto aes_range = sycl::nd_range(sycl::range(batch_eff, 8), sycl::range(aes_wg, 8));
      const auto aes_kernel = [=](sycl::nd_item<2> nd) {
        const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4, d_lpads16] =
            cn_views.pointers();
        const unsigned l1 = nd.get_local_id(1);
        aes_fill_tables(aes0, aes1, aes2, aes3, nd);

        nd.barrier(sycl::access::fence_space::local_space);

        const auto spad = d_spads4 + (nd.get_global_id(0) * (25 * 2));
        const sycl::uint4* const lpad = &d_lpads16[nd.get_global_id(0) * CN_MEMORY16];
        sycl::uint4 x;
        mo_vec_load(x, l1 + 4, spad);
        alignas(32) AesKey key;
        mo_vec_load(key.u8[0], 1, spad);
        aes_expend_key(key.u);
        const auto sg = nd.get_sub_group();
        const unsigned row_lane_base = sg.get_local_linear_id() & ~7u;

        // Subgroup shuffle replaces the old local-memory neighbor exchange on GPU.
        auto next_row_value = [&](const sycl::uint4 value) {
          const sycl::uint4 shifted = mom_shift_group_left(sg, value, 1);
          const sycl::uint4 wrapped = mom_select_from_group(sg, value, row_lane_base);
          return l1 == 7 ? wrapped : shifted;
        };

        sycl::uint4 x2s(0);

        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16 / 8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
          x ^= lpad[i1];
          const sycl::uint4 x2l = next_row_value(x2s);
          x ^= x2l;
          for (unsigned j = 0; j < 10; ++j)
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          x ^= lpad[i1 + 8];
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
          for (unsigned j = 0; j < 10; ++j)
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x2s = x;
        }

        const sycl::uint4 x2l = next_row_value(x2s);
        x ^= x2l;

        for (unsigned i = 0; i < 16; ++i) {
          for (unsigned j = 0; j < 10; ++j)
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
        }
        mo_vec_store(x, l1 + 4, spad);
      };
      if (use_native_width8)
        h.parallel_for(aes_range, [=](sycl::nd_item<2> nd) { aes_kernel(nd); });
      else
        h.parallel_for(aes_range,
                       [=](sycl::nd_item<2> nd) MOM_REQD_SG_16 { aes_kernel(nd); });
    });
  } else
#endif
  { // CPU/portable path keeps the barriered local-memory exchange for stable behavior.
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const AesTable aes0(sycl::range(256), h), aes1(sycl::range(256), h),
          aes2(sycl::range(256), h), aes3(sycl::range(256), h);
      const auto x1 = sycl::local_accessor<sycl::uint4, 2>(sycl::range(aes_wg, 8), h);
      const auto x2 = sycl::local_accessor<sycl::uint4, 2>(sycl::range(aes_wg, 8), h);
      h.parallel_for(
          sycl::nd_range(sycl::range(batch_eff, 8), sycl::range(aes_wg, 8)),
          [=](sycl::nd_item<2> nd) {
            const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4, d_lpads16] =
                cn_views.pointers();
            const unsigned l0 = nd.get_local_id(0), l1 = nd.get_local_id(1);
            aes_fill_tables(aes0, aes1, aes2, aes3, nd);

            nd.barrier(sycl::access::fence_space::local_space);

            const auto spad = d_spads4 + (nd.get_global_id(0) * (25 * 2));
            const sycl::uint4* const lpad = &d_lpads16[nd.get_global_id(0) * CN_MEMORY16];
            sycl::uint4 x;
            mo_vec_load(x, l1 + 4, spad);
            alignas(32) AesKey key;
            mo_vec_load(key.u8[0], 1, spad);
            aes_expend_key(key.u);
            sycl::uint4 &x1s = x1[l0][l1], &x1l = x1[l0][(l1 + 1) % 8], &x2s = x2[l0][l1],
                        &x2l = x2[l0][(l1 + 1) % 8];
            x2s = sycl::uint4(0);

            nd.barrier(sycl::access::fence_space::local_space);

            for (unsigned i = 0, i1 = l1; i < CN_MEMORY16 / 8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
              x ^= lpad[i1];
              x ^= x2l;
              for (unsigned j = 0; j < 10; ++j)
                aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
              x1s = x;
              nd.barrier(sycl::access::fence_space::local_space);
              x ^= lpad[i1 + 8];
              x ^= x1l;
              for (unsigned j = 0; j < 10; ++j)
                aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
              x2s = x;
              nd.barrier(sycl::access::fence_space::local_space);
            }

            x ^= x2l;

            for (unsigned i = 0; i < 16; ++i) {
              for (unsigned j = 0; j < 10; ++j)
                aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
              nd.barrier(sycl::access::fence_space::local_space);
              x1s = x;
              nd.barrier(sycl::access::fence_space::local_space);
              x ^= x1l;
            }
            mo_vec_store(x, l1 + 4, spad);
          });
    });
  }

  sycl::event final_event =
      q.submit([&](sycl::handler&
                       h) { // Final Keccak stays on device; the host scratchpad argument is unused.
        CnGpuActiveViews cn_views(state, h);
        h.parallel_for(sycl::nd_range(sycl::range(round_up_size(batch, WORKGROUP_SIZE)),
                                      sycl::range(WORKGROUP_SIZE)),
                       [=](sycl::nd_item<1> nd) {
                         const auto [d_inputs, d_spads, d_lpads, d_outputs, d_spads4, d_lpads4,
                                     d_lpads16] = cn_views.pointers();
                         const auto t = nd.get_global_id(0);
                         if (t >= batch)
                           return;

                         uint64_t* const spad = &d_spads[25 * t];
                         keccak(spad);

                         uint64_t* const out =
                             reinterpret_cast<uint64_t*>(d_outputs + HASH_LEN * t);
                         out[0] = spad[0];
                         out[1] = spad[1];
                         out[2] = spad[2];
                         out[3] = spad[3];
                       });
      });

  // Finish kernels before a pageable readback can block inside the runtime copy submission.
  sycl_wait_and_throw(final_event, state.device);
  if constexpr (mom_sycl_portable_opencl) {
    state.buffered_outputs.read(output, output_bytes);
  } else if (state.shared_io) {
    std::memcpy(output, d_outputs, output_bytes);
  } else {
    sycl_wait_and_throw(q.memcpy(output, d_outputs, output_bytes), state.device);
  }
}
