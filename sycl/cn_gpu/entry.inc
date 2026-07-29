// cn_gpu owns the full submit chain so the hot path is visible in one function.
void cn_gpu(
  const uint8_t* const inputs, const unsigned input_size, uint8_t* const output,
  void* const, const unsigned batch, const std::string& dev_str, const std::string& backend
) {
  CnGpuState& state = cn_gpu_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  // The FP recurrence packs two 16-thread hashes into one 32-wide group, so round
  // the in-flight batch up to even; the extra hash is computed but never read.
  const unsigned batch_eff = (batch + 1u) & ~1u;
  state.ensure_capacity(batch_eff, input_size);
  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  uint8_t* const d_inputs = state.inputs;
  uint8_t* const d_outputs = state.outputs;
  const unsigned input_bytes = input_size * batch;
  const unsigned output_bytes = HASH_LEN * batch;

  const auto batch_start = std::chrono::steady_clock::now();  // for the pre-read sleep EMA below

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
  q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range((batch + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE),
                                 sycl::range(WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const auto t = nd.get_global_id(0);
      if (t >= batch) return;

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
          q.wait_and_throw();
          state.cuda_expand.launch(state.spads, state.lpads, batch);
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
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const unsigned stripes = CN_MEMORY8 / 64; // 512-byte stripes per scratchpad
      mom_use_bundle(h, kb);
      const unsigned total_threads = batch * stripes;
      unsigned expand_wg = WORKGROUP_SIZE;
#if defined(_WIN32)
      // NVIDIA's Windows CUDA driver runs this register-heavy Keccak expansion 7.9% faster with
      // eight-thread blocks; Linux, Intel, and AMD retain the measured 16-thread default.
      if (mom_is_cuda(state.device)) expand_wg = 8;
#endif
      const unsigned wg_size = std::min(expand_wg, stripes);
      h.parallel_for(sycl::nd_range(
                       sycl::range((total_threads + wg_size - 1) / wg_size * wg_size),
                       sycl::range(wg_size)),
                     [=](sycl::nd_item<1> nd) {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
        const auto t = nd.get_global_id(0);
        if (t >= total_threads) return;

        const unsigned stripe = t % stripes;
        const unsigned hash_idx = t / stripes;
        const uint64_t* const spad = &d_spads[hash_idx * 25];
        uint64_t* const lpad = &d_lpads[hash_idx * CN_MEMORY8];
        generate512(stripe, spad, &lpad[stripe * 64]);
      });
    });
  }

  // Keep the long floating-point recurrence in the common SYCL pipeline on every backend.
  // The Intel OpenCL CPU runtime accepts the requested 16-wide subgroup but does not provide the
  // local-memory ordering this recurrence needs from subgroup-only barriers. Keep the measured
  // narrow barrier path on GPUs; the archive/generic-OpenCL CPU validation path uses the original
  // workgroup barrier. This is a uniform host-captured branch and does not alter GPU arithmetic.
  const bool use_narrow_recurrence_barriers =
    !mom_sycl_adaptivecpp && !mom_sycl_portable_opencl && state.device.is_gpu();
  q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    // The recurrence uses 16 threads/hash. Pack TWO independent hashes per workgroup so a native
    // NVIDIA/AMD 32-lane warp/wave is full; Intel requests one 16-lane subgroup per hash. Every
    // local-memory dependency stays inside its 16-lane slice. DPC++ therefore uses subgroup
    // barriers instead of paying four workgroup barriers in each of 49,152 rounds (+2.9% NVIDIA,
    // +5.1% AMD and +3.2% Intel in counterbalanced tests). AdaptiveCpp deliberately retains the
    // workgroup barriers: its generic/SSCP lowering was neutral on Windows AMD and 0.6-0.9% slower
    // on NVIDIA, so keeping the proven fallback avoids a compiler-specific regression.
    const auto vi0 = sycl::local_accessor<sycl::int4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    const auto vi1 = sycl::local_accessor<sycl::int4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    const auto vf0 = sycl::local_accessor<sycl::float4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range(batch_eff * WORKGROUP_SIZE),
                                  sycl::range(2 * WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) MOM_REQD_SG_16 {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const unsigned lid = nd.get_local_id();
      const unsigned sub = lid >> 4;        // which of the two warp-packed hashes
      const unsigned l = lid & 15U;         // 0..15 lane role within the hash
      const unsigned base16 = sub * 16U;    // int4 slice base for this hash
      const unsigned base32 = sub * 64U;    // int32/float slice base for this hash
      const unsigned ld = l / 4;
      const unsigned lm = l % 4;
      const unsigned b = ld * 16 + lm;
      const unsigned group_hash = nd.get_group().get_group_id() * 2U + sub;
      const uint32_t* const spad = &d_spads4[group_hash * (25 * 2)];
      int32_t* const lpad = &d_lpads4[group_hash * CN_MEMORY4];
      uint32_t s = *spad >> 8;
      sycl::float4 sf(0.0f);
      sycl::int4* const vi = &vi0[base16];
      int32_t* const vi4 = reinterpret_cast<int32_t*>(&vi0[0]) + base32;
      sycl::int4* const vo = &vi1[base16];
      int32_t* const vo4 = reinterpret_cast<int32_t*>(&vi1[0]) + base32;
      sycl::float4* const vf = &vf0[base16];
      float* const vf4 = reinterpret_cast<float*>(&vf0[0]) + base32;
      for (unsigned i = 0; i < CN_GPU_ITER; ++ i) {
        const int32_t xi = lpad_ptr(s, ld, lpad)[lm];
        vi4[l] = xi;

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        single_comupte_wrap(
          vi[cn_lane_select(l, 0)], vi[cn_lane_select(l, 1)],
          vi[cn_lane_select(l, 2)], vi[cn_lane_select(l, 3)],
          lm, cn_lane_count(l), sf, vf + l, vo + l
        );

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        { int32_t xo = vo4[b];
          for (unsigned dd = b + 4; dd < (ld + 1) * 16; dd += 4) xo ^= vo4[dd];
          lpad_ptr(s, ld, lpad)[lm] = xo ^ xi;
          vo4[l] = xo;
        }

        vf4[l] = (vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]);

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        const float xf = sycl::fabs((vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]));
        vo4[l] ^= vo4[l + 4] ^ vo4[l + 8] ^ vo4[l + 12] ^
                  static_cast<int32_t>(xf * 16777216.0f);
        vf4[l] = xf * 0.015625f;

        cn_recurrence_barrier(nd, use_narrow_recurrence_barriers);

        sf = vf[0];
        s = vo[0][0] ^ vo[0][1] ^ vo[0][2] ^ vo[0][3];
      }
    });
    });

  const unsigned aes_wg = WORKGROUP_SIZE / 4; // hashes per workgroup row; dim1 (8) splits each hash
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const bool use_gpu_aes = !mom_sycl_portable_opencl && state.device.is_gpu();
  if (use_gpu_aes) { // GPU path replaces the AES local-memory ring with subgroup shuffles.
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const AesTable aes0(sycl::range(256), h), aes1(sycl::range(256), h),
                     aes2(sycl::range(256), h), aes3(sycl::range(256), h);
      mom_use_bundle(h, kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch, 8), sycl::range(aes_wg, 8)),
                     [=](sycl::nd_item<2> nd) MOM_REQD_SG_16 {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
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

        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16/8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
          x ^= lpad[i1];
          const sycl::uint4 x2l = next_row_value(x2s);
          x ^= x2l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          x ^= lpad[i1 + 8];
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x2s = x;
        }

        const sycl::uint4 x2l = next_row_value(x2s);
        x ^= x2l;

        for (unsigned i = 0; i < 16; ++ i) {
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
        }
        mo_vec_store(x, l1 + 4, spad);
      });
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
      mom_use_bundle(h, kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch, 8), sycl::range(aes_wg, 8)),
                     [=](sycl::nd_item<2> nd) {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
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
        sycl::uint4 &x1s = x1[l0][l1], &x1l = x1[l0][(l1 + 1) % 8],
                    &x2s = x2[l0][l1], &x2l = x2[l0][(l1 + 1) % 8];
        x2s = sycl::uint4(0);

        nd.barrier(sycl::access::fence_space::local_space);

        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16/8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
          x ^= lpad[i1];
          x ^= x2l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);
          x ^= lpad[i1 + 8];
          x ^= x1l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x2s = x;
          nd.barrier(sycl::access::fence_space::local_space);
        }

        x ^= x2l;

        for (unsigned i = 0; i < 16; ++ i) {
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          nd.barrier(sycl::access::fence_space::local_space);
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);
          x ^= x1l;
        }
        mo_vec_store(x, l1 + 4, spad);
      });
    });
  }

  sycl::event final_event = q.submit([&](sycl::handler& h) { // Final Keccak stays on device; the host scratchpad argument is unused.
    CnGpuActiveViews cn_views(state, h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range((batch + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE),
                                 sycl::range(WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const auto t = nd.get_global_id(0);
      if (t >= batch) return;

      uint64_t* const spad = &d_spads[25 * t];
      keccak(spad);

      uint64_t* const out = reinterpret_cast<uint64_t*>(d_outputs + HASH_LEN * t);
      out[0] = spad[0]; out[1] = spad[1]; out[2] = spad[2]; out[3] = spad[3];
    });
  });

  // shared_io is false only for the device-memory GPUs that take the blocking D2H output read
  // below: Intel OpenCL (cn/gpu's policy backend) and the CUDA backend. On both, that read pins a
  // host core for the whole batch -- their queue waits return before the kernels finish, and the
  // CUDA D2H copy to pageable host memory is synchronous, so the real GPU sync happens inside the
  // read. (CPU and Level-Zero take the shared_io path instead: a poll-freed wait then a host-side
  // copy of already-ready data.) To stop pinning the core, sleep through most of the batch before
  // the read -- sized from an EMA of recent batch wall times -- leaving only a short spinning tail.
  // cn/gpu batches are long (hundreds of ms) and very stable, so a 90% sleep adds no latency (it
  // stays under the batch) while cutting steady-state host usage to ~10% of a core.
  const bool sleep_before_read = !state.shared_io;
  if (sleep_before_read && state.wait_ema_us > 2000.0) {
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(state.wait_ema_us * 0.90)));
  }

  if constexpr (mom_sycl_portable_opencl) {
    sycl_wait_and_throw(final_event, state.device);
    state.buffered_outputs.read(output, output_bytes);
  } else {
    if (state.shared_io) {
      sycl_wait_and_throw(final_event, state.device);
      std::memcpy(output, d_outputs, output_bytes);
    } else {
      sycl_wait_and_throw(q.memcpy(output, d_outputs, output_bytes), state.device);
    }
  }

  if (sleep_before_read) {
    // Track the full batch wall time (submit + sleep + read). With the sleep sized below the
    // batch this equals the true GPU time, so the EMA self-corrects: oversleep -> measured time
    // drops -> next sleep shrinks; undersleep -> read spins longer -> EMA rises toward the truth.
    const double batch_us = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - batch_start).count());
    state.wait_ema_us = state.wait_ema_us == 0.0 ? batch_us : state.wait_ema_us * 0.8 + batch_us * 0.2;
  }
}
