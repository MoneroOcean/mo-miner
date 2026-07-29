// ---- gen is FUSED into round-1 mix-scatter (see submit_mix_scatter, R==1): the leaf at slot==index is
// regenerated in-kernel via make_leaf, so no separate gen kernel and no level-0 DRAM arena. This elides
// a full level-0 write (~2 GiB) plus the round-1 read of it; scatter is DRAM-write bound so the extra
// siphash compute is free. Recovery still maps leaf index == level-0 slot directly. ----

// ---- Round r mix-scatter: read mixed[L] (L=r-1), applyMix(remLen), bucket by hi BUCKBITS of the low-24
// collision field, write a bucketed slot carrying the packed source linear index. ----
//
// SCATTER COALESCING (the dominant ~66%-of-solve cost). Naively, one lane writes its whole OUT_ST-u32
// (64..84 byte) record to a near-random global bucket slot; a warp's 32 lanes then hit 32 independent
// bucket regions, so each of the OUT_ST store instructions fans out to ~32 distinct cache lines -- a
// ~OUT_ST-x write-transaction amplification, and the kernel is DRAM-write-BW bound (WG sweep is flat ->
// not occupancy/atomic bound). An SLM-radix bucket grouping does NOT help: with 16384 buckets a WG of
// 256 survivors spreads ~1-per-bucket (run length ~1), so there is no same-bucket adjacency to batch.
// Instead we TRANSPOSE the wide-record COPY: each lane stages its computed record into SLM and reserves
// its global slot (atomic, unchanged), then the warp cooperatively copies one record at a time with
// consecutive lanes writing consecutive u32 of that record. Each ~84-byte record is then ONE coalesced
// transaction (~1 line) instead of ~OUT_ST scattered ones -> ~OUT_ST-x fewer write transactions. The
// bucket arena layout, bucket_ns semantics, NSLOTS and the collide kernel are all untouched; only the
// intra-bucket slot ASSIGNMENT order can change, which collide (rest-bin linked lists over ALL slots,
// order-independent) tolerates. Drops past NSLOTS / past in_count are honored via a sentinel.
template <unsigned R, bool DEVICE_COUNT = false> class BeamMixScatterKernel;
template <unsigned R, bool DEVICE_COUNT = false>
static sycl::event submit_mix_scatter(sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_mixed,
                                      uint32_t in_count, uint32_t* d_bucket, uint32_t* d_bucket_ns,
                                      unsigned WG, const uint64_t* d_prePow = nullptr,
                                      const uint32_t* d_in_count = nullptr) {
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_ST  = mixed_stride_u32(L);
  constexpr unsigned OUT_ST = bucket_stride_u32(L);
  // slot (<=NSLOTS<2^12) packed in low 12 bits, bucket (<2^14) above; SENTINEL = dropped/out-of-range.
  constexpr uint32_t SLOT_BITS = 12;
  constexpr uint32_t SENTINEL  = 0xffffffffu;
  static_assert(NSLOTS < (1u << SLOT_BITS), "NSLOTS must fit SLOT_BITS for the packed dst");
  static_assert(NBUCKETS <= (1u << (32 - SLOT_BITS)), "bucket must fit the packed dst");
  const size_t global = ((size_t)in_count + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    // Per-WG staging: each lane's OUT_ST-u32 record + its packed (bucket,slot) destination.
    sycl::local_accessor<uint32_t, 1> stage(sycl::range<1>((size_t)WG * OUT_ST), h);
    sycl::local_accessor<uint32_t, 1> dst(sycl::range<1>(WG), h);
    h.parallel_for<BeamMixScatterKernel<R, DEVICE_COUNT>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t s   = (uint32_t)it.get_global_id(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);
      uint32_t* st = &stage[(size_t)lid * OUT_ST];
      // A queued CUDA handoff dispatches the allocation bound and consumes the preceding collide
      // kernel's device-written count. The in-order queue is the memory dependency; avoiding a host
      // read/wait between rounds costs only one uniform, cache-hot counter load per work-item.
      uint32_t active_count = in_count;
      if constexpr (DEVICE_COUNT)
        active_count = std::min<uint32_t>(d_in_count[0], (uint32_t)MIXED_CAP);

      if (s < active_count) {
        uint64_t w[WORK_WORDS];
        for (unsigned i = 0; i < WORK_WORDS; ++i) w[i] = 0;
        uint32_t tree[LEVEL_NIDX[L] > 0 ? LEVEL_NIDX[L] : 1];
        if constexpr (R == 1) {
          // GEN FUSION: round 1's input is the leaf list, where slot == leaf index. Regenerate the
          // leaf in-kernel (make_leaf = LEVEL_WORDS[0] siphash24 calls) instead of reading a level-0
          // arena. This elides the separate gen kernel AND the level-0 DRAM write+read (~2 GiB each
          // way). Scatter is DRAM-write bound, so trading siphash compute for that read is free.
          uint64_t prePow[4];
          for (unsigned i = 0; i < 4; ++i) prePow[i] = d_prePow[i];
          make_leaf(prePow, s, w);
          tree[0] = s;   // the single leaf index (LEVEL_NIDX[0] == 1)
        } else {
          const uint32_t* rec = d_mixed + (size_t)s * IN_ST;
          for (unsigned i = 0; i < LEVEL_WORDS[L]; ++i)
            w[i] = (uint64_t)rec[2 * i] | ((uint64_t)rec[2 * i + 1] << 32);
          for (unsigned i = 0; i < LEVEL_NIDX[L]; ++i) tree[i] = rec[2 * LEVEL_WORDS[L] + i];
        }

        apply_mix(w, tree, LEVEL_NIDX[L], ROUND_REMLEN[R]);
        const uint32_t coll = collision_bits(w);
        const uint32_t bucket = coll >> RESTBITS;                 // hi BUCKBITS
        const uint32_t slot = dev_atomic_u32(d_bucket_ns[bucket]).fetch_add(1u);
        if (slot >= NSLOTS) {
          dst[lid] = SENTINEL;
        } else {
          // Stage the OUT_ST-u32 record into SLM (same byte layout the old direct write produced).
          for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i) {
            st[2 * i]     = (uint32_t)(w[i] & 0xffffffffull);
            st[2 * i + 1] = (uint32_t)(w[i] >> 32);
          }
          for (unsigned i = 0; i < BUCKET_NIDX[L]; ++i)
            st[2 * BUCKET_WORDS[L] + i] = tree[i];
          bucket_store_source<L>(st, s);
          dst[lid] = (bucket << SLOT_BITS) | slot;
        }
      } else {
        dst[lid] = SENTINEL;
      }
      sycl::group_barrier(it.get_group());

      // Warp-cooperative coalesced flush: each sub-group copies one whole record at a time with
      // consecutive lanes writing consecutive u32 -> one coalesced burst per ~84-byte record.
      if constexpr (mom_sycl_portable_opencl) {
        const uint32_t packed = dst[lid];
        if (packed != SENTINEL) {
          const uint32_t bucket = packed >> SLOT_BITS;
          const uint32_t slot   = packed & ((1u << SLOT_BITS) - 1u);
          uint32_t* o = d_bucket + ((size_t)bucket * NSLOTS + slot) * OUT_ST;
          const uint32_t* src = &stage[(size_t)lid * OUT_ST];
          for (unsigned k = 0; k < OUT_ST; ++k) o[k] = src[k];
        }
      } else {
        auto sg = it.get_sub_group();
        const unsigned sg_sz   = (unsigned)sg.get_max_local_range()[0];
        const unsigned lane     = (unsigned)sg.get_local_id()[0];
        const unsigned wg_warps = (lsz + sg_sz - 1) / sg_sz;
        const unsigned warp_id  = lid / sg_sz;
        for (unsigned r = warp_id; r < lsz; r += wg_warps) {
          const uint32_t packed = dst[r];
          if (packed == SENTINEL) continue;
          const uint32_t bucket = packed >> SLOT_BITS;
          const uint32_t slot   = packed & ((1u << SLOT_BITS) - 1u);
          uint32_t* o = d_bucket + ((size_t)bucket * NSLOTS + slot) * OUT_ST;
          const uint32_t* src = &stage[(size_t)r * OUT_ST];
          for (unsigned k = lane; k < OUT_ST; k += sg_sz) o[k] = src[k];
        }
      }
    });
  });
}

// ---- Round r collide/advance: one WG per bucket. Read bucketed[L], collide low RESTBITS in SLM,
// XOR-merge (mergeLen), build the next index prefix, and log the parents. For rounds 1..4, applyMix for
// the following round immediately and scatter the result into the alternate bucket arena. Only round 5
// writes a compact linear level-5 record. This removes four wide linear writes, four wide linear reads,
// and four standalone scatter launches from the steady solve.
template <unsigned R, bool FUSED, bool PREDECESSOR_ONLY> class BeamCollideKernel;
template <unsigned R, bool FUSED, bool PREDECESSOR_ONLY>
static sycl::event submit_collide(sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_bucket,
                                  const uint32_t* d_bucket_ns, uint32_t* d_final_out,
                                  uint32_t* d_out_count, uint32_t* d_tree, uint32_t* d_next_bucket,
                                  uint32_t* d_next_bucket_ns, unsigned wg) {
  constexpr unsigned L = R - 1;
  constexpr unsigned IN_ST   = bucket_stride_u32(L);
  constexpr unsigned OUT_L   = R;           // output level
  constexpr unsigned FINAL_ST = mixed_stride_u32(OUT_L);
  constexpr unsigned NEXT_ST  = bucket_stride_u32(OUT_L);
  constexpr unsigned mergeLen = ROUND_MERGELEN[R];
  const size_t global = (size_t)NBUCKETS * wg;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    sycl::local_accessor<uint32_t, 1> bin_head(sycl::range<1>(NRESTBINS), h);
    sycl::local_accessor<uint32_t, 1> bin_next(sycl::range<1>(NSLOTS), h);
    // Stage every slot's 24-bit collision key in SLM. The wide bucket records are read once here (the
    // inverse of the scatter-transpose); the rest-bin build and -- crucially -- the per-link compare in
    // the collide pass then hit SLM instead of re-striding the wide (60..84B) record in DRAM. Most link
    // traversals fail the full-24-bit match, so this removes the dominant scattered b[0],b[1] DRAM reads.
    sycl::local_accessor<uint32_t, 1> coll_key(sycl::range<1>(NSLOTS), h);
    h.parallel_for<BeamCollideKernel<R, FUSED, PREDECESSOR_ONLY>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
      const uint32_t bucket = (uint32_t)it.get_group(0);
      const unsigned lid = (unsigned)it.get_local_id(0);
      const unsigned lsz = (unsigned)it.get_local_range(0);

      uint32_t cnt = d_bucket_ns[bucket];
      if (cnt > NSLOTS) cnt = NSLOTS;
      for (unsigned r = lid; r < NRESTBINS; r += lsz) bin_head[r] = 0;

      const uint32_t* in_base = d_bucket + (size_t)bucket * NSLOTS * IN_ST;
      constexpr uint32_t COLL_MASK = (1u << COLLISION_BIT_SIZE) - 1u;

      // Stage 24-bit collision keys into SLM (one strided DRAM read per slot, vs. two later).
      for (unsigned s = lid; s < cnt; s += lsz)
        coll_key[s] = in_base[(size_t)s * IN_ST] & COLL_MASK;
      sycl::group_barrier(it.get_group());

      // Build the rest-bin linked lists (key = low RESTBITS of the collision field), from SLM keys.
      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t rb = coll_key[s] & (NRESTBINS - 1u);
        bin_next[s] = slm_atomic_u32(bin_head[rb]).exchange(s + 1u);
      }
      sycl::group_barrier(it.get_group());

      for (unsigned s = lid; s < cnt; s += lsz) {
        const uint32_t a_coll = coll_key[s];
        const uint32_t rb = a_coll & (NRESTBINS - 1u);

        const uint32_t* a = nullptr;   // a's wide record is read lazily, only when a partner matches.
        uint64_t aw[WORK_WORDS];

        // Level Zero benefits from following only this slot's predecessor chain: atomic insertion
        // order defines which endpoint emits each unordered pair, so the numeric t>=s test and the
        // remainder of the chain are unnecessary. On HIP this was neutral/slightly slower (DPC++
        // 14.68 vs 14.70--14.71 H/s). The analogous predecessor walk is also ~5% slower on NVIDIA
        // warps in Equihash, so CUDA conservatively retains its already-fast common-head kernel.
        // Keep both as compile-time variants selected once from the host device.
        uint32_t link = PREDECESSOR_ONLY ? bin_next[s] : bin_head[rb];
        for (; link != 0; link = bin_next[link - 1u]) {
          const uint32_t t = link - 1u;
          if constexpr (!PREDECESSOR_ONLY) if (t >= s) continue;
          if (coll_key[t] != a_coll) continue;                   // full 24-bit collision field must match (SLM)

          if (!a) {   // lazy: only now do we touch the wide records from DRAM.
            a = in_base + (size_t)s * IN_ST;
            for (unsigned i = 0; i < WORK_WORDS; ++i) aw[i] = 0;
            for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i)
              aw[i] = (uint64_t)a[2 * i] | ((uint64_t)a[2 * i + 1] << 32);
          }
          const uint32_t* b = in_base + (size_t)t * IN_ST;
          uint64_t bw[WORK_WORDS]; for (unsigned i = 0; i < WORK_WORDS; ++i) bw[i] = 0;
          for (unsigned i = 0; i < BUCKET_WORDS[L]; ++i)
            bw[i] = (uint64_t)b[2 * i] | ((uint64_t)b[2 * i + 1] << 32);

          // The two parents' first-NIDX[L] index lists + lead index decide the merge order
          // (lowest-index-first). Leader = the node whose lead index (tree[0]) is smaller.
          const uint32_t a_lead = a[2 * BUCKET_WORDS[L] + 0] & INDEX_MASK;
          const uint32_t b_lead = b[2 * BUCKET_WORDS[L] + 0] & INDEX_MASK;
          // distinct-index quick reject would require a full pairwise check; defer to recovery (final).

          uint64_t mw[WORK_WORDS];
          merge_workbits(aw, bw, mergeLen, mw);

          const uint32_t oslot = dev_atomic_u32(d_out_count[0]).fetch_add(1u);
          if (oslot >= MIXED_CAP) continue;

          // Build the merged first-NIDX[OUT_L] index tree from the two parents' stored prefixes in
          // lowest-index-first order. We have up to NIDX[L] per parent; the merged tree's first
          // NIDX[OUT_L] (<= 2*NIDX[L]) come from the leader's stored list then the follower's.
          uint32_t prefix[LEVEL_NIDX[OUT_L] > 0 ? LEVEL_NIDX[OUT_L] : 1];
          if (LEVEL_NIDX[OUT_L] > 0) {
            const uint32_t* lead = (a_lead < b_lead) ? a : b;
            const uint32_t* foll = (a_lead < b_lead) ? b : a;
            const unsigned base = 2 * BUCKET_WORDS[L];
            unsigned o_i = 0;
            for (unsigned i = 0; i < BUCKET_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              prefix[o_i++] = lead[base + i] & INDEX_MASK;
            for (unsigned i = 0; i < BUCKET_NIDX[L] && o_i < LEVEL_NIDX[OUT_L]; ++i)
              prefix[o_i++] = foll[base + i] & INDEX_MASK;
          }

          // A fused bucket record's source is this round's linear oslot, so the existing recovery
          // walk through tree[5..1] remains unchanged.
          const uint32_t a_src = bucket_load_source<L>(a);
          const uint32_t b_src = bucket_load_source<L>(b);
          uint32_t* tl = d_tree + (size_t)oslot * 2;
          tl[0] = (a_lead < b_lead) ? a_src : b_src;
          tl[1] = (a_lead < b_lead) ? b_src : a_src;

          if constexpr (FUSED && R < NUM_ROUNDS) {
            // Exactly the transform formerly performed by submit_mix_scatter<R+1>.
            apply_mix(mw, prefix, LEVEL_NIDX[OUT_L], ROUND_REMLEN[R + 1]);
            const uint32_t next_coll = collision_bits(mw);
            const uint32_t next_bucket = next_coll >> RESTBITS;
            const uint32_t next_slot = dev_atomic_u32(d_next_bucket_ns[next_bucket]).fetch_add(1u);
            if (next_slot >= NSLOTS) continue;
            uint32_t* o = d_next_bucket + ((size_t)next_bucket * NSLOTS + next_slot) * NEXT_ST;
            for (unsigned i = 0; i < BUCKET_WORDS[OUT_L]; ++i) {
              o[2 * i]     = (uint32_t)(mw[i] & 0xffffffffull);
              o[2 * i + 1] = (uint32_t)(mw[i] >> 32);
            }
            for (unsigned i = 0; i < BUCKET_NIDX[OUT_L]; ++i)
              o[2 * BUCKET_WORDS[OUT_L] + i] = prefix[i];
            bucket_store_source<OUT_L>(o, oslot);
          } else {
            uint32_t* o = d_final_out + (size_t)oslot * FINAL_ST;
            for (unsigned i = 0; i < LEVEL_WORDS[OUT_L]; ++i) {
              o[2 * i]     = (uint32_t)(mw[i] & 0xffffffffull);
              o[2 * i + 1] = (uint32_t)(mw[i] >> 32);
            }
            // The non-fused CUDA path feeds this linear record into the existing coalesced scatter.
            // Round 5 has no prefix, so the same loop compiles away for the fused final output.
            for (unsigned i = 0; i < LEVEL_NIDX[OUT_L]; ++i)
              o[2 * LEVEL_WORDS[OUT_L] + i] = prefix[i];
          }
        }
      }
    });
  });
}

// ---- Final: a level-5 survivor whose merged 24-bit workBits == 0 is a candidate. Walk tree[5..1] to
// its 32 leaf source indices (in mixed[0]), read each leaf's stored leaf index. ----
// Walk the tree from a level-OUT survivor down to its leaf indices. tree[L] maps a mixed[L] slot to its
// two parent mixed[L-1] slots. mixed[0] slot stores the leaf index at offset 2*LEVEL_WORDS[0].
