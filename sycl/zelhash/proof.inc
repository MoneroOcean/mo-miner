template <bool GPU_COMPACT>
// Return the events so their native handles stay owned until recovery completes the whole chain.
// Optional per-stage waits remain diagnostic/backend policy only.
static std::array<sycl::event, 4> run_collision_rounds(
    State& state, unsigned wg, bool wait_stages, bool prof, uint64_t t_round[4]) {
  sycl::queue& q = state.queue;
  auto r0 = submit_round<0, GPU_COMPACT>(q, state.level[0], state.level[1],
                                         state.nslots + 0u * NBUCKETS,
                                         state.nslots + 1u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages)
    sycl_wait_and_throw(r0, state.device);
  if (prof)
    t_round[0] = mom_now_ms();
  auto r1 = submit_round<1, GPU_COMPACT>(q, state.level[1], state.level[2],
                                         state.nslots + 1u * NBUCKETS,
                                         state.nslots + 2u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages)
    sycl_wait_and_throw(r1, state.device);
  if (prof)
    t_round[1] = mom_now_ms();
  auto r2 = submit_round<2, GPU_COMPACT>(q, state.level[2], state.level[3],
                                         state.nslots + 2u * NBUCKETS,
                                         state.nslots + 3u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages)
    sycl_wait_and_throw(r2, state.device);
  if (prof)
    t_round[2] = mom_now_ms();
  auto r3 = submit_round<3, GPU_COMPACT>(
      q, state.level[3], nullptr, state.nslots + 3u * NBUCKETS, nullptr, wg, state.slot_capacity,
      state.ref, state.ref_count, State::CAND_CAP);
  if (wait_stages)
    sycl_wait_and_throw(r3, state.device);
  if (prof)
    t_round[3] = mom_now_ms();
  return {r0, r1, r2, r3};
}

// Host-side finishing: device recovery already returned each candidate's 16 leaf indices, so the host
// only canonicalises their IndicesBefore order (equihash-pow.h) and packs the proof.

// CompressArray(26,0): pack 16 indices (26 bits each, big-endian) into 52 bytes. Exact inverse of the
// reference expandArray used by getIndicesFromMinimal.
static void compress_indices(const std::vector<uint32_t>& idx, uint8_t out[52]) {
  constexpr unsigned BIT_LEN = COLLISION_BIT_LENGTH + 1; // 26
  constexpr unsigned IN_WIDTH = ((BIT_LEN + 7) >> 3);    // 4 (each index expanded as 4 BE bytes)
  // Build the expanded byte stream (16 * 4 = 64 bytes, big-endian 26-bit fields) then compress.
  std::vector<uint8_t> in(idx.size() * IN_WIDTH, 0);
  for (size_t i = 0; i < idx.size(); ++i) {
    in[i * IN_WIDTH + 0] = (uint8_t)((idx[i] >> 24) & 0xFF);
    in[i * IN_WIDTH + 1] = (uint8_t)((idx[i] >> 16) & 0xFF);
    in[i * IN_WIDTH + 2] = (uint8_t)((idx[i] >> 8) & 0xFF);
    in[i * IN_WIDTH + 3] = (uint8_t)(idx[i] & 0xFF);
  }
  const size_t out_len = idx.size() * BIT_LEN / 8; // 52
  uint32_t acc_value = 0;
  unsigned acc_bits = 0;
  size_t j = 0;
  const uint32_t bit_mask = ((uint32_t)1 << BIT_LEN) - 1u;
  for (size_t i = 0; i < in.size(); ++i) {
    // CompressArray (Zcash util.cpp): read bit_len bits from each IN_WIDTH input element, low byte_pad=0.
    if ((i % IN_WIDTH) == 0) {
      // begin a new input element: take its BIT_LEN low bits
      uint32_t v = ((uint32_t)in[i] << 24) | ((uint32_t)in[i + 1] << 16) |
                   ((uint32_t)in[i + 2] << 8) | (uint32_t)in[i + 3];
      v &= bit_mask;
      acc_value = (acc_value << BIT_LEN) | v;
      acc_bits += BIT_LEN;
      while (acc_bits >= 8 && j < out_len) {
        acc_bits -= 8;
        out[j++] = (uint8_t)((acc_value >> acc_bits) & 0xFF);
      }
    }
  }
}
