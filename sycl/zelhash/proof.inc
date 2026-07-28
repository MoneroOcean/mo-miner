template <bool GPU_COMPACT>
static void run_collision_rounds(ZelHashState& state, unsigned wg, bool wait_stages,
                                 bool prof, uint64_t t_round[4]) {
  sycl::queue& q = state.queue;
  auto r0 = submit_round<0, GPU_COMPACT>(q, *state.bundle, state.level[0], state.level[1],
    state.nslots + 0u * NBUCKETS, state.nslots + 1u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r0, state.device);
  if (prof) t_round[0] = ZelHashState::now_ms();
  auto r1 = submit_round<1, GPU_COMPACT>(q, *state.bundle, state.level[1], state.level[2],
    state.nslots + 1u * NBUCKETS, state.nslots + 2u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r1, state.device);
  if (prof) t_round[1] = ZelHashState::now_ms();
  auto r2 = submit_round<2, GPU_COMPACT>(q, *state.bundle, state.level[2], state.level[3],
    state.nslots + 2u * NBUCKETS, state.nslots + 3u * NBUCKETS, wg, state.slot_capacity);
  if (wait_stages) sycl_wait_and_throw(r2, state.device);
  if (prof) t_round[2] = ZelHashState::now_ms();
  auto r3 = submit_round<3, GPU_COMPACT>(q, *state.bundle, state.level[3], nullptr,
    state.nslots + 3u * NBUCKETS, nullptr, wg, state.slot_capacity,
    state.ref, state.ref_count, ZelHashState::CAND_CAP);
  if (wait_stages) sycl_wait_and_throw(r3, state.device);
  if (prof) t_round[3] = ZelHashState::now_ms();
}

// ===========================================================================================
// Host-side finishing: the compact device recovery already returned each candidate's 16 leaf indices,
// so the host only canonicalises Zcash IndicesBefore order and CompressArray(26)s the proof. No arena
// copyback is required.
// ===========================================================================================

// orderindices: the Zcash GetIndices canonical ordering. The recovered 2^K leaves come out in tree
// order; IsValidSolution's IndicesBefore rule requires, at every internal node, that the subtree whose
// minimum (== first) index is smaller comes first. Re-derive that ordering bottom-up by swapping each
// sibling pair (and the blocks they head) so the smaller-leading half precedes. Matches fluxd's
// "if (indices[0] < indicesRight[0]) ... else swap".
static void order_indices(std::vector<uint32_t>& idx) {
  const size_t n = idx.size();
  for (size_t span = 1; span < n; span *= 2) {
    for (size_t i = 0; i + 2 * span <= n; i += 2 * span) {
      if (idx[i + span] < idx[i]) {  // right half leads with a smaller index -> it must come first
        for (size_t k = 0; k < span; ++k) std::swap(idx[i + k], idx[i + span + k]);
      }
    }
  }
}

// CompressArray(26,0): pack 16 indices (26 bits each, big-endian) into 52 bytes. Exact inverse of the
// reference expandArray used by getIndicesFromMinimal.
static void compress_indices(const std::vector<uint32_t>& idx, uint8_t out[52]) {
  constexpr unsigned BIT_LEN = COLLISION_BIT_LENGTH + 1;   // 26
  constexpr unsigned IN_WIDTH = ((BIT_LEN + 7) >> 3);      // 4 (each index expanded as 4 BE bytes)
  // Build the expanded byte stream (16 * 4 = 64 bytes, big-endian 26-bit fields) then compress.
  std::vector<uint8_t> in(idx.size() * IN_WIDTH, 0);
  for (size_t i = 0; i < idx.size(); ++i) {
    in[i * IN_WIDTH + 0] = (uint8_t)((idx[i] >> 24) & 0xFF);
    in[i * IN_WIDTH + 1] = (uint8_t)((idx[i] >> 16) & 0xFF);
    in[i * IN_WIDTH + 2] = (uint8_t)((idx[i] >> 8) & 0xFF);
    in[i * IN_WIDTH + 3] = (uint8_t)(idx[i] & 0xFF);
  }
  const size_t out_len = idx.size() * BIT_LEN / 8;        // 52
  uint32_t acc_value = 0; unsigned acc_bits = 0; size_t j = 0;
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

// ===========================================================================================
// Host-side SHA-256 (FIPS 180-4) + the Flux block-hash/PoW target test. The PoW hash is
// dSHA256(header(140) || compactSize(0x34) || solution(52)) = a 193-byte preimage. The 32-byte
// double-SHA output is read LITTLE-ENDIAN as a 256-bit integer and compared <= the 256-bit target.
// (fluxd pow.cpp: UintToArith256(hash) <= bnTarget; mom stores the target big-endian in m_target_bin,
// so the LE-read hash <= BE-stored target is a plain big-endian byte compare of the reversed hash.)
// Self-contained like the BLAKE2b above -- this rare host path runs only for the ~1.88 candidates/solve.
// ===========================================================================================
static constexpr uint32_t SHA256_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};
inline uint32_t sha256_rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

static void sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
  uint32_t h[8] = { 0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u };
  // Pad: msg || 0x80 || 0x00.. || 64-bit big-endian bit length, to a multiple of 64.
  const size_t total = ((len + 8) / 64 + 1) * 64;
  std::vector<uint8_t> buf(total, 0);
  std::memcpy(buf.data(), msg, len);
  buf[len] = 0x80;
  const uint64_t bits = (uint64_t)len * 8;
  for (unsigned i = 0; i < 8; ++i) buf[total - 1 - i] = (uint8_t)(bits >> (8 * i));
  for (size_t off = 0; off < total; off += 64) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; ++i)
      w[i] = ((uint32_t)buf[off + 4*i] << 24) | ((uint32_t)buf[off + 4*i + 1] << 16) |
             ((uint32_t)buf[off + 4*i + 2] << 8) | (uint32_t)buf[off + 4*i + 3];
    for (unsigned i = 16; i < 64; ++i) {
      const uint32_t s0 = sha256_rotr(w[i-15], 7) ^ sha256_rotr(w[i-15], 18) ^ (w[i-15] >> 3);
      const uint32_t s1 = sha256_rotr(w[i-2], 17) ^ sha256_rotr(w[i-2], 19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (unsigned i = 0; i < 64; ++i) {
      const uint32_t S1 = sha256_rotr(e,6) ^ sha256_rotr(e,11) ^ sha256_rotr(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + SHA256_K[i] + w[i];
      const uint32_t S0 = sha256_rotr(a,2) ^ sha256_rotr(a,13) ^ sha256_rotr(a,22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  for (unsigned i = 0; i < 8; ++i) {
    out[4*i]   = (uint8_t)(h[i] >> 24); out[4*i+1] = (uint8_t)(h[i] >> 16);
    out[4*i+2] = (uint8_t)(h[i] >> 8);  out[4*i+3] = (uint8_t)(h[i]);
  }
}

// dSHA256 of (header(140) || 0x34 || solution(52)) -> the 32-byte internal (LE) block hash.
static void flux_block_hash(const uint8_t* header140, const uint8_t solution[52], uint8_t out[32]) {
  uint8_t preimage[HEADER_LEN + 1 + 52];
  std::memcpy(preimage, header140, HEADER_LEN);
  preimage[HEADER_LEN] = 0x34;                     // compactSize(52)
  std::memcpy(preimage + HEADER_LEN + 1, solution, 52);
  uint8_t once[32];
  sha256(preimage, sizeof(preimage), once);
  sha256(once, 32, out);
}

// PoW target test: hash read little-endian <= target (m_target_bin is a 256-bit BIG-endian array).
// Equivalent big-endian byte compare of the reversed hash vs the target (cf. fishhash meets_target_be).
static bool flux_meets_target(const uint8_t hash_le[32], const uint8_t* target_be) {
  for (int i = 0; i < 32; ++i) {
    const uint8_t hb = hash_le[31 - i];             // most-significant byte of the LE-read hash first
    if (hb != target_be[i]) return hb < target_be[i];
  }
  return true;   // exactly equal still meets target
}

} // namespace mom_zelhash

using namespace mom_zelhash;
