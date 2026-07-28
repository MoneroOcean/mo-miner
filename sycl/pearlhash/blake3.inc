// ---- One-shot BLAKE3 (keyed + unkeyed) for host and SYCL device code ----
// Used by the PearlHash kernel for matrix-commitment roots, commitment hashes, the noise PRNG and the
// PoW jackpot. Ported from the BLAKE3 reference impl; validated against the python verifier.
// Device-friendly: no std::, no heap, no recursion; cv_stack is a fixed array.
namespace pearlhash_b3 {

constexpr uint32_t B3_CHUNK_START = 1u << 0;
constexpr uint32_t B3_CHUNK_END = 1u << 1;
constexpr uint32_t B3_PARENT = 1u << 2;
constexpr uint32_t B3_ROOT = 1u << 3;
constexpr uint32_t B3_KEYED = 1u << 4;
constexpr uint32_t B3_CHUNK_LEN = 1024;

inline void iv_words(uint32_t o[8]) {
  o[0] = 0x6a09e667; o[1] = 0xbb67ae85; o[2] = 0x3c6ef372; o[3] = 0xa54ff53a;
  o[4] = 0x510e527f; o[5] = 0x9b05688c; o[6] = 0x1f83d9ab; o[7] = 0x5be0cd19;
}
inline uint32_t b3_rotr(uint32_t w, int c) { return (w >> c) | (w << (32 - c)); }
inline uint32_t le32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24); }

inline void b3_g(uint32_t* s, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx; s[d] = b3_rotr(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];      s[b] = b3_rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + my; s[d] = b3_rotr(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];      s[b] = b3_rotr(s[b] ^ s[c], 7);
}
inline void b3_round(uint32_t* s, const uint32_t* m) {
  b3_g(s, 0, 4, 8, 12, m[0], m[1]);  b3_g(s, 1, 5, 9, 13, m[2], m[3]);
  b3_g(s, 2, 6, 10, 14, m[4], m[5]); b3_g(s, 3, 7, 11, 15, m[6], m[7]);
  b3_g(s, 0, 5, 10, 15, m[8], m[9]); b3_g(s, 1, 6, 11, 12, m[10], m[11]);
  b3_g(s, 2, 7, 8, 13, m[12], m[13]); b3_g(s, 3, 4, 9, 14, m[14], m[15]);
}

// compress -> full 16-word state.
inline void b3_compress(const uint32_t cv[8], const uint32_t block[16],
                        uint64_t counter, uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  static const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t s[16];
  for (int i = 0; i < 8; i++) s[i] = cv[i];
  s[8] = 0x6a09e667; s[9] = 0xbb67ae85; s[10] = 0x3c6ef372; s[11] = 0xa54ff53a;
  s[12] = (uint32_t)counter; s[13] = (uint32_t)(counter >> 32); s[14] = block_len; s[15] = flags;
  uint32_t m[16];
  for (int i = 0; i < 16; i++) m[i] = block[i];
  for (int r = 0; r < 7; r++) {
    b3_round(s, m);
    if (r < 6) { uint32_t p[16]; for (int i = 0; i < 16; i++) p[i] = m[PERM[i]]; for (int i = 0; i < 16; i++) m[i] = p[i]; }
  }
  for (int i = 0; i < 8; i++) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ cv[i]; }
}

inline void words_from_le(const uint8_t* b, uint32_t len, uint32_t out[16]) {
  for (int i = 0; i < 16; i++) {
    uint32_t w = 0;
    for (int j = 0; j < 4; j++) { uint32_t idx = i * 4 + j; if (idx < len) w |= (uint32_t)b[idx] << (8 * j); }
    out[i] = w;
  }
}

// A finalized node: its input CV + last block, enough to recompute root or chaining value.
struct Output {
  uint32_t input_cv[8];
  uint32_t block_words[16];
  uint64_t counter;
  uint32_t block_len;
  uint32_t flags;
};
inline void output_cv(const Output& o, uint32_t cv[8]) {
  uint32_t t[16]; b3_compress(o.input_cv, o.block_words, o.counter, o.block_len, o.flags, t);
  for (int i = 0; i < 8; i++) cv[i] = t[i];
}
inline void output_root(const Output& o, uint8_t out[32]) {
  uint32_t t[16]; b3_compress(o.input_cv, o.block_words, 0, o.block_len, o.flags | B3_ROOT, t);
  for (int i = 0; i < 8; i++) { out[i * 4] = t[i]; out[i * 4 + 1] = t[i] >> 8; out[i * 4 + 2] = t[i] >> 16; out[i * 4 + 3] = t[i] >> 24; }
}
inline void parent_output(const uint32_t left[8], const uint32_t right[8], const uint32_t key[8], uint32_t flags, Output& o) {
  for (int i = 0; i < 8; i++) { o.input_cv[i] = key[i]; o.block_words[i] = left[i]; o.block_words[i + 8] = right[i]; }
  o.counter = 0; o.block_len = 64; o.flags = B3_PARENT | flags;
}

// One-shot keyed (key32 != nullptr) or unkeyed BLAKE3 -> out[32].
inline void b3(const uint8_t* input, uint32_t len, const uint8_t* key32, uint8_t out[32]) {
  uint32_t key[8], base_flags = 0;
  if (key32) { for (int i = 0; i < 8; i++) key[i] = le32(key32 + i*4); base_flags = B3_KEYED; }
  else iv_words(key);

  uint32_t cv_stack[54][8];
  int stack_len = 0;
  uint64_t chunk_counter = 0;
  uint32_t off = 0;

  // chunk state
  uint32_t cv[8];
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  uint8_t block[64];
  uint32_t block_len = 0, blocks_compressed = 0;

  auto start_flag = [&]() -> uint32_t { return blocks_compressed == 0 ? B3_CHUNK_START : 0u; };

  // produce the final Output of the current chunk
  auto chunk_output = [&](Output& o) {
    uint32_t w[16]; words_from_le(block, block_len, w);
    for (int i = 0; i < 8; i++) o.input_cv[i] = cv[i];
    for (int i = 0; i < 16; i++) o.block_words[i] = w[i];
    o.counter = chunk_counter; o.block_len = block_len; o.flags = base_flags | start_flag() | B3_CHUNK_END;
  };
  auto reset_chunk = [&](uint64_t counter) {
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    block_len = 0; blocks_compressed = 0; chunk_counter = counter;
  };
  auto add_cv = [&](uint32_t new_cv[8], uint64_t total_chunks) {
    while ((total_chunks & 1) == 0) {
      Output po; parent_output(cv_stack[stack_len - 1], new_cv, key, base_flags, po);
      stack_len--; uint32_t merged[8]; output_cv(po, merged);
      for (int i = 0; i < 8; i++) new_cv[i] = merged[i];
      total_chunks >>= 1;
    }
    for (int i = 0; i < 8; i++) cv_stack[stack_len][i] = new_cv[i];
    stack_len++;
  };

  while (off < len) {
    // Finalize a full chunk before adding more input (keeps its 16th block as CHUNK_END).
    if (64u * blocks_compressed + block_len == B3_CHUNK_LEN) {
      Output o; chunk_output(o);
      uint32_t ccv[8]; output_cv(o, ccv);
      uint64_t total = chunk_counter + 1;
      add_cv(ccv, total);
      reset_chunk(total);
    }
    uint32_t want_chunk = B3_CHUNK_LEN - (64u * blocks_compressed + block_len);
    uint32_t avail = len - off;
    uint32_t take = want_chunk < avail ? want_chunk : avail;
    uint32_t t = take;
    while (t > 0) {
      if (block_len == 64) {  // compress a non-final block (more data follows in this chunk)
        uint32_t w[16]; words_from_le(block, 64, w);
        uint32_t cout[16]; b3_compress(cv, w, chunk_counter, 64, base_flags | start_flag(), cout);
        for (int i = 0; i < 8; i++) cv[i] = cout[i];
        blocks_compressed++; block_len = 0;
      }
      uint32_t want = 64 - block_len;
      uint32_t n = want < t ? want : t;
      for (uint32_t i = 0; i < n; i++) block[block_len + i] = input[off + i];
      block_len += n; off += n; t -= n;
    }
  }

  // finalize
  Output o; chunk_output(o);
  int rem = stack_len;
  while (rem > 0) {
    rem--;
    uint32_t ocv[8]; output_cv(o, ocv);
    parent_output(cv_stack[rem], ocv, key, base_flags, o);
  }
  output_root(o, out);
}

// ---- Merkle-style tree primitives (for parallel keyed-BLAKE3 of large buffers) ----
inline void load_key(const uint8_t* key32, uint32_t key[8], uint32_t& base_flags) {
  if (key32) { for (int i = 0; i < 8; i++) key[i] = le32(key32 + i*4); base_flags = B3_KEYED; }
  else { iv_words(key); base_flags = 0; }
}
inline void cv_to_bytes(const uint32_t cv[8], uint8_t out[32]) {
  for (int i = 0; i < 8; i++) { out[i*4]=cv[i]; out[i*4+1]=cv[i]>>8; out[i*4+2]=cv[i]>>16; out[i*4+3]=cv[i]>>24; }
}
// Non-root chaining value of one chunk (<=1024 bytes) at chunkIndex.
inline void chunk_cv(const uint8_t* data, uint32_t len, uint64_t chunkIndex, const uint8_t* key32, uint8_t out[32]) {
  uint32_t key[8], base; load_key(key32, key, base);
  uint32_t cv[8]; for (int i = 0; i < 8; i++) cv[i] = key[i];
  uint32_t blocks = (len + 63) / 64; if (blocks == 0) blocks = 1;
  for (uint32_t bi = 0; bi < blocks; bi++) {
    uint32_t bstart = bi * 64; uint32_t blen = (len > bstart) ? (len - bstart < 64 ? len - bstart : 64) : 0;
    uint32_t flags = base; if (bi == 0) flags |= B3_CHUNK_START; if (bi == blocks - 1) flags |= B3_CHUNK_END;
    uint32_t w[16]; words_from_le(data + bstart, blen, w);
    uint32_t t[16]; b3_compress(cv, w, chunkIndex, blen, flags, t);
    for (int i = 0; i < 8; i++) cv[i] = t[i];
  }
  cv_to_bytes(cv, out);
}
// Parent CV of two child CVs; isRoot applies the ROOT flag (top of tree).
inline void parent_cv(const uint8_t l[32], const uint8_t r[32], const uint8_t* key32, bool isRoot, uint8_t out[32]) {
  uint32_t key[8], base; load_key(key32, key, base);
  uint32_t block[16];
  for (int i = 0; i < 8; i++) { block[i] = le32(l + i*4); block[8+i] = le32(r + i*4); }
  uint32_t t[16]; b3_compress(key, block, 0, 64, base | B3_PARENT | (isRoot ? B3_ROOT : 0), t);
  cv_to_bytes(t, out);
}
// Merge numChunks contiguous CVs (cvs[i*32]) into the BLAKE3 root, applying ROOT at the top.
// Iterative largest-power-of-two-subtree split (matches pearl-blake3 merkle.rs / blake3.js).
inline void merge_root(uint8_t* cvs, int numChunks, const uint8_t* key32, uint8_t out[32]) {
  if (numChunks == 1) { for (int i = 0; i < 32; i++) out[i] = cvs[i]; return; }
  // explicit stack of half-open ranges to process post-order
  struct Fr { int lo, hi; bool done; };
  Fr stk[64]; int sp = 0;
  uint8_t res[64][32]; int rsp = 0;  // results stack of CVs
  stk[sp++] = {0, numChunks, false};
  while (sp > 0) {
    Fr& f = stk[sp - 1];
    if (f.hi - f.lo == 1) { for (int i = 0; i < 32; i++) res[rsp][i] = cvs[f.lo * 32 + i]; rsp++; sp--; continue; }
    if (!f.done) {
      f.done = true;
      int sz = f.hi - f.lo, p = 1; while (p * 2 < sz) p *= 2;
      int mid = f.lo + p;
      stk[sp++] = {mid, f.hi, false};  // right pushed first -> processed/popped after left
      stk[sp++] = {f.lo, mid, false};
    } else {
      // children results are top two of res stack: left then right (left pushed last -> on top)
      bool isRoot = (f.lo == 0 && f.hi == numChunks);
      uint8_t left[32], right[32];
      for (int i = 0; i < 32; i++) { left[i] = res[rsp - 2][i]; right[i] = res[rsp - 1][i]; }
      rsp -= 2;
      parent_cv(left, right, key32, isRoot, res[rsp]); rsp++;
      sp--;
    }
  }
  for (int i = 0; i < 32; i++) out[i] = res[0][i];
}

}  // namespace pearlhash_b3
