// 52-byte NoisyGEMM config block hashed with the 76-byte header into the per-job key (k, rank, and
// the two sparsity bytes at [9]/[15]); bytes 0..3 = k LE, 4..5 = rank LE.
static void config_bytes(uint8_t cfg[52], int k, int rank) {
  memset(cfg, 0, 52);
  cfg[0] = k & 0xff;
  cfg[1] = (k >> 8) & 0xff;
  cfg[2] = (k >> 16) & 0xff;
  cfg[3] = (k >> 24) & 0xff;
  cfg[4] = rank & 0xff;
  cfg[5] = (rank >> 8) & 0xff;
  cfg[9] = 15;
  cfg[15] = 15;
}
// Per-job key = unkeyed BLAKE3 of the 76-byte header concatenated with the 52-byte config block.
static void derive_key(const uint8_t* header76, int k, int rank, uint8_t key32[32]) {
  uint8_t cfg[52], kb[128];
  config_bytes(cfg, k, rank);
  memcpy(kb, header76, 76);
  memcpy(kb + 76, cfg, 52);
  pearlhash_b3::b3(kb, 128, nullptr, key32);
}

// ---- PlainProof construction on the host (the pool submission for a winning tile) ----
// On a winning tile the server builds the pool submission itself: it regenerates the A / Bt byte
// streams from the counter-RNG, builds the keyed-BLAKE3 Merkle tree, opens the 16 revealed rows
// (A) and cols (Bt), bincode-serializes and base64-encodes. Mirrors pearl-blake3 merkle.rs and is
// byte-identical to the pool's raw-root proof format. Certificate V3 changes only the noise-seed
// commitment chain, not these serialized roots. Runs once per found
// share (rare), so plain host code -- no GPU. CHUNK=1024; leaves use non-root chunk_cv, the root
// uses merge_root (ROOT at top); for power-of-two leaf counts the pairwise odd-carry layer tree
// (siblings) is identical to merge_root's tree.
using CV = std::array<uint8_t, 32>;
// Regenerate leaf `leaf` (1024 bytes, zero-padded) of A ('A') or Bt ('B') from the seed RNG.
static size_t gen_leaf(char which, uint32_t seed, size_t leaf, int m, int n, int k,
                       uint8_t out[1024]) {
  const size_t total = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves = checked_chunk_count(total, "PlainProof matrix");
  if (leaf >= num_leaves)
    throw std::string("Invalid pearlhash PlainProof leaf");
  const size_t offset = leaf * 1024;
  const size_t len = std::min(size_t{1024}, total - offset);
  const uint32_t tot = static_cast<uint32_t>(checked_elements(m, k, "M*K PlainProof"));
  for (size_t t = 0; t < len; t++) {
    const size_t byte = offset + t;
    if (which == 'A') {
      out[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(byte)));
    } else {
      const size_t source_offset =
          (byte % static_cast<size_t>(k)) * static_cast<size_t>(n) + byte / k;
      out[t] = static_cast<uint8_t>(gv(seed, tot + static_cast<uint32_t>(source_offset)));
    }
  }
  for (size_t t = len; t < 1024; t++)
    out[t] = 0;
  return len;
}
// Build non-root Merkle layers, carrying an unmatched node unchanged for proof openings.
static std::vector<std::vector<CV>> build_layers(char which, uint32_t seed, int numLeaves, int m,
                                                 int n, int k, const uint8_t* key) {
  std::vector<CV> l0(numLeaves);
  uint8_t chunk[1024];
  for (int i = 0; i < numLeaves; i++) {
    int len = gen_leaf(which, seed, i, m, n, k, chunk);
    pearlhash_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data());
  }
  std::vector<std::vector<CV>> layers;
  layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back();
    std::vector<CV> next;
    for (size_t i = 0; i < prev.size(); i += 2) {
      if (i + 1 < prev.size()) {
        CV p;
        pearlhash_b3::parent_cv(prev[i].data(), prev[i + 1].data(), key, false, p.data());
        next.push_back(p);
      } else
        next.push_back(prev[i]);
    }
    layers.push_back(std::move(next));
  }
  return layers;
}
static void put_u64le(std::vector<uint8_t>& v, uint64_t n) {
  for (int i = 0; i < 8; i++)
    v.push_back((uint8_t)(n >> (8 * i)));
}
static void put_bytes(std::vector<uint8_t>& v, const uint8_t* p, size_t n) {
  v.insert(v.end(), p, p + n);
}
// Append a serialized MatrixMerkleProof for the 16 contiguous `rows` of matrix `which` (cols wide).
static void ser_matrix_proof(std::vector<uint8_t>& out, char which, uint32_t seed, int m, int n,
                             int k, const uint8_t* key, int cols, const std::vector<int>& rows) {
  const size_t total_bytes = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves_size = checked_chunk_count(total_bytes, "PlainProof matrix");
  const int numLeaves = static_cast<int>(num_leaves_size);
  auto layers = build_layers(which, seed, numLeaves, m, n, k, key);
  uint8_t root[32];
  {
    std::vector<CV> tmp = layers[0];
    pearlhash_b3::merge_root((uint8_t*)tmp.data(), numLeaves, key, root);
  }
  std::set<int> leafset;
  for (int row : rows) {
    const size_t row_start = static_cast<size_t>(row) * static_cast<size_t>(cols);
    const size_t row_end = (static_cast<size_t>(row) + 1) * static_cast<size_t>(cols);
    const size_t first = row_start / 1024;
    const size_t last = (row_end - 1) / 1024;
    if (last >= num_leaves_size || last > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::string("Invalid pearlhash PlainProof row leaf range");
    for (size_t i = first; i <= last; i++)
      leafset.insert(static_cast<int>(i));
  }
  std::vector<int> leaves(leafset.begin(), leafset.end());
  // MerkleProof: leaf_data (Vec<[u8;1024]>), leaf_indices, total_leaves, root, siblings
  put_u64le(out, leaves.size());
  uint8_t chunk[1024];
  for (int li : leaves) {
    gen_leaf(which, seed, static_cast<size_t>(li), m, n, k, chunk);
    put_u64le(out, 1024);
    put_bytes(out, chunk, 1024);
  }
  put_u64le(out, leaves.size());
  for (int li : leaves)
    put_u64le(out, (uint64_t)li);
  put_u64le(out, (uint64_t)numLeaves);
  put_bytes(out, root, 32);
  // siblings: walk up, emit each node whose sibling is not itself in the active set
  std::set<int> cur(leafset);
  size_t levelLen = num_leaves_size;
  size_t level = 0;
  std::vector<CV> sib;
  while (levelLen > 1 && !cur.empty()) {
    auto& nodes = layers[level];
    for (int i : cur) {
      if (i % 2 == 1) {
        if (!cur.count(i - 1))
          sib.push_back(nodes[i - 1]);
      } else if (!cur.count(i + 1) && static_cast<size_t>(i + 1) < levelLen)
        sib.push_back(nodes[i + 1]);
    }
    std::set<int> nx;
    for (int i : cur)
      nx.insert(i >> 1);
    cur.swap(nx);
    levelLen = levelLen / 2 + levelLen % 2;
    level++;
  }
  put_u64le(out, sib.size());
  for (auto& s : sib)
    put_bytes(out, s.data(), 32);
  // MatrixMerkleProof tail: row_indices
  put_u64le(out, rows.size());
  for (int r : rows)
    put_u64le(out, (uint64_t)r);
}
// Encode the serialized PlainProof with standard base64 padding for pool submission.
static std::string base64(const std::vector<uint8_t>& d) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  for (size_t i = 0; i < d.size(); i += 3) {
    uint32_t x =
        d[i] << 16 | (i + 1 < d.size() ? d[i + 1] << 8 : 0) | (i + 2 < d.size() ? d[i + 2] : 0);
    o += T[(x >> 18) & 63];
    o += T[(x >> 12) & 63];
    o += (i + 1 < d.size()) ? T[(x >> 6) & 63] : '=';
    o += (i + 2 < d.size()) ? T[x & 63] : '=';
  }
  return o;
}
// Full PlainProof (base64) for a winning tile, from the seed + job key (no noised matrices needed).
static std::string build_plain_proof(uint32_t seed, int m, int n, int k, int rank,
                                     const uint8_t* key, int row, int col,
                                     uint32_t* adjustment_factor) {
  std::vector<int> aRows, btRows;
  for (int r = 0; r < 16; r++) {
    aRows.push_back(row + r);
    btRows.push_back(col + r);
  }
  if (!pearlhash_adjustment_factor(aRows.size(), btRows.size(), k, rank, adjustment_factor))
    return {};
  std::vector<uint8_t> buf;
  put_u64le(buf, (uint64_t)m);
  put_u64le(buf, (uint64_t)n);
  put_u64le(buf, (uint64_t)k);
  put_u64le(buf, (uint64_t)rank);
  ser_matrix_proof(buf, 'A', seed, m, n, k, key, k, aRows);
  ser_matrix_proof(buf, 'B', seed, m, n, k, key, k, btRows);
  buf.push_back(0x00); // moe: Option::None
  return base64(buf);
}
