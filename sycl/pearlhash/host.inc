// 52-byte NoisyGEMM config block hashed with the 76-byte header into the per-job key: bytes 0..3 =
// k LE, 4..5 = rank LE, then the six-byte row and column opening-pattern descriptors at [8]/[14].
static void config_bytes_pattern(uint8_t cfg[52], int k, int rank,
                                 const std::array<uint8_t, 6>& rows,
                                 const std::array<uint8_t, 6>& cols) {
  memset(cfg, 0, 52);
  cfg[0] = k & 0xff;
  cfg[1] = (k >> 8) & 0xff;
  cfg[2] = (k >> 16) & 0xff;
  cfg[3] = (k >> 24) & 0xff;
  cfg[4] = rank & 0xff;
  cfg[5] = (rank >> 8) & 0xff;
  std::copy(rows.begin(), rows.end(), cfg + 8);
  std::copy(cols.begin(), cols.end(), cfg + 14);
}
// Contiguous rows x cols openings encode only their extent minus one at pattern byte 1.
static void config_bytes(uint8_t cfg[52], int k, int rank, int rows = 16, int cols = 16) {
  config_bytes_pattern(cfg, k, rank, {0, static_cast<uint8_t>(rows - 1), 0, 0, 0, 0},
                       {0, static_cast<uint8_t>(cols - 1), 0, 0, 0, 0});
}
// Per-job key = unkeyed BLAKE3 of the 76-byte header concatenated with the 52-byte config block.
static void key_from_config(const uint8_t* header76, const uint8_t cfg[52], uint8_t key32[32]) {
  uint8_t kb[128];
  memcpy(kb, header76, 76);
  memcpy(kb + 76, cfg, 52);
  pearlhash_b3::b3(kb, 128, nullptr, key32);
}
static void derive_key(const uint8_t* header76, int k, int rank, uint8_t key32[32],
                       int rows = 16, int cols = 16) {
  uint8_t cfg[52];
  config_bytes(cfg, k, rank, rows, cols);
  key_from_config(header76, cfg, key32);
}

// Native CUDA rank-128 pairs use the verifier's canonical 8-by-8-by-2 opening
// pattern on both axes.  Keep this private: the public job/result ABI remains
// the same 16-by-16 proof area and factor.
static constexpr std::array<int, 16> kPairDist8Offsets = {
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57};
static constexpr std::array<uint8_t, 6> kPairDist8PatternBytes = {0, 1, 3, 7, 0, 0};

static bool pair_dist8_offset_valid(int offset, int dimension) {
  if (offset < 0 || dimension < 58 || offset > dimension - 58)
    return false;
  const int residue = offset % 64;
  return residue == 0 || residue == 2 || residue == 4 || residue == 6;
}

// Expand a pair-dist8 winner into its 16 opened A rows and Bt rows; false for an illegal origin.
static bool pair_dist8_lists(int row, int col, int m, int n,
                             std::vector<int>& a_rows, std::vector<int>& bt_rows) {
  if (!pair_dist8_offset_valid(row, m) || !pair_dist8_offset_valid(col, n))
    return false;
  a_rows.clear();
  bt_rows.clear();
  a_rows.reserve(kPairDist8Offsets.size());
  bt_rows.reserve(kPairDist8Offsets.size());
  for (const int offset : kPairDist8Offsets) {
    a_rows.push_back(row + offset);
    bt_rows.push_back(col + offset);
  }
  return true;
}

static void derive_key_pair_dist8(const uint8_t* header76, int k, int rank, uint8_t key32[32]) {
  uint8_t cfg[52];
  config_bytes_pattern(cfg, k, rank, kPairDist8PatternBytes, kPairDist8PatternBytes);
  key_from_config(header76, cfg, key32);
}

// ---- PlainProof construction on the host (the pool submission for a winning tile) ----
// On a winning tile the server builds the pool submission itself: it regenerates the A / Bt byte
// streams from the counter-RNG, builds the keyed-BLAKE3 Merkle trees, opens the selected rows
// (A) and columns (Bt), bincode-serializes and base64-encodes. Mirrors pearl-blake3 merkle.rs and is
// byte-identical to the pool's raw-root proof format. Certificate V3 changes only the noise-seed
// commitment chain, not these serialized roots. Runs once per found
// share. CHUNK=1024; leaves use non-root chunk_cv, while the root combines the retained top
// layer with ROOT set. Odd carries preserve the canonical left-power-of-two tree.
using CV = std::array<uint8_t, 32>;
// Regenerate leaf `leaf` (1024 bytes, zero-padded) of A ('A') or Bt ('B') from the seed RNG.
static size_t gen_leaf(char which, uint32_t seed, size_t leaf, int m, int n, int k,
                       uint8_t out[1024], bool positive_inputs = false,
                       bool seed_prefix_inputs = false) {
  const size_t total = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves = checked_chunk_count(total, "PlainProof matrix");
  if (leaf >= num_leaves)
    throw std::string("Invalid pearlhash PlainProof leaf");
  const size_t offset = leaf * 1024;
  const size_t len = std::min(size_t{1024}, total - offset);
  const uint32_t tot = static_cast<uint32_t>(checked_elements(m, k, "M*K PlainProof"));
  // B cannot alias A's seed prefix before uint32 indexing wraps. Fill the
  // constant region directly; other modes and wrapped streams retain the scalar path.
  if (positive_inputs && seed_prefix_inputs && n > 0 && tot >= 32 &&
      (uint64_t(m) + uint64_t(n)) * uint64_t(k) <= (uint64_t{1} << 32)) {
    std::memset(out, 64, len);
    if (which == 'A' && offset < 32)
      for (size_t t = 0; t < std::min(len, size_t{32} - offset); ++t)
        out[t] = static_cast<uint8_t>(63u + ((seed >> (offset + t)) & 1u));
    std::memset(out + len, 0, 1024 - len);
    return len;
  }
  for (size_t t = 0; t < len; t++) {
    const size_t byte = offset + t;
    if (which == 'A') {
      out[t] = static_cast<uint8_t>(gv(seed, static_cast<uint32_t>(byte),
                                        positive_inputs, seed_prefix_inputs));
    } else {
      const size_t source_offset =
          (byte % static_cast<size_t>(k)) * static_cast<size_t>(n) + byte / k;
      out[t] = static_cast<uint8_t>(
          gv(seed, tot + static_cast<uint32_t>(source_offset), positive_inputs, seed_prefix_inputs));
    }
  }
  for (size_t t = len; t < 1024; t++)
    out[t] = 0;
  return len;
}
// Build non-root Merkle layers, carrying an unmatched node unchanged for proof openings.
// `leaves`, when present, contains the exact raw non-root CVs for this seed/key; parent layers
// and the root remain rebuilt here so cached and regenerated proof paths share serialization.
static std::vector<std::vector<CV>> build_layers(char which, uint32_t seed, int numLeaves, int m,
                                                 int n, int k, const uint8_t* key,
                                                 const std::vector<CV>* leaves = nullptr,
                                                 bool positive_inputs = false,
                                                 bool seed_prefix_inputs = false) {
  if (leaves && leaves->size() != static_cast<size_t>(numLeaves))
    throw std::string("Invalid pearlhash PlainProof cached leaf count");
  std::vector<CV> l0;
  if (leaves) {
    l0 = *leaves;
  } else {
    l0.resize(numLeaves);
    uint8_t chunk[1024];
    for (int i = 0; i < numLeaves; i++) {
      int len = gen_leaf(which, seed, i, m, n, k, chunk, positive_inputs, seed_prefix_inputs);
      pearlhash_b3::chunk_cv(chunk, (uint32_t)len, (uint64_t)i, key, l0[i].data());
    }
  }
  std::vector<std::vector<CV>> layers;
  layers.push_back(std::move(l0));
  while (layers.back().size() > 2) {
    auto& prev = layers.back();
    std::vector<CV> next;
    next.reserve(prev.size() / 2 + prev.size() % 2);
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
// Append a MatrixMerkleProof for the selected rows of matrix `which` (cols wide).
static void ser_matrix_proof(std::vector<uint8_t>& out, char which, uint32_t seed, int m, int n,
                             int k, const uint8_t* key, int cols, const std::vector<int>& rows,
                             const std::vector<CV>* cached_leaves = nullptr,
                             bool positive_inputs = false,
                             const std::vector<std::vector<CV>>* cached_layers = nullptr,
                             bool seed_prefix_inputs = false) {
  const size_t total_bytes = checked_elements(which == 'A' ? m : n, k, "PlainProof matrix");
  const size_t num_leaves_size = checked_chunk_count(total_bytes, "PlainProof matrix");
  const int numLeaves = static_cast<int>(num_leaves_size);
  std::vector<std::vector<CV>> regenerated;
  if (!cached_layers)
    regenerated = build_layers(which, seed, numLeaves, m, n, k, key, cached_leaves,
                                 positive_inputs, seed_prefix_inputs);
  const auto& layers = cached_layers ? *cached_layers : regenerated;
  // Cache ownership binds key/seed separately. Validate its complete shape
  // here so an incomplete or stale-sized tree cannot cause a sibling overread.
  size_t expected = num_leaves_size, level_count = 0;
  do {
    if (level_count >= layers.size() || layers[level_count++].size() != expected)
      throw std::string("Invalid pearlhash PlainProof cached tree");
    if (expected <= 2) break;
    expected = expected / 2 + expected % 2;
  } while (true);
  if (level_count != layers.size())
    throw std::string("Invalid pearlhash PlainProof cached tree depth");
  uint8_t root[32];
  {
    // Parent layers already exist for the openings; only their final root remains to hash.
    std::vector<CV> tmp = layers.back();
    pearlhash_b3::merge_root((uint8_t*)tmp.data(), static_cast<int>(tmp.size()), key, root);
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
    gen_leaf(which, seed, static_cast<size_t>(li), m, n, k, chunk, positive_inputs, seed_prefix_inputs);
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
  const size_t groups = d.size() / 3 + (d.size() % 3 != 0);
  if (groups > o.max_size() / 4)
    throw std::string("PearlHash PlainProof base64 is too large");
  o.reserve(groups * 4);
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
// Full PlainProof (base64) for explicit selected rows/columns.  The old
// contiguous builder and pair-dist8 builder both delegate here so the Merkle
// serialization remains one implementation.
static std::string build_plain_proof_lists(
    uint32_t seed, int m, int n, int k, int rank, const uint8_t* key,
    const std::vector<int>& aRows, const std::vector<int>& btRows,
    uint32_t* adjustment_factor, const std::vector<CV>* cached_a = nullptr,
    const std::vector<CV>* cached_b = nullptr, bool positive_inputs = false,
    const uint32_t* b_seed = nullptr,
    const std::vector<std::vector<CV>>* a_layers = nullptr,
    const std::vector<std::vector<CV>>* b_layers = nullptr,
    bool seed_prefix_inputs = false) {
  auto valid_rows = [](const std::vector<int>& rows, int dimension) {
    return !rows.empty() && std::is_sorted(rows.begin(), rows.end()) &&
           std::adjacent_find(rows.begin(), rows.end()) == rows.end() &&
           rows.front() >= 0 && rows.back() < dimension;
  };
  if (!valid_rows(aRows, m) || !valid_rows(btRows, n))
    return {};
  if (!pearlhash_adjustment_factor(aRows.size(), btRows.size(), k, rank, adjustment_factor))
    return {};
  std::vector<uint8_t> buf;
  put_u64le(buf, (uint64_t)m);
  put_u64le(buf, (uint64_t)n);
  put_u64le(buf, (uint64_t)k);
  put_u64le(buf, (uint64_t)rank);
  ser_matrix_proof(buf, 'A', seed, m, n, k, key, k, aRows, cached_a, positive_inputs,
                     a_layers, seed_prefix_inputs);
  ser_matrix_proof(buf, 'B', b_seed ? *b_seed : seed, m, n, k, key, k, btRows, cached_b,
                   positive_inputs, b_layers, seed_prefix_inputs);
  buf.push_back(0x00); // moe: Option::None
  return base64(buf);
}

// Full PlainProof for the existing contiguous proof patterns.  This wrapper
// intentionally preserves the previous validation and byte path.
static std::string build_plain_proof(uint32_t seed, int m, int n, int k, int rank,
                                     const uint8_t* key, int row, int col,
                                     uint32_t* adjustment_factor,
                                     const std::vector<CV>* cached_a = nullptr,
                                     const std::vector<CV>* cached_b = nullptr,
                                     int rows = 16, int cols = 16,
                                     bool positive_inputs = false) {
  if (!((rows == 16 && cols == 16) || (rows == 8 && cols == 32) || (rows == 4 && cols == 64)) ||
      m < rows || n < cols || row < 0 || col < 0 ||
      row > m - rows || col > n - cols)
    return {};
  std::vector<int> aRows, btRows;
  for (int r = 0; r < rows; r++)
    aRows.push_back(row + r);
  for (int r = 0; r < cols; r++)
    btRows.push_back(col + r);
  return build_plain_proof_lists(seed, m, n, k, rank, key, aRows, btRows,
                                 adjustment_factor, cached_a, cached_b, positive_inputs);
}

// Pair-dist8's physical result offsets are the first elements of the canonical
// strided lists, not contiguous 16x16 tile origins.
static std::string build_pair_plain_proof(uint32_t seed, int m, int n, int k, int rank,
                                          const uint8_t* key, int row, int col,
                                          uint32_t* adjustment_factor,
                                          const std::vector<CV>* cached_a = nullptr,
                                          const std::vector<CV>* cached_b = nullptr,
                                          bool positive_inputs = false,
                                          const uint32_t* b_seed = nullptr) {
  std::vector<int> aRows, btRows;
  if (!pair_dist8_lists(row, col, m, n, aRows, btRows))
    return {};
  return build_plain_proof_lists(seed, m, n, k, rank, key, aRows, btRows,
                                 adjustment_factor, cached_a, cached_b, positive_inputs, b_seed);
}
