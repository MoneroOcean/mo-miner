// Host policy shared by compact admission, keying, and proof capture.
// Include after host.inc inside mom_pearlhash and job-boundary.h outside it;
// no SYCL or CUDA dependency.
struct CompactProfile {
  static constexpr int k = 65536, rank = 128, rows = 24, cols = 8;
  static constexpr std::array<uint8_t, 6> row_pattern = {0, 1, 7, 11, 0, 0};
  static constexpr std::array<uint8_t, 6> col_pattern = {7, 1, 3, 3, 0, 0};
  static constexpr ProofAxis row_axis = kCompactRows, col_axis = kCompactCols;
  static_assert(same_pattern(row_axis.bytes(), row_pattern) &&
                same_pattern(col_axis.bytes(), col_pattern) && row_axis.count() == rows &&
                col_axis.count() == cols);

  static bool supports(int m, int n, int requested_k, int requested_rank) {
    // B's wrapped global raw index must never alias A's seed-bearing prefix.
    return m > 0 && n > 0 && m % 192 == 0 && n % 256 == 0 &&
        requested_k == k && requested_rank == rank &&
        mom::job_boundary::valid_pearlhash_shape(m, n, requested_k, requested_rank) &&
        (uint64_t(m) + uint64_t(n)) * uint64_t(k) <= uint64_t(UINT32_MAX) + 1;
  }

  static bool row_valid(int row, int m) {
    // Eight even origins per 192-row CTA open twelve adjacent pairs, 16 rows apart.
    return row >= 0 && m >= 178 && row <= m - 178 && row % 192 < 16 && row % 2 == 0;
  }

  // Expand a winning compact origin into its proof's 24 A rows (adjacent pairs 16 apart) and 8 B
  // columns (pairs 8 apart, 64 between pairs). Returns false, with both lists empty, for an origin
  // the compact kernel can never report.
  static bool lists(int row, int col, int m, int n,
                    std::vector<int>& a, std::vector<int>& b) {
    a.clear(); b.clear();
    if (!row_valid(row, m) || !(col >= 0 && n >= 201 && col <= n - 201 && col % 256 < 56 && col % 16 < 8))
      return false;
    for (int i = 0; i < rows; ++i)
      a.push_back(row + 16 * (i / 2) + i % 2);
    for (int i = 0; i < cols; ++i)
      b.push_back(col + 64 * (i / 2) + 8 * (i % 2));
    return true;
  }

  // Job key: BLAKE3 of the 76-byte header followed by the compact profile's config bytes.
  static void derive_key(const uint8_t* header, uint8_t* key) {
    if (!header || !key) throw std::string("Missing compact job key input");
    uint8_t block[128];
    std::memcpy(block, header, 76);
    config_bytes_pattern(block + 76, k, rank, row_pattern, col_pattern);
    pearlhash_b3::b3(block, sizeof(block), nullptr, key);
  }
};

// Only A's first32 raw bytes change per seed; B is fixed. Keep their existing
// non-root trees for both first-leaf updates and winner serialization. A new
// key/shape must capture fresh trees after the complete producer has finished.
// Callers serialize under the state mutex, never retain mutable tree pointers.
class CompactRootPath {
  std::array<uint8_t, 32> key_{};
  std::vector<std::vector<CV>> a_, b_;
  int m_ = 0, n_ = 0;
  uint32_t seed_ = 0, b_seed_ = 0;
  bool valid_ = false;

  // Chaining value of A's first 1024-byte leaf, the only leaf that depends on the seed.
  static CV leaf(uint32_t seed, int m, int n, const uint8_t* key) {
    uint8_t bytes[1024];
    const size_t size = gen_leaf('A', seed, 0, m, n, CompactProfile::k, bytes, true, true);
    CV result{};
    pearlhash_b3::chunk_cv(bytes, uint32_t(size), 0, key, result.data());
    return result;
  }

  // Finish only the residual top that did not fit in the preserved even levels. Compact shape
  // admission bounds this suffix well below the full tree; larger parents remain GPU-produced.
  static CV finish_tree(std::vector<std::vector<CV>>& layers, const size_t leaf_count,
                        const char which, const uint32_t seed, const int m, const int n,
                        const int k, const uint8_t* key, const CV& gpu_root) {
    size_t count = leaf_count, index = 0;
    while (true) {
      if (index >= layers.size() || layers[index].size() != count)
        throw std::string("Invalid compact PearlHash retained level");
      ++index;
      if (count <= 2 || count % 2)
        break;
      count /= 2;
    }
    if (index != layers.size() || count > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::string("Invalid compact PearlHash retained prefix");
    if (count > 2) {
      auto top = build_layers(which, seed, static_cast<int>(count), m, n, k, key,
                              &layers.back(), true, true);
      if (top.empty() || top.front() != layers.back())
        throw std::string("Invalid compact PearlHash top reconstruction");
      for (size_t level = 1; level < top.size(); ++level)
        layers.push_back(std::move(top[level]));
    }
    if (layers.back().size() != 2)
      throw std::string("Invalid compact PearlHash root level");
    CV top[2] = {layers.back()[0], layers.back()[1]};
    CV root{};
    pearlhash_b3::merge_root(reinterpret_cast<uint8_t*>(top), 2, key, root.data());
    if (root != gpu_root)
      throw std::string("Invalid compact PearlHash GPU Merkle root");
    return root;
  }

 public:
  CV first_leaf{}, root{}; // State-owned upload sources survive asynchronous queue submission.

  void reset() noexcept {
    valid_ = false;
    m_ = n_ = 0;
    seed_ = b_seed_ = 0;
    key_.fill(0);
    a_.clear();
    b_.clear();
    first_leaf.fill(0);
    root.fill(0);
  }
  bool valid() const noexcept { return valid_; }

  // Retain the GPU-produced complete levels for a new key/shape and finish only the small top.
  // The first leaves and final roots are checked before the path becomes visible.
  void capture(const std::array<uint8_t, 32>& key, int m, int n, int k, int rank,
               uint32_t seed, std::vector<std::vector<CV>> a,
               std::vector<std::vector<CV>> b, const CV& gpu_root_a,
               const CV& gpu_root_b) {
    reset(); // A failed new-job capture must not expose the previous job's path.
    if (!CompactProfile::supports(m, n, k, rank))
      throw std::string("Invalid compact root capture");
    const size_t a_leaves = checked_chunk_count(checked_elements(m, k, "compact A"), "compact A");
    const size_t b_leaves = checked_chunk_count(checked_elements(n, k, "compact B"), "compact B");
    const CV a_root = finish_tree(a, a_leaves, 'A', seed, m, n, k, key.data(), gpu_root_a);
    static_cast<void>(finish_tree(b, b_leaves, 'B', seed, m, n, k, key.data(), gpu_root_b));
    if (a.empty() || a.front().empty() || a.front().front() != leaf(seed, m, n, key.data()))
      throw std::string("Invalid compact A capture");
    uint8_t bytes[1024];
    CV first_b{};
    const size_t len = gen_leaf('B',seed,0,m,n,k,bytes,true,true);
    pearlhash_b3::chunk_cv(bytes,uint32_t(len),0,key.data(),first_b.data());
    if (b.empty() || b.front().empty() || b.front().front() != first_b)
      throw std::string("Invalid compact B capture");
    key_ = key;
    m_ = m;
    n_ = n;
    b_seed_ = seed;
    a_ = std::move(a);
    b_ = std::move(b);
    first_leaf = a_.front().front();
    root = a_root;
    seed_ = seed;
    valid_ = true;
  }

  // Recompute A's root for a new seed by replacing its first leaf and rehashing only the leftmost
  // path. A key/shape that differs from the captured trees resets the path and throws.
  void update(const std::array<uint8_t, 32>& key, int m, int n, int k, int rank,
              uint32_t seed) {
    if (!valid_ || key != key_ || m != m_ || n != n_ ||
        !CompactProfile::supports(m, n, k, rank)) {
      reset();
      throw std::string("Stale compact root path");
    }
    first_leaf = leaf(seed, m, n, key.data());
    a_.front().front() = first_leaf;
    for (size_t i = 1; i < a_.size(); ++i) {
      pearlhash_b3::parent_cv(a_[i-1][0].data(),a_[i-1][1].data(),key.data(),
                             false,a_[i][0].data());
    }
    // Legal compact trees always end in two nodes. Keep the ROOT merge on
    // the stack, both to preserve non-root CVs and to avoid allocation in update.
    CV top[2] = {a_.back()[0],a_.back()[1]};
    pearlhash_b3::merge_root(reinterpret_cast<uint8_t*>(top),2,
                             key.data(),root.data());
    seed_ = seed;
  }

  // Serialize the PlainProof for a winning origin from the retained trees; empty when the origin
  // is not a legal compact proof. The trees must match this exact key, shape and seed.
  std::string proof(const std::array<uint8_t,32>& key,int m,int n,int k,int rank,
                     uint32_t seed,int row,int col,uint32_t* factor) {
    if (!valid_ || key != key_ || m != m_ || n != n_ || seed != seed_ ||
        !CompactProfile::supports(m,n,k,rank)) {
      reset();
      throw std::string("Stale compact proof tree");
    }
    std::vector<int> rows,cols;
    if (!CompactProfile::lists(row,col,m,n,rows,cols)) return {};
    return build_plain_proof_lists(seed,m,n,k,rank,key.data(),rows,cols,factor,
                                   nullptr,nullptr,true,&b_seed_,&a_,&b_,true);
  }
};
