#if defined(MOM_EQ_FORWARD_MAP)
inline uint32_t l0_invert_hash(uint32_t x) {
  return x >> 14;
}

class ZelHashL0InvertSetupKernel;
class ZelHashL0InvertKernel;
static sycl::event submit_l0_invert(
  sycl::queue& q, MomKernelBundle& kb, const uint32_t* d_forward, EqCandidate* d_cand,
  uint32_t target_count, uint32_t* d_target, uint32_t* d_next, uint32_t* d_head
) {
  constexpr unsigned WG = 256;
  q.memset(d_head, 0xff, (size_t)L0_INVERT_HEADS * sizeof(uint32_t));
  uint32_t* const output = reinterpret_cast<uint32_t*>(d_cand);
  q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashL0InvertSetupKernel>(
      sycl::nd_range<1>(sycl::range<1>(((size_t)target_count + WG - 1u) / WG * WG),
                        sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) {
        const uint32_t i = (uint32_t)it.get_global_id(0);
        if (i >= target_count) return;
        const uint32_t key = output[i];
        d_target[i] = key;
        output[i] = 0xffffffffu;
        d_next[i] = dev_atomic_u32(d_head[l0_invert_hash(key)]).exchange(i);
      });
  });
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<ZelHashL0InvertKernel>(
      sycl::nd_range<1>(sycl::range<1>(NUM_ENTRIES), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> it) {
        const uint32_t leaf = (uint32_t)it.get_global_id(0);
        const uint32_t key = d_forward[leaf];
        if (key == 0xffffffffu) return;
        for (uint32_t node = d_head[l0_invert_hash(key)];
             node != 0xffffffffu; node = d_next[node]) {
          if (d_target[node] == key) output[node] = leaf;
        }
      });
  });
}
#endif
