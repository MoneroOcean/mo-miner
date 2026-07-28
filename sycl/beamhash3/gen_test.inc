// M1 generation validation: dump the first BEAMHASH3_TEST_ROWS raw leaf rows on-device.
class BeamGenTestKernel;
static sycl::event submit_gen_test(sycl::queue& q, MomKernelBundle& kb, const uint64_t* d_prePow,
                                   uint8_t* d_rows) {
  constexpr unsigned WG = 64;
  const size_t global = (BEAMHASH3_TEST_ROWS + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<BeamGenTestKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t e = (uint32_t)it.get_global_id(0);
      if (e >= BEAMHASH3_TEST_ROWS) return;
      uint64_t prePow[4];
      for (unsigned i = 0; i < 4; ++i) prePow[i] = d_prePow[i];
      uint64_t w[WORK_WORDS];
      make_leaf(prePow, e, w);
      uint8_t* row = d_rows + (size_t)e * ROW_BYTES;
      for (unsigned i = 0; i < WORK_WORDS; ++i)
        for (unsigned b = 0; b < 8; ++b)
          row[i * 8 + b] = (uint8_t)((w[i] >> (8 * b)) & 0xff);
    });
  });
}
