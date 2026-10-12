// Recorded-vector verification and staged-backend policy for the host dispatcher.
static void test_recorded_vector(State& state, const uint8_t* input, uint8_t* output,
                                 uint64_t* nonce) {
  uint8_t serialized[49], mining_hash[32], final_hash[32];
  np_nonce_bytes(input, input + 32, kRecordedMinerNonce, serialized);
  nexapow_test_sha256d_49(state.queue, serialized, mining_hash);
  if (!np_equal(mining_hash, kRecordedMiningHash, sizeof(mining_hash)) ||
      !np_hash_one(input, input + 32, kRecordedMinerNonce, final_hash) ||
      !np_equal(final_hash, kRecordedHash, sizeof(final_hash)))
    throw std::string("NexaPoW recorded Echelon vector mismatch");
  *nonce = kRecordedMinerNonce;
  std::memcpy(output, final_hash, sizeof(final_hash));
}

// Resolve the existing environment contract: any nonempty value except literal 0 enables the option.
static bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value && *value && std::strcmp(value, "0");
}

// Describe the compiled field policy; AdaptiveCpp selects its representation at device JIT time.
static const char* staged_field_name(const sycl::device& device) {
#if defined(MOM_NEXAPOW_SYCL_NATIVE_FIELD)
  if (mom_is_cuda(device))
    return "nvptx-native";
  return "portable-10x26";
#elif defined(MOM_NEXAPOW_PORTABLE_FIELD32)
  (void)device;
  return "portable-8x32";
#elif defined(MOM_SYCL_ADAPTIVECPP_CUDA) && !defined(MOM_NEXAPOW_FORCE_PORTABLE)
  (void)device;
  return "jit-ptx-or-10x26";
#else
  (void)device;
  return "portable-10x26";
#endif
}
