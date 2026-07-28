bool Core::process_message(const std::string& type, const MessageValues& v) {
  if (type == "job") {
    m_is_bench = false;
    if (!v.contains("target"))    throw std::string("Missing target job key");
    if (!v.contains("pool_id"))   throw std::string("Missing pool_id job key");
    if (!v.contains("worker_id")) throw std::string("Missing worker_id job key");
    if (!v.contains("job_id"))    throw std::string("Missing job_id job key");
    const std::string new_target_str = strip_hex_prefix(v.at("target"));

    const std::string algo = v.contains("algo") ? v.at("algo") : "";
    const bool is_kawpow_target = algo == "kawpow";
    // etchash/autolykos2/pearlhash/fishhash use a full 32-byte target instead of a single 64-bit word
    // (fishhash: live Iron Fish sends a 256-bit big-endian target -> m_target_bin -> meets_target_be_dev)
    const bool is_big_target = algo == "etchash" || algo == "autolykos2" || algo == "pearlhash" ||
      algo == "fishhash" || algo == "zelhash" || algo == "beamhash3" ||
      // KarlsenHashV2 compares the 32-byte hash against a full 256-bit BE boundary via
      // m_target_bin; without this the bin stays zero and live mining finds no shares.
      algo == "karlsenhashv2";
    const uint64_t new_target = is_big_target ? 1 : parse_target_hex(new_target_str, is_kawpow_target);
    uint8_t new_target_bin[HASH_LEN]{};
    if (is_big_target) parse_big_target_hex(new_target_str, new_target_bin);
    // The KarlsenHashV2 kernel compares the little-endian-stored 32-byte hash
    // against a little-endian target array (index 31 = MSB), but the share target hex is big-endian.
    // Reverse the bytes so target[31] is the MSB, matching the hash's LE byte layout. (is_test passes a
    // zero target, which is symmetric, so the offline vector path is unaffected.)
    if (is_big_target && algo == "karlsenhashv2")
      for (unsigned i = 0; i < HASH_LEN / 2; ++i) {
        const uint8_t t = new_target_bin[i];
        new_target_bin[i] = new_target_bin[HASH_LEN - 1 - i];
        new_target_bin[HASH_LEN - 1 - i] = t;
      }

    const uint64_t prev_last_nonce = last_nonce();
    const std::string prev_pool_id = m_pool_id;
    set_job(true, true, v, [&]() {
      m_target    = new_target;
      if (is_big_target) std::memcpy(m_target_bin, new_target_bin, HASH_LEN);
      m_pool_id   = v.at("pool_id");
      m_worker_id = v.at("worker_id");
      m_job_id    = v.at("job_id");
      m_header_hash = v.contains("header_hash") ? v.at("header_hash") : "";
    });
    if (prev_last_nonce) send_last_nonce(prev_last_nonce, m_nonce_bytes, prev_pool_id);

  } else if (type == "bench") {
    debug_startup("process bench start");
    m_is_bench = true;
    set_job(true, true, v);
    debug_startup("process bench done");
    m_target = 0;

  } else if (type == "test") {
    debug_startup("process test start");
    m_is_bench = false;
    set_job(false, false, v);
    debug_startup("process test done");
    m_nonce32 = 0;
    m_nonce64 = 0;
    m_target  = 0;

  } else if (type == "pause") {
    ++ m_job_ref; // to stop rx threads if any
    set_fn(nullptr);

  } else if (type == "read_msr" || type == "write_msr") {
     // in case of unknown MSR mod support stop here
     const auto pi = msr_mods.find(ci.msrMod());
     if (pi == msr_mods.end()) throw std::string("Unsupported CPU");

     // build default_msr_items from input message params
     const MsrItems& msr_items = pi->second;
     MsrItems default_msr_items;
     for (const auto& vi : v) {
       const std::string& key = vi.first;
       if (key.starts_with("msr:0x")) {
         const std::string& value = vi.second;
         const uint32_t reg = strtoul(key.substr(6).c_str(), NULL, 16);
         auto parts = tokenize(value, ',');
         if (parts.size() != 2 || !parts[0].starts_with("0x") || !parts[1].starts_with("0x"))
           throw std::string("Wrong value,mask MSR value: " + value);
         default_msr_items[reg] = MsrValue(
           strtoul(parts[0].substr(2).c_str(), NULL, 16),
           strtoul(parts[1].substr(2).c_str(), NULL, 16)
         );
       }
     }

     auto msr = xmrig::Msr::get();

     if (type == "read_msr") { // read missing default_msr_items and return them all as result
       MessageValues result;
       for (const auto& i : msr_items) {
         const uint32_t reg = i.first;
         MsrValue value;
         const auto pi = default_msr_items.find(reg);
         if (pi == default_msr_items.end()) {
           const auto msr_item = msr->read(reg);
           if (!msr_item.isValid()) throw fmt::format("Can't read MSR register {:#x}", reg);
           value = MsrValue(msr_item.value(), msr_item.mask());
         } else value = pi->second;
         result[fmt::format("msr:{:#x}", reg)] =
           fmt::format("{:#x},{:#x}", value.value, value.mask);
       }
       send_msg("read_msr", result);

     } else { // write msr
       // set the MSR mod if the active algo needs it, otherwise reset MSRs to default values
       const std::string algo = v.contains("algo") ? v.at("algo") : "";
       const bool is_set_msr =
         algo.starts_with("rx/") || algo == "ghostrider" || algo == "cn-heavy/xhv";

       for (const auto& i : msr_items) {
         const uint32_t reg = i.first;
         xmrig::MsrItem msr_item;
         if (!is_set_msr) {
           const auto pi = default_msr_items.find(reg);
           if (pi == default_msr_items.end()) continue; // this item was not recorded before
           msr_item = xmrig::MsrItem(reg, pi->second.value, pi->second.mask);
         } else msr_item = xmrig::MsrItem(reg, i.second.value, i.second.mask);
         msr->write([&msr, msr_item, reg](const int32_t cpu) {
           if (!msr->write(msr_item, cpu))
             throw fmt::format("Can't write MSR register {:#x}", reg);
           return true;
         });
       }
     }

     return true;

  } else if (type == "close") {
    const uint64_t prev_last_nonce = last_nonce();
    if (prev_last_nonce) send_last_nonce(prev_last_nonce, m_nonce_bytes, m_pool_id);
    free_memory();
    return false; // stop processing messages

  } else if (type == "algo_params") {
    get_algo_params(v);
  }

  return true; // continue processing messages
}

// MOM_LOOP_STATS=1 prints, every ~10s on stderr, how the compute loop wall time splits
// between GPU/CPU dispatch calls and the host work around them (plus one-off stall lines).
