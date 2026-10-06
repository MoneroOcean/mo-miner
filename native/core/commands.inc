bool Core::process_message(const std::string& type, const MessageValues& v) {
  if (type == "job") {
    if (!v.contains("target"))    throw std::string("Missing target job key");
    if (!v.contains("pool_id"))   throw std::string("Missing pool_id job key");
    if (!v.contains("worker_id")) throw std::string("Missing worker_id job key");
    if (!v.contains("job_id"))    throw std::string("Missing job_id job key");
    if (!v.contains("job_token") || v.at("job_token").empty())
      throw std::string("Missing job_token job key");
    const std::string new_target_str = strip_hex_prefix(v.at("target"));
    if (new_target_str.size() & 1) {
      throw std::string("Bad target hex");
    }

    const std::string algo = v.contains("algo") ? v.at("algo") : "";
    const bool is_kawpow_target = algo == "kawpow";
    // etchash/autolykos2/pearlhash/fishhash/xelishashv3 use a full 32-byte target instead of a single 64-bit word
    // (fishhash: live Iron Fish sends a 256-bit big-endian target -> m_target_bin -> meets_target_be_dev)
    const bool is_big_target = algo == "etchash" || algo == "octopus" || algo == "autolykos2" || algo == "pearlhash" ||
      algo == "fishhash" || algo == "c30" || algo == "equihash192_7" ||
      algo == "zelhash" || algo == "zhash" || algo == "beamhash3" ||
      // Kaspa-family hashes and Verthash compare against a full 256-bit boundary via m_target_bin.
      algo == "karlsenhashv2" || algo == "hoohash" || algo == "walahash" || algo == "verthash" ||
      // XelisHashV3 compares the digest and target in consensus byte order (index 0 is the MSB).
      algo == "xelishashv3" || algo == "nexapow";
    const uint64_t new_target = is_big_target ? 1 : parse_target_hex(new_target_str, is_kawpow_target);
    uint8_t new_target_bin[HASH_LEN]{};
    if (is_big_target) {
      parse_big_target_hex(new_target_str, new_target_bin);
    }
    // The KarlsenHashV2 and Verthash kernels compare their little-endian-stored 32-byte hashes
    // against a little-endian target array (index 31 = MSB), but the share target hex is big-endian.
    // Reverse the bytes so target[31] is the MSB, matching the hash's LE byte layout. (is_test passes a
    // zero target, which is symmetric, so the offline vector path is unaffected.)
    if (is_big_target && (algo == "karlsenhashv2" || algo == "hoohash" || algo == "walahash" || algo == "verthash"))
      for (unsigned i = 0; i < HASH_LEN / 2; ++i) {
        const uint8_t t = new_target_bin[i];
        new_target_bin[i] = new_target_bin[HASH_LEN - 1 - i];
        new_target_bin[HASH_LEN - 1 - i] = t;
      }

    const uint64_t prev_last_nonce = last_nonce();
    const unsigned prev_nonce_bytes = m_nonce_bytes;
    const std::string prev_pool_id = m_pool_id;
    const std::string prev_job_id = m_job_id;
    const std::string prev_job_token = m_job_token;
    set_job(JobMode::mine, true, v, [&]() {
      m_target    = new_target;
      if (is_big_target) {
        std::memcpy(m_target_bin, new_target_bin, HASH_LEN);
      }
      m_pool_id   = v.at("pool_id");
      m_worker_id = v.at("worker_id");
      m_job_id    = v.at("job_id");
      m_job_token = v.at("job_token");
      m_header_hash = v.contains("header_hash") ? v.at("header_hash") : "";
    });
    if (prev_last_nonce) {
      send_last_nonce(prev_last_nonce, prev_nonce_bytes, prev_pool_id, prev_job_id, prev_job_token);
    }

  } else if (type == "bench" || type == "test") {
    const bool is_bench = type == "bench";
    debug_async_worker(is_bench ? "process bench start" : "process test start");
    const uint64_t prev_last_nonce = last_nonce();
    const unsigned prev_nonce_bytes = m_nonce_bytes;
    const std::string prev_pool_id = m_pool_id;
    const std::string prev_job_id = m_job_id;
    const std::string prev_job_token = m_job_token;
    set_job(is_bench ? JobMode::bench : JobMode::test, is_bench, v);
    if (prev_last_nonce) {
      send_last_nonce(prev_last_nonce, prev_nonce_bytes, prev_pool_id, prev_job_id, prev_job_token);
    }
    debug_async_worker(is_bench ? "process bench done" : "process test done");

  } else if (type == "pause") {
    stop_rx_threads();
    clear_fn();

  } else if (type == "read_msr" || type == "write_msr") {
    const auto pi = msr_mods.find(xmrig::Cpu::info()->msrMod());
    if (pi == msr_mods.end()) throw std::string("Unsupported CPU");

    const MsrItems& msr_items = pi->second;
    MsrItems default_msr_items;
    for (const auto& vi : v) {
      const std::string& key = vi.first;
      if (!key.starts_with("msr:0x")) continue;

      uint32_t reg = 0;
      uint64_t value = 0;
      uint64_t mask = 0;
      std::string_view value_text;
      std::string_view mask_text;
      if (!mom::job_boundary::parse_unsigned(
            std::string_view(key).substr(6), reg, 16) ||
          !mom::job_boundary::split_once(vi.second, ',', value_text, mask_text) ||
          !value_text.starts_with("0x") || !mask_text.starts_with("0x") ||
          !mom::job_boundary::parse_unsigned(value_text, value, 16, true) ||
          !mom::job_boundary::parse_unsigned(mask_text, mask, 16, true)) {
        throw std::string("Wrong value,mask MSR value: " + vi.second);
      }
      default_msr_items[reg] = MsrValue(value, mask);
    }

    auto msr = xmrig::Msr::get();
    if (!msr) throw std::string("MSR access is unavailable");

    if (type == "read_msr") { // read missing default_msr_items and return them all as result
      MessageValues result;
      for (const auto& i : msr_items) {
        const uint32_t reg = i.first;
        MsrValue value;
        const auto default_item = default_msr_items.find(reg);
        if (default_item == default_msr_items.end()) {
          const auto msr_item = msr->read(reg);
          if (!msr_item.isValid()) throw fmt::format("Can't read MSR register {:#x}", reg);
          value = MsrValue(msr_item.value(), msr_item.mask());
        } else {
          value = default_item->second;
        }
        result[fmt::format("msr:{:#x}", reg)] =
          fmt::format("{:#x},{:#x}", value.value, value.mask);
      }
      send_msg("read_msr", result);
    } else { // write msr
      // Set the MSR mod if the active algo needs it; otherwise restore recorded defaults.
      const std::string algo = v.contains("algo") ? v.at("algo") : "";
      const bool is_set_msr =
        algo.starts_with("rx/") || algo == "ghostrider" || algo == "cn-heavy/xhv";

      for (const auto& i : msr_items) {
        const uint32_t reg = i.first;
        xmrig::MsrItem msr_item;
        if (!is_set_msr) {
          const auto default_item = default_msr_items.find(reg);
          if (default_item == default_msr_items.end()) continue;
          msr_item = xmrig::MsrItem(
            reg, default_item->second.value, default_item->second.mask);
        } else {
          msr_item = xmrig::MsrItem(reg, i.second.value, i.second.mask);
        }
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
    if (prev_last_nonce) {
      send_last_nonce(prev_last_nonce, m_nonce_bytes, m_pool_id, m_job_id, m_job_token);
    }
    free_memory();
    return false; // stop processing messages

  } else if (type == "algo_params") {
    get_algo_params(v);
  } else {
    throw std::string("Unknown native command: ") + type;
  }

  return true; // continue processing messages
}

// MOM_LOOP_STATS=1 prints, every ~10s on stderr, worker-process CPU use and how the compute-loop wall
// time splits between GPU/CPU dispatch calls and the host work around them (plus one-off stall lines).
