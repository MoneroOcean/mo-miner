// Replace the current job and report how far the replaced pool job got, so the controller can resume
// it later. A job rejected by set_job()'s early validation leaves the previous job running untouched.
// Once set_job() has stopped the previous job (m_job_ref advances right after clear_fn() and after RX
// threads are stopped), a later rejection must still report the old progress and drop the
// half-applied ids; otherwise the next switch would report a partially written nonce under them.
void Core::replace_job(
  JobMode mode, const bool is_no_same_input, const MessageValues& v, std::function<void(void)> fn_extra_setup
) {
  const uint64_t prev_last_nonce = last_nonce();
  const unsigned prev_nonce_bytes = m_nonce_bytes;
  const std::string prev_pool_id = m_pool_id;
  const std::string prev_job_id = m_job_id;
  const std::string prev_job_token = m_job_token;
  const unsigned prev_job_ref = m_job_ref.load(std::memory_order_relaxed);
  try {
    set_job(mode, is_no_same_input, v, std::move(fn_extra_setup));
  } catch (...) {
    if (m_job_ref.load(std::memory_order_relaxed) == prev_job_ref)
      throw;
    if (prev_last_nonce)
      send_last_nonce(prev_last_nonce, prev_nonce_bytes, prev_pool_id, prev_job_id, prev_job_token);
    m_pool_id.clear();
    m_job_id.clear();
    m_job_token.clear();
    throw;
  }
  if (prev_last_nonce)
    send_last_nonce(prev_last_nonce, prev_nonce_bytes, prev_pool_id, prev_job_id, prev_job_token);
}

// Apply one controller command on the compute thread. Returns false only for "close"; command
// failures throw and are reported as "error" messages by the compute loop.
bool Core::process_message(const std::string& type, const MessageValues& v) {
  if (type == "job") {
    if (!v.contains("target"))    throw std::string("Missing target job key");
    if (!v.contains("pool_id"))   throw std::string("Missing pool_id job key");
    if (!v.contains("worker_id")) throw std::string("Missing worker_id job key");
    if (!v.contains("job_id"))    throw std::string("Missing job_id job key");
    if (!v.contains("job_token") || v.at("job_token").empty())
      throw std::string("Missing job_token job key");
    const std::string algo = v.contains("algo") ? v.at("algo") : "";
    mom::job_boundary::ShareTarget target;
    if (!mom::job_boundary::parse_share_target(
          algo, mom::job_boundary::strip_hex_prefix(v.at("target")), target))
      throw std::string("Bad target hex");
    // Preserve the original pool base target for Pearl's native pattern selection.
    // A final jackpot target has no base field and must never be scaled twice.
    const bool new_pearlhash_has_base_target = algo == "pearlhash" &&
        v.contains("pearlhash_base_target");
    uint8_t new_pearlhash_base_target[HASH_LEN]{};
    if (new_pearlhash_has_base_target) {
      std::string base(mom::job_boundary::strip_hex_prefix(v.at("pearlhash_base_target")));
      if (base.empty() || base.size() > HASH_LEN * 2)
        throw std::string("Bad PearlHash base target hex");
      if (base.size() & 1)
        base.insert(0, 1, '0');
      if (!mom::job_boundary::parse_big_target(base, new_pearlhash_base_target))
        throw std::string("Bad target hex");
    }

    replace_job(JobMode::mine, true, v, [&]() {
      m_target    = target.word;
      // This callback also runs on same-input retargets. Always replace/clear
      // both the flag and bytes so final-target jobs cannot reuse an old base.
      m_pearlhash_has_base_target = new_pearlhash_has_base_target;
      std::memcpy(m_pearlhash_base_target_bin, new_pearlhash_base_target, HASH_LEN);
      if (target.full) {
        std::memcpy(m_target_bin, target.bytes, HASH_LEN);
      }
      m_pool_id   = v.at("pool_id");
      m_worker_id = v.at("worker_id");
      m_job_id    = v.at("job_id");
      m_job_token = v.at("job_token");
      m_header_hash = v.contains("header_hash") ? v.at("header_hash") : "";
    });

  } else if (type == "bench" || type == "test") {
    const bool is_bench = type == "bench";
    debug_async_worker(is_bench ? "process bench start" : "process test start");
    replace_job(is_bench ? JobMode::bench : JobMode::test, is_bench, v);
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

