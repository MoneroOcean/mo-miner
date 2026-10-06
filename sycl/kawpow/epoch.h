#pragma once

#include <cstdint>

namespace mom_kawpow {

// Firo mainnet consensus (firoorg/firo@88d16dc7): nMaxPPEpoch=926 with terminal epoch 650.
// Heights at or above the terminal height clamp the DAG, cache, and seed epoch;
// the ProgPoW period remains height-based.
constexpr uint32_t FIROPOW_EPOCH_LENGTH = 1300;
constexpr uint32_t FIROPOW_TERMINAL_HEIGHT = 1205100;
constexpr uint32_t FIROPOW_TERMINAL_EPOCH = 650;

constexpr uint32_t firopow_epoch(const uint32_t height) {
  return height >= FIROPOW_TERMINAL_HEIGHT ? FIROPOW_TERMINAL_EPOCH : height / FIROPOW_EPOCH_LENGTH;
}

} // namespace mom_kawpow
