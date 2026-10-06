#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* mom_hoohash_math_error(void);
void mom_hoohash_canonical_hash(const uint8_t input[72], uint64_t nonce, uint8_t output[32]);

#ifdef __cplusplus
}
#endif
