#pragma once

// Only standalone GPU applications provide an admitted static JIT closure identity.
#ifdef MOM_NATIVE_ACPP_CACHE
#include "mom-jit-cache-tag.h"
extern "C" bool mom_acpp_set_appdb_cache_tag(const unsigned char *tag) noexcept;
#endif
