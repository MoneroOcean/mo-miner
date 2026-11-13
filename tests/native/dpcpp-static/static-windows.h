// export.hpp explicitly honors these overrides. Static consumers must not emit DLL imports
// or re-export the complete SYCL API from the application; Engineering DLL builds omit this file.
#define __SYCL_EXPORT
#define __SYCL_EXPORT_DEPRECATED(message)
#define __SYCL_DLL_LOCAL
