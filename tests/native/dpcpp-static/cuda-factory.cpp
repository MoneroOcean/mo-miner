// Only opaque Unified Runtime handles and its original dispatch table cross this boundary.
// Namespace-isolate the complete CUDA graph at link time; do not duplicate its function registry.
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>
extern "C" ur_result_t UR_APICALL urAllAddrTable(ur_api_version_t, ur_dditable_t*);
extern "C" ur_result_t mom_ur_cuda_factory(ur_dditable_t* table) {
  return table ? urAllAddrTable(UR_API_VERSION_CURRENT, table) : UR_RESULT_ERROR_INVALID_NULL_POINTER;
}
