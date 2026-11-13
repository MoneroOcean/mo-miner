// V1 and V2 expose identical C++ names. Keep V2 private and share only its original opaque
// dispatch table; upstream retains device-IP/default-version selection inside each adapter.
#include <unified-runtime/ur_api.h>
#include <unified-runtime/ur_ddi.h>
namespace ur::level_zero {ur_result_t urAdapterGetDdiTables(ur_dditable_t*);}
extern "C" ur_result_t mom_ur_level_zero_v2_factory(ur_dditable_t* table) {
  return table ? ur::level_zero::urAdapterGetDdiTables(table) : UR_RESULT_ERROR_INVALID_NULL_POINTER;
}
