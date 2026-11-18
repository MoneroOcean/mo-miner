// Exercise the real provider/allocator with CPU-backed Level Zero ABI stubs, never a GPU driver.
#include <umf/memory_provider.h>
#include <umf/providers/provider_level_zero.h>
#include <ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

static std::unordered_map<void*, size_t> live;
static unsigned allocations, frees, queries;
static int context;
extern "C" ze_result_t ZE_APICALL __wrap_zeMemAllocHost(
    ze_context_handle_t handle, const ze_host_mem_alloc_desc_t* desc, size_t bytes,
    size_t alignment, void** output) {
  if (handle != reinterpret_cast<ze_context_handle_t>(&context) || !desc ||
      desc->stype != ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC || !bytes || !output)
    return ZE_RESULT_ERROR_INVALID_ARGUMENT;
  void* memory = nullptr;
  if (alignment && (alignment & (alignment - 1))) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
  if (posix_memalign(&memory, alignment < sizeof(void*) ? sizeof(void*) : alignment, bytes))
    return ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  live.emplace(memory, bytes); ++allocations; *output = memory;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL __wrap_zeMemGetAllocProperties(
    ze_context_handle_t handle, const void* memory, ze_memory_allocation_properties_t* properties,
    ze_device_handle_t*) {
  if (handle != reinterpret_cast<ze_context_handle_t>(&context) || !properties || !live.count(const_cast<void*>(memory)))
    return ZE_RESULT_ERROR_INVALID_ARGUMENT;
  properties->pageSize = 4096; properties->type = ZE_MEMORY_TYPE_HOST; ++queries;
  return ZE_RESULT_SUCCESS;
}
extern "C" ze_result_t ZE_APICALL __wrap_zeMemFree(ze_context_handle_t handle, void* memory) {
  if (handle != reinterpret_cast<ze_context_handle_t>(&context) || !live.erase(memory))
    return ZE_RESULT_ERROR_INVALID_ARGUMENT;
  std::free(memory); ++frees; return ZE_RESULT_SUCCESS;
}
static void check(umf_result_t result, const char* operation) {
  if (result != UMF_RESULT_SUCCESS) {
    std::fprintf(stderr, "%s result=%d\n", operation, int(result));
    throw std::runtime_error(operation);
  }
}
int main() try {
  for (int iteration = 0; iteration < 3; ++iteration) {
    umf_level_zero_memory_provider_params_handle_t params = nullptr;
    check(umfLevelZeroMemoryProviderParamsCreate(&params), "params create");
    check(umfLevelZeroMemoryProviderParamsSetContext(params, reinterpret_cast<ze_context_handle_t>(&context)), "context");
    check(umfLevelZeroMemoryProviderParamsSetMemoryType(params, UMF_MEMORY_TYPE_HOST), "memory type");
    umf_memory_provider_handle_t provider = nullptr;
    check(umfMemoryProviderCreate(umfLevelZeroMemoryProviderOps(), params, &provider), "provider create");
    check(umfLevelZeroMemoryProviderParamsDestroy(params), "params destroy");
    void* memory = nullptr;
    check(umfMemoryProviderAlloc(provider, 8192, 128, &memory), "provider allocation");
    if (!memory || reinterpret_cast<uintptr_t>(memory) % 128 || live.at(memory) != 8192)
      throw std::runtime_error("Allocation ABI/size/alignment mismatch");
    std::memset(memory, 0x5a, 8192);
    check(umfMemoryProviderFree(provider, memory, 8192), "provider free");
    check(umfMemoryProviderDestroy(provider), "provider destroy");
    if (!live.empty()) throw std::runtime_error("Provider leaked driver allocations");
  }
  if (allocations != frees || allocations < 6 || queries < 3)
    throw std::runtime_error("Provider bypassed the actual allocation/property/free path");
  std::printf("Static UMF provider passed: allocations=%u frees=%u queries=%u; no GPU\n", allocations, frees, queries);
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what()); return 1;
}
