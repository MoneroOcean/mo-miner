// HIP device tables of 4 GiB and more, as one address range over smaller allocations.
//
// On Windows a single allocation that large serves random reads about 12% slower than the same bytes
// in allocations below 4 GiB (RX 9060 XT: 1.07 against 1.25 G 64-byte reads/s; 3.99 GiB is still
// fast, 4.01 GiB is not). Every DAG-class algorithm crosses that size, so mom_usm_alloc() reserves
// one range and backs it with allocations of at most 2 GiB: kernels keep a single pointer. Linux
// shows no such step and keeps single allocations; MOM_HIP_SPLIT_TABLES=0/1 overrides either
// default for comparisons.
//
// Two Windows rules shape the mapping. Access rights are set per mapping, because one call over the
// whole range is rejected. And every mapping has the same size, because the call is intermittently
// rejected for a final mapping smaller than the others; the range is rounded up by less than
// 2 MiB per mapping instead. Any failure leaves the request on the ordinary single allocation.
#if defined(MOM_SYCL_ADAPTIVECPP) && defined(MOM_SYCL_HAS_HIP)
#include <mutex>
#include <unordered_map>

namespace {

struct HipSplitTable {
  size_t part_bytes = 0;
  size_t parts = 0;
  hipDevice_t device = -1;
  std::vector<hipMemGenericAllocationHandle_t> handles;
};

constexpr size_t HIP_SPLIT_MAX_PART_BYTES = size_t{2} << 30;
constexpr size_t HIP_SPLIT_GRANULE_BYTES = size_t{2} << 20;

std::mutex hip_split_mutex;
std::unordered_map<void*, HipSplitTable> hip_split_tables;

// Unmap the first `mapped` parts, then drop every handle and the reservation.
void hip_split_release(void* const base, const HipSplitTable& table, const size_t mapped) noexcept {
  (void)hipSetDevice(table.device);
  for (size_t part = 0; part < mapped; ++part)
    (void)hipMemUnmap(static_cast<char*>(base) + part * table.part_bytes, table.part_bytes);
  for (const hipMemGenericAllocationHandle_t handle : table.handles)
    (void)hipMemRelease(handle);
  (void)hipMemAddressFree(base, table.parts * table.part_bytes);
}

} // namespace

// Null means the request stays on the ordinary allocation path (not wanted here, not HIP, or the
// range API failed).
void* mom_hip_alloc_split(sycl::queue& queue, const size_t bytes) {
#if defined(_WIN32)
  constexpr bool split_by_default = true;
#else
  constexpr bool split_by_default = false;
#endif
  static const bool split = mom_env_flag("MOM_HIP_SPLIT_TABLES").value_or(split_by_default);
  if (!split || !bytes || !mom_is_hip(queue.get_device()))
    return nullptr;
  HipSplitTable table;
  try {
    mom::hip_queue_device(queue, table.device);
  } catch (...) {
    return nullptr;
  }
  table.parts = (bytes - 1) / HIP_SPLIT_MAX_PART_BYTES + 1;
  table.part_bytes = ((bytes - 1) / table.parts / HIP_SPLIT_GRANULE_BYTES + 1) *
                     HIP_SPLIT_GRANULE_BYTES;
  const size_t part = table.part_bytes;
  void* base = nullptr;
  if (hipSetDevice(table.device) != hipSuccess ||
      hipMemAddressReserve(&base, table.parts * part, 0, nullptr, 0) != hipSuccess || !base)
    return nullptr;
  hipMemAllocationProp physical{};
  physical.type = hipMemAllocationTypePinned;
  physical.location.type = hipMemLocationTypeDevice;
  physical.location.id = table.device;
  hipMemAccessDesc access{};
  access.location = physical.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  size_t mapped = 0;
  while (mapped < table.parts) {
    void* const at = static_cast<char*>(base) + mapped * part;
    hipMemGenericAllocationHandle_t handle{};
    if (hipMemCreate(&handle, part, &physical, 0) != hipSuccess)
      break;
    table.handles.push_back(handle);
    if (hipMemMap(at, part, 0, handle, 0) != hipSuccess)
      break;
    ++mapped;
    if (hipMemSetAccess(at, part, &access, 1) != hipSuccess)
      break;
    if (mapped == table.parts) {
      const std::lock_guard<std::mutex> lock(hip_split_mutex);
      hip_split_tables.emplace(base, std::move(table));
      return base;
    }
  }
  hip_split_release(base, table, mapped);
  return nullptr;
}

// False when `pointer` is not a split table, so the caller frees it as ordinary USM.
bool mom_hip_free_split(void* const pointer) noexcept {
  HipSplitTable table;
  {
    const std::lock_guard<std::mutex> lock(hip_split_mutex);
    const auto found = hip_split_tables.find(pointer);
    if (found == hip_split_tables.end())
      return false;
    table = std::move(found->second);
    hip_split_tables.erase(found);
  }
  hip_split_release(pointer, table, table.handles.size());
  return true;
}
#endif
