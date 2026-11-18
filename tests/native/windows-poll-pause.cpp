#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using HANDLE = void*;
using DWORD = unsigned long;
struct LARGE_INTEGER { long long QuadPart; };
constexpr DWORD INFINITE = 0xffffffffUL, WAIT_OBJECT_0 = 0, WAIT_TIMEOUT = 258,
                WAIT_FAILED = 0xffffffffUL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 2,
                TIMER_MODIFY_STATE = 2, SYNCHRONIZE = 0x100000;
constexpr bool FALSE = false;
static const char* scenario;
static unsigned creates, arms, waits, sleeps, closes;
static DWORD limit;
static HANDLE active_handle;
static bool okay = true;
static bool is(const char* name) { return std::strcmp(scenario, name) == 0; }
static void check(bool condition) { okay = okay && condition; }

HANDLE CreateWaitableTimerExW(void* security, void* name, DWORD flags, DWORD access) {
  ++creates;
  check(!security && !name && flags == CREATE_WAITABLE_TIMER_HIGH_RESOLUTION &&
        access == (TIMER_MODIFY_STATE | SYNCHRONIZE));
  check(!active_handle);
  if (is("create-failure")) return nullptr;
  active_handle = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(creates));
  return active_handle;
}
bool SetWaitableTimer(HANDLE handle, const LARGE_INTEGER* due, long period,
                     void* callback, void* argument, bool resume) {
  ++arms;
  check(handle && handle == active_handle && due->QuadPart == -1000 &&
        period == 0 && !callback && !argument && !resume);
  return !is("arm-failure");
}
DWORD WaitForSingleObject(HANDLE handle, DWORD milliseconds) {
  ++waits;
  limit = milliseconds;
  check(handle && handle == active_handle);
  if (is("timeout-then-signaled")) {
    check(limit == 10);
    return waits == 1 ? WAIT_TIMEOUT : WAIT_OBJECT_0;
  }
  if (is("lost-signal")) {
    // Only the owning CPU test deadline can release the original unbounded wait.
    if (limit == INFINITE)
      for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
    check(limit == 10);
    return WAIT_TIMEOUT;
  }
  return is("wait-failure") ? WAIT_FAILED : WAIT_OBJECT_0;
}
void Sleep(DWORD milliseconds) { ++sleeps; check(milliseconds == 1); }
bool CloseHandle(HANDLE handle) {
  ++closes;
  check(handle && handle == active_handle);
  active_handle = nullptr;
  return true;
}

#include "poll_pause.inc"

int main(int argc, char** argv) {
  if (argc != 2) return 90;
  scenario = argv[1];
  if (!is("normal") && !is("lost-signal") && !is("create-failure") &&
      !is("arm-failure") && !is("wait-failure") && !is("timeout-then-signaled") &&
      !is("sequential-workers")) return 91;
  // Two calls on one real thread test reuse and destruction of the actual TLS timer.
  const unsigned workers = is("sequential-workers") ? 2u : 1u;
  // Join before the next worker: check per-thread ownership without mock counter races.
  for (unsigned i = 0; i < workers; ++i) {
    std::thread worker([] { mom_sycl_poll_pause(); mom_sycl_poll_pause(); });
    worker.join();
  }
  const bool create_failed = is("create-failure"), arm_failed = is("arm-failure");
  check(creates == workers && arms == (create_failed ? 0u : 2u * workers));
  check(waits == (create_failed || arm_failed ? 0u : 2u * workers));
  check(sleeps == (create_failed || arm_failed || is("wait-failure") ? 2u : 0u));
  check(closes == (create_failed ? 0u : workers) && !active_handle);
  std::printf("okay=%u creates=%u arms=%u waits=%u sleeps=%u closes=%u limit=%lu\n",
              okay, creates, arms, waits, sleeps, closes, limit);
  return okay ? 0 : 17;
}
