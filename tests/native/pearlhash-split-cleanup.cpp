#include <cstdio>
#include <exception>
#include <stdexcept>
struct Queue {
  bool pending = false, fault = false;
  unsigned waits = 0;
  void submit() {
    if (pending)
      throw std::logic_error("queue reused while pending");
    pending = true;
  }
  void wait_and_throw() {
    ++waits;
    pending = false;
    if (fault)
      throw std::runtime_error("queue cleanup failed");
  }
};
struct State { Queue queue, prep_queue; };
template <typename Fn> void sycl_cleanup_noexcept(const char*, Fn fn) noexcept {
  try { fn(); } catch (...) { }
}
static void require(bool ok, const char* message) {
  if (!ok)
    throw std::logic_error(message);
}
static void dispatch(State& st, std::exception_ptr original, bool queue_fault, bool prep_fault) {
  auto& q = st.queue;
  q.fault = queue_fault;
  st.prep_queue.fault = prep_fault;
  q.submit();
  st.prep_queue.submit();
  try {
    std::rethrow_exception(original);
#include "catch.inc"
}
static void reuse(State& st) {
  require(!st.queue.pending && !st.prep_queue.pending,
          "pending work survived before retry");
  st.prep_queue.submit();
  st.prep_queue.wait_and_throw();
}
static void run_case(bool queue_fault, bool prep_fault) {
  State st;
  const auto original = std::make_exception_ptr(std::runtime_error("original attempt failure"));
  try {
    dispatch(st, original, queue_fault, prep_fault);
    throw std::logic_error("attempt did not throw");
  } catch (...) {
    require(std::current_exception() == original, "cleanup replaced original exception");
  }
  require(st.queue.waits == 1 && st.prep_queue.waits == 1,
          "both queue cleanups were not attempted");
  require(!st.prep_queue.pending, "preparation work survived cleanup");
  st.queue.fault = st.prep_queue.fault = false;
  reuse(st);
}
int main(int argc, char** argv) {
  if (argc != 3)
    return 2;
  try {
    run_case(argv[1][0] == '1', argv[2][0] == '1');
    std::puts("PASS PearlHash split preparation cleanup");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
