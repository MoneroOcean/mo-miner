// GPU-free integration proof: the real generic SYCL translator and real HIPRTC share one LLVM.
// A test-only LLD wrapper observes overlapping entry and exits before a second linker can corrupt
// the process-global context. Positive runs compare actual generated images, not just counters.
#include "../../sycl/hiprtc-api.h"
#include <hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpuFactory.hpp>
#include <hipSYCL/common/hcf_container.hpp>
#include <lld/Common/Driver.h>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>
#include "mixed-compiler-resources.inc"

static std::atomic<bool> observe{false};
static std::atomic<unsigned> entries{0}, calls{0};
extern "C" lld::Result real_link(llvm::ArrayRef<const char*>, llvm::raw_ostream&,
    llvm::raw_ostream&, llvm::ArrayRef<lld::DriverDef>)
    asm("__real__ZN3lld7lldMainEN4llvm8ArrayRefIPKcEERNS0_11raw_ostreamES6_NS1_INS_9DriverDefEEE");
extern "C" lld::Result observed_link(llvm::ArrayRef<const char*>, llvm::raw_ostream&,
    llvm::raw_ostream&, llvm::ArrayRef<lld::DriverDef>)
    asm("__wrap__ZN3lld7lldMainEN4llvm8ArrayRefIPKcEERNS0_11raw_ostreamES6_NS1_INS_9DriverDefEEE");

extern "C" lld::Result observed_link(llvm::ArrayRef<const char*> args, llvm::raw_ostream& out,
    llvm::raw_ostream& err, llvm::ArrayRef<lld::DriverDef> drivers) {
  const bool active = observe.load();
  if (active) {
    if (entries.fetch_add(1) != 0) {
      std::fputs("MIXED_COMPILER_LINK_OVERLAP\n", stderr); std::fflush(stderr); std::_Exit(73);
    }
    ++calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Bounded overlap window, CPU-only test.
  }
  const auto result = real_link(args, out, err, drivers);
  if (active) --entries;
  return result;
}

static std::string sycl_compile(unsigned wave, bool fast, bool invalid = false) {
  hipsycl::common::hcf_container hcf{std::string(embedded_resource("generic.hcf"))};
  const auto* images = hcf.root_node()->get_subnode("images");
  const auto* names = hcf.root_node()->get_subnode("kernels");
  std::string ir;
  if (!images || !names || !hcf.get_binary_attachment(images->get_subnode("llvm-ir.global"), ir))
    throw std::runtime_error("Missing original generic SYCL input");
  auto compiler = hipsycl::compiler::createLLVMToAmdgpuTranslator(names->get_subnodes());
  if (!compiler->setBuildOption("amdgpu-target-device", std::string("gfx1200")) ||
      !compiler->setBuildOption("desired-subgroup-size", wave) || !compiler->setBuildOption("known-local-mem-size", 0) ||
      !compiler->setBuildOption("known-group-size-x", 64) || !compiler->setBuildOption("known-group-size-y", 1) ||
      !compiler->setBuildOption("known-group-size-z", 1) || (fast && !compiler->setBuildFlag("fast-math")))
    throw std::runtime_error("Compiler configuration rejected");
  std::string output;
  const bool ok = compiler->fullTransformation(invalid ? "not bitcode" : ir, output);
  if (invalid) {
    if (ok || !output.empty()) throw std::runtime_error("Invalid generic IR accepted");
  } else if (!ok) throw std::runtime_error(compiler->getErrorLogAsString());
  return output;
}

static std::string rtc_compile(bool invalid = false) {
  const std::string source = invalid ? "this is not HIP source" : std::string(
#include "../../sycl/pearlhash/hip_kernel.inc"
  );
  auto& api = mom::HiprtcApi::instance();
  if (!api.available()) throw std::runtime_error(api.error);
  mom::HiprtcProgram program(api);
  if (api.create_program(program.address(), source.c_str(), "mom-native.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS)
    throw std::runtime_error("HIPRTC program allocation failed");
  const char* options[] = {"-O3", "--std=c++17", "--offload-arch=gfx1200", "-mwavefrontsize64", "-DPEARLHASH_ER=13",
    "-DPEARLHASH_EC=4", "-DPEARLHASH_K=2048", "-DPEARLHASH_RANK=128", "-DPEARLHASH_K_UNROLL=1"};
  const auto status = api.compile_program(program.get(), std::size(options), options);
  if (invalid) {
    if (status == HIPRTC_SUCCESS) throw std::runtime_error("Invalid HIP source accepted");
    return {};
  }
  if (status != HIPRTC_SUCCESS) {
    size_t size = 0; api.get_log_size(program.get(), &size);
    std::string log(size, '\0'); if (size) api.get_log(program.get(), log.data());
    throw std::runtime_error(log);
  }
  size_t size = 0;
  if (api.get_code_size(program.get(), &size) != HIPRTC_SUCCESS || !size) throw std::runtime_error("Missing compiled HIP image");
  std::string output(size, '\0');
  if (api.get_code(program.get(), output.data()) != HIPRTC_SUCCESS) throw std::runtime_error("HIP image read failed");
  return output;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    // Exercise both compilers on every round, not COMGR's generated-image cache.
    if (setenv("AMD_COMGR_CACHE", "0", 1)) throw std::runtime_error("Cannot disable test compiler cache");
    const auto strict = sycl_compile(64, false), fast = sycl_compile(32, true), rtc = rtc_compile();
    if (strict == fast) throw std::runtime_error("Math/wave variants unexpectedly identical");
    const auto preserve = [&](const std::string& name, const std::string& bytes) {
      std::ofstream file(std::string(argv[1]) + "/" + name + ".hsaco", std::ios::binary);
      file.write(bytes.data(), bytes.size()); file.close();
      if (!file) throw std::runtime_error("Cannot preserve serial baseline");
    };
    preserve("sycl-strict", strict); preserve("sycl-fast", fast); preserve("hiprtc", rtc);
    std::barrier rounds(2);
    std::string sycl_error, rtc_error;
    observe = true;
    auto sycl = std::async(std::launch::async, [&] {
      for (unsigned round = 0; round < 3; ++round) {
        rounds.arrive_and_wait();
        try {
          if (sycl_compile(round % 2 ? 32 : 64, round % 2) != (round % 2 ? fast : strict))
            throw std::runtime_error("Generic SYCL output changed across SDKs");
        } catch (const std::exception& error) {sycl_error = error.what(); rounds.arrive_and_drop(); return;}
      }
    });
    auto hip = std::async(std::launch::async, [&] {
      for (unsigned round = 0; round < 3; ++round) {
        rounds.arrive_and_wait();
        try {if (rtc_compile() != rtc) throw std::runtime_error("HIPRTC output changed across SDKs");}
        catch (const std::exception& error) {rtc_error = error.what(); rounds.arrive_and_drop(); return;}
      }
    });
    sycl.get(); hip.get(); observe = false;
    if (!sycl_error.empty() || !rtc_error.empty()) throw std::runtime_error(sycl_error + rtc_error);
    sycl_compile(64, false, true); rtc_compile(true);
    const auto sycl_recovered = sycl_compile(64, false), rtc_recovered = rtc_compile();
    preserve("sycl-recovered", sycl_recovered); preserve("hiprtc-recovered", rtc_recovered);
    if (sycl_recovered != strict) throw std::runtime_error("Generic SYCL compiler recovery changed output");
    if (rtc_recovered != rtc) throw std::runtime_error("HIPRTC compiler recovery changed output");
    if (calls < 6) throw std::runtime_error("Insufficient observed link coverage: " + std::to_string(calls.load()));
    std::printf("Mixed SYCL/HIPRTC compiler parity passed; observed links=%u\n", calls.load());
    return 0;
  } catch (const std::exception& error) {std::fprintf(stderr, "%s\n", error.what()); return 1;}
}
