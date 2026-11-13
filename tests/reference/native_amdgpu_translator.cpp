// CPU-only qualification of the maintained generic SYCL AMD lowering and embedded device libs.
// Generated HSA data is inspected, never executed; actual GPU correctness/performance is separate.
#include <hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.hpp>
#include <hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpuFactory.hpp>
#include <hipSYCL/common/hcf_container.hpp>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>
#include <fstream>
#include <future>
#include <mutex>
#include <csignal>
#include <sys/resource.h>
#include <sys/stat.h>
#include <stdexcept>
#include <string>
#include "amdgpu-resources.inc"

// Preserve exactly the pre-emission IR for a build-time llvm-link/opt/llc oracle, without
// giving the runtime translator any disk source or device-library fallback.
class RecordedAmdTranslator : public hipsycl::compiler::LLVMToAmdgpuTranslator {
  std::string filename;
public:
  RecordedAmdTranslator(const std::vector<std::string> &kernels, std::string file)
      : LLVMToAmdgpuTranslator(kernels), filename(std::move(file)) {}
  bool translateToBackendFormat(llvm::Module &module, std::string &out) override {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    llvm::WriteBitcodeToFile(module, stream);
    std::ofstream output(filename, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); output.close();
    if (!output) throw std::runtime_error("Cannot preserve pre-emission IR");
    return LLVMToAmdgpuTranslator::translateToBackendFormat(module, out);
  }
};

// Fault only this sequential CPU proof, before the concurrent cases. Ignoring SIGXFSZ turns
// an actual short write into EFBIG; restore both process settings before any later case.
struct ShortWrite {
  rlimit saved{};
  using Handler = void (*)(int);
  Handler signal;
  ShortWrite() {
    if (getrlimit(RLIMIT_FSIZE, &saved)) throw std::runtime_error("Cannot query file limit");
    signal = std::signal(SIGXFSZ, SIG_IGN);
    if (signal == SIG_ERR) throw std::runtime_error("Cannot install short-write signal guard");
    auto limited = saved; limited.rlim_cur = 1;
    if (setrlimit(RLIMIT_FSIZE, &limited)) {
      std::signal(SIGXFSZ, signal); throw std::runtime_error("Cannot set short-write limit");
    }
  }
  ~ShortWrite() {
    if (setrlimit(RLIMIT_FSIZE, &saved) || std::signal(SIGXFSZ, signal) == SIG_ERR) std::abort();
  }
};

static void compile(const std::string &base, const std::string &target, unsigned wave, bool fast,
                    const std::string &fault = {}, const std::string &library = {}) {
  hipsycl::common::hcf_container hcf{std::string(embedded_resource("generic.hcf"))};
  const auto *images = hcf.root_node()->get_subnode("images"), *names = hcf.root_node()->get_subnode("kernels");
  const auto *image = images ? images->get_subnode("llvm-ir.global") : nullptr;
  std::string ir;
  if (!image || !names || names->subnodes.size() != 4 || !hcf.get_binary_attachment(image, ir))
    throw std::runtime_error("Invalid generic SYCL image");
  const auto kernels = names->get_subnodes();
  RecordedAmdTranslator translator(kernels, base + ".bc");
  if (fault != "no-resolver")
    translator.provideBitcodeLibraryResolver([&](const std::string &name) -> std::string_view {
      if (name == library && fault == "missing") return {};
      if (name == library && fault == "corrupt") return "not bitcode";
      return embedded_resource(name);
    });
  const auto configure = [&](hipsycl::compiler::LLVMToBackendTranslator &compiler) {
    if (!compiler.setBuildOption("amdgpu-target-device", target) ||
        !compiler.setBuildOption("desired-subgroup-size", wave) ||
        !compiler.setBuildOption("known-local-mem-size", 0) ||
        !compiler.setBuildOption("known-group-size-x", 64) ||
        !compiler.setBuildOption("known-group-size-y", 1) || !compiler.setBuildOption("known-group-size-z", 1) ||
        (fast && !compiler.setBuildFlag("fast-math")))
      throw std::runtime_error("Rejected build configuration");
  };
  configure(translator);
  std::string binary;
  bool ok;
  std::string errors;
  if (fault == "object-write") {
    auto actualFactory = hipsycl::compiler::createLLVMToAmdgpuTranslator(kernels);
    configure(*actualFactory);
    {ShortWrite limited; ok = actualFactory->fullTransformation(ir, binary);}
    errors = actualFactory->getErrorLogAsString();
  } else {
    ok = translator.fullTransformation(ir, binary);
    errors = translator.getErrorLogAsString();
  }
  if (!fault.empty()) {
    const std::string expected = fault == "no-resolver" ? "requires an embedded bitcode-library resolver" :
        fault == "missing" ? "Missing embedded bitcode library: " + library :
        fault == "corrupt" ? "Could not load LLVM module" : fault == "object-write" ? "cannot write generated object" : fault;
    if (ok || !binary.empty() || errors.find(expected) == std::string::npos)
      throw std::runtime_error("Negative case failed for wrong reason: " + fault + ": " + errors);
    llvm::outs() << "rejected " << target << " wave=" << wave << " " << fault << " " << library << '\n';
    return;
  }
  if (!ok) throw std::runtime_error(translator.getErrorLogAsString());
  auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(binary, "generated HSA data"));
  if (!object) {llvm::consumeError(object.takeError()); throw std::runtime_error("Malformed HSA output");}
  const auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object->get());
  if (!elf || elf->getELFFile().getHeader().e_machine != llvm::ELF::EM_AMDGPU ||
      elf->getELFFile().getHeader().e_type != llvm::ELF::ET_DYN ||
      elf->getELFFile().getHeader().e_ident[llvm::ELF::EI_OSABI] != llvm::ELF::ELFOSABI_AMDGPU_HSA)
    throw std::runtime_error("Wrong HSA object format");
  for (const auto &kernel : kernels) {
    bool found = false;
    for (const auto &symbol : object.get()->symbols()) {
      auto name = symbol.getName();
      if (!name) {llvm::consumeError(name.takeError()); throw std::runtime_error("Malformed symbol");}
      found |= *name == kernel;
    }
    if (!found) throw std::runtime_error("Lost generated kernel " + kernel);
  }
  auto factory = hipsycl::compiler::createLLVMToAmdgpuTranslator(kernels);
  configure(*factory);
  std::string repeated;
  if (!factory->fullTransformation(ir, repeated) || binary != repeated)
    throw std::runtime_error("Runtime factory output differs: " + factory->getErrorLogAsString());
  std::ofstream output(base + ".hsaco", std::ios::binary);
  output.write(binary.data(), static_cast<std::streamsize>(binary.size())); output.close();
  if (!output) throw std::runtime_error("Cannot preserve HSA image");
  static std::mutex outputMutex;
  std::lock_guard<std::mutex> lock(outputMutex);
  llvm::outs() << target << " wave=" << wave << " fast=" << fast << ": " << binary.size() << " bytes; factory matched\n";
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    // This child process alone deliberately has a permissive umask: the JIT must still
    // create private scratch atomically. Keep intentional fatal-negative controls core-free.
    umask(0);
    const rlimit noCore{0, 0};
    if (setrlimit(RLIMIT_CORE, &noCore)) throw std::runtime_error("Cannot disable proof core dumps");
    unsigned next = 0;
    const auto check = [&](const std::string &target, unsigned wave, bool fast,
                           const std::string &fault = {}, const std::string &library = {}) {
      compile(std::string(argv[1]) + "/" + std::to_string(next++), target, wave, fast, fault, library);
    };
    for (const auto fast : {false, true}) {
      check("gfx900", 64, fast); check("gfx906:xnack-:sramecc+", 64, fast);
      check("gfx1200", 32, fast); check("gfx1200", 64, fast);
    }
    check("gfx9999", 64, false, "unsupported target ID");
    check("gfx900", 32, false, "does not support wave32");
    check("gfx1200", 16, false, "Invalid wavefront size");
    check("gfx1200:xnack+", 32, false, "unsupported target ID");
    check("gfx900::xnack+", 64, false, "malformed target ID");
    check("gfx900:", 64, false, "malformed target ID");
    check("gfx900", 64, false, "no-resolver");
    for (const std::string library : {"libkernel-sscp-amdgpu-amdhsa-full.bc", "ocml.bc", "ockl.bc", "oclc_isa_version_900.bc"})
      for (const std::string fault : {"missing", "corrupt"}) check("gfx900", 64, false, fault, library);
    check("gfx900", 64, false, "object-write");
    check("gfx1200", 32, false); // rejected inputs must not contaminate later translation/linking
    auto legacy = std::async(std::launch::async, [&] {compile(std::string(argv[1]) + "/concurrent-legacy", "gfx900", 64, false);});
    auto modern = std::async(std::launch::async, [&] {compile(std::string(argv[1]) + "/concurrent-modern", "gfx1200", 32, true);});
    legacy.get(); modern.get(); // exercises serialized LLD contexts with independent LLVM contexts
    llvm::outs() << "AdaptiveCpp AMD generic lowering/linking passed (no GPU)\n";
  } catch (const std::exception &error) {llvm::errs() << error.what() << '\n'; return 1;}
}
