// CPU-only proof of AdaptiveCpp generic SYCL lowering with embedded device libraries.
// GPU dispatch, driver compatibility and performance remain separate acceptance gates.
#include <hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtx.hpp>
#include <hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtxFactory.hpp>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <hipSYCL/common/hcf_container.hpp>

#include <fstream>
#include <stdexcept>
#include <string>
#include "ptx-resources.inc"

// Save the exact pre-emission module so the build-time opt/llc oracle can use identical inputs.
// This is evidence output only: neither the translator nor its resolver reads these files.
class RecordedTranslator : public hipsycl::compiler::LLVMToPtxTranslator {
  std::string filename;
public:
  RecordedTranslator(const std::vector<std::string> &kernels, std::string file)
      : LLVMToPtxTranslator(kernels), filename(std::move(file)) {}
  bool translateToBackendFormat(llvm::Module &module, std::string &out) override {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    llvm::WriteBitcodeToFile(module, stream);
    std::ofstream output(filename, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Cannot preserve pre-emission IR");
    return LLVMToPtxTranslator::translateToBackendFormat(module, out);
  }
};

static void compile(const char *directory, unsigned sm, unsigned version, bool fast, bool full,
                    const std::string &fault = {}, const std::string &library = {}) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  // Assembly parsing requires a trailing sentinel; executable byte resources are length-delimited.
  auto input = llvm::MemoryBuffer::getMemBufferCopy(embedded_resource("device.ll"), "device");
  auto module = llvm::parseIR(input->getMemBufferRef(), diagnostic, context);
  if (!module) {diagnostic.print("native-ptx", llvm::errs()); throw std::runtime_error("Cannot parse maintained device IR fixture");}
  std::vector<std::string> kernels{"mom_device_blake3"};
  std::string generic;
  if (full) {
    hipsycl::common::hcf_container hcf{std::string(embedded_resource("generic.hcf"))};
    const auto *images = hcf.root_node()->get_subnode("images"), *names = hcf.root_node()->get_subnode("kernels");
    const auto *image = images ? images->get_subnode("llvm-ir.global") : nullptr;
    // AdaptiveCpp emits normal and offset dispatch variants for each of the two parallel_fors.
    if (!image || !names || names->subnodes.size() != 4 || !hcf.get_binary_attachment(image, generic))
      throw std::runtime_error("Invalid generic SYCL compiler image");
    kernels = names->get_subnodes();
  }
  const std::string filename = std::string(directory) + (full ? "/generic-sm_" : "/sm_") + std::to_string(sm) + (fast ? "-fast" : "");
  // A negative target/version case must not overwrite the positive oracle's input snapshot.
  const bool invalidTarget = sm == 999 || version == 999;
  RecordedTranslator translator(kernels, filename + (invalidTarget ? "-invalid.bc" : ".bc"));
  if (fault != "no-resolver")
    translator.provideBitcodeLibraryResolver([&](const std::string &name) -> std::string_view {
      if (name == library && fault == "missing") return {};
      if (name == library && fault == "corrupt") return "not bitcode";
      return embedded_resource(name);
    });
  const auto configure = [&](hipsycl::compiler::LLVMToBackendTranslator &compiler) {
    if (!compiler.setBuildOption("ptx-target-device", sm) || !compiler.setBuildOption("ptx-version", version) ||
        (fast && !compiler.setBuildFlag("fast-math"))) throw std::runtime_error("Target/math policy rejected");
    if (full && (!compiler.setBuildOption("known-group-size-x", 32) ||
        !compiler.setBuildOption("known-group-size-y", 1) || !compiler.setBuildOption("known-group-size-z", 1)))
      throw std::runtime_error("Workgroup specialization rejected");
  };
  configure(translator);
  std::string ptx;
  const bool ok = full ? translator.fullTransformation(generic, ptx) : translator.translateToBackendFormat(*module, ptx);
  if (!fault.empty()) {
    const auto errors = translator.getErrorLogAsString();
    const std::string expected = fault == "no-resolver" ? "requires an embedded bitcode-library resolver" :
        fault == "missing" ? "Missing embedded bitcode library: " + library : "Could not load LLVM module";
    if (ok || !ptx.empty() || errors.find(expected) == std::string::npos)
      throw std::runtime_error("Embedded resource failure did not fail closed: " + fault + ": " + errors);
    llvm::outs() << fault << ": " << library << " rejected\n";
    return;
  }
  if (invalidTarget) {
    if (ok || translator.getErrorLog().empty()) throw std::runtime_error("Unknown target/version accepted");
    return;
  }
  if (!ok) throw std::runtime_error(translator.getErrorLogAsString());
  if (full) {
    // The real CUDA queue uses this factory without any application-supplied resolver. Compare
    // its linked libraries with the independently embedded reference resources above.
    auto factory = hipsycl::compiler::createLLVMToPtxTranslator(kernels);
    configure(*factory);
    std::string actual;
    if (!factory->fullTransformation(generic, actual) || actual != ptx)
      throw std::runtime_error("Runtime factory resources differ: " + factory->getErrorLogAsString());
    llvm::outs() << "factory sm_" << sm << " fast=" << fast << " matched\n";
  }
  // Retain raw code even if a later assertion exposes a codegen regression.
  std::ofstream output(filename + ".ptx", std::ios::binary);
  output.write(ptx.data(), static_cast<std::streamsize>(ptx.size()));
  if (!output) throw std::runtime_error("Cannot preserve translated PTX");
  if (ptx.find(".target sm_" + std::to_string(sm)) == std::string::npos)
    throw std::runtime_error("PTX lost the requested device");
  for (const auto &kernel : kernels)
    if (ptx.find(".entry " + kernel) == std::string::npos)
      throw std::runtime_error("PTX lost a generated kernel");
  if (full && ptx.find(".maxntid 32, 1, 1") == std::string::npos)
    throw std::runtime_error("PTX lost runtime workgroup specialization");
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  try {
    for (const bool full : {false, true}) {
      for (const bool fast : {false, true}) {
        compile(argv[1], 61, 64, fast, full);
        compile(argv[1], 120, 87, fast, full);
      }
    }
    compile(argv[1], 999, 64, false, false);
    compile(argv[1], 61, 999, false, false);
    compile(argv[1], 61, 64, false, true, "no-resolver");
    for (const std::string library : {"libdevice.10.bc", "libkernel-sscp-ptx-full.bc"})
      for (const std::string fault : {"missing", "corrupt"})
        compile(argv[1], 61, 64, false, true, fault, library);
    llvm::outs() << "AdaptiveCpp in-process generic SYCL/PTX passed (maintained primitive; no GPU)\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return 1;
  }
}
