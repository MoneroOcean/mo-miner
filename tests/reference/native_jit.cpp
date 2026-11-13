// CPU-only feasibility test: LLVM optimizes and lowers device IR in this process. It neither
// loads a GPU driver nor dispatches kernels. Default AMD output is relocatable; the optional
// static-LLD lane additionally qualifies HSA image linking, not device execution.
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef MOM_NATIVE_AMDGPU_LINK_PROBE
#include <lld/Common/Driver.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>
LLD_HAS_DRIVER(elf)

// Link generated GPU objects as data, not application libraries. This opt-in primitive uses
// private evidence/cache files; it never invokes ld.lld or evaluates an external source file.
static std::string link_amdgpu(const std::string &input, const std::string &output,
                               bool reject = false) {
  const std::vector<std::string> arguments{"ld.lld", "-m", "elf64_amdgpu", "--shared",
      "--no-undefined", "--build-id=none", "--threads=1", "-o", output, input};
  std::vector<const char *> argv;
  for (const auto &argument : arguments) argv.push_back(argument.c_str());
  std::string errors, messages;
  llvm::raw_string_ostream stderr_stream(errors), stdout_stream(messages);
  const auto result = lld::lldMain(argv, stdout_stream, stderr_stream, {{lld::Gnu, lld::elf::link}});
  if (!result.canRunAgain) throw std::runtime_error("In-process ELF linker is no longer reusable");
  if (reject) {
    if (!result.retCode || errors.empty()) throw std::runtime_error("Malformed GPU object accepted");
    return errors;
  }
  if (result.retCode) throw std::runtime_error("In-process ELF link failed: " + errors);
  auto bytes = llvm::MemoryBuffer::getFile(output);
  if (!bytes) throw std::runtime_error("Cannot preserve linked HSA image");
  auto object = llvm::object::ObjectFile::createObjectFile(bytes.get()->getMemBufferRef());
  if (!object) {llvm::consumeError(object.takeError()); throw std::runtime_error("Invalid linked HSA image");}
  const auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object->get());
  if (!elf || elf->getELFFile().getHeader().e_type != llvm::ELF::ET_DYN ||
      elf->getELFFile().getHeader().e_machine != llvm::ELF::EM_AMDGPU ||
      elf->getELFFile().getHeader().e_ident[llvm::ELF::EI_OSABI] != llvm::ELF::ELFOSABI_AMDGPU_HSA) {
    throw std::runtime_error("Wrong GPU code-object format");
  }
  return bytes.get()->getBuffer().str();
}
#endif

struct Variant {const char *cpu, *features; bool ptx;};

// These are test targets spanning older/newer devices, not production tuning or dispatch policy.
static void compile(const Variant &variant, const std::string &directory) {
  llvm::LLVMContext context;
  const std::string triple = variant.ptx ? "nvptx64-nvidia-cuda" : "amdgcn-amd-amdhsa";
  const std::string ir = std::string("define ") + (variant.ptx ? "ptx_kernel" : "amdgpu_kernel") +
      " void @probe(ptr addrspace(1) %out, i32 %value) {\n"
      "entry:\n %sum = add i32 %value, 7\n store i32 %sum, ptr addrspace(1) %out\n ret void\n}\n";
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(ir, diagnostic, context);
  if (!module) {throw std::runtime_error("Cannot parse JIT fixture");}
  std::string error;
  const auto *target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {throw std::runtime_error(error);}
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(llvm::Triple(triple),
      variant.cpu, variant.features, llvm::TargetOptions{}, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Aggressive));
  if (!machine) {throw std::runtime_error("Cannot create JIT target");}
  module->setTargetTriple(llvm::Triple(triple));
  module->setDataLayout(machine->createDataLayout());

  // The public new-pass-manager pipeline replaces the external opt stage; target code emission
  // replaces llc. No temporary bitcode, shell, PATH lookup or external compiler is involved.
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager calls;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder builder(machine.get());
  builder.registerModuleAnalyses(modules);
  builder.registerCGSCCAnalyses(calls);
  builder.registerFunctionAnalyses(functions);
  builder.registerLoopAnalyses(loops);
  builder.crossRegisterProxies(loops, functions, calls, modules);
  auto optimization = builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  optimization.run(*module, modules);

  llvm::SmallVector<char> bytes;
  llvm::raw_svector_ostream output(bytes);
  llvm::legacy::PassManager emission;
  const auto format = variant.ptx ? llvm::CodeGenFileType::AssemblyFile : llvm::CodeGenFileType::ObjectFile;
  if (machine->addPassesToEmitFile(emission, output, nullptr, format)) {
    throw std::runtime_error("JIT target cannot emit requested format");
  }
  emission.run(*module);
  const std::string result(bytes.begin(), bytes.end());
  if (variant.ptx ? result.find(".target " + std::string(variant.cpu)) == std::string::npos ||
      result.find(".entry probe") == std::string::npos
      : result.size() < 4 || result.compare(0, 4, "\x7f" "ELF") != 0) {
    throw std::runtime_error("JIT fixture produced invalid device output");
  }
  std::ofstream file(directory + "/" + variant.cpu + (variant.ptx ? ".ptx" : ".o"), std::ios::binary);
  file.write(result.data(), static_cast<std::streamsize>(result.size()));
  file.close();
  if (!file) {throw std::runtime_error("Cannot preserve JIT output");}
  llvm::outs() << variant.cpu << ": " << result.size() << " bytes\n";
#ifdef MOM_NATIVE_AMDGPU_LINK_PROBE
  if (!variant.ptx) {
    const std::string base = directory + "/" + variant.cpu;
    const auto linked = link_amdgpu(base + ".o", base + ".hsaco");
    const std::string bad = directory + "/invalid.o";
    {std::ofstream malformed(bad, std::ios::binary); malformed << "not a valid GPU object\n";
      malformed.close(); if (!malformed) throw std::runtime_error("Cannot preserve malformed GPU object");}
    {std::ofstream diagnostic(directory + "/" + variant.cpu + "-invalid-link.log");
      diagnostic << link_amdgpu(bad, base + "-invalid.hsaco", true);
      diagnostic.close(); if (!diagnostic) throw std::runtime_error("Cannot preserve linker rejection");}
    if (linked != link_amdgpu(base + ".o", base + "-repeated.hsaco")) {
      throw std::runtime_error("A rejected link contaminated subsequent GPU output");
    }
    llvm::outs() << variant.cpu << ": " << linked.size() << " bytes linked HSA image\n";
  }
#endif
}

int main(int argc, char **argv) {
  if (argc != 2) {return 2;}
  try {
    LLVMInitializeNVPTXTargetInfo(); LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC(); LLVMInitializeNVPTXAsmPrinter();
    LLVMInitializeAMDGPUTargetInfo(); LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC(); LLVMInitializeAMDGPUAsmPrinter();
    for (const Variant variant : {Variant{"sm_61", "+ptx64", true}, {"sm_120", "+ptx87", true},
        {"gfx900", "", false}, {"gfx1200", "", false}}) {compile(variant, argv[1]);}
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n'; return 1;
  }
  return 0;
}
