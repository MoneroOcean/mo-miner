// Build-only NVRTC image worker. It compiles maintained raw-string kernels but never loads a
// CUDA driver, discovers a device, or executes an image.
#include <nvrtc.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace {

constexpr size_t kMaxImageBytes = 64U * 1024U * 1024U;
constexpr size_t kMaxLogBytes = 1U * 1024U * 1024U;
constexpr size_t kMaxIncludeRoots = 16;

const char cnSource[] =
#include "../sycl/cn-gpu/cuda-expand-kernel.inc"
;
const char pearlSource[] =
#include "../sycl/pearlhash/cuda-kernel.inc"
;
const char compactSource[] =
#include "../sycl/pearlhash/cuda-compact-kernel.inc"
;

struct Program {
  nvrtcProgram value = nullptr;
  Program() = default;
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  ~Program() { if (value) nvrtcDestroyProgram(&value); }
};

struct CompilerInfo { int major = 0, minor = 0; std::vector<int> architectures; };
struct Profile {
  const char* name;
  const char* sourcePath;
  const char* programName;
  const char* source;
  int rank;
  bool compact;
  bool pearl;
  std::vector<const char*> exports;
};
struct Target { int architecture; bool cubin; bool feature; };

std::string json(std::string_view value) {
  std::string result = "\"";
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') { result.push_back('\\'); result.push_back(static_cast<char>(c)); }
    else if (c < 0x20) {
      constexpr char digits[] = "0123456789abcdef";
      result += "\\u00";
      result.push_back(digits[c >> 4]);
      result.push_back(digits[c & 0x0f]);
    }
    else { result.push_back(static_cast<char>(c)); }
  }
  result.push_back('"');
  return result;
}

std::string errorText(nvrtcResult status) {
  const char* text = nvrtcGetErrorString(status);
  return text ? text : "unknown NVRTC error";
}

CompilerInfo compilerInfo() {
  CompilerInfo info;
  nvrtcResult status = nvrtcVersion(&info.major, &info.minor);
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("nvrtcVersion: " + errorText(status)); }
  if (info.major < 1 || info.minor < 0) { throw std::runtime_error("NVRTC returned an invalid version"); }
  int count = 0;
  status = nvrtcGetNumSupportedArchs(&count);
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("nvrtcGetNumSupportedArchs: " + errorText(status)); }
  if (count < 1 || count > 128) { throw std::runtime_error("NVRTC returned an invalid architecture count"); }
  info.architectures.resize(static_cast<size_t>(count));
  status = nvrtcGetSupportedArchs(info.architectures.data());
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("nvrtcGetSupportedArchs: " + errorText(status)); }
  if (std::any_of(info.architectures.begin(), info.architectures.end(), [](int sm) { return sm < 50 || sm > 999; })) {
    throw std::runtime_error("NVRTC returned an invalid architecture");
  }
  std::sort(info.architectures.begin(), info.architectures.end());
  if (std::adjacent_find(info.architectures.begin(), info.architectures.end()) != info.architectures.end()) {
    throw std::runtime_error("NVRTC returned duplicate architectures");
  }
  return info;
}

std::string compilerJson(const CompilerInfo& info) {
  std::ostringstream out;
  out << "\"nvrtc\":{\"major\":" << info.major << ",\"minor\":" << info.minor
      << "},\"supportedArchs\":[";
  for (size_t i = 0; i < info.architectures.size(); ++i) { out << (i ? "," : "") << info.architectures[i]; }
  out << "]";
  return out.str();
}

Profile profileFor(std::string_view name) {
  if (name == "cn-expand") {
    return {"cn-expand", "sycl/cn-gpu/cuda-expand-kernel.inc", "cn-gpu-expand.cu", cnSource,
      0, false, false, {"cn_gpu_expand_cuda"}};
  }
  if (name == "pearl-rank128") {
    return {"pearl-rank128", "sycl/pearlhash/cuda-kernel.inc", "pearlhash-cute.cu", pearlSource,
      128, false, true, {"pearlhash_cuda_gemm", "pearlhash_cuda_hash"}};
  }
  if (name == "pearl-rank256") {
    return {"pearl-rank256", "sycl/pearlhash/cuda-kernel.inc", "pearlhash-cute.cu", pearlSource,
      256, false, true, {"pearlhash_cuda_gemm", "pearlhash_cuda_hash"}};
  }
  if (name == "pearl-compact") {
    return {"pearl-compact", "sycl/pearlhash/cuda-compact-kernel.inc", "pearlhash-cute.cu", compactSource,
      128, true, true, {"compact_gemm", "selector_prepare"}};
  }
  throw std::runtime_error("unknown image profile");
}

std::vector<std::string> includeRoots(int argc, char** argv) {
  if ((argc - 5) % 2 != 0 || static_cast<size_t>((argc - 5) / 2) > kMaxIncludeRoots) {
    throw std::runtime_error("optional include roots must be repeated --include <absolute-directory> pairs");
  }
  std::vector<std::string> roots;
  for (int i = 5; i < argc; i += 2) {
    if (std::string_view(argv[i]) != "--include") { throw std::runtime_error("expected --include <absolute-directory>"); }
    const fs::path path(argv[i + 1]);
    if (!path.is_absolute() || !fs::is_directory(path)) { throw std::runtime_error("NVRTC include root must be an absolute directory"); }
    const std::string canonical = fs::canonical(path).string();
    if (std::find(roots.begin(), roots.end(), canonical) != roots.end()) { throw std::runtime_error("duplicate NVRTC include root"); }
    roots.push_back(canonical);
  }
  return roots;
}

Target parseTarget(std::string_view value) {
  bool cubin = false, feature = false;
  std::string_view digits;
  if (value.starts_with("sm_")) {
    cubin = true;
    digits = value.substr(3);
    if (!digits.empty() && digits.back() == 'a') { feature = true; digits.remove_suffix(1); }
  } else if (value.starts_with("compute_")) {
    digits = value.substr(8);
  } else {
    throw std::runtime_error("target must be sm_N or compute_N");
  }
  if (digits.size() < 2 || digits.size() > 3 ||
      !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    throw std::runtime_error("invalid NVRTC target architecture");
  }
  int architecture = 0;
  for (const char c : digits) { architecture = architecture * 10 + (c - '0'); }
  if (architecture < 50 || architecture > 999 || (feature && (!cubin || architecture != 120))) {
    throw std::runtime_error("unsupported NVRTC target architecture");
  }
  return {architecture, cubin, feature};
}

void validateSelection(const Profile& profile, const Target& target, std::string_view format,
                       const CompilerInfo& info) {
  if ((format == "cubin") != target.cubin || (format != "cubin" && format != "ptx")) {
    throw std::runtime_error("format must match sm_N/cubin or compute_N/ptx target");
  }
  if (target.architecture < 80 ||
      !std::binary_search(info.architectures.begin(), info.architectures.end(), target.architecture)) {
    throw std::runtime_error("target is not supported by this NVRTC compiler/profile");
  }
  if (profile.compact && (format != "cubin" || !target.feature || target.architecture != 120 ||
      info.major < 13 || (info.major == 13 && info.minor < 3))) {
    throw std::runtime_error("compact Pearl requires sm_120a CUBIN and NVRTC 13.3+");
  }
  if (!profile.compact && target.feature) { throw std::runtime_error("feature targets are reserved for compact Pearl"); }
}

std::string programLog(nvrtcProgram program) {
  size_t size = 0;
  if (nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS || !size || size > kMaxLogBytes) {
    return "<NVRTC log unavailable or exceeded 1 MiB bound>";
  }
  std::string result(size, '\0');
  if (nvrtcGetProgramLog(program, result.data()) != NVRTC_SUCCESS) { return "<NVRTC log read failed>"; }
  while (!result.empty() && result.back() == '\0') { result.pop_back(); }
  return result;
}

bool containsExport(const std::vector<char>& image, const char* name, bool ptx) {
  const size_t length = std::char_traits<char>::length(name);
  if (!ptx) { return std::search(image.begin(), image.end(), name, name + length) != image.end(); }
  const std::string_view text(image.data(), image.size());
  for (size_t offset = 0; (offset = text.find(".entry", offset)) != std::string_view::npos; ++offset) {
    size_t cursor = offset + 6;
    while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) { ++cursor; }
    if (cursor + length > text.size() || text.substr(cursor, length) != name) { continue; }
    const size_t end = cursor + length;
    if (end == text.size() || !(std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_')) { return true; }
  }
  return false;
}

std::string ptxVersion(const std::vector<char>& image) {
  const std::string_view text(image.data(), image.size());
  const size_t marker = text.find(".version ");
  if (marker == std::string_view::npos) { throw std::runtime_error("NVRTC PTX has no .version directive"); }
  size_t cursor = marker + 9, split = cursor;
  while (split < text.size() && std::isdigit(static_cast<unsigned char>(text[split]))) { ++split; }
  if (split == cursor || split >= text.size() || text[split] != '.') { throw std::runtime_error("invalid NVRTC PTX version"); }
  size_t end = split + 1;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) { ++end; }
  if (end == split + 1 || end - cursor > 8) { throw std::runtime_error("invalid NVRTC PTX version"); }
  return std::string(text.substr(cursor, end - cursor));
}

void writeImage(const fs::path& output, const std::vector<char>& image) {
  if (!output.is_absolute() || output.filename().empty() || !fs::is_directory(output.parent_path())) {
    throw std::runtime_error("output must be an absolute file path in an existing directory");
  }
  // Exclusive creation prevents replacing a prior artifact and does not follow a leaf symlink.
  std::FILE* file = std::fopen(output.string().c_str(), "wbx");
  if (!file) { throw std::runtime_error("cannot create image output exclusively"); }
  const size_t written = std::fwrite(image.data(), 1, image.size(), file);
  const int closeStatus = std::fclose(file);
  if (written != image.size() || closeStatus != 0) {
    throw std::runtime_error("cannot preserve complete NVRTC image");
  }
}

void compile(const Profile& profile, const Target& target, std::string_view format,
             const fs::path& output, const CompilerInfo& info, const std::vector<std::string>& roots) {
  if (profile.pearl && roots.empty()) { throw std::runtime_error("Pearl profiles require explicit NVRTC --include roots"); }
  if (!profile.pearl && !roots.empty()) { throw std::runtime_error("include roots are only valid for Pearl profiles"); }

  std::vector<std::string> optionStorage{"--std=c++17"};
  if (profile.rank) { optionStorage.emplace_back("-DPEARLHASH_RANK=" + std::to_string(profile.rank)); }
  optionStorage.emplace_back("--gpu-architecture=" + std::string(target.cubin ? "sm_" : "compute_") +
      std::to_string(target.architecture) + (target.feature ? "a" : ""));
  for (const auto& root : roots) { optionStorage.emplace_back("--include-path=" + root); }
  optionStorage.emplace_back("--device-as-default-execution-space");
  if (profile.pearl) { optionStorage.emplace_back("--use_fast_math"); }
  if (profile.compact) { optionStorage.emplace_back("-DDIRECT_RANK_STRIDE=1"); }
  std::vector<const char*> options;
  for (const auto& value : optionStorage) { options.push_back(value.c_str()); }

  Program program;
  nvrtcResult status = nvrtcCreateProgram(&program.value, profile.source, profile.programName,
      0, nullptr, nullptr);
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("nvrtcCreateProgram: " + errorText(status)); }
  status = nvrtcCompileProgram(program.value, static_cast<int>(options.size()), options.data());
  if (status != NVRTC_SUCCESS) {
    throw std::runtime_error("nvrtcCompileProgram: " + errorText(status) + "\n" + programLog(program.value));
  }
  size_t size = 0;
  status = target.cubin ? nvrtcGetCUBINSize(program.value, &size) : nvrtcGetPTXSize(program.value, &size);
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("NVRTC image size: " + errorText(status) + "\n" + programLog(program.value)); }
  if (!size || size > kMaxImageBytes) { throw std::runtime_error("NVRTC image is empty or exceeds 64 MiB"); }
  std::vector<char> image(size);
  status = target.cubin ? nvrtcGetCUBIN(program.value, image.data()) : nvrtcGetPTX(program.value, image.data());
  if (status != NVRTC_SUCCESS) { throw std::runtime_error("NVRTC image read: " + errorText(status) + "\n" + programLog(program.value)); }
  for (const char* name : profile.exports) {
    if (!containsExport(image, name, !target.cubin)) { throw std::runtime_error(std::string("NVRTC image is missing maintained export: ") + name); }
  }
  const std::string ptx = target.cubin ? std::string{} : ptxVersion(image);
  writeImage(output, image);

  std::cout << "{\"status\":\"ok\"," << compilerJson(info)
      << ",\"profile\":" << json(profile.name) << ",\"source\":" << json(profile.sourcePath)
      << ",\"program\":" << json(profile.programName)
      << ",\"target\":" << json(target.cubin ? "sm_" + std::to_string(target.architecture) + (target.feature ? "a" : "") :
          "compute_" + std::to_string(target.architecture))
      << ",\"format\":" << json(format) << ",\"ptxVersion\":"
      << (target.cubin ? "null" : json(ptx)) << ",\"options\":[";
  for (size_t i = 0; i < optionStorage.size(); ++i) { std::cout << (i ? "," : "") << json(optionStorage[i]); }
  std::cout << "],\"includeRoots\":[";
  for (size_t i = 0; i < roots.size(); ++i) { std::cout << (i ? "," : "") << json(roots[i]); }
  std::cout << "],\"exports\":[";
  for (size_t i = 0; i < profile.exports.size(); ++i) {
    std::cout << (i ? "," : "") << json(profile.exports[i]);
  }
  std::cout << "],\"outputBytes\":" << size << "}\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CompilerInfo info = compilerInfo();
    if (argc == 2 && std::string_view(argv[1]) == "--query") {
      std::cout << "{\"status\":\"ok\"," << compilerJson(info) << "}\n";
      return 0;
    }
    if (argc < 5 || ((argc - 5) % 2 != 0)) {
      throw std::runtime_error("usage: native-cuda-images <profile> <sm_N|compute_N> <cubin|ptx> <absolute-output-file> [--include <absolute-directory> ...] | --query");
    }
    const Profile profile = profileFor(argv[1]);
    const Target target = parseTarget(argv[2]);
    const std::string_view format(argv[3]);
    validateSelection(profile, target, format, info);
    const auto roots = includeRoots(argc, argv);
    compile(profile, target, format, argv[4], info, roots);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native-cuda-images: " << error.what() << "\n";
    return 1;
  }
}
