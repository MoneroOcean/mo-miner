// Application arguments belong to MoM, never to the semantic runtime's CLI parser. The exported
// AOT unit and this tiny entrypoint replace upstream's console main without interpreting JS.
#include "hermes/VM/static_h.h"
#include "hermes/hermes.h"
#include "process.h"
#include "windows-stack.h"
#include "jit-cache.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <csignal>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

struct SHConsoleContext;
extern "C" {
SHUnit *sh_export_mom_entry(void);
SHConsoleContext *init_console_bindings(SHRuntime *, int, const char *const *);
bool run_event_loop(SHRuntime *, SHConsoleContext *);
void release_console_bindings(SHConsoleContext *);
void free_console_context(SHConsoleContext *);
}

static int run(int argc, const char *const *argv) {
#ifdef MOM_NATIVE_ACPP_CACHE
  // HCF registration is static, but persistent JIT storage must not open before this tag.
  if (!mom_acpp_set_appdb_cache_tag(momNativeJitCacheTag)) {
    std::fputs("Cannot initialize native JIT cache identity\n", stderr);
    return 1;
  }
#endif
#ifdef SIGPIPE
  // Node reports broken pipe writes as EPIPE. Match that process-local disposition instead of
  // dying before the I/O layer can report an error and close native workers/resources.
  if (std::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    std::fputs("Cannot configure native pipe error handling\n", stderr);
    return 1;
  }
#endif
  char *error = nullptr;
  SHRuntime *runtime = _sh_init_with_error(0, nullptr, &error);
  if (!runtime) {
    std::fprintf(stderr, "Native runtime initialization failed: %s\n", error ? error : "unknown error");
    std::free(error);
    return 1;
  }
  SHConsoleContext *context = nullptr;
  int status = 1;
  try {
    context = init_console_bindings(runtime, argc, argv);
    if (_sh_initialize_units(runtime, 1, sh_export_mom_entry) && run_event_loop(runtime, context)) {
      status = mom::runtime::processExitCode(*_sh_get_hermes_runtime(runtime));
    }
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "Native application failed: %s\n", failure.what());
  }
  // I/O callbacks/JS references must be released before the language runtime, even after failure.
  if (context) {release_console_bindings(context);}
  _sh_done(runtime);
  if (context) {free_console_context(context);}
  return status;
}

#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
  try {
    std::vector<std::string> utf8;
    for (int i = 0; i < argc; i++) {
      const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
      if (!size) {throw std::runtime_error("Invalid UTF-16 command line");}
      std::string value(static_cast<size_t>(size), '\0');
      if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr)) {
        throw std::runtime_error("Cannot encode command line as UTF-8");
      }
      value.pop_back();
      utf8.push_back(std::move(value));
    }
    std::vector<const char *> pointers;
    for (const auto &value : utf8) {pointers.push_back(value.c_str());}
    return run(argc, pointers.data());
  } catch (const std::exception &failure) {
    std::fprintf(stderr, "Native argument conversion failed: %s\n", failure.what());
    return 1;
  }
}
#else
int main(int argc, char **argv) {return run(argc, argv);}
#endif
