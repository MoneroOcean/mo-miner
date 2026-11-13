#pragma once
#include <string>
#include <vector>

namespace facebook::jsi { class Runtime; }

namespace mom::runtime {
// Install OS primitives only; argument layout and Node coercions live in the compiled JS adapter.
void installProcess(facebook::jsi::Runtime &runtime, int argc, const char *const *argv);
int processExitCode(facebook::jsi::Runtime &runtime);
// Native Linux workers use a non-writable loader cwd, retaining caller-relative data/cache paths.
const char *prepareWorkerDirectory(std::vector<std::string> &environment);
}
