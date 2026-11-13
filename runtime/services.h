#pragma once
namespace facebook::jsi { class Runtime; }
namespace mom::runtime {
// OS/crypto/compression primitives used by the maintained controller. No external tools/DSOs.
void installServices(facebook::jsi::Runtime &runtime);
}
