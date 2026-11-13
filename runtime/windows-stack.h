#pragma once

// LLVM's Windows executables reserve 10,000,000 bytes for compiler recursion. Static
// embedders do not inherit that executable policy from its archives; the PE default
// is only 1 MiB. Match upstream for the native main/worker and its host-JIT fixture.
// This reserves address space, not an eagerly committed 10-MB allocation per thread.
#if defined(_WIN32) && defined(_MSC_VER)
#pragma comment(linker, "/STACK:10000000")
#endif
