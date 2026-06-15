{
  "variables": {
    # SYCL implementation selector. Default "dpcpp" (Intel oneAPI DPC++) keeps the
    # existing Intel Linux and Windows builds byte-for-byte unchanged. The NVIDIA
    # build runs `node-gyp configure -- -Dmom_sycl_impl=dpcpp-cuda` to compile
    # the same SYCL sources with the DPC++ CUDA backend.
    "mom_sycl_impl%": "dpcpp",
    # CUDA AOT target arch for mom_sycl_impl=dpcpp-cuda. A single low arch (sm_80, Ampere), NOT a
    # multi-arch list: the nightly clang mis-selects a multi-nvptx-target fatbin at runtime (every
    # algo CUDA_ERROR_NO_BINARY on an sm_89 device), whereas a single sm_80 image's forward-compatible
    # PTX is JIT'd by the driver to the actual GPU at load -- so one sm_80 build runs natively on
    # Ampere/Ada/Hopper (verified on an L4/sm_89 at full perf). sm_80 is also the floor for pearl's
    # int8 mma.m16n8k32. Override e.g. -Dmom_cuda_arch=nvidia_gpu_sm_90 to pin a single newer arch.
    # (Blackwell sm_120 needs a CUDA 12.8+ toolkit.)
    "mom_cuda_arch%": "nvidia_gpu_sm_80"
  },
  "targets": [
    {
      "target_name": "mom",
      "sources": [
        "native/core.cpp",
        "native/xmrig-compat.cpp",
        "native/job.cpp",

        "xmrig/crypto/common/VirtualMemory.cpp",
        "xmrig/crypto/common/HugePagesInfo.cpp",
        "xmrig/base/crypto/keccak.cpp",
        "xmrig/base/crypto/sha3.cpp",
        "xmrig/backend/cpu/Cpu.cpp",
        "xmrig/3rdparty/libethash/ethash_internal.c",

        "xmrig/crypto/cn/CnCtx.cpp",
        "xmrig/crypto/cn/CnHash.cpp",
        "xmrig/crypto/cn/c_blake256.c",
        "xmrig/crypto/cn/c_groestl.c",
        "xmrig/crypto/cn/c_jh.c",
        "xmrig/crypto/cn/c_skein.c",

        "xmrig/crypto/randomx/aes_hash.cpp",
        "xmrig/crypto/randomx/bytecode_machine.cpp",
        "xmrig/crypto/randomx/dataset.cpp",
        "xmrig/crypto/randomx/soft_aes.cpp",
        "xmrig/crypto/randomx/virtual_memory.cpp",
        "xmrig/crypto/randomx/vm_interpreted.cpp",
        "xmrig/crypto/randomx/allocator.cpp",
        "xmrig/crypto/randomx/randomx.cpp",
        "xmrig/crypto/randomx/superscalar.cpp",
        "xmrig/crypto/randomx/vm_compiled.cpp",
        "xmrig/crypto/randomx/vm_interpreted_light.cpp",
        "xmrig/crypto/randomx/blake2_generator.cpp",
        "xmrig/crypto/randomx/instructions_portable.cpp",
        "xmrig/crypto/randomx/reciprocal.c",
        "xmrig/crypto/randomx/virtual_machine.cpp",
        "xmrig/crypto/randomx/vm_compiled_light.cpp",
        "xmrig/crypto/randomx/blake2/blake2b.c",
        "xmrig/crypto/randomx/panthera/KangarooTwelve.c",
        "xmrig/crypto/randomx/panthera/KeccakP-1600-reference.c",
        "xmrig/crypto/randomx/panthera/KeccakSpongeWidth1600.c",
        "xmrig/crypto/randomx/panthera/sha256.c",
        "xmrig/crypto/randomx/panthera/yespower-opt.c",

        "xmrig/crypto/ghostrider/sph_blake.c",
        "xmrig/crypto/ghostrider/sph_bmw.c",
        "xmrig/crypto/ghostrider/sph_cubehash.c",
        "xmrig/crypto/ghostrider/sph_echo.c",
        "xmrig/crypto/ghostrider/sph_fugue.c",
        "xmrig/crypto/ghostrider/sph_groestl.c",
        "xmrig/crypto/ghostrider/sph_hamsi.c",
        "xmrig/crypto/ghostrider/sph_hamsi_helper.c",
        "xmrig/crypto/ghostrider/sph_jh.c",
        "xmrig/crypto/ghostrider/sph_keccak.c",
        "xmrig/crypto/ghostrider/sph_luffa.c",
        "xmrig/crypto/ghostrider/sph_sha2.c",
        "xmrig/crypto/ghostrider/sph_shabal.c",
        "xmrig/crypto/ghostrider/sph_shavite.c",
        "xmrig/crypto/ghostrider/sph_simd.c",
        "xmrig/crypto/ghostrider/sph_skein.c",
        "xmrig/crypto/ghostrider/sph_whirlpool.c",
        "xmrig/crypto/ghostrider/ghostrider.cpp",

        "xmrig/3rdparty/argon2/lib/argon2.c",
        "xmrig/3rdparty/argon2/lib/core.c",
        "xmrig/3rdparty/argon2/lib/encoding.c",
        "xmrig/3rdparty/argon2/lib/genkat.c",
        "xmrig/3rdparty/argon2/lib/impl-select.c",
        "xmrig/3rdparty/argon2/lib/blake2/blake2.c",

        "xmrig/3rdparty/fmt/format.cc"
      ],
      "include_dirs": [
        ".",
        "xmrig",
        "xmrig/3rdparty/argon2/include",
        "xmrig/3rdparty/argon2/lib"
      ],
      "defines": [
        "NDEBUG",
        "HAVE_ROTR",
        "XMRIG_ALGO_CN_LITE",
        "XMRIG_ALGO_CN_HEAVY",
        "XMRIG_ALGO_CN_PICO",
        "XMRIG_ALGO_CN_FEMTO",
        "XMRIG_ALGO_ARGON2",
        "XMRIG_ALGO_GHOSTRIDER"
      ],
      "cflags!": [ "-O3" ],
      "cflags_cc!": [ "-std=gnu++1y", "-std=gnu++17", "-fno-exceptions" ],
      # The SYCL static/shared lib is always required; per-arch argon2/blake2b
      # libs are appended below for x86_64.
      "dependencies": [ "sycl" ],
      "conditions": [
        [ "OS=='win'", {
          "sources": [
            "xmrig/crypto/common/VirtualMemory_win.cpp",
            "xmrig/backend/cpu/platform/BasicCpuInfo.cpp",
            "xmrig/hw/msr/Msr.cpp",
            "xmrig/hw/msr/Msr_win.cpp",
            "xmrig/crypto/rx/RxFix_win.cpp",
            "xmrig/crypto/randomx/jit_compiler_x86.cpp",
            "xmrig/crypto/randomx/jit_compiler_x86_static.asm",
            "xmrig/crypto/cn/r/CryptonightR_gen.cpp",
            "xmrig/crypto/cn/asm/win64/cn_main_loop.asm",
            "xmrig/crypto/cn/asm/win64/CryptonightR_template.asm",
            "xmrig/3rdparty/argon2/arch/generic/lib/argon2-arch.c"
          ],
          "defines": [
            "NOMINMAX",
            "WIN32_LEAN_AND_MEAN",
            "XMRIG_FEATURE_ASM"
          ],
          "configurations": {
            "Release": {
              "msvs_settings": {
                "VCCLCompilerTool": {
                  "RuntimeLibrary": 0
                }
              }
            },
            "Debug": {
              "msvs_settings": {
                "VCCLCompilerTool": {
                  "RuntimeLibrary": 1
                }
              }
            }
          },
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "LanguageStandard": "stdcpp20",
              "AdditionalOptions": [
                "/O2",
                "/fp:strict"
              ]
            },
            "VCLinkerTool": {
              "DelayLoadDLLs": [
                "sycl.dll"
              ],
              "AdditionalDependencies": [
                "delayimp.lib",
                "%(AdditionalDependencies)"
              ],
              "AdditionalOptions": [
                "/DELAYLOAD:sycl.dll"
              ]
            }
          }
        }, {
          "sources": [
            "xmrig/crypto/common/VirtualMemory_unix.cpp",
            "xmrig/crypto/cn/r/CryptonightR_gen.cpp",
            "<!@(./scripts/cpu-feature.sh x86_64 && ("
            "     echo \"xmrig/backend/cpu/platform/BasicCpuInfo.cpp\""
            "     echo \"xmrig/hw/msr/Msr.cpp\""
            "     echo \"xmrig/hw/msr/Msr_linux.cpp\""
            "     echo \"xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-arch.c\""
            "     echo \"xmrig/crypto/rx/RxFix_linux.cpp\""
            "     echo \"xmrig/crypto/cn/asm/cn_main_loop.S\""
            "     echo \"xmrig/crypto/cn/asm/CryptonightR_template.S\""
            "     echo \"xmrig/crypto/randomx/jit_compiler_x86_static.S\""
            "     echo \"xmrig/crypto/randomx/jit_compiler_x86.cpp\""
            "   ) || ("
            "     echo \"xmrig/backend/cpu/platform/BasicCpuInfo_arm.cpp\""
            "     echo \"xmrig/3rdparty/argon2/arch/generic/lib/argon2-arch.c\""
            "     echo \"xmrig/crypto/randomx/jit_compiler_a64_static.S\""
            "     echo \"xmrig/crypto/randomx/jit_compiler_a64.cpp\""
            "   ))"
          ],
          "cflags+": [
            "<!@(./scripts/cpu-cflags.sh)",
            "<!@(./scripts/cpu-feature.sh avx512f && echo \"-DHAVE_AVX512F\" || echo)",
            "<!@(./scripts/cpu-feature.sh avx2    && echo \"-DHAVE_AVX2 -DXMRIG_FEATURE_AVX2\" || echo)",
            "<!@(./scripts/cpu-feature.sh xop     && echo \"-DHAVE_XOP\" || echo)",
            "<!@(./scripts/cpu-feature.sh sse4_1  && echo \"-DXMRIG_FEATURE_SSE4_1\" || echo)",
            "<!@(./scripts/cpu-feature.sh ssse3   && echo \"-DHAVE_SSSE3\" || echo)",
            "<!@(./scripts/cpu-feature.sh sse2    && echo \"-DHAVE_SSE2\" || echo)",
            "<!@(./scripts/cpu-feature.sh msr     && echo \"-DXMRIG_FEATURE_MSR\" || echo)",
            "<!@(./scripts/cpu-feature.sh vaes    && echo \"-DHAVE_VAES\" || echo)",
            "<!@(./scripts/cpu-optflags.sh cflags)"
          ],
          "cflags_cc+": [ "-std=c++20" ],
          "ldflags+": [ "<!@(./scripts/cpu-optflags.sh ldflags)" ]
        } ],
        [ "OS!='win'", {
          # rpath order $$ORIGIN, lib, mom lets the .node find libsycl.so next to
          # itself, in build/Release/lib, or in build/Release/mom.
          "ldflags+": [
            "-fsycl",
            "-Wl,--disable-new-dtags",
            "-Wl,-rpath,'$$ORIGIN'",
            "-Wl,-rpath,'$$ORIGIN/lib'",
            "-Wl,-rpath,'$$ORIGIN/mom'"
          ],
          "conditions": [
            [ "mom_sycl_impl=='dpcpp-cuda'", {
              "ldflags+": [ "-fsycl-targets=<(mom_cuda_arch)" ]
            } ]
          ]
        } ],
        [ "OS!='win' and target_arch=='x64'", {
          "dependencies": [
            "argon2_x86_sse2",
            "argon2_x86_ssse3",
            "argon2_x86_xop",
            "argon2_x86_avx2",
            "argon2_x86_avx512f",
            "randomx_blake2b_sse41",
            "randomx_blake2b_avx2"
          ]
        } ]
      ]
    },
    {
      "target_name": "sycl",
      "type": "static_library",
      "win_delay_load_hook": "false",
      "sources": [
        "sycl/lib.cpp",
        "sycl/ethash.cpp",
        "sycl/etchash.cpp",
        "sycl/autolykos2.cpp",
        "sycl/pearl.cpp",
        "sycl/c29.cpp",
        "sycl/cn-gpu.cpp",
        "sycl/kawpow.cpp"
      ],
      "include_dirs": [
        "xmrig"
      ],
      "cflags!": [ "-O3" ],
      "cflags_cc!": [ "-fno-rtti", "-fno-exceptions", "-std=gnu++1y", "-std=gnu++17" ],
      "conditions": [
        [ "OS=='win'", {
          "type": "shared_library",
          # Release/Debug differ only in RuntimeLibrary (MD vs MDd); the toolset and
          # linker flags below are shared via the target-level msvs_settings.
          "configurations": {
            "Release": {
              "msbuild_toolset": "Intel(R) oneAPI DPC++ Compiler 2026",
              "msvs_settings": { "VCCLCompilerTool": { "RuntimeLibrary": 2 } }
            },
            "Debug": {
              "msbuild_toolset": "Intel(R) oneAPI DPC++ Compiler 2026",
              "msvs_settings": { "VCCLCompilerTool": { "RuntimeLibrary": 3 } }
            }
          },
          "sources": [
            "sycl/blake2b.cpp",
            "xmrig/crypto/randomx/blake2/blake2b.c",
            "xmrig/base/crypto/keccak.cpp",
            "xmrig/base/crypto/sha3.cpp"
          ],
          "defines": [
            "MOM_SYCL_BUILD"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "LanguageStandard": "stdcpp20",
              "AdditionalOptions": [
                "/O2",
                "/fsycl",
                "/DNDEBUG",
                "/DPEARL_ESIMD",
                "/clang:-fno-strict-aliasing"
              ]
            },
            "VCLinkerTool": {
              "AdditionalLibraryDirectories": [
                "$(ICInstallDir)lib",
                "%(AdditionalLibraryDirectories)"
              ],
              "AdditionalOptions": [
                "/DLL",
                "/fsycl"
              ]
            }
          }
        }, {
          "conditions": [
            [ "mom_sycl_impl=='dpcpp-cuda'", {
              # NVIDIA-only via the intel/llvm DPC++ CUDA backend (AOT to nvptx). MOM_SYCL_HAS_CUDA
              # compiles the CUDA-capable host paths (pearl mma.sync search_cuda, kawpow source JIT);
              # device-specific code is gated per compilation pass on the compiler's __NVPTX__.
              # -ffp-contract=off keeps the cn/gpu FP recurrence deterministic; -fsycl-embed-ir for the
              # kawpow runtime kernel-compiler. Multi-arch AOT, NVIDIA-wide (Ampere/Ada/Hopper). No ESIMD.
              "cflags_cc!": [ "-std=gnu++20" ],
              "cflags+": [
                "-std=c++20 -O3 -ffp-contract=off -fsycl -fsycl-embed-ir -fsycl-targets=<(mom_cuda_arch) -DNDEBUG -DMOM_SYCL_HAS_CUDA"
              ]
            } ],
            [ "mom_sycl_impl=='dpcpp-combined'", {
              # Combined Intel+NVIDIA: one mom.node carrying both spir64 + nvptx device images. The
              # cxx-combined.sh wrapper compiles these TUs with the intel/llvm nightly clang and adds
              # -fsycl-targets=spir64,<cuda archs>; the flags here are the shared ones. MOM_SYCL_HAS_CUDA
              # compiles the CUDA host paths (pearl mma.sync, kawpow JIT); __NVPTX__ gates device code
              # per pass. Intel pearl keeps its full-speed ESIMD path, but ESIMD can't share a
              # -fsycl-targets with nvptx, so it is compiled spir64-only in the separate pearl_esimd.cpp
              # TU (the wrapper gives that file -fsycl-targets=spir64) and linked into the same binary;
              # MOM_PEARL_HAS_ESIMD tells pearl.cpp to call that external search_esimd.
              "sources": [ "sycl/pearl_esimd.cpp" ],
              "cflags_cc!": [ "-std=gnu++20" ],
              "cflags+": [
                "-std=c++20 -O3 -ffp-contract=off -fsycl -fsycl-embed-ir -DNDEBUG -DMOM_SYCL_HAS_CUDA -DMOM_PEARL_HAS_ESIMD"
              ]
            } ],
            [ "mom_sycl_impl=='dpcpp'", {
              "cflags+": [
                "-std=c++20 -O3 -fsycl -DNDEBUG -DPEARL_ESIMD"
              ]
            } ]
          ]
        } ]
      ]
    },
    {
      "target_name": "argon2_x86_sse2",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-sse2.c" ],
          "include_dirs": [
            "xmrig",
            "xmrig/3rdparty/argon2/include",
            "xmrig/3rdparty/argon2/lib"
          ],
          "defines": [ "HAVE_ROTR", "HAVE_SSE2" ],
          "cflags+": [ "-O3", "-fno-lto", "-msse2" ]
        } ]
      ]
    },
    {
      "target_name": "argon2_x86_ssse3",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-ssse3.c" ],
          "include_dirs": [
            "xmrig",
            "xmrig/3rdparty/argon2/include",
            "xmrig/3rdparty/argon2/lib"
          ],
          "defines": [ "HAVE_ROTR", "HAVE_SSSE3" ],
          "cflags+": [ "-O3", "-fno-lto", "-mssse3" ]
        } ]
      ]
    },
    {
      "target_name": "argon2_x86_xop",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-xop.c" ],
          "include_dirs": [
            "xmrig",
            "xmrig/3rdparty/argon2/include",
            "xmrig/3rdparty/argon2/lib"
          ],
          "defines": [ "HAVE_ROTR", "HAVE_XOP" ],
          "cflags+": [ "-O3", "-fno-lto", "-mxop" ]
        } ]
      ]
    },
    {
      "target_name": "argon2_x86_avx2",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-avx2.c" ],
          "include_dirs": [
            "xmrig",
            "xmrig/3rdparty/argon2/include",
            "xmrig/3rdparty/argon2/lib"
          ],
          "defines": [ "HAVE_ROTR", "HAVE_AVX2" ],
          "cflags+": [ "-O3", "-fno-lto", "-mavx2" ]
        } ]
      ]
    },
    {
      "target_name": "argon2_x86_avx512f",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/3rdparty/argon2/arch/x86_64/lib/argon2-avx512f.c" ],
          "include_dirs": [
            "xmrig",
            "xmrig/3rdparty/argon2/include",
            "xmrig/3rdparty/argon2/lib"
          ],
          "defines": [ "HAVE_ROTR", "HAVE_AVX512F" ],
          "cflags+": [ "-O3", "-fno-lto", "-mavx512f" ]
        } ]
      ]
    },
    {
      "target_name": "randomx_blake2b_sse41",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/crypto/randomx/blake2/blake2b_sse41.c" ],
          "include_dirs": [ "xmrig" ],
          "cflags+": [ "-O3", "-fno-lto", "-msse4.1" ]
        } ]
      ]
    },
    {
      "target_name": "randomx_blake2b_avx2",
      "type": "none",
      "conditions": [
        [ "OS!='win' and target_arch=='x64'", {
          "type": "static_library",
          "sources": [ "xmrig/crypto/randomx/blake2/avx2/blake2b_avx2.c" ],
          "include_dirs": [ "xmrig" ],
          "cflags+": [ "-O3", "-fno-lto", "-mavx2" ]
        } ]
      ]
    }
  ]
}
