"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"),
  vm = require("node:vm");
const {pins, archiveNames, hsakmtInstallArchiveName, xxdPin, validateConfig, rocmHostPlan, hostEnvironment, withPrivateBuildDirectories,
  parsePackageOrigin, compileDepfiles, readCompileDatabase, useStableHipPchStdin, patchExportedHipPchScript,
  useExplicitRocrDeviceLibPath, patchExportedRocrDeviceLibPath, buildRocmHostRuntime} =
  require("../scripts/build-native-rocm-host-runtime");
const {prepareRocmHostAdmission} = require("../scripts/prepare-native-ci");

const host = {cc: "/intel/2026.2/bin/icx", cxx: "/intel/2026.2/bin/icpx"};
const config = () => ({version: 1,
  sources: {clr: "/sources/clr", hip: "/sources/hip", rocr: "/sources/rocr"},
  tools: {ar: "/usr/bin/x86_64-linux-gnu-ar", cmake: "/usr/bin/cmake", dpkgQuery: "/usr/bin/dpkg-query", git: "/usr/bin/git", llvmObjcopy: "/opt/llvm21-ubuntu24/bin/llvm-objcopy",
    ninja: "/usr/bin/ninja", perl: "/usr/bin/perl", pkgConfig: "/usr/bin/pkgconf", python3: "/usr/bin/python3.12",
    ranlib: "/usr/bin/x86_64-linux-gnu-ranlib", tar: "/usr/bin/tar", xxd: "/usr/bin/xxd"},
  hostCompilers: host, pkgConfigDir: "/usr/lib/x86_64-linux-gnu/pkgconfig",
  dependencies: {drm: "/usr/lib/x86_64-linux-gnu/libdrm.a", drmAmdgpu: "/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.a",
    elf: "/usr/lib/x86_64-linux-gnu/libelf.a", numa: "/usr/lib/x86_64-linux-gnu/libnuma.a"}, comgrReceipt: "/comgr/result.json"});
const llvm = "/llvm-21", comgrPackage = "/comgr/build/lib/cmake/amd_comgr", device = "/amd device libs",
  sdkRoots = {llvm, comgrPackage, device};

test.describe("Native original ROCm host-runtime recipe",
  {skip: process.platform !== "linux" ? "requires Linux ROCr and POSIX path semantics" : false}, () => {
    test("source pins are exact and source components remain separate", () => {
      assert.deepEqual(pins, {clr: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50", hip: "993b610351d78993091c0057fcf67636169b7467",
        rocr: "4e4d7abeb786e903ac2a46815ae2565de331cbd4"});
      assert.deepEqual(archiveNames, {amdhip64: "libamdhip64.a", hiprtc: "libhiprtc.a", hsa: "libhsa-runtime64.a", hsakmt: "libhsakmt-staticdrm.a"});
      assert.equal(hsakmtInstallArchiveName, "libhsakmt.a");
      assert.deepEqual(xxdPin, {path: "/usr/bin/xxd", package: "xxd", packageVersion: "2:9.1.0016-1ubuntu7.20",
        version: "xxd 2023-10-25 by Juergen Weigert et al."});
    });
    test("Linux SDK consumers restore original ROCm admission inputs before strict receipt readers", () => {
      const paths = {env: {}, runnerTemp: "/fixture/runner-temp"}, state = {pythonArchive: "/fixture/parser.tar"}, parser = {installed: true}, calls = /** @type {string[]} */ ([]),
        prepare = vm.runInNewContext("(" + prepareRocmHostAdmission.toString() + ")", {
          provisionRocmHostPackages: /** @type {(actualPaths: typeof paths) => void} */ (actualPaths => {assert.equal(actualPaths, paths); calls.push("packages");}),
          installRocmParserSource: /** @type {(actualPaths: typeof paths, actualState: typeof state) => typeof parser} */ ((actualPaths, actualState) => {
            assert.equal(actualPaths, paths); assert.equal(actualState, state); calls.push("parser"); return parser;
          }),
        });
      assert.equal(prepare(paths, state), parser);
      assert.deepEqual(calls, ["packages", "parser"], "the existing pinned package and parser helpers run in producer order");

      const source = fs.readFileSync(path.join(__dirname, "../scripts/prepare-native-ci.js"), "utf8"),
        body = (/** @type {string} */ name) => {
          const start = source.indexOf("function " + name), end = source.indexOf("\n/**", start + 1);
          if (start < 0 || end <= start) {throw new Error(name + " remains a declared CI frontdoor");}
          return source.slice(start, end);
        };
      const consumers = /** @type {Array<[string, string, string]>} */ ([
        ["buildLinuxNativeSdkDescriptor", "restoreLinuxNativeSdkDescriptorInputs(state)", "nativeSdkConfig(config, repo)"],
        ["restoreLinuxNativeDescriptorHandoff", "restoreLinuxNativeSdkDescriptorInputs(state)", "nativeSdkConfig(config, repo)"],
        ["buildLinuxNativeApplication", "restoreLinuxNativeSdkDescriptorInputs(descriptor)", "auditLinuxNativeSdkDescriptorLineage(descriptor)"],
      ]);
      for (const [name, restore, admission] of consumers) {
        const section = body(name), restored = section.indexOf(restore), prepared = section.indexOf("prepareRocmHostAdmission(paths, "), admitted = section.indexOf(admission);
        assert.ok(restored >= 0 && prepared > restored && admitted > prepared,
          name + " prepares original host packages and pinned parser source after restore and before strict reader admission");
      }
      assert.match(body("buildRocmHostSdk"), /const parser = prepareRocmHostAdmission\(paths, state\)/,
        "the original producer and strict consumers share the same package/parser preparation path");
      const descriptor = body("buildLinuxNativeSdkDescriptor");
      assert.ok(descriptor.indexOf("provisionOneApi(paths") < descriptor.indexOf("verifyLinuxLlvmAdmissionCi(paths)") &&
        descriptor.indexOf("verifyLinuxLlvmAdmissionCi(paths)") < descriptor.indexOf("prepareRocmHostAdmission(paths, state.rocm)"),
      "the descriptor preserves the producer's signed-tool admission before host package preparation");
    });
    test("ROCr xxd generator is pinned, included in PATH and provenance inputs", () => {
      const c = config(), p = rocmHostPlan(c, "/output", 2, sdkRoots);
      assert.equal(validateConfig(c).tools.xxd, xxdPin.path);
      assert.ok(Object.values(p.tools).includes(xxdPin.path));
      assert.ok(hostEnvironment({}, c, p, "/output", llvm)["PATH"]?.split(path.delimiter).includes("/usr/bin"));
      assert.throws(() => validateConfig({...c, tools: {...c.tools, xxd: "/usr/local/bin/xxd"}}), /exact pinned roots/);
    });
    test("ROCr image kernels use the admitted source-built COMGR device libraries", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots), dollar = "$",
        source = `add_custom_command(OUTPUT ${dollar}{OUTPUT_FILE} COMMAND clang ${dollar}{CLANG_ARG_LIST} DEPENDS ${dollar}{INPUT_FILE})`,
        marker = `COMMAND clang ${dollar}{CLANG_ARG_LIST}`,
        replacement = `COMMAND clang "--rocm-device-lib-path=${dollar}{MOM_ROCM_DEVICE_LIBS_PATH}" ${dollar}{CLANG_ARG_LIST}`;
      assert.equal(p.deviceLibPath, path.join(device, "amdgcn/bitcode"));
      assert.ok(p.commands.rocrConfigure.includes("-DMOM_ROCM_DEVICE_LIBS_PATH=" + p.deviceLibPath));
      const patched = useExplicitRocrDeviceLibPath(source);
      assert.equal(patched, source.replace(marker, replacement));
      assert.ok(patched.includes(replacement));
      assert.ok(!patched.includes("-nogpulib"));
      assert.throws(() => useExplicitRocrDeviceLibPath(source.replace("COMMAND clang", "COMMAND other")), /command has changed/);
      assert.throws(() => rocmHostPlan(config(), "/output", 2, {llvm, comgrPackage}), /device-library roots/);
    });
    test("preflight compiler temporary directories exist before the first probe", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-tmp-"));
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const temporary = {home: path.join(root, "home"), tmp: path.join(root, "tmp")};
      fs.mkdirSync(temporary.home); fs.mkdirSync(temporary.tmp);
      const output = path.join(root, "output"), p = rocmHostPlan(config(), output, 2, sdkRoots);
      const env = hostEnvironment({}, config(), p, output, llvm, temporary);
      assert.equal(env["HOME"], temporary.home);
      assert.equal(env["TMPDIR"], temporary.tmp);
      assert.ok(fs.statSync(env["HOME"] || "").isDirectory());
      assert.ok(fs.statSync(env["TMPDIR"] || "").isDirectory());
      assert.equal(fs.existsSync(output), false);
    });
    test("private preflight directories are removed when an input probe fails", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-preflight-"));
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const before = fs.readdirSync(root);
      assert.throws(() => withPrivateBuildDirectories(root, temporary => {
        assert.ok(fs.statSync(temporary.home).isDirectory());
        assert.ok(fs.statSync(temporary.tmp).isDirectory());
        throw new Error("simulated compiler input probe failure");
      }), /simulated compiler input probe failure/);
      assert.deepEqual(fs.readdirSync(root), before);
    });
    test("ROCr installs its package exports before CLR/HIP configuration", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      assert.deepEqual(Object.keys(p.commands), ["rocrConfigure", "rocrBuild", "rocrInstallBinary", "rocrInstallDev", "clrConfigure", "clrBuild", "clrInstallBinary", "clrInstallDev"]);
      assert.ok(p.commands.rocrInstallDev.includes("dev"));
      assert.ok(p.commands.clrConfigure.includes("-Dhsa-runtime64_DIR=/output/rocr-install/lib/cmake/hsa-runtime64"));
    });
    test("all runtime C and C++ sources use Intel while original LLVM remains the ROCm assembler", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      for (const command of [p.commands.rocrConfigure, p.commands.clrConfigure]) {
        assert.ok(command.includes("-DCMAKE_C_COMPILER=" + host.cc));
        assert.ok(command.includes("-DCMAKE_CXX_COMPILER=" + host.cxx));
        assert.ok(!command.includes("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"));
        assert.ok(command.includes("-DPython3_EXECUTABLE=/usr/bin/python3.12"));
        assert.ok(command.includes("-DCMAKE_POSITION_INDEPENDENT_CODE=ON"));
        assert.ok(command.includes("-DBUILD_SHARED_LIBS=OFF"));
      }
      assert.ok(p.commands.rocrConfigure.includes("-DHSA_ASSEMBLER=" + llvm + "/bin/clang-21"));
      assert.ok(p.commands.rocrConfigure.includes("-DHSA_OBJCOPY=/opt/llvm21-ubuntu24/bin/llvm-objcopy"));
      assert.ok(p.commands.rocrConfigure.includes("-DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu"));
      assert.ok(p.commands.rocrConfigure.includes("-DLLVM_DIR=/opt/llvm21-ubuntu24/lib/cmake/llvm"));
      assert.ok(p.commands.rocrConfigure.includes("-DClang_DIR=" + path.join(llvm, "lib/cmake/clang")));
      assert.ok(p.commands.clrConfigure.includes("-DHIP_COMMON_DIR=/output/source/hip"));
      assert.ok(p.commands.clrConfigure.includes("-DHIPCC_BIN_DIR="));
      assert.ok(p.commands.clrConfigure.includes("-Damd_comgr_DIR=" + comgrPackage));
    });
    test("HIP PCH helpers use the image LLVM tool root selected by llvm-objcopy", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      assert.equal(p.imageLlvmRoot, "/opt/llvm21-ubuntu24");
      assert.ok(p.commands.clrConfigure.includes("-DHIP_LLVM_ROOT=" + p.imageLlvmRoot));
    });
    test("HIP PCH source paths are stable while scratch directories remain unique", () => {
      const source = ["tmp=$tmpdir/hip_pch.$$",
        "$LLVM_DIR/bin/clang -O3 -x hip $tmp/hip_pch.h -E >$tmp/pch_wave32.cui &&",
        "$LLVM_DIR/bin/clang -O3 -x hip $tmp/hip_pch.h -E >$tmp/pch_wave64.cui &&"].join("\n");
      const patched = useStableHipPchStdin(source);
      assert.match(patched, /^tmp=\$tmpdir\/hip_pch\.\$\$$/m);
      assert.equal(patched.split("-E - <$tmp/hip_pch.h").length - 1, 2);
      assert.ok(patched.includes("-E - <$tmp/hip_pch.h >$tmp/pch_wave32.cui &&"));
      assert.ok(patched.includes("-E - <$tmp/hip_pch.h >$tmp/pch_wave64.cui &&"));
      assert.throws(() => useStableHipPchStdin(source.replace("pch_wave32", "pch_wave31")), /unexpected preprocessor commands/);
    });
    test("HIP PCH source patch rejects unrecognized exported source without modifying it", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-pch-patch-"));
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const script = path.join(root, "hipamd/src/hip_embed_pch.sh");
      fs.mkdirSync(path.dirname(script), {recursive: true}); fs.writeFileSync(script, "unrecognized upstream source\n");
      assert.throws(() => patchExportedHipPchScript(root), /Pinned exported HIP PCH helper changed/);
      assert.equal(fs.readFileSync(script, "utf8"), "unrecognized upstream source\n");
    });
    test("ROCr image-kernel source patch rejects unrecognized exported CMake without modifying it", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-rocr-patch-"));
      t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const cmake = path.join(root, "runtime/hsa-runtime/image/blit_src/CMakeLists.txt");
      fs.mkdirSync(path.dirname(cmake), {recursive: true}); fs.writeFileSync(cmake, "unrecognized upstream source\n");
      assert.throws(() => patchExportedRocrDeviceLibPath(root), /Pinned exported ROCr image-kernel CMake source changed/);
      assert.equal(fs.readFileSync(cmake, "utf8"), "unrecognized upstream source\n");
    });
    test("COMGR 3.0 fallback searches the selected build prefix before coinstalled ROCm", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      assert.equal(p.comgrPrefix, "/comgr/build");
      assert.equal(p.cmakePrefixPath, "/output/rocr-install;/comgr/build");
      assert.ok(p.commands.clrConfigure.includes("-DCMAKE_PREFIX_PATH=" + p.cmakePrefixPath));
      assert.ok(p.commands.clrConfigure.includes("-Damd_comgr_DIR=" + comgrPackage));
    });
    test("CLR uses a receipt-bound private header overlay and upstream runtime/dev install components", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      assert.equal(p.comgrHeaderInclude, "/output/comgr-install-include");
      assert.equal(p.comgrHeader, "/output/comgr-install-include/amd_comgr/amd_comgr.h");
      assert.equal(p.clrCxxFlags, '-I"/output/comgr-install-include"');
      assert.ok(p.commands.clrConfigure.includes("-DCMAKE_CXX_FLAGS=" + p.clrCxxFlags));
      assert.ok(!p.clrCxxFlags.includes("/comgr/"));
      assert.deepEqual(p.commands.clrInstallBinary.slice(-2), ["--component", "binary"]);
      assert.deepEqual(p.commands.clrInstallDev.slice(-2), ["--component", "dev"]);
      assert.ok(![...p.commands.clrInstallBinary, ...p.commands.clrInstallDev].some(arg => /hipcc|hipconfig/.test(arg)));
    });
    test("only original static ROCr and CLR targets are requested", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      assert.deepEqual(p.commands.rocrBuild.slice(-7), ["--target", "hsa-runtime64_static", "hsakmt", "hsakmt-staticdrm", "--", "-d", "keepdepfile"]);
      assert.ok(p.commands.clrBuild.includes("amdhip64")); assert.ok(p.commands.clrBuild.includes("hiprtc"));
      assert.equal(Object.values(p.commands).flat().some(arg => /rocminfo|nvidia-smi|--device|qemu|--install.*\/opt\/rocm/.test(arg)), false);
      assert.deepEqual(Object.keys(config().dependencies).sort(), ["drm", "drmAmdgpu", "elf", "numa"]);
    });
    test("explicit host dependencies and tools survive as literal argv", () => {
      const c = config(), p = rocmHostPlan(c, "/output with spaces", 2, sdkRoots);
      assert.ok(p.commands.rocrConfigure.includes("-DLIBELF_LIBRARIES=" + c.dependencies.elf));
      assert.ok(p.commands.rocrConfigure.includes("-DNUMA=" + c.dependencies.numa));
      assert.ok(p.commands.clrConfigure.includes("-DNUMA_LIBRARIES=" + c.dependencies.numa));
      assert.ok(p.commands.rocrConfigure.includes("-DCMAKE_AR=" + c.tools.ar));
      assert.ok(p.commands.rocrConfigure.includes("-DCMAKE_RANLIB=" + c.tools.ranlib));
      assert.ok(p.commands.clrConfigure.includes('-DCMAKE_CXX_FLAGS=-I"/output with spaces/comgr-install-include"'));
    });
    test("malformed config, shell/list syntax, relative paths and unknown fields fail", () => {
      assert.throws(() => validateConfig({...config(), extra: true}), /exact pinned roots/);
      assert.throws(() => validateConfig({...config(), sources: {...config().sources, hip: "relative"}}), /exact pinned roots/);
      assert.throws(() => validateConfig({...config(), sources: {...config().sources, clr: "/source;escape"}}), /exact pinned roots/);
      assert.throws(() => validateConfig({...config(), tools: {...config().tools, old: "/usr/bin/cc"}}), /exact pinned roots/);
      assert.throws(() => validateConfig({...config(), tools: {...config().tools, python3: "/usr/bin/python3"}}), /exact pinned roots/);
    });
    for (const jobs of [0, 17, 1.5, NaN, Infinity]) {
      test("invalid parallelism fails in the pure plan: " + jobs, () => {assert.throws(() => rocmHostPlan(config(), "/output", jobs), /jobs/);});
    }
    test("output cannot overlap or contain any source or selected input", () => {
      for (const root of ["/sources/clr", "/sources/hip", "/sources/rocr", "/comgr/result.json", "/comgr/build",
        comgrPackage, device, path.join(device, "amdgcn/bitcode"), "/opt/llvm21-ubuntu24/lib/cmake/llvm", "/usr/bin", "/"]) {
        assert.throws(() => rocmHostPlan(config(), root, 2, sdkRoots), /must not overlap/);
      }
    });
    test("hostile inherited Git and compiler hooks are removed before commands", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      const env = hostEnvironment({PATH: "/attacker/bin", GIT_DIR: "/attacker/repo", GIT_CONFIG_GLOBAL: "/attacker/config",
        GIT_CONFIG_NOSYSTEM: "0", GIT_ALTERNATE_OBJECT_DIRECTORIES: "/attacker/objects", CMAKE_TOOLCHAIN_FILE: "/attacker/toolchain",
        CC: "/attacker/cc", CFLAGS: "-DATTACKER", PKG_CONFIG_LIBDIR: "/attacker/pkgconfig"},
      config(), p, "/output", llvm);
      assert.equal(env["GIT_DIR"], undefined);
      assert.equal(env["GIT_ALTERNATE_OBJECT_DIRECTORIES"], undefined);
      assert.equal(env["GIT_CONFIG_GLOBAL"], "/dev/null");
      assert.equal(env["GIT_CONFIG_NOSYSTEM"], "1");
      assert.equal(env["CMAKE_TOOLCHAIN_FILE"], undefined);
      assert.equal(env["CFLAGS"], undefined);
      assert.equal(env["PKG_CONFIG_LIBDIR"], config().pkgConfigDir);
      assert.equal(env["PYTHONDONTWRITEBYTECODE"], "1");
      assert.ok(env["PATH"]?.startsWith("/intel/2026.2/bin"));
    });
    test("preserves the selected SDK shared-library runtime path", () => {
      const p = rocmHostPlan(config(), "/output", 2, sdkRoots);
      const loaderPath = "/opt/ubuntu24-libs";
      const env = hostEnvironment({LD_LIBRARY_PATH: loaderPath}, config(), p, "/output", llvm);
      assert.equal(env["LD_LIBRARY_PATH"], loaderPath);
    });
  });

test.describe("Native ROCm OS archive origin evidence", () => {
  test("multiarched dpkg owner matches the explicit package and architecture fields", () => {
    assert.deepEqual(parsePackageOrigin("/usr/lib/x86_64-linux-gnu/libdrm.a",
      "libdrm-dev:amd64: /usr/lib/x86_64-linux-gnu/libdrm.a\n",
      "libdrm-dev\tamd64\t2.4.131-1\tii \n", "a".repeat(64), "libdrm-dev"),
    {path: "/usr/lib/x86_64-linux-gnu/libdrm.a", sha256: "a".repeat(64), package: "libdrm-dev", architecture: "amd64", version: "2.4.131-1", status: "ii"});
  });
  test("unqualified package owners remain valid for all-architecture packages", () => {
    assert.equal(parsePackageOrigin("/usr/lib/x86_64-linux-gnu/libelf.a", "libelf-dev: /usr/lib/x86_64-linux-gnu/libelf.a",
      "libelf-dev\tall\t0.190-1\tii ", "b".repeat(64), "libelf-dev").architecture, "all");
  });
  test("an architecture-specific owner may be unqualified in dpkg-query", () => {
    assert.deepEqual(parsePackageOrigin("/usr/bin/xxd", "xxd: /usr/bin/xxd",
      "xxd\tamd64\t2:9.1.0016-1ubuntu7.20\tii ", "c".repeat(64), "xxd"),
    {path: "/usr/bin/xxd", sha256: "c".repeat(64), package: "xxd", architecture: "amd64", version: "2:9.1.0016-1ubuntu7.20", status: "ii"});
  });
  test("missing, ambiguous or mismatched package ownership is refused", () => {
    const file = "/usr/lib/libdrm.a", hash = "c".repeat(64);
    assert.throws(() => parsePackageOrigin(file, "", "", hash, "libdrm-dev"), /exactly one/);
    assert.throws(() => parsePackageOrigin(file, "other:amd64: /usr/lib/libdrm.a", "other\tamd64\t1\tii", hash, "libdrm-dev"), /ownership is ambiguous/);
    assert.throws(() => parsePackageOrigin(file, "libdrm-dev:amd64: /usr/lib/libdrm.a", "libdrm-dev\tarm64\t1\tii", hash, "libdrm-dev"), /not installed/);
    assert.throws(() => parsePackageOrigin(file, "libdrm-dev:amd64: /usr/lib/libdrm.a", "libdrm-dev\tamd64\t1\tun", hash, "libdrm-dev"), /not installed/);
  });
});

test.describe("Native ROCm CMake dependency guards",
  {skip: process.platform !== "linux" ? "requires Linux CMake output paths" : false}, () => {
    test("compile database depfiles stay within their exact build tree", () => {
      const rows = Array.from({length: 8}, (_, i) => ({directory: "/build", output: `CMakeFiles/runtime/${i}.o`}));
      assert.equal(compileDepfiles("/build", rows)[0], "/build/CMakeFiles/runtime/0.o.d");
      for (const output of ["../outside.o", "/outside.o", "nested/../escape.o", "bad\\path.o", "x.d", ""]) {
        const invalid = rows.map(row => ({...row})); if (invalid[0]) {invalid[0].output = output;}
        assert.throws(() => compileDepfiles("/build", invalid), /compile output/);
      }
      const duplicate = rows.map(row => ({...row})); if (duplicate[1]) {duplicate[1].output = duplicate[0]?.output || "";}
      assert.throws(() => compileDepfiles("/build", duplicate), /Duplicate/);
      assert.throws(() => compileDepfiles("/build", rows.slice(1)), /compile database/);
    });
    test("compile database requires every row to use the selected Intel pair", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-db-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const rows = Array.from({length: 8}, (_, i) => ({directory: root, output: `CMakeFiles/runtime/${i}.o`, command: `"${host.cxx}" -c source${i}.cpp`}));
      const file = path.join(root, "compile_commands.json"); fs.writeFileSync(file, JSON.stringify(rows));
      assert.equal(readCompileDatabase(file, root, host).depfiles.length, 8);
      rows[3] = {directory: root, output: "CMakeFiles/runtime/3.o", command: "/usr/bin/g++ -c source3.cpp"}; fs.writeFileSync(file, JSON.stringify(rows));
      assert.throws(() => readCompileDatabase(file, root, host), /selected Intel compiler/);
      for (const define of ["-DCOMGR_DYN_DLL=0", "-DROCR_DYN_DLL"]) {
        rows[3] = {directory: root, output: "CMakeFiles/runtime/3.o", command: `"${host.cxx}" ${define} -c source3.cpp`};
        fs.writeFileSync(file, JSON.stringify(rows));
        assert.throws(() => readCompileDatabase(file, root, host), /dynamic DLL lookup shims/);
      }
    });
    test("build-only admission fails before source/tool access", () => {
      assert.throws(() => buildRocmHostRuntime(config(), "/missing/output", {}), /BUILD_ONLY/);
    });
    test("an existing output is refused without changing its prior evidence", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-existing-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const output = path.join(root, "existing-output"), evidence = path.join(output, "result.json");
      fs.mkdirSync(output); fs.writeFileSync(evidence, "prior receipt\n");
      assert.throws(() => buildRocmHostRuntime(config(), output, {MOM_AOT_BUILD_ONLY: "1"}), /never overwritten/);
      assert.equal(fs.readFileSync(evidence, "utf8"), "prior receipt\n");
      assert.deepEqual(fs.readdirSync(root), ["existing-output"]);
    });
    test("a protected output overlap is refused without creating scratch in that tree", t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-overlap-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const protectedRoot = path.join(root, "source"); fs.mkdirSync(protectedRoot);
      const c = config(); c.sources.clr = protectedRoot;
      const output = path.join(protectedRoot, "new-output");
      assert.throws(() => buildRocmHostRuntime(c, output, {MOM_AOT_BUILD_ONLY: "1"}), /overlap/);
      assert.deepEqual(fs.readdirSync(protectedRoot), []);
      assert.equal(fs.existsSync(output), false);
    });
    test("a source-root symlink is rejected before output creation", {skip: process.platform !== "linux"}, t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-rocm-host-paths-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      fs.mkdirSync(path.join(root, "real")); fs.symlinkSync(path.join(root, "real"), path.join(root, "source-link"));
      const c = config(); c.sources.clr = path.join(root, "source-link");
      const output = path.join(root, "new-output");
      assert.throws(() => buildRocmHostRuntime(c, output, {MOM_AOT_BUILD_ONLY: "1"}), /canonical directory/);
      assert.equal(fs.existsSync(output), false);
      assert.deepEqual(fs.readdirSync(root).sort(), ["real", "source-link"]);
    });
  });
