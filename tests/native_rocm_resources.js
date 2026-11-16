"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {validateConfig, hiprtcHeaders, resourcePlan, buildResources} = require("../scripts/build-native-rocm-resources");
const config = (target = "win32") => ({version: 1, target, source: "/source", clr: "/clr", llvm: "/llvm", cmake: "/cmake", hipHeaders: "/headers"});
const list = (rows = Array.from({length: 16}, (_, i) => "${HIP_COMMON_INCLUDE_DIR}/hip/file" + i + ".h")) => "set(HIPRTC_HEADERS\n" + rows.join("\n") + "\n)\n"; // eslint-disable-line no-template-curly-in-string -- Literal pinned CMake syntax.

test.describe("Native ROCm target ABI recipe", {skip: process.platform !== "linux"}, () => {
  test("Windows device libraries use upstream's short-wchar setting", () => {assert.ok(resourcePlan(config(), "/new", 2).configure.includes("-DCLANG_OPTIONS_APPEND=-fshort-wchar"));});
  test("Linux retains its existing wchar ABI", () => {assert.ok(resourcePlan(config("linux"), "/new", 2).configure.includes("-DCLANG_OPTIONS_APPEND="));});
  test("device-only HIPRTC preprocessing explicitly selects the Windows auxiliary target", () => {assert.ok(resourcePlan(config(), "/new", 2).preprocess.includes("--target=x86_64-pc-windows-msvc"));});
  test("Linux preprocessing cannot accidentally borrow Windows typedefs", () => {assert.ok(resourcePlan(config("linux"), "/new", 2).preprocess.includes("--target=x86_64-unknown-linux-gnu"));});
  test("headers used by preprocessing produce compiler dependency evidence", () => {const p = resourcePlan(config(), "/new", 2); assert.ok(p.preprocess.includes("-dependency-file")); assert.ok(p.preprocess.includes("-sys-header-deps")); assert.ok(p.preprocess.includes("/new/hipRTC.d"));});
  test("original device libraries, not a hand-rewritten metadata pass, are compiled", () => {const p = resourcePlan(config(), "/new", 2); assert.ok(p.compile.includes("rocm-device-libs")); assert.ok(p.configure.includes("/new/source/device"));});
  test("no build command executes a target binary or discovers a GPU", () => {const p = resourcePlan(config(), "/new", 2); assert.equal([p.configure, p.compile, p.preprocess].flat().some(s => /\.exe$|nvidia-smi|rocminfo|qemu|--device/.test(s)), false);});
  test("ROCm autodetection cannot import a coinstalled package", () => {assert.ok(resourcePlan(config(), "/new", 2).configure.includes("-DCMAKE_DISABLE_FIND_PACKAGE_ROCM=ON"));});
  test("explicit tools and output paths remain separate arguments with spaces", () => {const c = config(); c.llvm = "/llvm path"; const p = resourcePlan(c, "/new output", 2); assert.equal(p.preprocess[0], "/llvm path/bin/clang"); assert.ok(p.preprocess.includes("/new output/hipRTC_header.h"));});
  test("unsupported and missing release targets are rejected", () => {for (const target of ["", "darwin", "windows", "host"]) {assert.throws(() => validateConfig(config(target)), /explicit/);}});
  test("unknown fields cannot select an unreviewed ABI override", () => {assert.throws(() => validateConfig({...config(), flags: "-Doverride"}), /explicit/);});
  test("missing, relative and interpolated paths fail before output mutation", () => {for (const source of ["", "relative", "/bad;list", "/bad\npath", "/bad${value}", '/bad"path']) {assert.throws(() => validateConfig({...config(), source}), /literal/);}}); // eslint-disable-line no-template-curly-in-string -- Negative CMake syntax.
  test("the upstream header-list generator's whitespace limitation is explicit", () => {assert.throws(() => validateConfig({...config(), hipHeaders: "/headers path"}), /literal/);});
  test("parallelism stays finite and bounded", () => {for (const jobs of [0, 17, 1.5, NaN, Infinity]) {assert.throws(() => resourcePlan(config(), "/new", jobs), /jobs/);}});
  test("resource production needs explicit build-only admission", () => {assert.throws(() => buildResources(config(), "/missing", {}), /BUILD_ONLY/);});
  test("CLI without arguments starts no source or compiler work", () => {const r = spawnSync(process.execPath, [path.resolve(__dirname, "../scripts/build-native-rocm-resources.js")], {encoding: "utf8", timeout: 5000}); assert.equal(r.status, 1); assert.match(r.stderr, /Usage:/); assert.equal(r.stdout, "");});
});

test.describe("Native ROCm resource host compiler selection", {skip: process.platform !== "linux"}, () => {
  const hostCompilers = {cc: "/opt/intel/oneapi/compiler/2026.2/bin/icx", cxx: "/opt/intel/oneapi/compiler/2026.2/bin/icpx"};
  test("the build-time host helper uses the selected compiler without changing device commands", () => {
    for (const target of ["linux", "win32"]) {
      const baseline = resourcePlan(config(target), "/new", 2), selected = resourcePlan({...config(target), hostCompilers}, "/new", 2);
      assert.deepEqual(selected.hostCompilers, hostCompilers);
      assert.ok(selected.configure.includes("-DCMAKE_C_COMPILER=" + hostCompilers.cc));
      assert.ok(selected.configure.includes("-DCMAKE_CXX_COMPILER=" + hostCompilers.cxx));
      assert.deepEqual(selected.preprocess, baseline.preprocess);
      assert.equal(selected.clang, baseline.clang);
      assert.deepEqual(selected.configure.filter(arg => !/^-DCMAKE_C(?:XX)?_COMPILER=/.test(arg)),
        baseline.configure.filter(arg => !/^-DCMAKE_C(?:XX)?_COMPILER=/.test(arg)));
    }
  });
  test("an absent override retains the SDK host compiler pair", () => {
    assert.deepEqual(resourcePlan(config(), "/new", 1).hostCompilers, {cc: "/llvm/bin/clang", cxx: "/llvm/bin/clang++"});
  });
  test("incomplete, nonliteral and extra host compiler selections fail before output mutation", () => {
    for (const pair of [null, {}, {cc: hostCompilers.cc}, {...hostCompilers, extra: "/tool"},
      {...hostCompilers, cxx: "relative"}, {...hostCompilers, cc: "/bad;command"}, {...hostCompilers, cxx: "/bad\ncommand"}]) {
      assert.throws(() => validateConfig({...config(), hostCompilers: pair}), /compilers|literal/);
    }
  });
});

test.describe("Native HIPRTC original header membership", () => {
  test("common header references resolve inside the selected package", () => {assert.equal(hiprtcHeaders(list(), "/headers")[0], path.join("/headers", "hip/file0.h"));});
  test("CLR include references use the same complete selected header cohort", () => {assert.equal(hiprtcHeaders(list(Array(16).fill("${PROJECT_SOURCE_DIR}/include/hip/a.h")), "/headers")[0], path.join("/headers", "hip/a.h"));}); // eslint-disable-line no-template-curly-in-string -- Pinned CMake syntax.
  test("repeated upstream members retain their order rather than being curated", () => {const rows = hiprtcHeaders(list(Array(16).fill("${HIP_COMMON_INCLUDE_DIR}/hip/a.h")), "/headers"); assert.equal(rows.length, 16); assert.equal(new Set(rows).size, 1);}); // eslint-disable-line no-template-curly-in-string -- Pinned CMake syntax.
  test("missing or truncated upstream declarations fail", () => {for (const text of ["", "set(HIPRTC_HEADERS\n", list([]), list(Array(129).fill("x"))]) {assert.throws(() => hiprtcHeaders(text, "/headers"), /header/);}});
  for (const member of ["/outside.h", "${OTHER}/hip/a.h", "${HIP_COMMON_INCLUDE_DIR/hip/a.h", "${HIP_COMMON_INCLUDE_DIR}/hip/../a.h", "${HIP_COMMON_INCLUDE_DIR}/hip/a.h;extra", "${HIP_COMMON_INCLUDE_DIR}/hip/a.h)"]) { // eslint-disable-line no-template-curly-in-string -- Negative literal CMake syntax.
    test("nonliteral or escaping member is refused: " + member, () => {const rows = Array(16).fill(member); assert.throws(() => hiprtcHeaders(list(rows), "/headers"), /Nonliteral/);});
  }
  test("header root aliases cannot be shell or CMake fragments", () => {for (const root of ["relative", "/bad;root", "/bad\nroot", "/with space"]) {assert.throws(() => hiprtcHeaders(list(), root), /absolute/);}});
});

test.describe("Native HIPRTC actual cross-preprocessor ABI regression", () => {
  const clang = "/usr/lib/llvm-21/bin/clang";
  for (const target of ["linux", "win32"]) {
    test(target + " HIPRTC preprocessing keeps supplied headers without an installed HIP runtime", {skip: process.platform !== "linux" || !fs.existsSync(clang)}, t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hiprtc-no-runtime-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
      const headers = path.join(root, "headers"), supplied = path.join(headers, "hip", "supplied.h");
      fs.mkdirSync(path.dirname(supplied), {recursive: true});
      const targetGuard = target === "win32" ? "#if !defined(_WIN32)\n#error expected the Windows target ABI\n#endif\n" :
        "#if defined(_WIN32)\n#error expected the Linux target ABI\n#endif\n";
      fs.writeFileSync(supplied, targetGuard + "typedef __SIZE_TYPE__ hiprtc_size_t;\n" +
        "namespace __hip_internal {\n  typedef unsigned long size_t;\n}\nsize_t amd_dbgapi_get_build_id();\n");
      fs.writeFileSync(path.join(root, "hipRTC_header.h"), "#include <hip/supplied.h>\n");
      const c = {...config(target), llvm: path.dirname(path.dirname(clang)), hipHeaders: headers}, plan = resourcePlan(c, root, 1);
      const args = plan.preprocess.slice(1); args.splice(args.indexOf("-nogpulib") + 1, 0, "--hip-path=/nonexistent/mom-hip-probe");
      const r = spawnSync(clang, args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      assert.equal(r.status, 0, r.error?.message || r.stderr);
      const generated = fs.readFileSync(path.join(root, "hipRTC"), "utf8");
      const dependencies = fs.readFileSync(path.join(root, "hipRTC.d"), "utf8");
      assert.match(dependencies, /hip\/supplied\.h/);
      assert.ok(dependencies.includes(plan.hipRuntimeWrapper), "the compiler-owned wrapper remains in the dependency receipt");
      const wrapperSize = generated.search(/^typedef [^\n;]*\bsize_t;$/m), firstSizeUse = generated.indexOf("size_t amd_dbgapi_get_build_id");
      assert.ok(wrapperSize >= 0 && wrapperSize < firstSizeUse,
        "the generated HIPRTC header must retain the compiler wrapper's global size_t before upstream declarations");
      const check = spawnSync(clang, ["--target=" + plan.target, "--offload-arch=gfx900", "--cuda-device-only", "-nogpulib", "-nogpuinc",
        "-D__HIPCC_RTC__", "-D__HIP_PLATFORM_AMD__", "-x", "hip", "-std=c++17", "-fsyntax-only", "-include", path.join(root, "hipRTC"), "-"],
      {input: 'static_assert(__is_same(size_t, __SIZE_TYPE__), "HIPRTC size_t ABI");\n' +
        'static_assert(__is_same(hiprtc_size_t, __SIZE_TYPE__), "HIPRTC supplied-header ABI");\n', encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      assert.equal(check.status, 0, check.error?.message || check.stderr);
    });
  }
  test("Windows generation passes the real type check while the former Linux-host generation fails", {skip: process.platform !== "linux" || !fs.existsSync(clang)}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-hiprtc-abi-")); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(path.join(root, "hipRTC_header.h"), "typedef __SIZE_TYPE__ size_t;\n");
    for (const target of ["linux", "win32"]) {
      const c = {...config(target), llvm: path.dirname(path.dirname(clang)), hipHeaders: root};
      const plan = resourcePlan(c, root, 1), r = spawnSync(clang, plan.preprocess.slice(1), {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      assert.equal(r.status, 0, r.error?.message || r.stderr);
      assert.match(fs.readFileSync(path.join(root, "hipRTC.d"), "utf8"), /hipRTC_header\.h/);
      const check = spawnSync(clang, ["--target=x86_64-pc-windows-msvc", "--offload-arch=gfx900", "--cuda-device-only", "-nogpulib", "-nogpuinc", "-D__HIPCC_RTC__", "-D__HIP_PLATFORM_AMD__", "-x", "hip", "-std=c++17", "-fsyntax-only", "-include", path.join(root, "hipRTC"), "-"], {input: 'static_assert(__is_same(size_t, __SIZE_TYPE__), "HIPRTC size_t ABI");\n', encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      assert.equal(check.status, target === "win32" ? 0 : 1, check.error?.message || check.stderr);
      if (target === "linux") {assert.match(check.stderr, /typedef redefinition|HIPRTC size_t ABI/);}
    }
  });
});
