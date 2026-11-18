"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {adaptiveCppCompileCommands, nativeIncludeRoots, gpuProfile, writeAdaptiveCppUmbrella} = require("../scripts/native-gpu-build");
const {dpcppCompileCommands, oneapiCompilerInputs, buildInputs} = require("../scripts/build-native-gpu");
const {windowsGpuCompileCommands, windowsGpuHostFlags, windowsGpuDeviceFlags, windowsGpuSourceFlags, validateWindowsGpuConfig} = require("../scripts/build-native-windows-gpu");

const repo = path.resolve(__dirname, "..");

test.describe("Native GPU compile planning", () => {
  test("AdaptiveCpp umbrella retains relative SDK includes after cache relocation", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-acpp-umbrella-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const sdk = path.join(root, "sdk with spaces"), include = path.join(sdk, "include"), output = path.join(root, "generated"), legacy = path.join(root, "copied");
    for (const dir of [path.join(include, "SYCL"), path.join(include, "hipSYCL/sycl"), output, path.join(legacy, "sycl")]) {
      fs.mkdirSync(dir, {recursive: true});
    }
    const upstream = path.join(include, "SYCL/sycl.hpp"), input = path.join(root, "fixture.cpp");
    const umbrella = '#include "../hipSYCL/sycl/sycl.hpp"\nnamespace sycl { using namespace hipsycl::sycl; }\n';
    fs.writeFileSync(upstream, umbrella);
    fs.writeFileSync(path.join(include, "hipSYCL/sycl/sycl.hpp"), "namespace hipsycl::sycl { constexpr int sdk_value = 17; }\n");
    fs.writeFileSync(input, "#include <sycl/sycl.hpp>\nstatic_assert(sycl::sdk_value == 17);\n");
    const compile = (/** @type {string} */ generated) => spawnSync("c++", ["-std=c++17", "-fsyntax-only", "-I", generated, "-isystem", include, input],
      {encoding: "utf8", timeout: 30000, maxBuffer: 64 << 10});
    // Replay the old producer operation, not a deliberately invalid replacement header.
    fs.copyFileSync(upstream, path.join(legacy, "sycl/sycl.hpp"));
    const before = compile(legacy);
    assert.equal(before.status, 1, before.error?.message || before.stderr);
    assert.match(before.stderr, /hipSYCL\/sycl\/sycl\.hpp/);
    // Model Windows case-folding on Linux: SYCL and sycl resolve to the same directory.
    fs.symlinkSync("sycl", path.join(legacy, "SYCL"), "dir");
    fs.writeFileSync(path.join(legacy, "sycl/sycl.hpp"), "#include <SYCL/sycl.hpp>\n");
    const recursive = compile(legacy);
    assert.equal(recursive.status, 1, recursive.error?.message || recursive.stderr);
    assert.match(recursive.stderr, /include.*(?:depth|nested)|nested.*include/i);
    fs.writeFileSync(path.join(legacy, "sycl/sycl.hpp"), "#include <hipSYCL/sycl/sycl.hpp>\n");
    const missingNamespace = compile(legacy);
    assert.equal(missingNamespace.status, 1, missingNamespace.error?.message || missingNamespace.stderr);
    assert.match(missingNamespace.stderr, /sycl/);
    writeAdaptiveCppUmbrella(sdk, output);
    fs.symlinkSync("sycl", path.join(output, "SYCL"), "dir");
    const after = compile(output);
    assert.equal(after.status, 0, after.error?.message || after.stderr);
    assert.equal(fs.lstatSync(path.join(output, "sycl/sycl.hpp")).isSymbolicLink(), false);
    assert.equal(fs.readFileSync(upstream, "utf8"), umbrella);
    assert.throws(() => writeAdaptiveCppUmbrella(sdk, output), /EEXIST/);
    const quoted = path.join(root, 'sdk"quote'), rejected = path.join(root, "rejected");
    fs.mkdirSync(path.join(quoted, "include/SYCL"), {recursive: true}); fs.mkdirSync(rejected);
    fs.writeFileSync(path.join(quoted, "include/SYCL/sycl.hpp"), umbrella);
    assert.throws(() => writeAdaptiveCppUmbrella(quoted, rejected), /cannot be represented/);
    assert.deepEqual(fs.readdirSync(rejected), []);
    const windows = fs.readFileSync(path.join(repo, "scripts/build-native-windows-gpu.js"), "utf8");
    assert.match(windows, /if \(config\.acpp\) \{writeAdaptiveCppUmbrella\(config\.acpp\.source, output\);\}/);
  });

  test("AdaptiveCpp cache plan preserves source order and CUDA header exceptions", () => {
    const output = path.join(os.tmpdir(), "acpp-candidate");
    const files = ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp", "sycl/kawpow/kawpow.cpp", "sycl/hoohash/host-math.c",
      "runtime/driver-library.cpp"];
    const flags = ["-std=c++20", "-O3", "-I", output];
    const commands = adaptiveCppCompileCommands(files, "/sdk/bin/clang++", repo, output, flags, true, "cuda");
    assert.equal(commands.length, files.length);
    assert.deepEqual(commands.map(command => path.relative(repo, command.source).split(path.sep).join("/")), files);
    assert.deepEqual(commands[0]?.args.slice(0, flags.length + 1), [...flags, "-DMOM_SYCL_HAS_CUDA"]);
    assert.deepEqual(commands[1]?.args.slice(0, flags.length + 2), [...flags, "-DMOM_SYCL_HAS_CUDA", "-DMOM_EMBEDDED_NVRTC_HEADERS"]);
    assert.deepEqual(commands[2]?.generatedInputs, [path.join(output, "sycl/sycl.hpp")]);
    assert.deepEqual(commands[0]?.generatedInputs, [path.join(output, "sycl/sycl.hpp"), path.join(output, "nvrtc-api/nvrtc.h")]);
    assert.deepEqual(commands[1]?.generatedInputs, [path.join(output, "sycl/sycl.hpp"), path.join(output, "nvrtc-api/nvrtc.h"),
      path.join(output, "nvrtc-headers.inc")]);
    assert.deepEqual(commands[3]?.args.slice(0, 5), ["-x", "c", "-O3", "-ffp-contract=off", "-fPIC"]);
    for (const index of [0, 1, 2]) {
      const namespace = commands[index]?.args.indexOf("-acpp-sscp-hcf-object-id-namespace=acpp-cuda") ?? -1;
      assert.ok(namespace >= 1);
      assert.equal(commands[index]?.args[namespace - 1], "-mllvm");
    }
    for (const index of [3, 4]) {
      assert.ok(!commands[index]?.args.some(flag => flag.startsWith("-acpp-sscp-hcf-object-id-namespace=")));
    }
    assert.deepEqual(commands[3]?.generatedInputs, []);
    assert.deepEqual(commands[4]?.generatedInputs, []);
    assert.deepEqual(commands.map(command => path.basename(command.object)), ["0.o", "1.o", "2.o", "3.o", "4.o"]);
    const hip = adaptiveCppCompileCommands(files, "/sdk/bin/clang++", repo, output, flags, false, "hip");
    const combined = adaptiveCppCompileCommands(files, "/sdk/bin/clang++", repo, output, flags, true, "cuda-hip");
    assert.ok(hip[0]?.args.includes("-acpp-sscp-hcf-object-id-namespace=acpp-hip"));
    assert.ok(combined[0]?.args.includes("-acpp-sscp-hcf-object-id-namespace=acpp-cuda-hip"));
    assert.throws(() => adaptiveCppCompileCommands(files, "/sdk/bin/clang++", repo, output, flags, true, "omp"), /HCF identity profile/);
  });

  test("DPC++ cache plan preserves mixed compiler selection and only declares consumed generated inputs", () => {
    const output = path.join(os.tmpdir(), "dpcpp-candidate"), cuda = "/sdk/cuda";
    const files = ["sycl/c29/c29.cpp", "sycl/kawpow/kawpow.cpp", "sycl/pearlhash/pearlhash.cpp", "sycl/nexapow/validation.cpp",
      "sycl/hoohash/host-math.c"];
    const commands = dpcppCompileCommands({sources: files, repo, output, sdk: "/sdk", compiler: "/sdk/bin/clang++",
      flags: ["-fsycl", "-DMOM_EMBEDDED_SYCL_JIT"], wrapper: "", profile: "oneapi", cuda, oneapiCompiler: "/oneapi/bin/icpx"});
    assert.equal(commands.length, files.length);
    assert.deepEqual(commands.map(command => path.relative(repo, command.source).split(path.sep).join("/")), files);
    assert.equal(commands[0]?.command, "/oneapi/bin/icpx");
    assert.equal(commands[1]?.command, "/oneapi/bin/icpx");
    assert.equal(commands[2]?.command, "/oneapi/bin/icpx");
    assert.equal(commands[3]?.command, "/oneapi/bin/icpx", "NexaPoW SHA-49 validation must use the qualified Intel frontend");
    assert.deepEqual(commands[1]?.generatedInputs, [path.join(output, "kawpow-jit-resources.inc")]);
    assert.deepEqual(commands[2]?.generatedInputs, [path.join(output, "nvrtc-headers.inc")]);
    assert.equal(commands[4]?.command, path.join("/sdk", "bin/clang"));
    assert.deepEqual(commands[4]?.args.slice(0, 5), ["-x", "c", "-O3", "-ffp-contract=off", "-fPIC"]);
    assert.deepEqual(commands.map(command => path.basename(command.object)), ["0.o", "1.o", "2.o", "3.o", "4.o"]);
    for (const profile of ["dpcpp", "dpcpp-opencl"]) {
      const command = dpcppCompileCommands({sources: ["sycl/nexapow/validation.cpp"], repo, output, sdk: "/sdk", compiler: "/sdk/bin/clang++",
        flags: ["-fsycl"], wrapper: "", profile})[0];
      assert.equal(command?.command, "/sdk/bin/clang++", `${profile} must retain its selected DPC++ frontend`);
    }
  });

  test("oneAPI mixed frontend keys its bounded driver helper and configuration closure", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-gpu-oneapi-inputs-")), compilerRoot = path.join(root, "toolchain"),
      bin = path.join(compilerRoot, "bin");
    const helperDirectory = path.join(bin, "compiler"), compiler = path.join(bin, "icpx"), config = path.join(bin, "icpx.cfg");
    fs.mkdirSync(helperDirectory, {recursive: true}); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(compiler, "compiler fixture\n", {mode: 0o755}); fs.writeFileSync(config, "driver config\n");
    const names = ["clang", "clang-offload-bundler", "sycl-post-link", "llvm-spirv", "clang-offload-wrapper",
      "clang-linker-wrapper", "llvm-link", "llvm-foreach", "llc"];
    const helpers = Object.fromEntries(names.map(name => {
      const file = path.join(helperDirectory, name); fs.writeFileSync(file, "helper fixture\n", {mode: 0o755}); return [name, file];
    }));
    /** @type {Array<{command: string, args: string[]}>} */
    const calls = [];
    /** @type {NonNullable<Parameters<typeof oneapiCompilerInputs>[1]>} */
    const spawn = (command, args) => {
      calls.push({command, args});
      const stdout = (helpers[args[0]?.slice("-print-prog-name=".length) || ""] || "") + "\n";
      return /** @type {import("node:child_process").SpawnSyncReturns<string>} */ ({status: 0, signal: null, pid: 1,
        stdout, stderr: "", output: [null, stdout, ""]});
    };
    const admitted = {compiler, version: "Intel oneAPI\nConfiguration file: " + config + "\n"};
    const inputs = oneapiCompilerInputs(admitted, spawn);
    assert.deepEqual(inputs, [...names.map(name => fs.realpathSync(helpers[name] || "")), fs.realpathSync(config)]);
    assert.deepEqual(calls.map(call => call.args[0]), names.map(name => "-print-prog-name=" + name));
    assert.ok(calls.every(call => call.command === compiler));
    const outside = path.join(root, "outside", "llc"); fs.mkdirSync(path.dirname(outside), {recursive: true});
    fs.writeFileSync(outside, "external helper\n", {mode: 0o755});
    const escapedSpawn = /** @type {NonNullable<Parameters<typeof oneapiCompilerInputs>[1]>} */ ((command, args) => {
      const result = spawn(command, args, {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
      return args[0] === "-print-prog-name=llc" ? {...result, stdout: outside + "\n", output: [null, outside + "\n", ""]} : result;
    });
    assert.throws(() => oneapiCompilerInputs(admitted, escapedSpawn), /not a regular compiler-root file/);
  });

  test("Windows cache plan preserves host/device argv and generated header mappings", () => {
    const output = path.join(os.tmpdir(), "windows-candidate");
    const config = /** @type {Parameters<typeof windowsGpuCompileCommands>[0]["config"]} */ ({version: 1, profile: "acpp-cuda",
      tools: {clang: "/sdk/bin/clang++", clangC: "/sdk/bin/clang"},
      includeRoots: ["/sdk/include"], overlay: "/sdk/vfs.json", cuda: {sdk: "/sdk/cuda", nvrtcInclude: "/sdk/nvrtc", headers: "/sdk/nvrtc-headers.inc"},
      acpp: {source: "/acpp/source", generatedInclude: "/acpp/generated", plugin: "/acpp/plugin"}});
    assert.ok(nativeIncludeRoots(windowsGpuHostFlags(config, output), repo, output).includes(repo));
    const files = ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp", "sycl/kawpow/kawpow.cpp", "sycl/hoohash/host-math.c",
      "xmrig/base/crypto/keccak.cpp"];
    const commands = windowsGpuCompileCommands({sourceFiles: files, deviceCount: 3, repo, output, config,
      gpuFlags: ["--target=x86_64-pc-windows-msvc", "-O3", "-DNDEBUG", "-DDEVICE"],
      hostFlags: ["--target=x86_64-pc-windows-msvc", "-O3", "-DNDEBUG", "-DHOST"], perSourceFlags: [["-DCN"], ["-DPEARL"], []]});
    assert.equal(commands.length, files.length);
    assert.deepEqual(commands.map(command => path.relative(repo, command.source).split(path.sep).join("/")), files);
    assert.deepEqual(commands[0]?.args.slice(0, 6), ["--target=x86_64-pc-windows-msvc", "-O3", "-DNDEBUG", "-DDEVICE", "-DCN", "-MD"]);
    assert.deepEqual(commands[3]?.args.slice(0, 4), ["--target=x86_64-pc-windows-msvc", "-mno-incremental-linker-compatible", "-O3", "-DNDEBUG"]);
    assert.deepEqual(commands[0]?.generatedInputs, [path.join(output, "sycl/sycl.hpp")]);
    assert.deepEqual(commands[1]?.generatedInputs, [path.join(output, "sycl/sycl.hpp"), path.join(output, "nvrtc-headers.inc")]);
    assert.deepEqual(commands[3]?.generatedInputs, []);
    assert.deepEqual(commands[4]?.generatedInputs, []);
    for (const [index, command] of commands.entries()) {
      assert.equal(command.dependencyFormat, "nmake");
      assert.deepEqual(command.args.slice(-8), ["-MD", "-MV", "-MF", path.join(output, `${index}.d`), "-c", path.join(repo, files[index] || ""),
        "-o", path.join(output, `${index}.obj`)]);
    }
    const dpcppConfig = /** @type {Parameters<typeof windowsGpuCompileCommands>[0]["config"]} */ ({version: config.version, profile: "dpcpp",
      tools: config.tools, includeRoots: config.includeRoots, overlay: config.overlay, cuda: config.cuda});
    const dpcppCommands = windowsGpuCompileCommands({sourceFiles: files, deviceCount: 3, repo, output, config: dpcppConfig,
      gpuFlags: ["-fsycl", "-DMOM_EMBEDDED_SYCL_JIT"], hostFlags: ["-DHOST"], perSourceFlags: [[], [], []]});
    assert.deepEqual(dpcppCommands[1]?.generatedInputs, [path.join(output, "nvrtc-headers.inc")]);
    assert.deepEqual(dpcppCommands[2]?.generatedInputs, [path.join(output, "kawpow-jit-resources.inc")]);
    assert.deepEqual(dpcppCommands[4]?.generatedInputs, []);
    assert.deepEqual(dpcppCommands[3]?.args.slice(-12), ["-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-builtin",
      "-MD", "-MV", "-MF", path.join(output, "3.d"), "-c", path.join(repo, files[3] || ""), "-o", path.join(output, "3.obj")]);
  });

  test("include-root membership retains bounded repository root and excludes output/source subtrees", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-gpu-includes-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const localRepo = path.join(root, "repo"), output = path.join(root, "candidate"), sdk = path.join(root, "sdk"), external = path.join(root, "external");
    for (const directory of [localRepo, output, path.join(sdk, "include"), external]) {fs.mkdirSync(directory, {recursive: true});}
    const roots = nativeIncludeRoots(["-I", external, "-isystem", path.join(sdk, "include"), "--cuda-path=" + sdk,
      "-I", output, "-I", localRepo], localRepo, output);
    assert.deepEqual(roots, [external, path.join(sdk, "include"), localRepo]);
    assert.ok(!roots.includes(path.join(localRepo, "native")));
  });

  test("all object batches precede the fresh link or isolation step and finalize complete live graphs", () => {
    for (const [file, after] of /** @type {Array<[string, string]>} */ ([
      ["scripts/native-gpu-build.js", "isolateGpuObjects(inputs.cxx"],
      ["scripts/build-native-gpu.js", "const linked = run(compiler"],
      ["scripts/build-native-windows-gpu.js", "if (!acpp && run(config.tools.clang, deviceLink"],
    ])) {
      const source = fs.readFileSync(path.join(repo, file), "utf8");
      assert.ok(source.indexOf("compileNativeObjects(") >= 0 && source.indexOf("compileNativeObjects(") < source.indexOf(after), file);
      assert.match(source, /finalize: true,[\s\S]*?liveObjects:/, file);
      assert.match(source, /keyFiles:\s*compileKeyFiles/, file);
      assert.doesNotMatch(source, /cudaImageCompileCommand|imageCommand/, "the factory must not own a CUDA image registry object");
    }
    const windows = fs.readFileSync(path.join(repo, "scripts/build-native-windows-gpu.js"), "utf8");
    assert.ok(windows.indexOf("factoryBatch = compileNativeObjects(") < windows.indexOf("const isolated = isolateCoffObjects("));
    assert.match(windows, /liveObjects: \[\.\.\.compileCommands\.map\(item => item\.object\), factoryObject\]/);
    assert.equal((windows.match(/keyFiles:\s*compileKeyFiles/g) || []).length, 2);
    assert.match(windows, /\.\.\.Object.keys\(imageInputs\)/, "CUDA bundle and sidecar remain admitted SDK inputs");
    assert.match(windows, /PATH: \[cudaBin, \.\.\.\(env\["PATH"\]/);
    assert.match(windows, /run\(config\.tools\.clang, deviceLink, "device-link", 900000, deviceLinkEnvironment\)/);
    assert.match(fs.readFileSync(path.join(repo, "scripts/build-native-gpu.js"), "utf8"), /nvvm\/libdevice\/libdevice\.10\.bc/);
    assert.match(fs.readFileSync(path.join(repo, "scripts/build-native-gpu.js"), "utf8"), /oneapiCompilerInputs\(oneapi\)/);
  });

  test("CUDA image bundles remain hash-bound inputs and are not copied into per-factory objects", () => {
    for (const file of ["scripts/build-native-gpu.js", "scripts/native-gpu-build.js", "scripts/build-native-windows-gpu.js"]) {
      const source = fs.readFileSync(path.join(repo, file), "utf8");
      assert.match(source, /cudaImageInputs\(images\)\.files/, file);
      assert.match(source, /\.\.\.\(images \? \{cudaImages: images\} : \{\}\)/, file + " must attest image mode with the selected include path");
      assert.doesNotMatch(source, /cudaImageCompileCommand|imageCommand|cuda-images\.cpp|copyFileSync\(images,\s*path\.join\(output,\s*["']cuda-images\.inc/,
        file + " must leave payload emission to the shared application registry");
    }
  });

  test("native image mode removes NVRTC from the CUDA compile graph without altering SYCL source order", () => {
    const output = path.join(os.tmpdir(), "cuda-image-candidate"), files = ["sycl/cn-gpu/cn-gpu.cpp", "sycl/pearlhash/pearlhash.cpp"];
    const flags = gpuProfile("cuda", true).definitions;
    assert.ok(flags.includes("-DMOM_EMBEDDED_CUDA_IMAGES"));
    assert.ok(!flags.includes("-DMOM_STATIC_NVRTC"));
    assert.ok(gpuProfile("cuda").definitions.includes("-DMOM_STATIC_NVRTC"), "transitional compiler mode stays explicit");
    const commands = adaptiveCppCompileCommands(files, "/sdk/bin/clang++", repo, output, flags, true, "cuda", true);
    for (const command of commands) {
      assert.ok(command.args.includes("-DMOM_SYCL_HAS_CUDA"));
      assert.doesNotMatch(command.args.join(" "), /NVRTC/);
      assert.ok(command.args.includes("-acpp-sscp-hcf-object-id-namespace=acpp-cuda"));
      assert.deepEqual(command.generatedInputs, [path.join(output, "sycl/sycl.hpp")]);
    }
    const dpc = dpcppCompileCommands({sources: files, repo, output, sdk: "/sdk", compiler: "/sdk/bin/clang++",
      flags: ["-fsycl", "-DMOM_EMBEDDED_CUDA_IMAGES"], wrapper: "", profile: "dpcpp", cuda: "/cuda", images: "/cuda-images.inc"});
    assert.ok(dpc.every(command => command.generatedInputs.length === 0));
    assert.deepEqual(dpc.map(command => path.basename(command.object)), ["0.o", "1.o"]);
  });

  test("CUDA image selection is explicit and cannot leak to an Intel or portable profile", () => {
    const sdk = path.resolve("/sdk"), cuda = path.resolve("/cuda"), output = path.resolve("/new-output"), images = path.resolve("/images.inc");
    assert.deepEqual(buildInputs({MOM_AOT_CUDA_IMAGES: images}, [sdk, cuda, output]),
      {sdk, cuda, output, images, nvrtc: undefined, headers: undefined});
    assert.throws(() => buildInputs({MOM_AOT_CUDA_IMAGES: "relative"}, [sdk, cuda, output]), /absolute/);
    for (const profile of ["oneapi", "dpcpp-opencl"]) {
      assert.throws(() => buildInputs({MOM_AOT_DPCPP_PROFILE: profile, MOM_AOT_CUDA_IMAGES: images}, [sdk, output]), /only valid/);
    }
  });

  test("Windows image descriptors omit runtime compiler headers for both CUDA profiles", () => {
    const gyp = fs.readFileSync(path.join(repo, "binding.gyp"), "utf8");
    const wrapper = fs.readFileSync(path.join(repo, "scripts/cxx-combined.sh"), "utf8");
    for (const profile of ["dpcpp", "acpp-cuda"]) {
      /** @type {Parameters<typeof windowsGpuDeviceFlags>[0]} */
      const config = {version: 1, profile, tools: {clang: "/sdk/clang", clangC: "/sdk/clang-c", llvmNm: "/sdk/nm", llvmReadobj: "/sdk/readobj", llvmObjcopy: "/sdk/objcopy"},
        overlay: "/sdk/vfs.json", includeRoots: ["/sdk/include"], cuda: {sdk: "/cuda", images: "/images.inc"},
        ...(profile === "acpp-cuda" ? {acpp: {source: "/acpp", generatedInclude: "/generated", plugin: "/plugin"}} : {})};
      validateWindowsGpuConfig(config);
      const flags = windowsGpuDeviceFlags(config, [], gyp, wrapper);
      assert.ok(flags.includes("-DMOM_EMBEDDED_CUDA_IMAGES"));
      assert.doesNotMatch(flags.join(" "), /NVRTC/);
      assert.doesNotMatch(windowsGpuSourceFlags(config, "sycl/pearlhash/pearlhash.cpp", wrapper).join(" "), /NVRTC/);
      assert.throws(() => validateWindowsGpuConfig({...config, cuda: {...config.cuda, headers: "/old.inc"}}), /CUDA SDK/);
    }
  });

  test("AdaptiveCpp uses the explicit hipSYCL include root rather than an output-directory symlink", () => {
    const source = fs.readFileSync(path.join(repo, "scripts/native-gpu-build.js"), "utf8");
    assert.match(source, /"-isystem", path\.join\(inputs\.source, "include"\)/);
    assert.ok(source.includes("writeAdaptiveCppUmbrella(inputs.source, output);"), "Linux uses the tested shared forwarding header");
    assert.doesNotMatch(source, /fs\.symlinkSync/);
  });
});
