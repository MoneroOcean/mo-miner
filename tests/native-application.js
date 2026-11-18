"use strict";
const test = require("node:test"), assert = require("node:assert/strict"), path = require("node:path"), fs = require("node:fs"), os = require("node:os");
const {spawnSync} = require("node:child_process"), vm = require("node:vm");
const {controllerEntry, nativeApplicationGpu, nativeGpuLinkArgs, nativeApplicationLinkPlan, nativeApplicationInputs,
  nativeApplicationLinkCacheInputs, nativeApplicationBuildInputs} = require("../scripts/probe-native-controller");
const {fileHash, nativeSourceInputs} = require("../scripts/build-source-inputs");
const {admittedJitCacheTag} = require("../scripts/native-jit-cache-tag");

test.describe("Native default-application compilation boundary", () => {
  const repo = path.resolve(__dirname, ".."), script = path.join(repo, "scripts/probe-native-controller.js");
  const source = ['extern "C" const mom::runtime::GpuTable* mom_native_gpu_dpcpp();',
    'extern "C" const mom::runtime::GpuTable* mom_native_gpu_acpp_cuda();'].join("\n");
  const inputs = {defaultCompiler: "dpcpp", flags: [], links: []};
  test("application entry is only the unchanged maintained controller", () => {
    assert.equal(controllerEntry(repo, true), `require(${JSON.stringify(path.join(repo, "mom.js"))});\n`);
  });
  test("controller receipts freeze the dependency lock without invalidating CPU or GPU source receipts", () => {
    const state = nativeApplicationBuildInputs(repo);
    for (const name of ["package.json", "package-lock.json"]) {assert.equal(state.graph[name], fileHash(path.join(repo, name)));}
    for (const scope of /** @type {const} */ (["cpu", "gpu"])) {
      assert.equal(Object.hasOwn(nativeSourceInputs(repo, scope).graph, "package-lock.json"), false);
    }
  });
  test("application entry cannot select a probe through inherited environment flags", () => {
    /** @type {string[]} */
    const seen = [];
    vm.runInNewContext(controllerEntry(repo, true), {require: (/** @type {string} */ file) => seen.push(file),
      process: {env: {MOM_AOT_CHILD_TASK: "yes", MOM_AOT_FILE_FAULT: "yes", MOM_AOT_RUN_CONTROLLER: "0"}}});
    assert.deepEqual(seen, [path.join(repo, "mom.js")]);
    assert.doesNotMatch(controllerEntry(repo, true), /MOM_AOT_|tests\/reference|process\.env/);
  });
  test("Unicode, quotes and newlines in source paths remain literal data", () => {
    const root = path.join(repo, 'é-💎-"\n');
    /** @type {string[]} */
    const seen = [];
    vm.runInNewContext(controllerEntry(root, true), {require: (/** @type {string} */ file) => seen.push(file)});
    assert.deepEqual(seen, [path.join(root, "mom.js")]);
  });
  test("default probe mode retains its independent test routing", () => {
    assert.match(controllerEntry(repo), /MOM_AOT_CHILD_TASK/);
    assert.match(controllerEntry(repo), /native-controller\.js/);
  });
  test("factories and their compile definitions follow maintained declarations", () => {
    assert.deepEqual(nativeApplicationGpu(source, inputs), {factories: ["mom_native_gpu_dpcpp", "mom_native_gpu_acpp_cuda"],
      definitions: ['-DMOM_NATIVE_GPU_DEFAULT="dpcpp"', "-DMOM_NATIVE_GPU_DPCPP", "-DMOM_NATIVE_GPU_ACPP_CUDA"],
      linkFlags: ["-Wl,--wrap=dlopen", "-Wl,--wrap=dlerror", "-Wl,--undefined=mom_native_gpu_dpcpp", "-Wl,--undefined=mom_native_gpu_acpp_cuda"]});
  });
  test("the actual dispatch source yields a nonempty complete unique factory set", () => {
    const actual = fs.readFileSync(path.join(repo, "runtime/gpu-dispatch.cpp"), "utf8");
    const {factories, definitions} = nativeApplicationGpu(actual, {...inputs, defaultCompiler: "oneapi"});
    assert.equal(factories.length, (actual.match(/^extern "C" const mom::runtime::GpuTable/gm) || []).length);
    assert.equal(new Set(factories).size, factories.length);
    assert.equal(definitions.length, factories.length + 1);
  });
  test("a default must name a linked declared compiler", () => {
    for (const defaultCompiler of [undefined, null, "", "oneapi", "dpcpp\n", 'dpcpp"', "$(false)"]) {
      assert.throws(() => nativeApplicationGpu(source, {...inputs, defaultCompiler}), /explicit linked default/);
    }
  });
  test("an alternate declared default retains every factory", () => {
    const selected = nativeApplicationGpu(source, {...inputs, defaultCompiler: "acpp-cuda"});
    assert.equal(selected.factories.length, 2); assert.equal(selected.definitions[0], '-DMOM_NATIVE_GPU_DEFAULT="acpp-cuda"');
  });
  test("duplicate declarations cannot conceal a missing graph", () => {
    assert.throws(() => nativeApplicationGpu(source + "\n" + source, inputs), /factory declarations/);
  });
  test("empty or changed declaration syntax fails closed", () => {
    for (const text of ["", source.replace("GpuTable*", "GpuTable *")]) {
      assert.throws(() => nativeApplicationGpu(text, inputs), /factory declarations/);
    }
  });
  test("probe main wrappers are refused in either linker input channel", () => {
    for (const key of ["flags", "links"]) {assert.throws(() => nativeApplicationGpu(source, {...inputs, [key]: ["-Wl,--wrap=main"]}), /no main wrapper/);}
    assert.equal(nativeApplicationGpu(source, {...inputs, flags: ["-Wl,--wrap=dlopen"]}).factories.length, 2);
  });
  test("excessive factory sets are not accepted as a build plan", () => {
    const text = Array.from({length: 65}, (_, i) => `extern "C" const mom::runtime::GpuTable* mom_native_gpu_x${i}();`).join("\n");
    assert.throws(() => nativeApplicationGpu(text, {...inputs, defaultCompiler: "x0"}), /factory declarations/);
  });
  test("application mode requires build-only complete CPU/GPU/I/O inputs before invoking tools", {skip: process.platform !== "linux"}, () => {
    for (const missing of ["MOM_AOT_CPU_CORE", "MOM_AOT_GPU_INPUTS", "MOM_AOT_IO_SDK", "MOM_AOT_BUILD_ONLY"]) {
      const env = /** @type {NodeJS.ProcessEnv} */ ({...process.env, PATH: "/unavailable", MOM_AOT_CPU_CORE: "/missing/core",
        MOM_AOT_GPU_INPUTS: "/missing/gpu", MOM_AOT_IO_SDK: "/missing/io", MOM_AOT_BUILD_ONLY: "1"}); delete env[missing];
      const result = spawnSync(process.execPath, [script, "/missing/source", "/missing/build", "--app"], {encoding: "utf8", timeout: 5000, env});
      assert.equal(result.status, 1); assert.match(result.stderr, /Application build requires the full CPU\/GPU\/I\/O SDK inputs/);
      assert.doesNotMatch(result.stderr, /spawn|ENOENT/); assert.equal(result.stdout, "");
    }
  });
  test("unrecognized CLI flags fail before toolchain or output access", () => {
    const result = spawnSync(process.execPath, [script, "/missing/source", "/missing/build", "--unknown"],
      {encoding: "utf8", timeout: 5000, env: {...process.env, PATH: "/unavailable"}});
    assert.equal(result.status, 1); assert.match(result.stderr, /Usage/); assert.equal(result.stdout, "");
  });
  test("importing application-build helpers never performs a build", () => {
    const result = spawnSync(process.execPath, ["-e", `require(${JSON.stringify(script)})`],
      {encoding: "utf8", timeout: 5000, env: {...process.env, PATH: "/unavailable", MOM_AOT_CPU_CORE: "/missing", MOM_AOT_GPU_INPUTS: "/missing"}});
    assert.equal(result.status, 0); assert.equal(result.stdout, ""); assert.equal(result.stderr, "");
  });
  test("every Linux controller and fault-injection object is explicitly position independent", () => {
    const text = fs.readFileSync(script, "utf8");
    const commands = [...text.matchAll(/run\((?:cxx|process\.env\["CC"\] \|\| "cc"), \[([\s\S]*?)\],\s*path\.join\(output/g)]
      .map(match => match[1] || "").filter(args => args.includes('"-c"'));
    assert.equal(commands.length, 9, "inspect every maintained host compilation, including looped sources");
    assert.ok(commands.some(args => args.includes('path.join(repo, "sycl/cuda-images.cpp")')),
      "the embedded CUDA image table must be compiled into the application");
    for (const args of commands) {assert.match(args, /"-fPIC"/, args.slice(0, 120));}
  });
  test("Linux executable links explicitly retain ASLR with Intel and distro compilers", () => {
    const text = fs.readFileSync(script, "utf8");
    assert.ok(/const linkArgs = \["-pie",/.test(text), "the shared link arguments must request PIE");
    assert.equal(/"-no-pie"|"-fno-pic"/.test(text), false);
  });
  test("only the final application link gets an exact path-bound GNU/LLD map", () => {
    const output = path.join(repo, "out with,comma"), map = path.join(output, "controller-native.map"), args = ["-pie", "input.a"];
    assert.deepEqual(nativeApplicationLinkPlan(output, args, true), {args: ["-Xlinker", "-Map=" + map, ...args], map});
    assert.deepEqual(nativeApplicationLinkPlan(output, args, false), {args, map: null});
    const text = fs.readFileSync(script, "utf8");
    assert.match(text, /const linkPlan = nativeApplicationLinkPlan\(output, linkArgs, application\)/);
    assert.match(text, /linkMap: linkPlan\.map/);
    assert.match(text, /linkMapReceipt = \{path: linkPlan\.map, bytes: stat\.size, sha256: fileHash\(linkPlan\.map\)\}/);
  });
  test("the real static SYCL probe also compiles PIC and links PIE", () => {
    const text = fs.readFileSync(path.join(repo, "scripts/probe-static-sycl.js"), "utf8");
    const command = text.match(/run\(cxx, \[([\s\S]*?)\], path\.join\(output, "build\.log"\)\)/)?.[1] || "";
    assert.ok(command.includes('"-fPIC"'), "the probe must compile PIC objects");
    assert.ok(command.includes('"-pie"'), "the probe must link a PIE executable");
    assert.equal(/"-no-pie"|"-fno-pic"/.test(command), false);
  });
  test("static runtime libraries precede every image-bearing graph within the link group", () => {
    assert.deepEqual(nativeGpuLinkArgs({flags: [], libraries: ["runtime.a"], links: ["dependency.a"],
      objects: [{object: "image1.o"}, {object: "image2.o"}]}),
    ["-Wl,--start-group", "runtime.a", "dependency.a", "image1.o", "image2.o", "-Wl,--end-group"]);
  });
  test("link options preserve order without admitting compile-only flags", () => {
    assert.deepEqual(nativeGpuLinkArgs({flags: ["-O2", "--ld-path=/sdk/ld.lld", "-DMACRO", "-Wl,--gc-sections"],
      libraries: [], links: ["-Wl,--whole-archive", "runtime.a", "-Wl,--no-whole-archive"], objects: [{object: "image.o"}]}),
    ["--ld-path=/sdk/ld.lld", "-Wl,--gc-sections", "-Wl,--start-group", "-Wl,--whole-archive", "runtime.a",
      "-Wl,--no-whole-archive", "image.o", "-Wl,--end-group"]);
  });
  test("SDK initialization objects precede image registrations after their runtime libraries", () => {
    assert.deepEqual(nativeGpuLinkArgs({flags: [], libraries: ["runtime.a"], links: [],
      sdkObjects: [{object: "adapter.o"}], objects: [{object: "image.o"}]}),
    ["-Wl,--start-group", "runtime.a", "adapter.o", "image.o", "-Wl,--end-group"]);
  });
  test("driver-only loading and real factory retention do not depend on SDK-provided link flags", () => {
    for (const links of [[], ["-Wl,--wrap=dlopen"]]) {
      const selected = nativeApplicationGpu(source, {...inputs, links});
      assert.equal(selected.linkFlags.filter(flag => flag === "-Wl,--wrap=dlopen").length, 1);
      for (const factory of selected.factories) {assert.ok(selected.linkFlags.includes("-Wl,--undefined=" + factory));}
    }
  });
  test("application link admission treats -L operands as relocatable directories, not files", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-app-links-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const app = path.join(root, "app"), cpu = path.join(app, "core.a"), gpu = path.join(app, "gpu.o"), receipt = path.join(app, "gpu.json");
    fs.mkdirSync(app); fs.writeFileSync(cpu, "core"); fs.writeFileSync(gpu, "gpu");
    const core = {gpu: true, portable: true, objectSha256: fileHash(cpu), sourceState: nativeSourceInputs(repo, "cpu")};
    fs.writeFileSync(path.join(app, "inputs.json"), JSON.stringify(core));
    fs.writeFileSync(receipt, JSON.stringify({factory: "mom_native_gpu_dpcpp", object: gpu,
      objectSha256: fileHash(gpu), sourceState: nativeSourceInputs(repo, "gpu")}));
    const libA = path.join(root, "relocated-a", "sdk", "lib"), libB = path.join(root, "relocated-b", "sdk", "lib");
    fs.mkdirSync(libA, {recursive: true}); fs.mkdirSync(libB, {recursive: true});
    const archiveA = path.join(libA, "libLLVMCore.a"), archiveB = path.join(libB, "libLLVMCore.a");
    fs.writeFileSync(archiveA, "same archive"); fs.writeFileSync(archiveB, "same archive");
    const appInputs = (/** @type {string} */ archive) => ({objects: [{object: gpu}], receipts: [receipt], sdkObjects: [], libraries: [],
      links: ["-Wl,--start-group", "-L", path.dirname(archive), archive, "-Wl,--end-group"]});
    const tag = (/** @type {string} */ archive) => nativeApplicationInputs(repo, cpu, appInputs(archive), ["mom_native_gpu_dpcpp"]).jitCacheTag;
    assert.equal(tag(archiveA), tag(archiveB), "moving same-byte archives and their search directories preserves identity");
    fs.writeFileSync(archiveB, "changed archive");
    assert.notEqual(tag(archiveA), tag(archiveB), "changing an explicitly linked archive changes identity");
  });
  test("link cache planning preserves order and rejects malformed search directories", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-app-link-plan-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const lib = path.join(root, "lib"), archive = path.join(lib, "libcore.a"), file = path.join(root, "not-a-directory");
    fs.mkdirSync(lib); fs.writeFileSync(archive, "archive"); fs.writeFileSync(file, "file");
    const plan = nativeApplicationLinkCacheInputs(["-Wl,--start-group", "-L", lib, archive, "-Wl,--end-group"]);
    assert.deepEqual(plan, {files: [archive], linkPlan: ["-Wl,--start-group", "-L", "@link-search/0", "@link/0", "-Wl,--end-group"]});
    const joined = nativeApplicationLinkCacheInputs(["-Wl,--start-group", "-L" + lib, archive, "-Wl,--end-group"]);
    assert.deepEqual(joined, plan, "attached and separated -L forms normalize to the same ordered plan");
    assert.equal(admittedJitCacheTag(repo, "linux-x64", {link: joined.files}, joined.linkPlan),
      admittedJitCacheTag(repo, "linux-x64", {link: plan.files}, plan.linkPlan));
    for (const links of [["-L"], ["-L", "relative"], ["-L", "-pthread"], ["-L", file], ["-L" + path.join(root, "missing")], ["-Lrelative"]]) {
      assert.throws(() => nativeApplicationLinkCacheInputs(links), /Native application link search/);
    }
    const directoryAsFile = nativeApplicationLinkCacheInputs([lib]);
    assert.deepEqual(directoryAsFile.files, [lib]);
    assert.throws(() => admittedJitCacheTag(repo, "linux-x64", {link: directoryAsFile.files}, directoryAsFile.linkPlan), /not a regular file/);
  });
});
