"use strict";

const {test, describe} = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {fetchSdkSources, fetchArchiveCohort, fetchIoArchives, fetchBuildArchives, fetchWindowsOpenSslArchives, fetchNvrtcArchives, fetchWindowsHostSdkArchives,
    verifyBuildArchives, verifyWindowsOpenSslArchives, verifyIoArchives, verifyNvrtcArchives, verifyWindowsHostSdkArchives,
    nvrtcArchiveCatalog, nvrtcSourceRecipe, fetchZstdArchive, verifyZstdArchive, zstdArchiveCatalog, zstdSourceRecipe,
    ioArchiveUrls, buildArchiveUrls, origins, runGit, gitSourceComponents, sourceRecipe, buildSourceRecipe,
    windowsHostSdkCatalog, windowsHostSdkArchiveCatalog} = require("../scripts/fetch-native-sdk-sources"),
  {sdkComponents, validateSourceRoots} = require("../scripts/prepare-native-sdk-sources"), {ioPins} = require("../scripts/build-native-io"),
  {archiveName, archiveSha256} = require("../scripts/native-llvm-sdk"), {opensslPin, opensslBuildRecipe} = require("../scripts/build-native-openssl"),
  {hermesPins} = require("../scripts/build-native-hermes"), {zstdPin, zstdBuildRecipe} = require("../scripts/build-native-zstd");
/** @typedef {Pick<Response, "status" | "url" | "body"> & {headers: Pick<Headers, "get">}} TestArchiveResponse */
/** @typedef {(url: string, options: {redirect: "manual", signal: AbortSignal}) => Promise<TestArchiveResponse>} TestArchiveFetch */
/** @typedef {Record<"libuv" | "zlib", Buffer>} TestIoPayloads */
/** @typedef {Record<"libuv" | "zlib", {directory: string, sha256: string}>} TestIoPins */
/** @typedef {Record<"llvm" | "openssl", Buffer>} TestBuildPayloads */
/** @typedef {Record<"llvm" | "openssl", {sha256: string}>} TestBuildHashes */

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-sdk-fetch-")), git = path.join(root, "git");
  fs.writeFileSync(git, "#!/bin/sh\nprintf 'git version 2.45.0\\n'\n"); fs.chmodSync(git, 0o755);
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return {root, git, output: path.join(root, "fetched")};
}

/** @param {{mutateTool?: boolean, failFetch?: boolean, invalidOrigin?: boolean, promisor?: boolean, shallow?: boolean}} [options] */
function mockGit(options = {}) {
  /** @type {{git: string, args: string[], env: NodeJS.ProcessEnv, log: string, deadline: number}[]} */ const calls = [];
  /** @type {Map<string, {origin: string, revision?: string}>} */ const pins = new Map();
  /** @param {string} git @param {string[]} args @param {NodeJS.ProcessEnv} env @param {string} log @param {number} deadline @returns {Promise<void>} */
  async function run(git, args, env, log, deadline) {
    calls.push({git, args, env, log, deadline});
    if (options.mutateTool && calls.length === 1) {fs.writeFileSync(git, "mutated git tool\n");}
    if (args[0] === "init") {
      const root = args.at(-1) || ""; fs.mkdirSync(path.join(root, "objects", "pack"), {recursive: true});
      fs.writeFileSync(path.join(root, "config"), "[core]\n\tbare = true\n");
    } else if (args[0] === "-C" && args[2] === "remote") {
      const root = args[1] || "", origin = args.at(-1) || ""; pins.set(root, {origin});
      const selectedOrigin = options.invalidOrigin ? "https://github.com/attacker/fake.git" : origin;
      fs.appendFileSync(path.join(root, "config"), `[remote "origin"]\n\turl = ${selectedOrigin}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n${options.promisor ? "\tpromisor = true\n" : ""}`);
    } else if (args[0] === "-C" && args[2] === "fetch") {
      const root = args[1] || "", pin = args.at(-1) || "";
      if (options.failFetch) {throw new Error("mock fetch failure");}
      const selected = pins.get(root); assert.ok(selected); selected.revision = pin;
      if (options.shallow !== false) {fs.writeFileSync(path.join(root, "shallow"), pin + "\n");}
      if (options.promisor) {fs.writeFileSync(path.join(root, "objects", "pack", "test.promisor"), "promisor\n");}
    } else if (args[0] === "-C" && args[2] === "cat-file") {
      const root = args[1] || ""; assert.equal(args.at(-1), pins.get(root)?.revision + "^{commit}");
    } else if (args[0] === "-C" && args[2] === "update-ref") {
      const root = args[1] || ""; assert.equal(args[3], "refs/heads/mom-native-source"); assert.equal(args[4], pins.get(root)?.revision);
      fs.mkdirSync(path.join(root, "refs/heads"), {recursive: true}); fs.writeFileSync(path.join(root, "refs/heads/mom-native-source"), args[4] + "\n");
    } else if (args[0] === "-C" && args[2] === "symbolic-ref") {
      const root = args[1] || ""; assert.deepEqual(args.slice(3), ["HEAD", "refs/heads/mom-native-source"]);
      fs.writeFileSync(path.join(root, "HEAD"), "ref: refs/heads/mom-native-source\n");
    } else if (args[0] === "-C" && args[2] === "fsck") {
      assert.ok(pins.has(args[1] || ""));
    }
  }
  return {run, calls};
}

/** @param {string} url @param {number} status @param {Buffer | string} body @param {Record<string, string>} [headerValues]
 * @returns {TestArchiveResponse} */
function archiveResponse(url, status, body, headerValues = {}) {
  const bytes = Buffer.isBuffer(body) ? body : Buffer.from(body); let delivered = false;
  const stream = new ReadableStream({pull(controller) {
    if (delivered) {controller.close(); return;}
    delivered = true; if (bytes.length) {controller.enqueue(bytes);} controller.close();
  }});
  return {status, url, body: stream, headers: {get: (/** @type {string} */ name) => headerValues[name.toLowerCase()] ?? null}};
}

/** @param {Buffer} bytes @param {TestArchiveFetch} fetchImpl @param {{target?: string, fileName?: string,
 *   url?: string, origins?: string[], maxBytes?: number}} [options]
 * @returns {Parameters<typeof fetchArchiveCohort>[3]} */
function pinnedArchiveFixture(bytes, fetchImpl, options = {}) {
  const pins = {payload: {sha256: createHash("sha256").update(bytes).digest("hex")}};
  return {target: options.target || "windows-sdk-supplemental", sourceLabel: "Windows SDK supplemental",
    archiveLabel: "Windows SDK payload", specs: {payload: {fileName: options.fileName || "fixture.vsix",
      url: options.url || "https://download.visualstudio.microsoft.com/package/fixture.vsix",
      origins: options.origins || ["https://download.visualstudio.microsoft.com"]}},
    pins, maxBytes: options.maxBytes ?? 1024, timeoutMs: 1000,
    recipeFor: () => ({files: {}, graph: {"scripts/windows-sdk-payload-lock.js": "a".repeat(64)}}),
    pinSnapshot: () => JSON.stringify(pins), fetchImpl};
}

/** @param {TestIoPayloads} payloads @returns {TestIoPins} */
function testIoPins(payloads) {
  return {libuv: {...ioPins.libuv, sha256: createHash("sha256").update(payloads.libuv).digest("hex")},
    zlib: {...ioPins.zlib, sha256: createHash("sha256").update(payloads.zlib).digest("hex")}};
}

/** @param {TestIoPayloads} payloads @param {{calls?: Array<{url: string, redirect: string}>}} [state] @returns {TestArchiveFetch} */
function mockArchiveFetch(payloads, state = {}) {
  const calls = state.calls || (state.calls = []);
  return async (/** @type {string} */ url, /** @type {{redirect: "manual", signal: AbortSignal}} */ options) => {
    calls.push({url, redirect: options.redirect});
    const name = url.includes("dist.libuv.org") ? "libuv" : url === ioArchiveUrls.zlib ? "zlib" : undefined;
    if (!name) {throw new Error("Unexpected fake archive URL: " + url);}
    const body = payloads[name];
    if (!body) {throw new Error("Missing fake archive payload: " + name);}
    return archiveResponse(url, 200, body, {"content-length": String(body.length)});
  };
}

/** @param {TestBuildPayloads} payloads @returns {TestBuildHashes} */
function testBuildHashes(payloads) {
  return {llvm: {sha256: createHash("sha256").update(payloads.llvm).digest("hex")},
    openssl: {sha256: createHash("sha256").update(payloads.openssl).digest("hex")}};
}

/** @param {TestBuildPayloads} payloads @param {{calls?: Array<{url: string, redirect: string}>}} [state] @returns {TestArchiveFetch} */
function mockBuildArchiveFetch(payloads, state = {}) {
  const calls = state.calls || (state.calls = []), assets = {
    llvm: "https://release-assets.githubusercontent.com/native-sdk/llvm-project-21.1.8.src.tar.xz",
    openssl: "https://release-assets.githubusercontent.com/native-sdk/openssl-3.5.9.tar.gz",
  };
  return async (/** @type {string} */ url, /** @type {{redirect: "manual", signal: AbortSignal}} */ options) => {
    calls.push({url, redirect: options.redirect});
    const name = /** @type {Array<"llvm" | "openssl">} */ (["llvm", "openssl"]).find(key => url === buildArchiveUrls[key] || url === assets[key]);
    if (!name) {throw new Error("Unexpected fake build archive URL: " + url);}
    if (url === buildArchiveUrls[name]) {return archiveResponse(url, 302, "", {location: assets[name]});}
    const body = payloads[name];
    return archiveResponse(url, 200, body, {"content-length": String(body.length)});
  };
}

describe("native SDK source acquisition", () => {
  test("ordinary fetcher import does not load the ACPP, LLVM JIT, or release dependency graph", {skip: process.platform !== "linux"}, () => {
    const file = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js"), source = `
      const Module = require("node:module"), path = require("node:path"), original = Module._load;
      const forbidden = new Set(["probe-static-sycl", "probe-static-nvrtc", "build-native-sycl-sdk", "build-native-release", "native-release-sdk-inputs"]), loaded = [];
      Module._load = function(request, parent, isMain) {
        const name = path.basename(String(request)).replace(/\\.js$/, "");
        if (forbidden.has(name)) {loaded.push(name);}
        return original.apply(this, arguments);
      };
      require(${JSON.stringify(file)});
      process.stdout.write(JSON.stringify(loaded));
    `;
    const result = spawnSync(process.execPath, ["-e", source], {cwd: path.resolve(__dirname, ".."), encoding: "utf8", env: {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}});
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "[]");
  });

  test("fetch creates roots.json accepted by the offline runtime preparer", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), callerEnv = {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1", GIT_DIR: "/wrong", GIT_CONFIG_COUNT: "1"};
    const result = await fetchSdkSources("linux", f.git, f.output, callerEnv, true, {run: mock.run});
    const pins = sdkComponents("linux"), roots = /** @type {Record<string, string>} */ (JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8")));
    assert.deepEqual(Object.keys(roots).sort(), Object.keys(pins).sort());
    assert.deepEqual(Object.keys(result.components).sort(), Object.keys(pins).sort());
    for (const [name, component] of Object.entries(pins)) {
      const selected = result.components[name], root = roots[name]; assert.ok(selected); assert.ok(root);
      assert.equal(selected.revision, component.revision); assert.equal(selected.origin, origins[/** @type {keyof typeof origins} */ (name)]);
      assert.equal(selected.root, root); assert.ok(fs.statSync(root).isDirectory());
    }
    assert.equal(result.status, "PASS"); assert.equal(result.compiled, false); assert.match(result.recipe.sha256, /^[0-9a-f]{64}$/);
    assert.equal(result.recipe.files["scripts/prepare-native-sdk-sources.js"], sourceRecipe("linux").graph["scripts/prepare-native-sdk-sources.js"]);
    assert.equal(callerEnv.GIT_DIR, "/wrong");
    for (const call of mock.calls) {
      assert.equal(call.env["GIT_NO_LAZY_FETCH"], "1"); assert.equal(call.env["GIT_CONFIG_NOSYSTEM"], "1");
      assert.equal(call.env["GIT_CONFIG_GLOBAL"], "/dev/null"); assert.equal(call.env["GIT_DIR"], undefined);
      assert.ok(call.deadline > Date.now()); assert.equal(call.git, fs.realpathSync(f.git));
    }
    assert.equal(mock.calls.filter(call => call.args[0] === "init").length, 8);
    assert.equal(mock.calls.filter(call => call.args[0] === "-C" && call.args[2] === "fetch").length, 8);
    assert.ok(mock.calls.every(call => !call.args.includes("checkout") && !call.args.some(arg => arg.startsWith("--filter"))));
    assert.ok(mock.calls.filter(call => call.args[0] === "init").every(call => call.args.some(arg => arg.startsWith("--template="))));
    assert.ok(mock.calls.filter(call => call.args[0] === "-C" && call.args[2] === "fetch").every(call =>
      call.args.includes("--depth=1") && call.args.includes("--no-tags") && call.args.includes("--no-recurse-submodules")));
  });

  test("Windows runtime source cohort is the same pinned eight-root contract", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), result = await fetchSdkSources("windows", f.git, f.output,
      {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mock.run});
    const expected = sdkComponents("windows"), roots = /** @type {Record<string, string>} */ (JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8")));
    assert.equal(Object.keys(roots).length, 8); assert.deepEqual(Object.keys(result.components).sort(), Object.keys(expected).sort());
    assert.doesNotThrow(() => validateSourceRoots(roots, path.join(f.root, "prepared"), expected));
  });

  test("compiler source roots include the two additional pinned repositories", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), result = await fetchSdkSources("compiler", f.git, f.output,
      {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mock.run});
    const components = sdkComponents("compiler"), roots = /** @type {Record<string, string>} */ (JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8")));
    const vc = result.components["vcIntrinsics"], spirv = result.components["spirvHeaders"];
    assert.ok(vc); assert.ok(spirv); assert.equal(Object.keys(roots).length, 10); assert.equal(roots["vcIntrinsics"], vc.root);
    assert.equal(roots["spirvHeaders"], spirv.root);
    assert.doesNotThrow(() => validateSourceRoots(roots, path.join(f.root, "prepared"), components));
  });

  test("ROCm/COMGR fetch uses four exact producer-pinned official Git roots", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), selected = gitSourceComponents("rocm"),
      result = await fetchSdkSources("rocm", f.git, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mock.run}),
      roots = /** @type {Record<string, string>} */ (JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8"))),
      recipe = sourceRecipe("rocm");
    assert.deepEqual(selected, {
      rocmLlvm: {revision: "27682a16360e33e37c4f3cc6adf9a620733f8fe1", patches: ["rocm-comgr-llvm21.patch", "rocm-comgr-resources.patch"]},
      clr: {revision: "5b591b2ff64aac4c1c08be879b0d5f89fd713d50", patches: []},
      hip: {revision: "993b610351d78993091c0057fcf67636169b7467", patches: []},
      rocr: {revision: "4e4d7abeb786e903ac2a46815ae2565de331cbd4", patches: []},
    });
    assert.deepEqual({rocmLlvm: origins.rocmLlvm, clr: origins.clr, hip: origins.hip, rocr: origins.rocr}, {
      rocmLlvm: "https://github.com/ROCm/llvm-project.git", clr: "https://github.com/ROCm/clr.git",
      hip: "https://github.com/ROCm/hip.git", rocr: "https://github.com/ROCm/ROCR-Runtime.git",
    });
    assert.deepEqual(Object.keys(roots).sort(), Object.keys(selected).sort());
    assert.deepEqual(Object.keys(result.components).sort(), Object.keys(selected).sort());
    for (const [name, component] of Object.entries(selected)) {
      const fetched = result.components[name], root = roots[name], origin = origins[/** @type {keyof typeof origins} */ (name)];
      assert.ok(fetched); assert.ok(root); assert.ok(origin);
      assert.equal(fetched.revision, component.revision); assert.equal(fetched.origin, origin);
      assert.equal(fetched.root, root); assert.equal(fs.realpathSync(root), root);
      assert.equal(fetched.gitDirectory, "bare"); assert.equal(fetched.shallow, true); assert.equal(fetched.partialClone, false);
    }
    for (const file of ["scripts/build-native-comgr-sdk.js", "scripts/build-native-rocm-host-runtime.js", "scripts/build-native-rocm-resources.js",
      "scripts/native-comgr-inputs.js", "runtime/rocm-comgr-llvm21.patch", "runtime/rocm-comgr-resources.patch"]) {
      assert.equal(result.recipe.files[file], recipe.graph[file], "the fetch receipt binds the exact consumer/pin recipe and patches");
    }
    assert.equal(result.status, "PASS"); assert.equal(result.networked, true); assert.equal(result.compiled, false);
    assert.equal(result.publicationQualified, false);
    assert.deepEqual(mock.calls.filter(call => call.args[0] === "-C" && call.args[2] === "fetch").map(call => call.args.at(-1)).sort(),
      Object.values(selected).map(component => component.revision).sort());
    assert.ok(mock.calls.every(call => !call.args.includes("checkout") && !call.args.includes("apply")));
  });

  test("unknown target fails before creating output", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t);
    await assert.rejects(fetchSdkSources(/** @type {"linux" | "windows" | "compiler"} */ ("unknown"), f.git, f.output,
      {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mockGit().run}), /SDK source target/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("relative and control-character paths are refused before output", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), env = {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"};
    await assert.rejects(fetchSdkSources("linux", "relative-git", f.output, env, true, {run: mockGit().run}), /absolute executable Git/);
    await assert.rejects(fetchSdkSources("linux", f.git, "relative-output", env, true, {run: mockGit().run}), /absolute executable Git/);
    await assert.rejects(fetchSdkSources("linux", f.git + "\n", f.output, env, true, {run: mockGit().run}), /absolute executable Git/);
    await assert.rejects(fetchSdkSources("linux", f.git, f.output + "\0", env, true, {run: mockGit().run}), /absolute executable Git/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("an existing symlink output is refused without touching its target", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), target = path.join(f.root, "preserved"); fs.mkdirSync(target); fs.writeFileSync(path.join(target, "keep"), "safe");
    fs.symlinkSync(target, f.output, "dir");
    await assert.rejects(fetchSdkSources("linux", f.git, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mockGit().run}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(target, "keep"), "utf8"), "safe");
  });

  test("network permission, build-only mode, and a new output are mandatory", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), env = {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"};
    await assert.rejects(fetchSdkSources("linux", f.git, f.output, env, false, {run: mockGit().run}), /explicit --allow-network/);
    await assert.rejects(fetchSdkSources("linux", f.git, f.output, {...env, MOM_AOT_BUILD_ONLY: "0"}, true, {run: mockGit().run}), /MOM_AOT_BUILD_ONLY=1/);
    fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "unchanged");
    await assert.rejects(fetchSdkSources("linux", f.git, f.output, env, true, {run: mockGit().run}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "unchanged");
  });

  test("CLI without --allow-network refuses before invoking Git", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    const result = spawnSync(process.execPath, [script, "linux", f.git, f.output], {encoding: "utf8", env: {...process.env, MOM_AOT_BUILD_ONLY: "1"}});
    assert.notEqual(result.status, 0); assert.match(result.stderr, /Usage: fetch-native-sdk-sources/); assert.equal(fs.existsSync(f.output), false);
  });

  test("invalid remotes, promisor repositories, and missing shallow state are rejected", {skip: process.platform !== "linux"}, async t => {
    for (const options of [{invalidOrigin: true}, {promisor: true}, {shallow: false}]) {
      const f = fixture(t), mock = mockGit(options);
      await assert.rejects(fetchSdkSources("linux", f.git, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mock.run}), /Fetched SDK Git root/);
      assert.equal(fs.existsSync(path.join(f.output, "roots.json")), false);
    }
  });

  test("source-preparer recipe drift cannot emit roots.json", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(); let recipeReads = 0;
    const driftingRecipe = (/** @type {"linux" | "windows" | "compiler" | "hermes" | "acpp" | "rocm"} */ target) => {
      const current = sourceRecipe(target); recipeReads++;
      return recipeReads === 1 ? current : {...current, graph: {...current.graph, drift: "0".repeat(64)}};
    };
    await assert.rejects(fetchSdkSources("linux", f.git, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true,
      {run: mock.run, sourceRecipe: driftingRecipe}), /recipe changed/);
    assert.equal(fs.existsSync(path.join(f.output, "roots.json")), false);
    assert.equal(recipeReads, 2);
  });

  test("a failed fetch or mutated Git tool never writes a usable roots manifest", {skip: process.platform !== "linux"}, async t => {
    for (const options of [{failFetch: true}, {mutateTool: true}]) {
      const f = fixture(t), mock = mockGit(options);
      await assert.rejects(fetchSdkSources("linux", f.git, f.output, {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"}, true, {run: mock.run}));
      assert.equal(fs.existsSync(path.join(f.output, "roots.json")), false);
    }
  });

  test("runGit propagates a fake child failure and keeps output in its disk log", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), log = path.join(f.root, "child.log");
    await assert.rejects(runGit(process.execPath, ["-e", "process.stderr.write('fake git failure');process.exit(9)"], process.env, log, Date.now() + 5000), /git failed \(9\)/);
    assert.match(fs.readFileSync(log, "utf8"), /fake git failure/);
  });

  test("runGit kills a timed-out process group even when its leader exits on TERM", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), pidFile = path.join(f.root, "descendant.pid"), log = path.join(f.root, "child.log");
    const descendant = 'process.on("SIGTERM",()=>{});setInterval(()=>{},1000);';
    const leader = `const{spawn}=require("node:child_process"),fs=require("node:fs");const c=spawn(process.execPath,["-e",${JSON.stringify(descendant)}],{stdio:"ignore"});fs.writeFileSync(${JSON.stringify(pidFile)},String(c.pid));process.on("SIGTERM",()=>process.exit(0));setInterval(()=>{},1000);`;
    const pending = runGit(process.execPath, ["-e", leader], process.env, log, Date.now() + 1000);
    for (let attempt = 0; attempt < 50 && !fs.existsSync(pidFile); attempt++) {await new Promise(resolve => setTimeout(resolve, 20));}
    assert.ok(fs.existsSync(pidFile), "fake child launched its TERM-ignoring descendant");
    const pid = Number(fs.readFileSync(pidFile, "utf8")); assert.ok(Number.isSafeInteger(pid) && pid > 0);
    await assert.rejects(pending, /10 minute SDK source component deadline exceeded/);
    let state;
    for (let attempt = 0; attempt < 50; attempt++) {
      try {const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8"); state = stat.slice(stat.lastIndexOf(")") + 2).split(" ")[0];} catch {state = undefined; break;}
      if (state === "Z") {break;}
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    assert.ok(state === undefined || state === "Z", `descendant process state after timeout: ${state}`);
  });
});

describe("native NVRTC vendor-input acquisition", {skip: process.platform !== "linux"}, () => {
  const env = () => ({MOM_AOT_BUILD_ONLY: "1"});
  const {nvrtcPins, nvrtcCohortInfo} = require("../scripts/probe-static-nvrtc");
  const payloads = () => ({nvrtc: Buffer.from("fixture NVRTC vendor input"), nvptxcompiler: Buffer.from("fixture PTX compiler vendor input")});
  /** @param {ReturnType<typeof payloads>} data */
  const hashesFor = data => Object.fromEntries(Object.entries(data).map(([name, data]) => [name, {sha256: createHash("sha256").update(data).digest("hex")}]));
  /** @param {ReturnType<typeof payloads>} data @returns {TestArchiveFetch} */
  const fetchFor = data => async url => {
    const body = url.includes("/cuda_nvrtc/") ? data.nvrtc : data.nvptxcompiler;
    return archiveResponse(url, 200, body, {"content-length": String(body.length)});
  };
  const platforms = /** @type {const} */ (["linux", "windows"]);

  test("pins the currently qualified OS-specific vendor versions, not a silent upgrade", () => {
    assert.equal(nvrtcPins.linux.version, "13.3.33"); assert.equal(nvrtcPins.windows.version, "13.1.115");
    assert.match(nvrtcPins.linux.manifest, /\/redistrib_13\.3\.0\.json$/);
    assert.match(nvrtcPins.windows.manifest, /\/redistrib_13\.1\.1\.json$/);
    assert.equal(nvrtcPins.linux.archives.nvrtc.sha256, "9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a");
    assert.equal(nvrtcPins.windows.archives.nvrtc.sha256, "4ddd5a1e34fd62bb41e78c0725edfbf5609f4ecedd7fe118eddf2148b097fc91");
    assert.equal(nvrtcPins["linux-12.6"].version, "12.6.85");
    assert.match(nvrtcPins["linux-12.6"].manifest, /\/redistrib_12\.6\.3\.json$/);
    assert.equal(nvrtcPins["linux-12.6"].archives.nvrtc.sha256, "a4289ba86f0deeccf4e1d94c58559eae4c8184fcb6eaade8fcab45584e6cf402");
    assert.ok(Object.isFrozen(nvrtcPins["linux-12.6"]) && Object.isFrozen(nvrtcPins["linux-12.6"].archives.nvrtc));
    assert.ok(Object.isFrozen(nvrtcPins) && Object.isFrozen(nvrtcPins.windows.archives.nvrtc));
    for (const [platform, installer] of [["linux", "install-nvrtc-current.sh"], ["windows", "install-dev.ps1"]]) {
      const source = fs.readFileSync(path.join(__dirname, "../scripts/" + installer), "utf8");
      const pin = nvrtcPins[/** @type {"linux" | "windows"} */ (platform)];
      assert.ok(source.includes(pin.version) && source.includes(pin.archives.nvrtc.sha256), "Engineering/native NVRTC inputs drifted on " + platform);
    }
  });

  test("CUDA 12.6 is an explicit Linux-only one-archive compatibility cohort", async t => {
    const f = fixture(t), bytes = Buffer.from("fixture NVRTC 12.6 archive"), sha256 = createHash("sha256").update(bytes).digest("hex"),
      hashes = {nvrtc: {sha256}}, catalog = nvrtcArchiveCatalog("linux", hashes, "linux-12.6"); let calls = 0;
    assert.equal(catalog["nvrtc"]?.fileName, "cuda_nvrtc-linux-x86_64-12.6.85-archive.tar.xz");
    assert.equal(catalog["nvrtc"]?.sha256, sha256); assert.match(catalog["nvrtc"]?.url || "", /\/cuda_nvrtc\/linux-x86_64\/cuda_nvrtc-linux-x86_64-12\.6\.85-archive\.tar\.xz$/);
    const fetch = /** @type {TestArchiveFetch} */ (async url => {calls++; return archiveResponse(url, 200, bytes, {"content-length": String(bytes.length)});});
    const result = await fetchNvrtcArchives("linux", f.output, env(), true, {cohort: "linux-12.6", hashes, fetch});
    assert.equal(result.target, "nvrtc-linux-12.6"); assert.deepEqual(Object.keys(result.archives), ["nvrtc"]);
    assert.deepEqual(verifyNvrtcArchives("linux", f.output, {cohort: "linux-12.6", hashes}), result); assert.equal(calls, 1);
    assert.throws(() => nvrtcArchiveCatalog("windows", undefined, "linux-12.6"), /not available/);
    assert.throws(() => nvrtcArchiveCatalog("linux", undefined, /** @type {"linux"} */ ("")), /not available/);
    assert.throws(() => nvrtcCohortInfo(/** @type {"linux"} */ ("other")), /platform must be linux or windows/);
    assert.throws(() => nvrtcArchiveCatalog("linux", {nvrtc: {sha256}, nvptxcompiler: {sha256}}, "linux-12.6"), /exact pinned compiler archive cohort/);
    assert.equal(nvrtcArchiveCatalog("linux")["nvrtc"]?.fileName, "cuda_nvrtc-linux-x86_64-13.3.33-archive.tar.xz");
  });

  for (const platform of platforms) {
    test(platform + " fetch and offline verification reuse the original bounded archive contract", async t => {
      const f = fixture(t), data = payloads(), hashes = hashesFor(data), catalog = nvrtcArchiveCatalog(platform, hashes);
      const result = await fetchNvrtcArchives(platform, f.output, env(), true, {fetch: fetchFor(data), hashes});
      assert.equal(result.target, "nvrtc-" + platform); assert.equal(result.compiled, false); assert.equal(result.publicationQualified, false);
      assert.deepEqual(Object.keys(result.archives).sort(), ["nvptxcompiler", "nvrtc"]);
      for (const [name, row] of Object.entries(result.archives)) {
        assert.equal(row.url, catalog[name]?.url); assert.equal(path.basename(row.path), catalog[name]?.fileName);
        assert.match(row.url, new RegExp("^https://developer\\.download\\.nvidia\\.com/compute/cuda/redist/[^/]+/" + platform + "-x86_64/"));
        assert.equal(row.sha256, hashes[name]?.sha256); assert.equal(fs.statSync(row.path).mode & 0o222, 0);
      }
      assert.deepEqual(verifyNvrtcArchives(platform, f.output, {hashes}), result);
      assert.deepEqual(Object.keys(result.recipe.files).sort(), ["scripts/build-source-inputs.js", "scripts/probe-static-nvrtc.js"]);
    });
  }

  test("unknown target, incomplete digests and extra compiler components fail before network", async t => {
    const f = fixture(t); let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 200, "unused");};
    await assert.rejects(fetchNvrtcArchives(/** @type {"linux"} */ ("other"), f.output, env(), true, {fetch}), /target must be linux or windows/);
    for (const hashes of [{nvrtc: {sha256: "0".repeat(64)}}, {...hashesFor(payloads()), injected: {sha256: "0".repeat(64)}},
      {...hashesFor(payloads()), nvrtc: {sha256: "x"}}]) {
      await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true, {fetch, hashes}), /exact pinned compiler archive cohort/);
    }
    assert.equal(calls, 0); assert.equal(fs.existsSync(f.output), false);
  });

  test("an explicitly malformed cohort never falls back to the platform default", async t => {
    const f = fixture(t), data = {nvrtc: Buffer.from("unused")}, hashes = {nvrtc: {sha256: createHash("sha256").update(data.nvrtc).digest("hex")}}; let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 200, data.nvrtc);};
    await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true,
      {cohort: /** @type {"linux"} */ (""), hashes, fetch}), /cohort is not available/);
    assert.equal(calls, 0); assert.equal(fs.existsSync(f.output), false);
  });

  test("network permission, build-only mode and production limit ceilings are enforced", async t => {
    const f = fixture(t), data = payloads(), hashes = hashesFor(data), fetch = fetchFor(data);
    await assert.rejects(fetchNvrtcArchives("windows", f.output, env(), false, {fetch, hashes}), /explicit --allow-network/);
    await assert.rejects(fetchNvrtcArchives("windows", f.output, {}, true, {fetch, hashes}), /MOM_AOT_BUILD_ONLY=1/);
    await assert.rejects(fetchNvrtcArchives("windows", f.output, env(), true, {fetch, hashes, maxBytes: (320 << 20) + 1}), /may only be narrowed/);
    await assert.rejects(fetchNvrtcArchives("windows", f.output, env(), true, {fetch, hashes, timeoutMs: 600001}), /may only be narrowed/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("no-clobber and canonical output checks protect pre-existing files", async t => {
    const f = fixture(t), data = payloads(), dependencies = {fetch: fetchFor(data), hashes: hashesFor(data)};
    await assert.rejects(fetchNvrtcArchives("linux", "relative-output", env(), true, dependencies), /new absolute directory/);
    fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "preserved");
    await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true, dependencies), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "preserved");
  });

  test("redirects cannot leave the official vendor origin or downgrade to HTTP", async t => {
    for (const location of ["https://attacker.example/sdk", "http://developer.download.nvidia.com/sdk"]) {
      const f = fixture(t); let calls = 0;
      /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 302, "", {location});};
      await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true, {fetch, hashes: hashesFor(payloads())}), /left its approved HTTPS origin/);
      assert.equal(calls, 1); assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("changed bytes remove the partial cohort, including an already promoted sibling", async t => {
    const f = fixture(t), data = payloads(); let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {
      calls++; return archiveResponse(url, 200, url.includes("/cuda_nvrtc/") ? "wrong bytes" : data.nvptxcompiler);
    };
    await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true, {fetch, hashes: hashesFor(data)}), /SHA256 differs/);
    assert.equal(calls, 2); assert.equal(fs.existsSync(f.output), false);
  });

  test("vendor archives remain bounded with and without Content-Length", async t => {
    for (const headers of [{"content-length": "999"}, {}]) {
      const f = fixture(t);
      /** @type {TestArchiveFetch} */ const fetch = async url => archiveResponse(url, 200, "too many bytes", headers);
      await assert.rejects(fetchNvrtcArchives("windows", f.output, env(), true, {fetch, hashes: hashesFor(payloads()), maxBytes: 3}), /configured archive bound/);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("an aborted download respects the narrowed deadline without a partial receipt", async t => {
    const f = fixture(t);
    /** @type {TestArchiveFetch} */ const fetch = async (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("aborted")), {once: true});
    });
    await assert.rejects(fetchNvrtcArchives("linux", f.output, env(), true, {fetch, hashes: hashesFor(payloads()), timeoutMs: 20}), /component deadline exceeded/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("offline admission rejects receipt/recipe changes and other-platform substitution", async t => {
    const data = payloads(), hashes = hashesFor(data);
    for (const fault of ["target", "fetcher", "recipe"]) {
      const f = fixture(t); await fetchNvrtcArchives("linux", f.output, env(), true, {fetch: fetchFor(data), hashes});
      const receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
      if (fault === "target") {receipt.target = "nvrtc-windows";}
      if (fault === "fetcher") {receipt.fetcherSha256 = "not-a-sha256";}
      if (fault === "recipe") {receipt.recipe.sha256 = "0".repeat(64);}
      fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, JSON.stringify(receipt)); fs.chmodSync(receiptPath, 0o444);
      assert.throws(() => verifyNvrtcArchives("linux", f.output, {hashes}), /invalid build-only|valid build-only|(?:stale|invalid) .* provenance/);
      assert.throws(() => verifyNvrtcArchives("windows", f.output, {hashes}), /only its receipt and pinned archive cohort/);
    }
    assert.ok(nvrtcSourceRecipe().graph["scripts/probe-static-nvrtc.js"]);
  });

  test("offline admission rejects writable, linked and unexpected archive inputs", async t => {
    for (const fault of ["writable", "linked", "extra"]) {
      const f = fixture(t), data = payloads(), hashes = hashesFor(data);
      const result = await fetchNvrtcArchives("windows", f.output, env(), true, {fetch: fetchFor(data), hashes});
      const archive = result.archives["nvrtc"]?.path; assert.ok(archive);
      if (fault === "writable") {fs.chmodSync(archive, 0o644);}
      if (fault === "linked") {fs.linkSync(archive, path.join(f.root, "second-link"));}
      if (fault === "extra") {fs.writeFileSync(path.join(f.output, "extra"), "not admitted");}
      assert.throws(() => verifyNvrtcArchives("windows", f.output, {hashes}), /read-only, single-link|only its receipt/);
    }
  });

  test("both CLI targets require network opt-in and their verifier refuses network flags", t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    for (const platform of platforms) {
      const fetch = spawnSync(process.execPath, [script, "nvrtc-" + platform, f.output], {encoding: "utf8", env: env()});
      assert.equal(fetch.status, 1); assert.match(fetch.stderr, /Usage: fetch-native-sdk-sources.js nvrtc-linux/);
      const verify = spawnSync(process.execPath, [script, "verify-nvrtc-" + platform, f.output, "--allow-network"], {encoding: "utf8", env: env()});
      assert.equal(verify.status, 1); assert.match(verify.stderr, /Usage: fetch-native-sdk-sources.js verify-nvrtc-linux/);
    }
    assert.equal(fs.existsSync(f.output), false);
  });
});

describe("native zstd source archive acquisition", {skip: process.platform !== "linux"}, () => {
  const env = () => ({MOM_AOT_BUILD_ONLY: "1"});
  const payload = Buffer.from("tiny synthetic zstd archive body; not an upstream archive\n");
  const sha256 = createHash("sha256").update(payload).digest("hex");
  const hashes = {zstd: {sha256}};
  const catalog = zstdArchiveCatalog(hashes);
  /** @param {{calls?: Array<{url: string, redirect: string}>}} [state] @returns {TestArchiveFetch} */
  const fetchFor = (state = {}) => {
    const calls = state.calls || (state.calls = /** @type {Array<{url: string, redirect: string}>} */ ([]));
    return async (/** @type {string} */ url, /** @type {{redirect: "manual", signal: AbortSignal}} */ options) => {
      calls.push({url, redirect: options.redirect});
      return archiveResponse(url, 200, payload, {"content-length": String(payload.length)});
    };
  };

  test("uses producer-owned zstd pin and recipe with the shared bounded archive path", async t => {
    const f = fixture(t), calls = /** @type {Array<{url: string, redirect: string}>} */ ([]),
      fetched = await fetchZstdArchive(f.output, env(), true, {fetch: fetchFor({calls}), hashes});
    const row = fetched.archives["zstd"];
    assert.equal(zstdPin.version, "1.5.7");
    assert.equal(zstdPin.sha256, "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3");
    assert.ok(Object.isFrozen(zstdPin));
    assert.equal(fetched.target, "zstd"); assert.equal(fetched.compiled, false); assert.equal(fetched.publicationQualified, false);
    assert.equal(row?.path, path.join(f.output, catalog["zstd"].fileName));
    assert.equal(row?.url, catalog["zstd"].url); assert.equal(row?.sha256, sha256); assert.equal(row?.bytes, payload.length);
    assert.equal(fs.statSync(row.path).mode & 0o222, 0);
    assert.deepEqual(calls, [{url: catalog["zstd"].url, redirect: "manual"}]);
    assert.deepEqual(verifyZstdArchive(f.output, {hashes}), fetched);
    assert.deepEqual(fetched.recipe.files, zstdBuildRecipe().graph);
    assert.deepEqual(zstdSourceRecipe(), zstdBuildRecipe());
  });

  test("zstd archive identity is fixed and test seams cannot add members or origins", async t => {
    const f = fixture(t); let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 200, payload);};
    assert.equal(zstdArchiveCatalog()["zstd"].url, "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz");
    for (const bad of [{}, {zstd: {sha256: "x"}}, {...hashes, injected: {sha256: "0".repeat(64)}}]) {
      await assert.rejects(fetchZstdArchive(f.output, env(), true, {fetch, hashes: bad}), /exact pinned archive cohort/);
    }
    assert.equal(calls, 0); assert.equal(fs.existsSync(f.output), false);
  });

  test("network opt-in, source bound, redirect allowlist, and no-clobber remain enforced", async t => {
    const f = fixture(t), successful = fetchFor();
    await assert.rejects(fetchZstdArchive(f.output, env(), false, {fetch: successful, hashes}), /explicit --allow-network/);
    await assert.rejects(fetchZstdArchive(f.output, {}, true, {fetch: successful, hashes}), /MOM_AOT_BUILD_ONLY=1/);
    await assert.rejects(fetchZstdArchive(f.output, env(), true, {fetch: successful, hashes, maxBytes: (16 << 20) + 1}), /may only be narrowed/);
    const occupied = path.join(f.root, "occupied"); fs.mkdirSync(occupied); fs.writeFileSync(path.join(occupied, "keep"), "preserved");
    await assert.rejects(fetchZstdArchive(occupied, env(), true, {fetch: successful, hashes}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(occupied, "keep"), "utf8"), "preserved");
    for (const location of ["https://attacker.example/zstd.tar.gz", "http://github.com/zstd.tar.gz"]) {
      const denied = fixture(t); let calls = 0;
      /** @type {TestArchiveFetch} */ const redirect = async url => {calls++; return archiveResponse(url, 302, "", {location});};
      await assert.rejects(fetchZstdArchive(denied.output, env(), true, {fetch: redirect, hashes}), /left its approved HTTPS origin/);
      assert.equal(calls, 1); assert.equal(fs.existsSync(denied.output), false);
    }
  });

  test("zstd CLI fetch requires explicit network opt-in and verification refuses network flags", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js"),
      fetch = spawnSync(process.execPath, [script, "zstd", f.output], {encoding: "utf8", env: env()}),
      verify = spawnSync(process.execPath, [script, "verify-zstd", f.output, "--allow-network"], {encoding: "utf8", env: env()});
    assert.notEqual(fetch.status, 0); assert.match(fetch.stderr, /Usage: fetch-native-sdk-sources\.js zstd .*--allow-network/);
    assert.notEqual(verify.status, 0); assert.match(verify.stderr, /Usage: fetch-native-sdk-sources\.js verify-zstd/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("changed archive bytes, stale provenance, and unexpected members fail offline admission", async t => {
    const f = fixture(t), wrong = Buffer.alloc(payload.length, 0x78);
    /** @type {TestArchiveFetch} */ const badHash = async url => archiveResponse(url, 200, wrong);
    await assert.rejects(fetchZstdArchive(f.output, env(), true, {fetch: badHash, hashes}), /SHA256 differs/);
    assert.equal(fs.existsSync(f.output), false);
    await fetchZstdArchive(f.output, env(), true, {fetch: fetchFor(), hashes});
    const archive = path.join(f.output, catalog["zstd"].fileName), receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8")),
      goodReceipt = JSON.stringify(receipt);
    receipt.recipe.sha256 = "0".repeat(64);
    fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, JSON.stringify(receipt)); fs.chmodSync(receiptPath, 0o444);
    assert.throws(() => verifyZstdArchive(f.output, {hashes}), /stale builder recipe provenance/);
    fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, goodReceipt); fs.chmodSync(receiptPath, 0o444);
    fs.writeFileSync(path.join(f.output, "extra.tar.gz"), "unexpected");
    assert.throws(() => verifyZstdArchive(f.output, {hashes}), /only its receipt and pinned archive cohort/);
    fs.unlinkSync(path.join(f.output, "extra.tar.gz"));
    fs.chmodSync(archive, 0o644); fs.writeFileSync(archive, wrong); fs.chmodSync(archive, 0o444);
    assert.throws(() => verifyZstdArchive(f.output, {hashes}), /digest differs from its pin/);
    fs.chmodSync(archive, 0o644); fs.writeFileSync(archive, payload); fs.chmodSync(archive, 0o444);
    fs.chmodSync(archive, 0o644);
    assert.throws(() => verifyZstdArchive(f.output, {hashes}), /read-only, single-link regular file/);
  });
});

describe("native controller source acquisition", () => {
  const env = () => ({PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "1"});

  test("AdaptiveCpp source fetch reuses the admission pin and records only clean original Git", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), callerEnv = {...env(), GIT_DIR: "/ambient/repo", GIT_CONFIG_COUNT: "1",
        GIT_CONFIG_KEY_0: "include.path", GIT_CONFIG_VALUE_0: "/ambient/config", TAR_OPTIONS: "--to-command=unsafe"},
      result = await fetchSdkSources("acpp", f.git, f.output, callerEnv, true, {run: mock.run}),
      {syclRevision, admitSyclSource} = require("../scripts/probe-static-sycl"),
      roots = JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8")), root = path.join(f.output, "roots/acpp"),
      recipe = sourceRecipe("acpp"), patchText = ["adaptivecpp-static.patch", "adaptivecpp-jit.patch"]
        .map(name => fs.readFileSync(path.resolve(__dirname, "../runtime", name), "utf8")).join("");
    assert.equal(origins["acpp"], "https://github.com/AdaptiveCpp/AdaptiveCpp.git");
    assert.deepEqual(Object.keys(result.components), ["acpp"]);
    assert.deepEqual(roots, {acpp: root});
    assert.deepEqual(result.components["acpp"], {origin: origins["acpp"], revision: syclRevision, root,
      gitDirectory: "bare", shallow: true, partialClone: false});
    assert.deepEqual(mock.calls.filter(call => call.args[0] === "-C" && call.args[2] === "fetch").map(call => call.args.at(-1)), [syclRevision]);
    assert.equal(fs.readFileSync(path.join(root, "refs/heads/mom-native-source"), "utf8").trim(), syclRevision);
    assert.equal(fs.readFileSync(path.join(root, "HEAD"), "utf8"), "ref: refs/heads/mom-native-source\n");
    assert.equal(fs.existsSync(path.join(root, "source")), false, "source ingress remains a bare Git root without a worktree or patches");
    assert.deepEqual(Object.keys(result.recipe.files).sort(), ["runtime/adaptivecpp-jit.patch", "runtime/adaptivecpp-static.patch",
      "scripts/build-environment.js", "scripts/build-native-sycl-sdk.js", "scripts/build-source-inputs.js", "scripts/native-audit.js",
      "scripts/native-hcf.js", "scripts/native-resources.js", "scripts/prepare-native-sdk-sources.js", "scripts/probe-native-jit.js", "scripts/probe-static-sycl.js"]);
    assert.equal(result.recipe.files["scripts/probe-static-sycl.js"], recipe.graph["scripts/probe-static-sycl.js"]);
    assert.equal(result.recipe.files["scripts/build-native-sycl-sdk.js"], recipe.graph["scripts/build-native-sycl-sdk.js"]);
    assert.equal(result.recipe.files["scripts/prepare-native-sdk-sources.js"], recipe.graph["scripts/prepare-native-sdk-sources.js"]);
    assert.ok(mock.calls.every(call => !call.args.includes("checkout") && !call.args.includes("apply")));
    assert.ok(mock.calls.every(call => call.env["GIT_DIR"] === undefined && call.env["TAR_OPTIONS"] === undefined && call.env["GIT_CONFIG_GLOBAL"] === "/dev/null"));
    assert.equal(callerEnv.GIT_DIR, "/ambient/repo"); assert.equal(callerEnv.GIT_CONFIG_COUNT, "1");
    assert.equal(callerEnv.GIT_CONFIG_VALUE_0, "/ambient/config"); assert.equal(callerEnv.TAR_OPTIONS, "--to-command=unsafe");
    const admitted = admitSyclSource("/source", "/output", true, (command, args) => {
      assert.equal(command, "git");
      if (args.includes("rev-parse")) {return syclRevision + "\n";}
      if (args.includes("diff")) {return patchText;}
      return "";
    });
    assert.equal(admitted.revision, syclRevision, "the fetch pin is the existing producer/consumer admission pin");
    assert.equal(result.compiled, false); assert.equal(result.publicationQualified, false);
  });

  test("AdaptiveCpp source fetch refuses missing opt-in, invalid origin, and existing output", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit();
    await assert.rejects(fetchSdkSources("acpp", f.git, f.output, env(), false, {run: mock.run}), /explicit --allow-network/);
    await assert.rejects(fetchSdkSources("acpp", f.git, f.output, {...env(), MOM_AOT_BUILD_ONLY: "0"}, true, {run: mock.run}), /MOM_AOT_BUILD_ONLY=1/);
    assert.equal(mock.calls.length, 0); assert.equal(fs.existsSync(f.output), false);
    const invalid = fixture(t), invalidMock = mockGit({invalidOrigin: true});
    await assert.rejects(fetchSdkSources("acpp", invalid.git, invalid.output, env(), true, {run: invalidMock.run}), /Fetched SDK Git root/);
    assert.equal(fs.existsSync(path.join(invalid.output, "fetch.json")), false);
    fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "safe");
    await assert.rejects(fetchSdkSources("acpp", f.git, f.output, env(), true, {run: mock.run}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "safe");
  });

  test("AdaptiveCpp CLI target enforces build-only and network opt-in before any Git fetch", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js"),
      result = spawnSync(process.execPath, [script, "acpp", f.git, f.output, "--allow-network"],
        {encoding: "utf8", env: {PATH: process.env["PATH"], MOM_AOT_BUILD_ONLY: "0"}});
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /MOM_AOT_BUILD_ONLY=1/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("Hermes fetch uses the producer's exact commit and original public repository", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), result = await fetchSdkSources("hermes", f.git, f.output, env(), true, {run: mock.run});
    const roots = JSON.parse(fs.readFileSync(path.join(f.output, "roots.json"), "utf8"));
    assert.deepEqual(Object.keys(result.components), ["hermes"]);
    assert.deepEqual(roots, {hermes: path.join(f.output, "roots/hermes")});
    assert.deepEqual(result.components["hermes"], {origin: "https://github.com/facebook/hermes.git", revision: hermesPins.revision,
      root: roots.hermes, gitDirectory: "bare", shallow: true, partialClone: false});
    assert.deepEqual(mock.calls.filter(call => call.args[2] === "fetch").map(call => call.args.at(-1)), [hermesPins.revision]);
    assert.equal(fs.readFileSync(path.join(roots.hermes, "refs/heads/mom-native-source"), "utf8").trim(), hermesPins.revision);
    assert.equal(fs.readFileSync(path.join(roots.hermes, "HEAD"), "utf8"), "ref: refs/heads/mom-native-source\n");
    assert.equal(result.compiled, false); assert.equal(result.publicationQualified, false);
  });

  test("Hermes acquisition binds both OS patch recipes without applying them", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(), result = await fetchSdkSources("hermes", f.git, f.output, env(), true, {run: mock.run});
    assert.deepEqual(Object.keys(result.recipe.files).sort(), ["runtime/hermes-clang-cl.patch", "runtime/hermes-host-rejections.patch",
      "scripts/build-environment.js", "scripts/build-native-hermes.js", "scripts/build-source-inputs.js"]);
    assert.ok(mock.calls.every(call => !call.args.includes("checkout") && !call.args.includes("apply")));
    assert.equal(result.recipe.files["scripts/build-native-hermes.js"], sourceRecipe("hermes").graph["scripts/build-native-hermes.js"]);
  });

  test("Hermes keeps opt-in networking and refuses existing outputs", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit();
    await assert.rejects(fetchSdkSources("hermes", f.git, f.output, env(), false, {run: mock.run}), /explicit --allow-network/);
    await assert.rejects(fetchSdkSources("hermes", f.git, f.output, {...env(), MOM_AOT_BUILD_ONLY: "0"}, true, {run: mock.run}), /MOM_AOT_BUILD_ONLY=1/);
    assert.equal(mock.calls.length, 0); assert.equal(fs.existsSync(f.output), false);
    fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "safe");
    await assert.rejects(fetchSdkSources("hermes", f.git, f.output, env(), true, {run: mock.run}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "safe");
  });

  test("Hermes refuses changed origins, incomplete blobs and failed fetches", {skip: process.platform !== "linux"}, async t => {
    for (const options of [{invalidOrigin: true}, {promisor: true}, {shallow: false}, {failFetch: true}, {mutateTool: true}]) {
      const f = fixture(t), mock = mockGit(options);
      await assert.rejects(fetchSdkSources("hermes", f.git, f.output, env(), true, {run: mock.run}));
      assert.equal(fs.existsSync(path.join(f.output, "roots.json")), false);
      assert.equal(fs.existsSync(path.join(f.output, "fetch.json")), false);
    }
  });

  test("Hermes cannot publish a receipt after a mid-fetch recipe change", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), mock = mockGit(); let reads = 0;
    const recipe = (/** @type {"linux" | "windows" | "compiler" | "hermes" | "acpp" | "rocm"} */ target) => {
      const current = sourceRecipe(target); return ++reads === 1 ? current : {...current, graph: {...current.graph, changed: "0".repeat(64)}};
    };
    await assert.rejects(fetchSdkSources("hermes", f.git, f.output, env(), true, {run: mock.run, sourceRecipe: recipe}), /recipe changed/);
    assert.equal(fs.existsSync(path.join(f.output, "roots.json")), false);
  });

  test("Hermes CLI rejects missing network opt-in and extra arguments before output", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    for (const extra of [[], ["--allow-network", "unexpected"]]) {
      const result = spawnSync(process.execPath, [script, "hermes", f.git, f.output, ...extra], {encoding: "utf8", env: env()});
      assert.notEqual(result.status, 0); assert.match(result.stderr, /Usage: fetch-native-sdk-sources/);
      assert.equal(fs.existsSync(f.output), false);
    }
  });
});

describe("native I/O archive acquisition", () => {
  const env = () => ({MOM_AOT_BUILD_ONLY: "1"});
  /** @returns {TestIoPayloads} */
  const payloads = () => ({libuv: Buffer.from("test libuv source archive"), zlib: Buffer.from("test zlib source archive")});

  test("fetches only pinned official HTTPS archives, promotes read-only files, and records provenance", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {Array<{url: string, redirect: string}>} */ ([]);
    const result = await fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data, {calls}), pins: testIoPins(data)});
    assert.deepEqual(calls.map(call => call.url), Object.values(ioArchiveUrls));
    assert.ok(calls.every(call => call.redirect === "manual"));
    assert.deepEqual(Object.keys(result.archives).sort(), ["libuv", "zlib"]);
    for (const name of /** @type {Array<"libuv" | "zlib">} */ (Object.keys(result.archives))) {
      const row = result.archives[name];
      assert.ok(row);
      const bytes = fs.readFileSync(row.path), stat = fs.statSync(row.path);
      assert.equal(bytes.toString(), data[name].toString()); assert.equal(row.bytes, bytes.length);
      assert.equal(row.sha256, testIoPins(data)[name].sha256); assert.equal(stat.mode & 0o222, 0);
    }
    assert.equal(JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).status, "PASS");
    assert.equal(fs.readdirSync(f.output).some(name => name.includes("build") || name.includes("source")), false);
  });

  test("offline I/O admission uses the acquisition cohort without network or mutation", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), pins = testIoPins(data);
    const result = await fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins});
    const hashes = Object.fromEntries(Object.entries(pins).map(([name, pin]) => [name, {sha256: pin.sha256}]));
    const before = fs.statSync(path.join(f.output, "fetch.json"), {bigint: true});
    assert.deepEqual(verifyIoArchives(f.output, {hashes}), result);
    const after = fs.statSync(path.join(f.output, "fetch.json"), {bigint: true});
    assert.equal(after.mtimeNs, before.mtimeNs); assert.equal(after.ctimeNs, before.ctimeNs);
    assert.throws(() => verifyBuildArchives(f.output), /pinned archive cohort/);
    assert.throws(() => verifyIoArchives(f.output, {hashes: {other: {sha256: "0".repeat(64)}}}), /exact pinned libuv\/zlib/);
  });

  test("offline I/O admission rejects altered archives, receipt targets and writable inputs", {skip: process.platform !== "linux"}, async t => {
    for (const fault of ["archive", "target", "writable"]) {
      const f = fixture(t), data = payloads(), pins = testIoPins(data);
      await fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins});
      const hashes = Object.fromEntries(Object.entries(pins).map(([name, pin]) => [name, {sha256: pin.sha256}]));
      const archive = path.join(f.output, pins.zlib.directory + ".tar.gz"), receipt = path.join(f.output, "fetch.json");
      fs.chmodSync(fault === "target" ? receipt : archive, 0o644);
      if (fault === "archive") {fs.writeFileSync(archive, Buffer.alloc(data.zlib.length, 0x61)); fs.chmodSync(archive, 0o444);}
      if (fault === "target") {const record = JSON.parse(fs.readFileSync(receipt, "utf8")); record.target = "build"; fs.writeFileSync(receipt, JSON.stringify(record)); fs.chmodSync(receipt, 0o444);}
      assert.throws(() => verifyIoArchives(f.output, {hashes}), fault === "archive" ? /digest differs/ : fault === "target" ? /valid build-only/ : /read-only/);
    }
  });

  test("CLI I/O verification is offline and refuses a network flag", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    await fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins: testIoPins(data)});
    const verify = spawnSync(process.execPath, [script, "verify-io", f.output], {encoding: "utf8", timeout: 30000});
    assert.notEqual(verify.status, 0); assert.match(verify.stderr, /metadata does not match the pinned libuv archive/);
    const network = spawnSync(process.execPath, [script, "verify-io", f.output, "--allow-network"], {encoding: "utf8", timeout: 30000});
    assert.notEqual(network.status, 0); assert.match(network.stderr, /Usage:/);
  });

  test("accepts a bounded same-origin HTTPS redirect", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {Array<{url: string, redirect: string}>} */ ([]), pins = testIoPins(data);
    /** @type {TestArchiveFetch} */ const fetch = async (url, options) => {
      calls.push({url, redirect: options.redirect});
      if (url === ioArchiveUrls.libuv) {return archiveResponse(url, 302, "", {location: "/dist/v1.51.0/mirror.tar.gz"});}
      if (url.includes("mirror.tar.gz")) {return archiveResponse(url, 200, data.libuv);}
      return archiveResponse(url, 200, data.zlib);
    };
    await fetchIoArchives(f.output, env(), true, {fetch, pins});
    assert.equal(calls.length, 3); assert.equal(calls[1]?.url, "https://dist.libuv.org/dist/v1.51.0/mirror.tar.gz");
    assert.ok(calls.every(call => call.redirect === "manual"));
  });

  test("zlib uses the unchanged upstream release pin and only GitHub's asset redirect", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {string[]} */ ([]),
      asset = "https://release-assets.githubusercontent.com/example/zlib-1.3.1.tar.gz";
    assert.equal(ioArchiveUrls.zlib, "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz");
    assert.equal(ioPins.zlib.sha256, "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23");
    /** @type {TestArchiveFetch} */ const fetch = async url => {
      calls.push(url);
      if (url === ioArchiveUrls.libuv) {return archiveResponse(url, 200, data.libuv);}
      if (url === ioArchiveUrls.zlib) {return archiveResponse(url, 302, "", {location: asset});}
      assert.equal(url, asset); return archiveResponse(url, 200, data.zlib);
    };
    await fetchIoArchives(f.output, env(), true, {fetch, pins: testIoPins(data)});
    assert.deepEqual(calls, [ioArchiveUrls.libuv, ioArchiveUrls.zlib, asset]);
  });

  test("rejects a cross-origin redirect before requesting its target", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {string[]} */ ([]);
    /** @type {TestArchiveFetch} */ const fetch = async (url, _options) => {
      calls.push(url); return archiveResponse(url, 302, "", {location: "https://attacker.example/payload"});
    };
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch, pins: testIoPins(data)}), /left its approved HTTPS origin/);
    assert.equal(calls.length, 1); assert.equal(fs.existsSync(f.output), false);
  });

  test("rejects HTTPS downgrade, redirect loops, and response URLs outside the approved origin", {skip: process.platform !== "linux"}, async t => {
    /** @type {Array<{fetch: TestArchiveFetch, match: RegExp}>} */
    const cases = [
      {fetch: async url => archiveResponse(url, 302, "", {location: "http://dist.libuv.org/cleartext"}), match: /left its approved HTTPS origin/},
      {fetch: async url => archiveResponse(url, 302, "", {location: "/again"}), match: /redirect is missing or excessive/},
      {fetch: async _url => archiveResponse("https://attacker.example/payload", 200, "x"), match: /response left its approved HTTPS origin/},
    ];
    for (const row of cases) {
      const f = fixture(t);
      await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch: row.fetch, pins: testIoPins(payloads())}), row.match);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("rejects a failed status, absent body, and malformed Content-Length", {skip: process.platform !== "linux"}, async t => {
    /** @type {Array<{fetch: TestArchiveFetch, match: RegExp}>} */
    const cases = [
      {fetch: async url => archiveResponse(url, 503, "unavailable"), match: /complete response/},
      {fetch: async url => ({status: 200, url, headers: {get: () => null}, body: null}), match: /complete response/},
      {fetch: async url => archiveResponse(url, 200, "x", {"content-length": "1e9"}), match: /invalid Content-Length/},
    ];
    for (const row of cases) {
      const f = fixture(t);
      await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch: row.fetch, pins: testIoPins(payloads())}), row.match);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("rejects declared and streamed size excess before promoting an archive", {skip: process.platform !== "linux"}, async t => {
    const fetches = /** @type {TestArchiveFetch[]} */ ([
      async url => archiveResponse(url, 200, "x", {"content-length": "4"}),
      async url => archiveResponse(url, 200, "more than three bytes"),
    ]);
    for (const fetch of fetches) {
      const f = fixture(t);
      await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch, pins: testIoPins(payloads()), maxBytes: 3}), /configured archive bound/);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("rejects incorrect pinned bytes and removes the partial output", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads();
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins: ioPins}), /SHA256 differs from its pinned original/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("removes the first archive if the second archive fails verification", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), pins = testIoPins(data);
    /** @type {TestArchiveFetch} */ const fetch = async url => archiveResponse(url, 200, url === ioArchiveUrls.zlib ? "wrong zlib" : data.libuv);
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch, pins}), /SHA256 differs from its pinned original/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("times out a stalled archive and cleans its new output", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads();
    /** @type {TestArchiveFetch} */ const fetch = (_url, {signal}) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {once: true});
    });
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch, pins: testIoPins(data), timeoutMs: 20}), /component deadline exceeded/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("rejects existing directories, symlink outputs, and paths inside the repository without touching them", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), target = path.join(f.root, "preserved"); fs.mkdirSync(target);
    fs.writeFileSync(path.join(target, "keep"), "safe"); fs.symlinkSync(target, f.output, "dir");
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins: testIoPins(data)}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(target, "keep"), "utf8"), "safe");
    const existing = path.join(f.root, "existing"); fs.mkdirSync(existing); fs.writeFileSync(path.join(existing, "keep"), "untouched");
    await assert.rejects(fetchIoArchives(existing, env(), true, {fetch: mockArchiveFetch(data), pins: testIoPins(data)}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(existing, "keep"), "utf8"), "untouched");
    const inside = path.join(__dirname, "fetch-io-never-create");
    await assert.rejects(fetchIoArchives(inside, env(), true, {fetch: mockArchiveFetch(data), pins: testIoPins(data)}), /outside the repository/);
    assert.equal(fs.existsSync(inside), false);
  });

  test("requires build-only and explicit network permission and refuses limits above production bounds", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads();
    await assert.rejects(fetchIoArchives(f.output, env(), false, {fetch: mockArchiveFetch(data), pins: testIoPins(data)}), /explicit --allow-network/);
    await assert.rejects(fetchIoArchives(f.output, {...env(), MOM_AOT_BUILD_ONLY: "0"}, true, {fetch: mockArchiveFetch(data), pins: testIoPins(data)}), /MOM_AOT_BUILD_ONLY=1/);
    await assert.rejects(fetchIoArchives(f.output, env(), true, {fetch: mockArchiveFetch(data), pins: testIoPins(data), maxBytes: (128 << 20) + 1}), /may only be narrowed/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("CLI io mode requires --allow-network and keeps existing Git CLI syntax", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    const io = spawnSync(process.execPath, [script, "io", f.output], {encoding: "utf8", env: {...process.env, MOM_AOT_BUILD_ONLY: "1"}});
    assert.notEqual(io.status, 0); assert.match(io.stderr, /Usage: fetch-native-sdk-sources.js io/); assert.equal(fs.existsSync(f.output), false);
    const git = spawnSync(process.execPath, [script, "linux", f.git, f.output], {encoding: "utf8", env: {...process.env, MOM_AOT_BUILD_ONLY: "1"}});
    assert.notEqual(git.status, 0); assert.match(git.stderr, /Usage: fetch-native-sdk-sources.js linux/); assert.equal(fs.existsSync(f.output), false);
  });
});

describe("native SDK build-source archive acquisition", () => {
  test("source archive receipts exclude downstream LLVM build policy", () => {
    assert.deepEqual(Object.keys(buildSourceRecipe().graph).sort(), ["scripts/build-native-openssl.js",
      "scripts/build-source-inputs.js", "scripts/native-llvm-sdk.js"].sort());
  });
  const env = () => ({MOM_AOT_BUILD_ONLY: "1"});
  /** @returns {TestBuildPayloads} */
  const payloads = () => ({llvm: Buffer.from("test LLVM 21 source archive"), openssl: Buffer.from("test OpenSSL 3.5.9 source archive")});

  test("fetches pinned official LLVM/OpenSSL sources through the approved release asset redirect", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {Array<{url: string, redirect: string}>} */ ([]), hashes = testBuildHashes(data);
    const result = await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data, {calls}), hashes});
    assert.deepEqual(Object.keys(buildArchiveUrls).sort(), ["llvm", "openssl"]);
    assert.deepEqual(buildArchiveUrls, {
      llvm: "https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/llvm-project-21.1.8.src.tar.xz",
      openssl: "https://github.com/openssl/openssl/releases/download/openssl-3.5.9/openssl-3.5.9.tar.gz",
    });
    assert.deepEqual(calls.map(call => call.url), Object.values(buildArchiveUrls).flatMap((url, index) =>
      [url, index === 0 ? "https://release-assets.githubusercontent.com/native-sdk/llvm-project-21.1.8.src.tar.xz" :
        "https://release-assets.githubusercontent.com/native-sdk/openssl-3.5.9.tar.gz"]));
    assert.ok(calls.every(call => call.redirect === "manual"));
    assert.deepEqual(Object.keys(result.archives).sort(), ["llvm", "openssl"]);
    assert.equal(result.target, "build"); assert.equal(result.status, "PASS"); assert.equal(result.compiled, false);
    for (const name of /** @type {Array<"llvm" | "openssl">} */ (Object.keys(result.archives))) {
      const row = result.archives[name]; assert.ok(row);
      const bytes = fs.readFileSync(row.path), stat = fs.statSync(row.path);
      assert.equal(path.basename(row.path), name === "llvm" ? archiveName : "openssl-" + opensslPin.version + ".tar.gz");
      assert.deepEqual(bytes, data[name]); assert.equal(row.bytes, bytes.length);
      assert.equal(row.sha256, hashes[name].sha256); assert.equal(stat.mode & 0o222, 0);
      assert.equal(row.url, buildArchiveUrls[name]);
    }
    const receipt = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8"));
    assert.equal(receipt.status, "PASS"); assert.equal(fs.statSync(path.join(f.output, "fetch.json")).mode & 0o222, 0);
    assert.deepEqual(Object.keys(receipt.recipe.files).sort(), ["scripts/build-native-openssl.js",
      "scripts/build-source-inputs.js", "scripts/native-llvm-sdk.js"].sort());
    assert.equal(archiveSha256, "4633a23617fa31a3ea51242586ea7fb1da7140e426bd62fc164261fe036aa142");
    assert.equal(opensslPin.sha256, "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a");
    assert.equal(verifyBuildArchives(f.output, {hashes})["status"], "PASS");
  });

  test("acquires and warmly admits the original Windows OpenSSL receipt", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), bytes = Buffer.from("fixture Windows OpenSSL 3.5.9 archive"), digest = createHash("sha256").update(bytes).digest("hex"),
      hashes = {openssl: {sha256: digest}}, calls = /** @type {Array<{url: string, redirect: string}>} */ ([]);
    /** @type {TestArchiveFetch} */ const fetch = async (url, options) => {
      calls.push({url, redirect: options.redirect});
      assert.equal(url, buildArchiveUrls.openssl);
      return archiveResponse(url, 200, bytes, {"content-length": String(bytes.length)});
    };
    const fetched = await fetchWindowsOpenSslArchives(f.output, env(), true, {fetch, hashes, timeoutMs: 5000}),
      receiptPath = path.join(f.output, "fetch.json"), before = fs.readFileSync(receiptPath), receipt = JSON.parse(before.toString("utf8")),
      archive = fetched.archives["openssl"];
    assert.ok(archive);
    assert.equal(fetched.target, "windows-openssl"); assert.equal(fetched.status, "PASS"); assert.equal(fetched.compiled, false);
    assert.deepEqual(calls, [{url: buildArchiveUrls.openssl, redirect: "manual"}]);
    assert.deepEqual(Object.keys(receipt).sort(), ["archives", "buildOnly", "compiled", "fetcherSha256", "networked", "publicationQualified", "recipe", "status", "target", "version"]);
    assert.equal(receipt.target, "windows-openssl"); assert.equal(receipt.networked, true); assert.equal(receipt.buildOnly, true);
    assert.equal(receipt.publicationQualified, false); assert.deepEqual(receipt.recipe.files, opensslBuildRecipe().graph);
    assert.equal(receipt.recipe.sha256, createHash("sha256").update(JSON.stringify(opensslBuildRecipe())).digest("hex"));
    assert.equal(path.basename(archive.path), "openssl-" + opensslPin.version + ".tar.gz");
    assert.equal(archive.url, buildArchiveUrls.openssl); assert.equal(archive.sha256, digest); assert.equal(archive.bytes, bytes.length);
    assert.deepEqual(fs.readFileSync(archive.path), bytes); assert.equal(fs.statSync(archive.path).mode & 0o222, 0);
    assert.equal(fs.statSync(receiptPath).mode & 0o222, 0);
    const admitted = verifyWindowsOpenSslArchives(f.output, {hashes});
    assert.deepEqual(admitted, fetched); assert.deepEqual(fs.readFileSync(receiptPath), before);
    assert.equal(calls.length, 1, "warm verification must stay offline");
  });

  test("Windows OpenSSL acquisition requires opt-in and offline admission rejects tampering", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), bytes = Buffer.from("fixture Windows OpenSSL archive"), digest = createHash("sha256").update(bytes).digest("hex"),
      hashes = {openssl: {sha256: digest}}; let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 200, bytes);};
    const denied = path.join(f.root, "network-denied");
    await assert.rejects(fetchWindowsOpenSslArchives(denied, env(), false, {fetch, hashes}), /explicit --allow-network/);
    assert.equal(calls, 0); assert.equal(fs.existsSync(denied), false);
    await fetchWindowsOpenSslArchives(f.output, env(), true, {fetch, hashes});
    const archive = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).archives["openssl"].path;
    fs.chmodSync(archive, 0o644); fs.writeFileSync(archive, Buffer.alloc(bytes.length, 0x58)); fs.chmodSync(archive, 0o444);
    assert.throws(() => verifyWindowsOpenSslArchives(f.output, {hashes}), /digest differs from its pin/);
    assert.equal(calls, 1);
  });

  test("offline verification rejects an archive changed after acquisition", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const archive = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).archives.llvm.path;
    fs.chmodSync(archive, 0o644);
    const fd = fs.openSync(archive, "r+");
    try {fs.writeSync(fd, Buffer.from("X"), 0, 1, 0);} finally {fs.closeSync(fd);}
    fs.chmodSync(archive, 0o444);
    assert.throws(() => verifyBuildArchives(f.output, {hashes}), /digest differs from its pin/);
  });

  test("offline verification rejects a tampered acquisition receipt", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
    receipt.compiled = true;
    fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, JSON.stringify(receipt) + "\n"); fs.chmodSync(receiptPath, 0o444);
    assert.throws(() => verifyBuildArchives(f.output, {hashes}), /not a valid build-only, unpublished acquisition/);
  });

  test("offline verification retains historical fetcher provenance without restamping pinned sources", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
    const historical = "363a8dea6258dc5714b3cdde05065e1f12f729f9de1f5df25989e9a259ec5fc8";
    receipt.fetcherSha256 = historical;
    fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, JSON.stringify(receipt) + "\n"); fs.chmodSync(receiptPath, 0o444);
    const before = fs.readFileSync(receiptPath), stat = fs.statSync(receiptPath);
    assert.equal(verifyBuildArchives(f.output, {hashes}).fetcherSha256, historical);
    assert.deepEqual(fs.readFileSync(receiptPath), before);
    assert.equal(fs.statSync(receiptPath).mtimeMs, stat.mtimeMs);
    // A historical acquisition identity never substitutes for checking the actual pinned bytes.
    const archive = receipt.archives.llvm.path;
    fs.chmodSync(archive, 0o644); fs.writeFileSync(archive, "changed"); fs.chmodSync(archive, 0o444);
    assert.throws(() => verifyBuildArchives(f.output, {hashes}), /metadata does not match|digest differs from its pin/);
  });

  test("offline verification rejects malformed fetcher provenance", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8"));
    receipt.fetcherSha256 = "not-a-sha256";
    fs.chmodSync(receiptPath, 0o644); fs.writeFileSync(receiptPath, JSON.stringify(receipt) + "\n"); fs.chmodSync(receiptPath, 0o444);
    assert.throws(() => verifyBuildArchives(f.output, {hashes}), /invalid fetcher provenance/);
  });

  test("offline verification preserves historical builder fingerprints for immutable pinned archives", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data), current = buildSourceRecipe();
    const staleRecipe = () => ({...current, graph: {...current.graph, "scripts/build-native-openssl.js": "0".repeat(64)}});
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes, recipe: staleRecipe});
    const receipt = path.join(f.output, "fetch.json"), before = fs.readFileSync(receipt);
    assert.equal(verifyBuildArchives(f.output, {hashes}).recipe.files["scripts/build-native-openssl.js"], "0".repeat(64));
    assert.deepEqual(fs.readFileSync(receipt), before, "historical acquisition metadata is never restamped");
  });

  test("historical builder metadata still rejects malformed fingerprints, membership and checksums", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const file = path.join(f.output, "fetch.json"), original = fs.readFileSync(file, "utf8");
    for (const change of [
      (/** @type {any} */ r) => {r.recipe.files["scripts/build-native-openssl.js"] = "not-a-hash";},
      (/** @type {any} */ r) => {delete r.recipe.files["scripts/build-native-openssl.js"];},
      (/** @type {any} */ r) => {r.recipe.files["unexpected.js"] = "a".repeat(64);},
    ]) {
      const r = JSON.parse(original); change(r);
      r.recipe.sha256 = createHash("sha256").update(JSON.stringify({files: {}, graph: r.recipe.files})).digest("hex");
      fs.chmodSync(file, 0o644); fs.writeFileSync(file, JSON.stringify(r) + "\n"); fs.chmodSync(file, 0o444);
      assert.throws(() => verifyBuildArchives(f.output, {hashes}), /stale builder recipe provenance/);
    }
    const r = JSON.parse(original); r.recipe.files["scripts/build-native-openssl.js"] = "0".repeat(64);
    fs.chmodSync(file, 0o644); fs.writeFileSync(file, JSON.stringify(r) + "\n"); fs.chmodSync(file, 0o444);
    assert.throws(() => verifyBuildArchives(f.output, {hashes}), /stale builder recipe provenance/);
  });

  test("offline verification rejects a receipt rewritten and restored after reading", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const receipt = path.join(f.output, "fetch.json"), original = fs.readFileSync(receipt), before = fs.statSync(receipt);
    const recipe = () => {
      fs.chmodSync(receipt, 0o644); fs.writeFileSync(receipt, original);
      fs.utimesSync(receipt, before.atime, before.mtime); fs.chmodSync(receipt, 0o444);
      return buildSourceRecipe();
    };
    assert.throws(() => verifyBuildArchives(f.output, {hashes, recipe}), /fetch receipt changed during verification/);
    assert.deepEqual(fs.readFileSync(receipt), original, "unchanged bytes alone must not admit a raced receipt");
  });

  test("offline verification rejects writable, linked, symlinked, and extra files", {skip: process.platform !== "linux"}, async t => {
    const data = payloads(), hashes = testBuildHashes(data), fetch = () => mockBuildArchiveFetch(data);
    {
      const f = fixture(t); await fetchBuildArchives(f.output, env(), true, {fetch: fetch(), hashes});
      const archive = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).archives.llvm.path;
      fs.chmodSync(archive, 0o644);
      try {
        assert.throws(() => verifyBuildArchives(f.output, {hashes}), /read-only, single-link regular file/);
      } finally {
        fs.chmodSync(archive, 0o444);
      }
    }
    {
      const f = fixture(t); await fetchBuildArchives(f.output, env(), true, {fetch: fetch(), hashes});
      const archive = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).archives.llvm.path;
      fs.linkSync(archive, path.join(f.root, "second-link"));
      assert.throws(() => verifyBuildArchives(f.output, {hashes}), /read-only, single-link regular file/);
    }
    {
      const f = fixture(t); await fetchBuildArchives(f.output, env(), true, {fetch: fetch(), hashes});
      const archive = JSON.parse(fs.readFileSync(path.join(f.output, "fetch.json"), "utf8")).archives.llvm.path;
      const saved = path.join(f.root, "archive-target"); fs.renameSync(archive, saved); fs.symlinkSync(saved, archive);
      assert.throws(() => verifyBuildArchives(f.output, {hashes}), /read-only, single-link regular file/);
    }
    {
      const f = fixture(t); await fetchBuildArchives(f.output, env(), true, {fetch: fetch(), hashes});
      fs.writeFileSync(path.join(f.output, "extra"), "unexpected");
      assert.throws(() => verifyBuildArchives(f.output, {hashes}), /only its receipt and pinned archive cohort/);
    }
  });

  test("refuses unapproved or downgraded redirects before requesting the target", {skip: process.platform !== "linux"}, async t => {
    const cases = [
      {location: "https://attacker.example/payload"},
      {location: "http://release-assets.githubusercontent.com/payload"},
    ];
    for (const row of cases) {
      const f = fixture(t), data = payloads(), calls = /** @type {string[]} */ ([]);
      /** @type {TestArchiveFetch} */ const fetch = async url => {calls.push(url); return archiveResponse(url, 302, "", {location: row.location});};
      await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes: testBuildHashes(data)}), /left its approved HTTPS origin/);
      assert.equal(calls.length, 1); assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("rejects a response URL outside the fixed official host allowlist", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads();
    /** @type {TestArchiveFetch} */ const fetch = async _url => archiveResponse("https://attacker.example/payload", 200, "x");
    await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes: testBuildHashes(data)}), /response left its approved HTTPS origin/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("bounds streamed source bytes and rejects a changed hash without retaining a partial cohort", {skip: process.platform !== "linux"}, async t => {
    const data = payloads(), hashes = testBuildHashes(data);
    {
      const f = fixture(t), calls = /** @type {string[]} */ ([]);
      /** @type {TestArchiveFetch} */ const fetch = async url => {calls.push(url); return archiveResponse(url, 200, "too large", {"content-length": "9"});};
      await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes, maxBytes: 3}), /configured archive bound/);
      assert.equal(fs.existsSync(f.output), false);
    }
    {
      const f = fixture(t);
      /** @type {TestArchiveFetch} */ const fetch = async url => archiveResponse(url, 200, "oversized streamed source");
      await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes, maxBytes: 3}), /configured archive bound/);
      assert.equal(fs.existsSync(f.output), false);
    }
    {
      const f = fixture(t);
      /** @type {TestArchiveFetch} */ const fetch = async url => {
        const name = url.includes("llvm-project") ? "llvm" : "openssl", asset = "https://release-assets.githubusercontent.com/native-sdk/" + name;
        if (url === buildArchiveUrls[name]) {return archiveResponse(url, 302, "", {location: asset});}
        return archiveResponse(url, 200, name === "openssl" ? Buffer.from("wrong OpenSSL bytes") : data.llvm);
      };
      await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes}), /SHA256 differs from its pinned original/);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("validates pins, explicit network permission, and a new output before fetching", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), data = payloads(), calls = /** @type {string[]} */ ([]);
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls.push(url); return archiveResponse(url, 503, "unexpected");};
    await assert.rejects(fetchBuildArchives(f.output, env(), false, {fetch, hashes: testBuildHashes(data)}), /explicit --allow-network/);
    await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes: {llvm: {sha256: "x"}, openssl: {sha256: "0".repeat(64)}}}), /exact pinned LLVM\/OpenSSL/);
    await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes: testBuildHashes(data), maxBytes: (192 << 20) + 1}), /may only be narrowed/);
    assert.equal(calls.length, 0); assert.equal(fs.existsSync(f.output), false);
    fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "preserve");
    await assert.rejects(fetchBuildArchives(f.output, env(), true, {fetch, hashes: testBuildHashes(data)}), /new absolute directory/);
    assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "preserve"); assert.equal(calls.length, 0);
  });

  test("CLI build mode requires explicit --allow-network", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    const result = spawnSync(process.execPath, [script, "build", f.output], {encoding: "utf8", env: {...process.env, MOM_AOT_BUILD_ONLY: "1"}});
    assert.notEqual(result.status, 0); assert.match(result.stderr, /Usage: fetch-native-sdk-sources.js build/); assert.equal(fs.existsSync(f.output), false);
  });

  test("CLI build verification is offline and rejects network flags", async t => {
    const f = fixture(t), script = path.resolve(__dirname, "../scripts/fetch-native-sdk-sources.js");
    const data = payloads(), hashes = testBuildHashes(data);
    await fetchBuildArchives(f.output, env(), true, {fetch: mockBuildArchiveFetch(data), hashes});
    const verify = spawnSync(process.execPath, [script, "verify-build", f.output], {encoding: "utf8"});
    assert.notEqual(verify.status, 0); assert.match(verify.stderr, /receipt metadata does not match the pinned llvm archive/);
    const network = spawnSync(process.execPath, [script, "verify-build", f.output, "--allow-network"], {encoding: "utf8"});
    assert.notEqual(network.status, 0); assert.match(network.stderr, /Usage: fetch-native-sdk-sources.js verify-build/);
  });
});

describe("trusted pinned archive-cohort API", {skip: !["linux", "win32"].includes(process.platform)}, () => {
  const env = () => ({MOM_AOT_BUILD_ONLY: "1"});

  test("acquires a static exact-pin cohort and binds its read-only receipt to the caller recipe", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned Windows SDK package"), calls = /** @type {string[]} */ ([]);
    /** @type {TestArchiveFetch} */ const fetch = async (url, options) => {
      calls.push(url); assert.equal(options.redirect, "manual");
      return archiveResponse(url, 200, bytes, {"content-length": String(bytes.length)});
    };
    const cohort = pinnedArchiveFixture(bytes, fetch), result = await fetchArchiveCohort(f.output, env(), true, cohort),
      receiptPath = path.join(f.output, "fetch.json"), receipt = JSON.parse(fs.readFileSync(receiptPath, "utf8")),
      archive = receipt.archives["payload"];
    assert.equal(calls.length, 1); assert.equal(result.target, "windows-sdk-supplemental");
    assert.deepEqual(receipt.recipe.files, cohort.recipeFor().graph);
    assert.equal(receipt.recipe.sha256, createHash("sha256").update(JSON.stringify(cohort.recipeFor())).digest("hex"));
    assert.deepEqual(receipt.archives["payload"], result.archives["payload"]);
    assert.equal(fs.readFileSync(archive.path).compare(bytes), 0);
    assert.equal(fs.statSync(archive.path).mode & 0o222, 0);
    assert.equal(fs.statSync(receiptPath).mode & 0o222, 0);
    assert.equal(fs.statSync(archive.path).nlink, 1);
    assert.equal(fs.statSync(receiptPath).nlink, 1);
    assert.deepEqual(fs.readdirSync(f.output).sort(), ["fetch.json", "fixture.vsix"]);
  });

  test("rejects a changed pin, a response outside the pinned origin, and streamed size excess", async t => {
    const bytes = Buffer.from("pinned bytes"), changed = Buffer.from("different bytes"),
      expectedSha256 = createHash("sha256").update(bytes).digest("hex"), actualSha256 = createHash("sha256").update(changed).digest("hex"),
      mismatch = new RegExp(`SHA256 differs from its pinned original: payload \\(expected SHA256 ${expectedSha256}, actual SHA256 ${actualSha256}, downloaded ${changed.length} bytes\\)`),
      cases = /** @type {Array<{fetch: TestArchiveFetch, maxBytes: number, match: RegExp}>} */ ([
        {fetch: async (url, _options) => archiveResponse(url, 200, changed), maxBytes: 1024, match: mismatch},
        {fetch: async _url => archiveResponse("https://attacker.example/fixture.vsix", 200, bytes), maxBytes: 1024, match: /response left its approved HTTPS origin/},
        {fetch: async url => archiveResponse(url, 200, bytes), maxBytes: 3, match: /configured archive bound/},
      ]);
    for (const row of cases) {
      const f = fixture(t);
      /** @type {TestArchiveFetch} */ const fetch = row.fetch;
      await assert.rejects(fetchArchiveCohort(f.output, env(), true,
        pinnedArchiveFixture(bytes, fetch, {maxBytes: row.maxBytes})), row.match);
      assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("rejects unsafe or colliding names and never overwrites an existing output", async t => {
    const bytes = Buffer.from("pinned bytes"); let calls = 0;
    /** @type {TestArchiveFetch} */ const fetch = async url => {calls++; return archiveResponse(url, 200, bytes);};
    for (const name of [".", "..", "nested/payload.vsix", "nested\\payload.vsix", "NUL.msi", "con.cab", "COM1.zip", "LPT9", "AUX.data", "PRN.tar", "file."]) {
      const f = fixture(t), cohort = pinnedArchiveFixture(bytes, fetch, {fileName: name});
      await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /exact pinned archive cohort/);
      assert.equal(fs.existsSync(f.output), false);
    }
    {
      const f = fixture(t), cohort = pinnedArchiveFixture(bytes, fetch);
      const spec = cohort.specs["payload"], pin = cohort.pins["payload"];
      assert.ok(spec); assert.ok(pin);
      cohort.specs = {first: spec, second: {...spec}};
      cohort.pins = {first: pin, second: {...pin}};
      await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /unique safe leaf names/);
      assert.equal(fs.existsSync(f.output), false);
    }
    {
      const f = fixture(t); fs.mkdirSync(f.output); fs.writeFileSync(path.join(f.output, "keep"), "untouched");
      await assert.rejects(fetchArchiveCohort(f.output, env(), true, pinnedArchiveFixture(bytes, fetch)), /new absolute directory/);
      assert.equal(fs.readFileSync(path.join(f.output, "keep"), "utf8"), "untouched");
    }
    assert.equal(calls, 0);
  });

  test("archive acquisition requires build-only mode and explicit network opt-in", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes"); let calls = 0;
    const cohort = pinnedArchiveFixture(bytes, async url => {calls++; return archiveResponse(url, 200, bytes);});
    await assert.rejects(fetchArchiveCohort(f.output, env(), false, cohort), /explicit --allow-network/);
    await assert.rejects(fetchArchiveCohort(f.output, {}, true, cohort), /MOM_AOT_BUILD_ONLY/);
    assert.equal(calls, 0); assert.equal(fs.existsSync(f.output), false);
  });

  test("archive rollback removes an already-promoted read-only sibling", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes"); let calls = 0;
    const cohort = pinnedArchiveFixture(bytes, async url => archiveResponse(url, 200, ++calls === 1 ? bytes : "wrong bytes"));
    const spec = cohort.specs["payload"], pin = cohort.pins["payload"]; assert.ok(spec && pin);
    cohort.specs = {first: {...spec, fileName: "first.zip"}, second: {...spec, fileName: "second.zip"}};
    cohort.pins = {first: pin, second: {...pin}};
    await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /SHA256 differs/);
    assert.equal(calls, 2); assert.equal(fs.existsSync(f.output), false);
  });

  test("archive promotion never overwrites a foreign file created during download", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes"), target = path.join(f.output, "fixture.vsix");
    const cohort = pinnedArchiveFixture(bytes, async url => {
      fs.writeFileSync(target, "preserve me", {flag: "wx"}); return archiveResponse(url, 200, bytes);
    });
    await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /EEXIST/);
    assert.equal(fs.readFileSync(target, "utf8"), "preserve me");
    assert.equal(fs.existsSync(path.join(f.output, "fetch.json")), false);
  });

  test("archive redirects cannot downgrade HTTPS or leave their official origin", async t => {
    for (const location of ["http://download.visualstudio.microsoft.com/payload.zip", "https://attacker.example/payload.zip"]) {
      const f = fixture(t), bytes = Buffer.from("pinned bytes"); let calls = 0;
      const cohort = pinnedArchiveFixture(bytes, async url => {calls++; return archiveResponse(url, 302, "", {location});});
      await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /approved HTTPS origin/);
      assert.equal(calls, 1); assert.equal(fs.existsSync(f.output), false);
    }
  });

  test("archive recipe drift cannot leave promoted bytes or a passing receipt", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes"); let version = "a";
    const cohort = pinnedArchiveFixture(bytes, async url => {version = "b"; return archiveResponse(url, 200, bytes);});
    cohort.recipeFor = () => ({files: {}, graph: {"fixture.js": version.repeat(64)}});
    await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /recipe changed/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("archive deadlines cancel a stalled request and remove its private output", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes");
    const cohort = pinnedArchiveFixture(bytes, (_url, {signal}) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {once: true});
    }));
    cohort.timeoutMs = 20;
    await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /component deadline exceeded/);
    assert.equal(fs.existsSync(f.output), false);
  });

  test("archive metadata rejects malformed Content-Length before saving bytes", async t => {
    const f = fixture(t), bytes = Buffer.from("pinned bytes");
    const cohort = pinnedArchiveFixture(bytes, async url => archiveResponse(url, 200, bytes, {"content-length": "1e3"}));
    await assert.rejects(fetchArchiveCohort(f.output, env(), true, cohort), /Content-Length/);
    assert.equal(fs.existsSync(f.output), false);
  });
});

describe("pinned Windows host-SDK archive cohort", () => {
  test("lock binds only fixed selected roots, package origins, and MSI Media CAB relations", () => {
    const lock = windowsHostSdkCatalog(), catalog = windowsHostSdkArchiveCatalog();
    assert.equal(lock.target, "windows-host"); assert.equal(lock.sourceCatalog.publisherAuthenticated, false);
    assert.equal(Object.keys(lock.packages).length, 14); assert.equal(Object.keys(lock.cabinets).length, 28);
    assert.equal(Object.keys(lock.msiCabinets).length, 9); assert.equal(lock.selections.length, 19);
    assert.equal(Object.keys(catalog).length, 42);
    assert.deepEqual(lock.roots, ["sdk/include/ucrt", "sdk/include/shared", "sdk/include/um", "sdk/include/winrt",
      "sdk/lib/ucrt", "sdk/lib/um", "msvc/include", "msvc/lib"]);
    assert.equal(Object.keys(lock.packages).some(key => key.includes("Tools")), false);
    assert.ok(lock.selections.every(row => lock.packages[row.package]));
    assert.deepEqual(new Set(lock.selections.map(row => row.destinationPrefix.split("/").slice(0, 2).join("/"))).size, 4);
    assert.equal(lock.selections.find(row => row.destinationPrefix === "msvc/lib/oldnames.lib")?.sourcePrefix,
      "Contents/VC/Tools/MSVC/14.44.35207/lib/onecore/x64/oldnames.lib");
    assert.equal(lock.selections.find(row => row.destinationPrefix === "msvc/lib/comsuppw.lib")?.sourcePrefix,
      "Contents/VC/Tools/MSVC/14.44.35207/lib/x64/comsuppw.lib");
    assert.equal(lock.selections.find(row => row.destinationPrefix === "msvc/lib/delayimp.lib")?.sourcePrefix,
      "Contents/VC/Tools/MSVC/14.44.35207/lib/x64/delayimp.lib");
    const storeLibs = lock.selections.filter(row => row.package === "windows-sdk-store-libs-x86");
    assert.deepEqual(storeLibs.map(row => [row.sourcePrefix, row.destinationPrefix]), [[
      "Program Files/Windows Kits/10/Lib/10.0.26100.0/um/x64", "sdk/lib/um",
    ]]);
    for (const [msiKey, relations] of Object.entries(lock.msiCabinets)) {
      assert.equal(lock.packages[msiKey]?.format, "msi");
      for (const relation of relations) {
        const cabinet = lock.cabinets[relation.cabinet];
        assert.ok(cabinet); assert.equal(cabinet.msiKey, msiKey); assert.equal(cabinet.diskId, relation.diskId);
      }
    }
    for (const row of Object.values(catalog)) {
      assert.equal(new URL(row.url).origin, "https://download.visualstudio.microsoft.com");
    }
    assert.throws(() => windowsHostSdkArchiveCatalog({...Object.fromEntries(Object.keys(catalog).map(key => [key, {sha256: "a".repeat(64)}])),
      injected: {sha256: "b".repeat(64)}}), /exact pinned package\/CAB cohort/);
  });

  test("uses the shared bounded fetcher for synthetic payloads and verifies its offline receipt", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), payload = Buffer.from("synthetic Windows host SDK archive fixture\n"), digest = createHash("sha256").update(payload).digest("hex"),
      catalog = windowsHostSdkArchiveCatalog(), hashes = /** @type {Record<string, {sha256: string}>} */ ({}), calls = /** @type {string[]} */ ([]);
    for (const key of Object.keys(catalog)) {hashes[key] = {sha256: digest};}
    /** @type {TestArchiveFetch} */ const fetch = async (url, options) => {
      calls.push(url); assert.equal(options.redirect, "manual");
      return archiveResponse(url, 200, payload, {"content-length": String(payload.length)});
    };
    const receipt = await fetchWindowsHostSdkArchives(f.output, {MOM_AOT_BUILD_ONLY: "1"}, true, {fetch, hashes});
    assert.equal(receipt.target, "windows-host"); assert.equal(receipt.archives && Object.keys(receipt.archives).length, 42);
    assert.equal(calls.length, 42); assert.equal(receipt.compiled, false); assert.equal(receipt.publicationQualified, false);
    assert.deepEqual(fs.readdirSync(f.output).sort(), ["fetch.json", ...Object.values(catalog).map(row => row.fileName)].sort());
    for (const [key, row] of Object.entries(receipt.archives)) {
      const expected = catalog[key]; assert.ok(expected);
      assert.equal(row.path, path.join(f.output, expected.fileName)); assert.equal(row.sha256, digest); assert.equal(row.bytes, payload.length);
      assert.equal(fs.readFileSync(row.path).compare(payload), 0); assert.equal(fs.statSync(row.path).mode & 0o222, 0);
    }
    const verified = verifyWindowsHostSdkArchives(f.output, {hashes});
    assert.equal(verified.target, "windows-host"); assert.equal(verified.status, "PASS");
    assert.equal(fs.statSync(path.join(f.output, "fetch.json")).mode & 0o222, 0);
  });

  test("fails closed on missing network opt-in and untrusted response origin", {skip: process.platform !== "linux"}, async t => {
    const f = fixture(t), catalog = windowsHostSdkArchiveCatalog(); let calls = 0;
    await assert.rejects(fetchWindowsHostSdkArchives(f.output, {MOM_AOT_BUILD_ONLY: "1"}, false), /explicit --allow-network/);
    assert.equal(fs.existsSync(f.output), false);
    const payload = Buffer.from("synthetic payload"), digest = createHash("sha256").update(payload).digest("hex"),
      hashes = Object.fromEntries(Object.keys(catalog).map(key => [key, {sha256: digest}]));
    /** @type {TestArchiveFetch} */ const fetch = async _url => {
      calls++; return archiveResponse("https://attacker.example/payload", 200, payload);
    };
    await assert.rejects(fetchWindowsHostSdkArchives(f.output, {MOM_AOT_BUILD_ONLY: "1"}, true, {fetch, hashes}), /response left its approved HTTPS origin/);
    assert.equal(calls, 1); assert.equal(fs.existsSync(f.output), false);
  });

  test("CLI fetch requires explicit network opt-in and offline verify rejects network flags", {skip: process.platform !== "linux"}, t => {
    const f = fixture(t), script = require.resolve("../scripts/fetch-native-sdk-sources"),
      fetch = spawnSync(process.execPath, [script, "windows-host-sdk", f.output], {encoding: "utf8", env: {MOM_AOT_BUILD_ONLY: "1"}}),
      verify = spawnSync(process.execPath, [script, "verify-windows-host-sdk", f.output, "--allow-network"], {encoding: "utf8", env: {MOM_AOT_BUILD_ONLY: "1"}});
    assert.notEqual(fetch.status, 0); assert.match(fetch.stderr, /Usage: fetch-native-sdk-sources\.js windows-host-sdk/);
    assert.notEqual(verify.status, 0); assert.match(verify.stderr, /Usage: fetch-native-sdk-sources\.js verify-windows-host-sdk/);
    assert.equal(fs.existsSync(f.output), false);
  });
});
