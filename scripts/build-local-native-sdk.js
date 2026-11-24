"use strict";

// Explicit Linux source/LLVM bootstrap slice. This is not a release build or package step.
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"),
  {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {createHash} = require("node:crypto");
const {prepareSdkSources, verifyPreparedSdkSources, sdkComponents} = require("./prepare-native-sdk-sources");
const {canonicalBuildDirectory, hasExactBuildFields, buildPathsOverlap, fileHash, readBuildJson, nativeBuildJobs: localBuildJobs} = require("./build-source-inputs");
const {archiveName} = require("./native-llvm-sdk");
const {verifyLinuxLlvmAdmission} = require("./build-native-llvm-sdk");
const {ioPins, nativeIoInputs} = require("./build-native-io");
const {opensslPin, nativeOpenSslInputs} = require("./build-native-openssl");
const {zstdPin} = require("./build-native-zstd");
const {verifyBuildArchives, verifyIoArchives, verifyZstdArchive, verifySysrootPackages} = require("./fetch-native-sdk-sources");
const {prepareLinuxSysroot, applyLinuxSysrootEnvironment, linuxSysrootToolchain} = require("./native-linux-sysroot");
const ciJit = require("./native-ci-linux-jit"),
  ciSources = require("./native-ci-linux-sources"),
  ciSupport = require("./native-ci-linux-support"),
  dpcppInputs = require("./native-dpcpp-cuda-inputs");

const repo = fs.realpathSync(path.resolve(__dirname, "..")), markerName = ".mom-native-sdk-workspace.json",
  gib = 1024 ** 3, oneapiCompilerRoot = "/opt/intel/oneapi/compiler";
/** @param {string} name */ const producer = name => path.join(repo, "scripts", name);

/** Producers that run CI's own support/ROCm recipes, whose admissions accept only CI's 2 or 4 jobs. @param {number} jobs */
function ciRecipeJobs(jobs) {return jobs >= 4 ? 4 : 2;}

/** Aggregate no-swap scope cap in bytes: the proven 4 GiB for two jobs plus 3 GiB for each further job. Release
 * LLVM/DPC++ compiles peak near 1.5 GiB, while their static tool and SYCL links take 2-3 GiB and are not pooled
 * separately, so every added Ninja slot may hold a link. A cap above physical memory could not protect the host.
 * @param {NodeJS.ProcessEnv} [environment] @param {number} [totalMemory] */
function localMemoryMax(environment = process.env, totalMemory = os.totalmem()) {
  const jobs = localBuildJobs(environment), bytes = (4 + 3 * Math.max(0, jobs - 2)) * gib;
  if (bytes > totalMemory) {
    throw new Error(`MOM_BUILD_JOBS=${jobs} needs a ${bytes / gib} GiB memory scope, but this host has ${(totalMemory / gib).toFixed(1)} GiB`);
  }
  return bytes;
}

/** Read the job count a producer recorded only as `--parallel N` in its compile command. @param {unknown} command */
function recordedParallelJobs(command) {
  const args = Array.isArray(command) ? command : [], index = args.indexOf("--parallel"), value = index < 0 ? undefined : args[index + 1];
  if (typeof value !== "string" || !/^[1-9]\d?$/.test(value)) {throw new Error("Native SDK receipt does not record its build parallelism");}
  return Number(value);
}

function defaultWorkspace() {
  let parent;
  try {parent = fs.realpathSync(os.homedir());} catch {throw new Error("Cannot resolve the home directory for the default native SDK workspace");}
  for (const name of ["cache", "mom"]) {
    const next = path.join(parent, name), stat = fs.lstatSync(next, {throwIfNoEntry: false});
    if (!stat) {fs.mkdirSync(next, {mode: 0o700});} else if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(next) !== next) {
      throw new Error("Default native SDK workspace parent must be a real directory: " + next);
    }
    parent = next;
  }
  return path.join(parent, "native-sdk");
}

/** Resolve PATH tools to immutable absolute executable paths. @param {string} name @param {string} searchPath */
function findExecutable(name, searchPath) {
  for (const directory of searchPath.split(path.delimiter)) {
    if (!directory || !path.isAbsolute(directory)) {continue;}
    const candidate = path.join(directory, name);
    try {
      const real = fs.realpathSync(candidate), stat = fs.statSync(real);
      fs.accessSync(real, fs.constants.X_OK);
      if (stat.isFile()) {return real;}
    } catch { /* try the next PATH entry */ }
  }
  throw new Error("Missing executable prerequisite: " + name);
}

/** @param {string} file */
function executableVersion(file) {
  const result = spawnSync(file, ["--version"], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20});
  if (result.error || result.signal || result.status !== 0) {throw new Error("Cannot identify required compiler: " + file, {cause: result.error});}
  return (result.stdout || "") + (result.stderr || "");
}

function trustedSystemdRun() {
  const file = "/usr/bin/systemd-run", stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file || stat.uid !== 0 || (stat.mode & 0o022) !== 0) {
    throw new Error("A root-owned, non-writable /usr/bin/systemd-run is required to enforce native SDK limits");
  }
  fs.accessSync(file, fs.constants.X_OK);
  return file;
}

/** @param {NodeJS.ProcessEnv} environment @param {typeof executableVersion} versionOf */
function localTools(environment, versionOf = executableVersion) {
  let compilerDirectory;
  try {compilerDirectory = fs.realpathSync(path.join(oneapiCompilerRoot, "latest", "bin"));} catch {throw new Error("Intel oneAPI DPC++ compiler 2026.2 is missing; run scripts/install-dev.sh --component base,oneapi --jobs 2 --keep-workspace explicitly.");}
  const expectedDirectory = path.join(oneapiCompilerRoot, "2026.2", "bin");
  if (compilerDirectory !== expectedDirectory) {
    throw new Error("Intel oneAPI compiler/latest must resolve to 2026.2; run scripts/install-dev.sh --component base,oneapi --jobs 2 --keep-workspace explicitly.");
  }
  /** @param {string} name */
  const compiler = name => {
    const file = path.join(compilerDirectory, name);
    try {if (!fs.statSync(file).isFile()) {throw new Error("not a file");} fs.accessSync(file, fs.constants.X_OK);} catch {throw new Error("Intel oneAPI DPC++ compiler 2026.2 is incomplete; run scripts/install-dev.sh --component base,oneapi --jobs 2 --keep-workspace explicitly.");}
    if (!/Intel\(R\) oneAPI DPC\+\+\/C\+\+ Compiler 2026\.2(?:\.|\b)/.test(versionOf(file))) {
      throw new Error("Expected Intel oneAPI DPC++ compiler 2026.2 at " + file + "; run scripts/install-dev.sh --component base,oneapi --jobs 2 --keep-workspace explicitly.");
    }
    // Keep icpx as the selected invocation alias even if it resolves to icx.
    return file;
  };
  return {git: findExecutable("git", environment["PATH"] || ""), cmake: findExecutable("cmake", environment["PATH"] || ""),
    ninja: findExecutable("ninja", environment["PATH"] || ""), systemdRun: trustedSystemdRun(),
    cc: compiler("icx"), cxx: compiler("icpx")};
}

/** Create or admit only the marked workspace for this checkout; never clear unknown contents.
 * @param {string} workspace @param {string} checkout */
function admitWorkspace(workspace, checkout) {
  if (!path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || buildPathsOverlap(workspace, checkout) || /[\r\n\0]/.test(workspace)) {
    throw new Error("Native SDK workspace must be a canonical absolute directory outside the checkout");
  }
  const parent = path.dirname(workspace);
  if (fs.realpathSync(parent) !== parent) {throw new Error("Native SDK workspace parent must not use symlink aliases");}
  const existing = fs.lstatSync(workspace, {throwIfNoEntry: false});
  if (!existing) {
    fs.mkdirSync(workspace, {mode: 0o700});
    fs.writeFileSync(path.join(workspace, markerName), JSON.stringify({version: 1, checkout}) + "\n", {flag: "wx", mode: 0o600});
  } else {
    if (!existing.isDirectory() || existing.isSymbolicLink() || fs.realpathSync(workspace) !== workspace) {
      throw new Error("Native SDK workspace must be a real directory");
    }
    const marker = path.join(workspace, markerName), stat = fs.lstatSync(marker, {throwIfNoEntry: false});
    if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(marker) !== marker ||
        fs.readFileSync(marker, "utf8") !== JSON.stringify({version: 1, checkout}) + "\n") {
      throw new Error("Native SDK workspace is unmarked or belongs to another checkout; select an empty dedicated workspace");
    }
  }
  const sources = path.join(workspace, "sources"), sourceStat = fs.lstatSync(sources, {throwIfNoEntry: false});
  if (!sourceStat) {fs.mkdirSync(sources, {mode: 0o700});} else if (!sourceStat.isDirectory() || sourceStat.isSymbolicLink() || fs.realpathSync(sources) !== sources) {
    throw new Error("Native SDK workspace sources path must be a real directory");
  }
  return workspace;
}

/** @param {string} file @param {string} label */
function regularFile(file, label) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || fs.realpathSync(file) !== file) {throw new Error(label + " is missing or not a regular file: " + file);}
}

/** Bind every compiler source component to its expected, non-link directory before and after inventory admission.
 * @param {string} file @param {string} output */
function verifyCompilerSourceRoots(file, output) {
  verifyPreparedSourceRoots(file, output, "compiler");
}

/** @param {string} file @param {string} output @param {typeof verifyPreparedSdkSources} verify */
function verifyCompilerSources(file, output, verify) {
  verifyCompilerSourceRoots(file, output);
  verify(file, "compiler");
  verifyCompilerSourceRoots(file, output);
}

/** Use the exact Linux support-SDK tool sets maintained by the CI recipes. */
function localSupportTools() {
  return {hermes: ciSupport.hermesBuildTools(), cmake: fs.realpathSync("/usr/bin/cmake"), ninja: fs.realpathSync("/usr/bin/ninja"),
    openssl: ciSupport.linuxOpenSslTools(), zstd: ciSupport.linuxZstdTools(), openmp: ciSupport.openmpBuildTools()};
}

/** @param {string} workspace */
function supportPaths(workspace) {
  const compilerFetch = path.join(workspace, "compiler-fetch"), compilerSources = path.join(workspace, "compiler-sources");
  return {workspace, compilerFetch, compilerSources, linuxSources: path.join(workspace, "linux-sources"),
    buildInputs: path.join(workspace, "llvm-inputs"), llvmSdk: path.join(workspace, "llvm21"),
    dpcppCudaRoot: path.join(workspace, "dpcpp-cuda-inputs"),
    gnuHeadersRoot: path.join(workspace, "gnu-headers"),
    hermesSource: path.join(workspace, "hermes-fetch"), hermesSdk: path.join(workspace, "hermes-sdk"),
    ioSource: path.join(workspace, "io-fetch"), ioSdk: path.join(workspace, "io-sdk"), opensslSdk: path.join(workspace, "openssl-sdk"),
    opensslTools: path.join(workspace, "openssl-tools.json"), zstdSource: path.join(workspace, "zstd-fetch"),
    zstdSdk: path.join(workspace, "zstd-sdk"), zstdTools: path.join(workspace, "zstd-tools.json"),
    openmpSdk: path.join(workspace, "openmp-sdk"), openmpTools: path.join(workspace, "openmp-tools.json"),
    linuxJitResources: path.join(workspace, "linux-jit-resources"), linuxJitTools: path.join(workspace, "linux-jit-tools.json"),
    sysrootFetch: path.join(workspace, "sysroot-fetch"), sysroot: path.join(workspace, "linux-sysroot"),
    linuxJitAdmission: path.join(workspace, "linux-jit-admission.json")};
}

/** Fetch the pinned glibc sysroot packages once (networked, from the mirror release), prepare or
 * re-admit the sysroot inside the workspace, and return the environment that selects it for every
 * Linux producer. The sysroot tree is deterministic, so an existing one is admitted, never rebuilt.
 * @param {string} workspace @param {NodeJS.ProcessEnv} environment @param {typeof runCommand} [run] */
function ensureLocalLinuxSysroot(workspace, environment, run = runCommand) {
  const paths = supportPaths(workspace);
  if (!canonicalBuildDirectory(paths.sysrootFetch, "Linux sysroot package cohort", true)) {
    run(process.execPath, [producer("fetch-native-sdk-sources.js"), "sysroot", paths.sysrootFetch, "--allow-network"],
      {label: "pinned glibc sysroot package fetch", env: {...environment, MOM_AOT_BUILD_ONLY: "1"}, cwd: repo});
  }
  verifySysrootPackages(paths.sysrootFetch);
  if (!canonicalBuildDirectory(paths.sysroot, "Linux sysroot", true)) {prepareLinuxSysroot(paths.sysrootFetch, paths.sysroot);}
  return applyLinuxSysrootEnvironment({...environment, MOM_LINUX_SYSROOT: paths.sysroot});
}

/** Workspace paths shaped for the shared DPC++/CUDA producer and auditor.
 * @param {ReturnType<typeof supportPaths>} paths @returns {import("./native-dpcpp-cuda-inputs").DpcppCudaInputState} */
function localDpcppCudaState(paths) {
  const root = paths.dpcppCudaRoot;
  return {workspaceRoot: paths.workspace, root, name: path.basename(root), dpcppRoot: path.join(root, "dpcpp"), cudaRoot: path.join(root, "cuda"),
    manifest: path.join(root, "inputs.json"), cudaVersionEvidence: path.join(root, "cuda-nvcc-version.txt"),
    cudaProbeEvidence: path.join(root, "cuda-nvcc-probe.txt"), cudaCmakeSource: path.join(root, "cuda-cmake-probe"),
    cudaCmakeLists: path.join(root, "cuda-cmake-probe", "CMakeLists.txt"), cudaCmakeEvidence: path.join(root, "cuda-cmake-probe.json"),
    cudaCmakeBuild: path.join(paths.workspace, ".dpcpp-cuda-cmake-probe"),
    linux: {source: paths.compilerFetch, sourceName: path.basename(paths.compilerFetch),
      fetchReceipt: path.join(paths.compilerFetch, "fetch.json"), rootsFile: path.join(paths.compilerFetch, "roots.json"),
      linuxPrepared: paths.linuxSources, compilerPrepared: paths.compilerSources,
      preparedReceipt: path.join(paths.linuxSources, "sources.json"), compilerPreparedReceipt: path.join(paths.compilerSources, "sources.json")}};
}

/** Shape stable local paths for the maintained Linux JIT resource builder/auditor.
 * @param {ReturnType<typeof supportPaths>} paths @returns {import("./native-ci-linux-jit").LinuxJitResourceTreeState} */
function localLinuxJitResourceState(paths) {
  const root = paths.linuxJitResources, dpcpp = localDpcppCudaState(paths);
  return {root, emitter: path.join(root, "emitter"), bootstrap: path.join(root, "bootstrap"), bootstrapBuild: path.join(root, "bootstrap/build"),
    bootstrapOverlay: path.join(root, "bootstrap-overlay"), bootstrapOverlayEvidence: path.join(root, "bootstrap-overlay.json"),
    base: path.join(root, "base"), profile: path.join(root, "resource-profile.json"), resources: path.join(root, "resources"),
    configureEvidence: path.join(root, "bootstrap-configure.json"), linux: dpcpp.linux,
    dpcpp, gnu: {root: paths.gnuHeadersRoot}};
}

/** Installed vendor SDKs are prerequisites, never silently provisioned by the local bootstrap.
 * @param {{dpcpp:string,cuda:string}} [roots] */
function localDpcppCudaRoots(roots = {dpcpp: "/opt/dpcpp", cuda: "/usr/local/cuda-12.6"}) {
  if (!roots || !path.isAbsolute(roots.dpcpp) || !path.isAbsolute(roots.cuda)) {throw new Error("DPC++/CUDA roots must be absolute installed paths");}
  for (const [name, root] of Object.entries(roots)) {
    const stat = fs.lstatSync(root, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(root) !== root) {
      throw new Error(`Pinned ${name === "dpcpp" ? "DPC++" : "CUDA 12.6"} inputs are missing; install them explicitly with scripts/install-dev.sh --component dpcpp,cuda --jobs 4 --keep-workspace.`);
    }
  }
  return roots;
}

/** Reuse the compiler-fetch superset roots to prepare Linux runtime sources without changing
 * the already-admitted compiler source tree. All Git/tar subprocesses inherit the caller's scope.
 * @param {string} workspace @param {{environment?: NodeJS.ProcessEnv,
 *   assertFetch?: (...args: any[]) => unknown, verifyCompiler?: (file: string, output: string, verify: typeof verifyPreparedSdkSources) => unknown,
 *   prepare?: (target: string, input: Record<string, string>, output: string, environment: NodeJS.ProcessEnv) => unknown,
 *   verifyLinux?: (...args: any[]) => unknown}} [dependencies] */
function prepareOrAdmitLinuxSources(workspace, dependencies = {}) {
  canonicalBuildDirectory(workspace, "Native SDK workspace", true);
  const marker = path.join(workspace, markerName);
  regularFile(marker, "Native SDK workspace marker");
  if (fs.readFileSync(marker, "utf8") !== JSON.stringify({version: 1, checkout: repo}) + "\n") {
    throw new Error("Native SDK workspace marker does not bind this checkout");
  }
  const paths = supportPaths(workspace), state = {source: paths.compilerFetch,
      fetchReceipt: path.join(paths.compilerFetch, "fetch.json"), rootsFile: path.join(paths.compilerFetch, "roots.json"),
      linuxPrepared: paths.linuxSources, compilerPrepared: paths.compilerSources,
      preparedReceipt: path.join(paths.linuxSources, "sources.json"), compilerPreparedReceipt: path.join(paths.compilerSources, "sources.json")},
    env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"}, git = fs.realpathSync("/usr/bin/git"),
    assertFetch = dependencies.assertFetch || ciSources.assertSdkFetchProvenance,
    verifyCompiler = dependencies.verifyCompiler || verifyCompilerSources,
    prepare = dependencies.prepare || prepareSdkSources,
    verifyLinux = dependencies.verifyLinux || ciSources.verifyLinuxSources;
  fs.accessSync(git, fs.constants.X_OK);
  canonicalBuildDirectory(paths.compilerFetch, "Pinned compiler source roots", true);
  regularFile(state.fetchReceipt, "Pinned compiler source fetch receipt"); regularFile(state.rootsFile, "Pinned compiler source roots receipt");
  const receipt = readBuildJson(state.fetchReceipt), roots = readBuildJson(state.rootsFile);
  assertFetch(receipt, roots, "compiler", state.source);
  const plan = ciSources.linuxSourcePreparationPlan(/** @type {Parameters<typeof ciSources.linuxSourcePreparationPlan>[0]} */
      (/** @type {unknown} */ (state)), roots), linuxPlan = plan[0];
  if (!linuxPlan || linuxPlan.target !== "linux" || linuxPlan.output !== paths.linuxSources) {
    throw new Error("Pinned Linux source preparation plan does not bind the stable workspace output");
  }
  verifyCompiler(state.compilerPreparedReceipt, state.compilerPrepared, verifyPreparedSdkSources);
  if (canonicalBuildDirectory(paths.linuxSources, "Prepared Linux sources", true)) {
    verifyPreparedSourceRoots(state.preparedReceipt, paths.linuxSources, "linux");
    verifyLinux({env}, state, git);
  } else {
    prepare("linux", /** @type {Record<string, string>} */ (/** @type {unknown} */ (linuxPlan.roots)), linuxPlan.output, env);
    verifyPreparedSourceRoots(state.preparedReceipt, paths.linuxSources, "linux");
    verifyLinux({env}, state, git);
  }
  verifyCompiler(state.compilerPreparedReceipt, state.compilerPrepared, verifyPreparedSdkSources);
  verifyPreparedSourceRoots(state.preparedReceipt, paths.linuxSources, "linux");
  return paths.linuxSources;
}

/** Prepare the Windows-patched view from the same admitted compiler-fetch superset.
 * Keep it separate from Linux: their reviewed patch lists differ. Original fetch and
 * compiler receipts are never rewritten, and an occupied invalid output is preserved.
 * @param {string} workspace
 * @param {{environment?:NodeJS.ProcessEnv,admitLinux?:typeof prepareOrAdmitLinuxSources,
 * prepare?:typeof prepareSdkSources,verify?:typeof verifyPreparedSdkSources}} [dependencies] */
function prepareOrAdmitWindowsSources(workspace, dependencies = {}) {
  const admit = dependencies.admitLinux || prepareOrAdmitLinuxSources;
  admit(workspace, {environment: dependencies.environment || process.env});
  const paths = supportPaths(workspace), output = path.join(workspace, "windows-sources"),
    receipt = path.join(output, "sources.json"), rootsFile = path.join(paths.compilerFetch, "roots.json"),
    fetchFile = path.join(paths.compilerFetch, "fetch.json"),
    unchanged = [rootsFile, fetchFile, path.join(paths.compilerSources, "sources.json")].map(file => [file, fileHash(file)]),
    roots = readBuildJson(rootsFile), selected = Object.fromEntries(Object.keys(sdkComponents("windows")).map(name => [name, roots[name]])),
    env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    verify = dependencies.verify || verifyPreparedSdkSources;
  if (!canonicalBuildDirectory(output, "Prepared Windows sources", true)) {
    (dependencies.prepare || prepareSdkSources)("windows", selected, output, env);
  }
  verifyPreparedSourceRoots(receipt, output, "windows");
  verify(receipt, "windows");
  const receiptSha256 = fileHash(receipt);
  // Recheck the source cohort after preparation as well as before invoking Git/tar.
  admit(workspace, {environment: env});
  for (const [file, sha256] of unchanged) {
    if (!file || fileHash(file) !== sha256) {throw new Error("Original compiler source receipt changed during Windows preparation");}
  }
  verifyPreparedSourceRoots(receipt, output, "windows");
  verify(receipt, "windows");
  if (fileHash(receipt) !== receiptSha256) {throw new Error("Windows source receipt changed during admission");}
  return {workspace, root: output, receipt, receiptSha256};
}

/** Bind every prepared component to its expected real directory before and after source admission.
 * @param {string} file @param {string} output @param {"linux" | "windows" | "compiler"} target */
function verifyPreparedSourceRoots(file, output, target) {
  regularFile(file, "Prepared " + target + " source receipt");
  const receipt = /** @type {{version?: number, target?: string, status?: string, prepared?: Record<string, {root?: string}>}} */ (readBuildJson(file)),
    names = Object.keys(sdkComponents(target)).sort(), prepared = receipt?.prepared;
  if (receipt?.version !== 2 || receipt.target !== target || receipt.status !== "PASS" || !prepared ||
      !hasExactBuildFields(prepared, names)) {
    throw new Error("Prepared " + target + " source receipt has an unexpected component cohort");
  }
  for (const name of names) {
    const expected = path.join(output, name), stat = fs.lstatSync(expected, {throwIfNoEntry: false});
    if (prepared[name]?.root !== expected || !stat?.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(expected) !== expected) {
      throw new Error("Prepared " + target + " source root is not its expected real directory: " + name);
    }
  }
}

/** Prepare/admit source receipts around the shared DPC++/CUDA input producer or its stable cache.
 * @param {string} workspace
 * @param {{environment?:NodeJS.ProcessEnv, installedRoots?:{dpcpp:string,cuda:string},
 *   prepareSources?:(workspace:string)=>unknown, verifyLinux?:(paths:{env:NodeJS.ProcessEnv},state:import("./native-ci-linux-sources").LinuxPreparedSources,git:string)=>unknown,
 *   stageInputs?:(state:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState,roots:{dpcpp:string,cuda:string},
 *     origin:import("./native-dpcpp-cuda-inputs").PreparedDpcppCudaSourceOrigin,env:NodeJS.ProcessEnv)=>unknown,
 *   auditInputs?:(state:import("./native-dpcpp-cuda-inputs").DpcppCudaInputState,record:any)=>unknown}} [dependencies] */
function stageLocalCompilerInputs(workspace, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local DPC++/CUDA input staging currently requires Linux x64");}
  canonicalBuildDirectory(workspace, "Native SDK workspace", true);
  const root = admitWorkspace(workspace, repo), paths = supportPaths(root), state = localDpcppCudaState(paths),
    env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"}, git = fs.realpathSync("/usr/bin/git"),
    prepareSources = dependencies.prepareSources || prepareOrAdmitLinuxSources,
    verifyLinux = dependencies.verifyLinux || ciSources.verifyLinuxSources,
    stageInputs = dependencies.stageInputs || dpcppInputs.stageDpcppCudaInputs,
    auditInputs = dependencies.auditInputs || dpcppInputs.auditDpcppCudaInputs;
  fs.accessSync(git, fs.constants.X_OK);
  prepareSources(root);
  const admitSources = () => verifyLinux({env}, state.linux, git), audit = () => auditInputs(state, readBuildJson(state.manifest));
  admitSources();
  const reused = canonicalBuildDirectory(state.root, "Local DPC++/CUDA inputs", true);
  if (reused) {
    regularFile(state.manifest, "Local DPC++/CUDA input manifest");
    audit();
  } else {
    stageInputs(state, localDpcppCudaRoots(dependencies.installedRoots), dpcppInputs.preparedDpcppCudaSourceOrigin(state.linux, root), env);
  }
  admitSources(); audit();
  return {workspace: root, dpcppRoot: state.dpcppRoot, cudaRoot: state.cudaRoot, manifest: state.manifest, reused};
}

/** Write a stable build descriptor once; never rewrite a cache input or repair a missing receipt.
 * A recorded job count is parallelism provenance, so it may differ from this run's; the producer's audit bounds it.
 * @param {string} file @param {{version: number, jobs?: number}} value @param {boolean} outputExists */
function admitToolDescriptor(file, value, outputExists) {
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (stat) {
    regularFile(file, "Native SDK tool descriptor");
    const recorded = readBuildJson(file);
    if (!isDeepStrictEqual(recorded, value.jobs === undefined ? value : {...value, jobs: recorded?.jobs})) {
      throw new Error("Native SDK tool descriptor changed: " + file);
    }
    return recorded;
  }
  if (outputExists) {throw new Error("Native SDK output exists without its original tool descriptor; preserve it and investigate: " + file);}
  fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n", {flag: "wx", mode: 0o600});
  return value;
}

/** Re-admit one already-produced I/O SDK before proceeding to OpenSSL.
 * @param {{ioSource: string, ioSdk: string}} paths @param {{cmake: string, ninja: string, openssl: {cc: string}}} tools */
function verifyIoSdk(paths, tools) {
  const io = nativeIoInputs(paths.ioSdk, "linux"), receipt = readBuildJson(io.receipt),
    expectedInputs = {[path.join(paths.ioSource, "libuv-v1.51.0.tar.gz")]: ioPins.libuv.sha256,
      [path.join(paths.ioSource, "zlib-1.3.1.tar.gz")]: ioPins.zlib.sha256},
    expectedTools = Object.fromEntries([tools.cmake, tools.openssl.cc, tools.ninja].map(file => [file, fileHash(file)]));
  if (!isDeepStrictEqual(receipt.inputs, expectedInputs) || !isDeepStrictEqual(receipt.fingerprints, expectedTools) ||
      Object.entries(expectedInputs).some(([file, hash]) => fileHash(file) !== hash)) {
    throw new Error("Native I/O SDK does not bind this run's pinned source archives and exact tool paths");
  }
  return io;
}

/** Re-admit an existing OpenSSL SDK independently when a prior run stopped between the two producers.
 * @param {{buildInputs: string, opensslSdk: string, opensslTools: string}} paths @param {Record<string,string>} tools */
function verifyOpenSslSdk(paths, tools) {
  const sdk = nativeOpenSslInputs(paths.opensslSdk, "linux"), receipt = readBuildJson(sdk.receipt),
    descriptor = readBuildJson(paths.opensslTools), expectedTools = Object.fromEntries(Object.entries(tools).map(([name, file]) =>
      [name, {path: file, sha256: fileHash(file)}]));
  if (![2, 4].includes(descriptor?.jobs) || !isDeepStrictEqual(descriptor, {version: 1, jobs: descriptor.jobs, tools}) || receipt.archive?.path !== path.join(paths.buildInputs, "openssl-" + opensslPin.version + ".tar.gz") ||
      receipt.archive?.sha256 !== opensslPin.sha256 || receipt.config?.sha256 !== fileHash(paths.opensslTools) ||
      !isDeepStrictEqual(receipt.tools, expectedTools) || fileHash(receipt.archive.path) !== opensslPin.sha256) {
    throw new Error("Static OpenSSL SDK does not bind this run's pinned archive, descriptor and tools");
  }
  return sdk;
}

/** Run one existing CI admission helper in a bounded child scope. Tests may supply an audit seam.
 * @param {string} kind @param {ReturnType<typeof supportPaths>} paths
 * @param {(label: string, command: string, args: string[]) => unknown} invoke
 * @param {{auditSupport?: (kind: string, paths: ReturnType<typeof supportPaths>) => unknown}} dependencies */
function admitSupport(kind, paths, invoke, dependencies) {
  if (dependencies.auditSupport) {dependencies.auditSupport(kind, paths); return;}
  invoke(kind + " support SDK re-admission", process.execPath, [__filename, "--audit-support", kind, paths.workspace]);
}

/** Build or reuse the static support SDKs required by the eventual release graph.
 * The context must come from an already completed source-LLVM slice; this does not yet compose a release.
 * @param {{workspace: string, compilerFetch: string, compilerSources: string, buildInputs: string, llvmSdk: string}} context
 * @param {{run?: typeof runCommand, tools?: ReturnType<typeof localTools>, supportTools?: ReturnType<typeof localSupportTools>,
 *   resolveTools?: typeof localTools, versionOf?: typeof executableVersion, environment?: NodeJS.ProcessEnv,
 *   auditSupport?: (kind: string, paths: ReturnType<typeof supportPaths>) => unknown,
 *   verifyIo?: (paths: any, tools: any) => unknown, verifyOpenSsl?: (paths: any, tools: any) => unknown}} [dependencies] */
function buildLocalNativeSupportSdk(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local native support SDK bootstrap currently requires Linux x64");}
  if (!context || typeof context.workspace !== "string") {throw new Error("Native support SDK bootstrap needs an admitted source-LLVM context");}
  canonicalBuildDirectory(context.workspace, "Admitted native SDK workspace", true);
  const root = admitWorkspace(context.workspace, repo), paths = supportPaths(root);
  if (context.compilerFetch !== paths.compilerFetch || context.compilerSources !== paths.compilerSources ||
      context.buildInputs !== paths.buildInputs || context.llvmSdk !== paths.llvmSdk) {
    throw new Error("Native support SDK context must use this workspace's admitted compiler sources, LLVM inputs and SDK paths");
  }
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    runnerTools = dependencies.tools || (dependencies.resolveTools || localTools)(env, dependencies.versionOf || executableVersion),
    support = dependencies.supportTools || localSupportTools(),
    invoke = scopedInvoker(runnerTools, env, dependencies.run || runCommand),
    admitIo = dependencies.verifyIo || verifyIoSdk, admitOpenSsl = dependencies.verifyOpenSsl || verifyOpenSslSdk,
    archive = path.join(paths.buildInputs, archiveName), fetcher = producer("fetch-native-sdk-sources.js"),
    hermes = {source: paths.hermesSource, sdkRoot: paths.hermesSdk},
    hermesReused = canonicalBuildDirectory(paths.hermesSdk, "Hermes SDK", true),
    jobs = localBuildJobs(env), supportJobs = ciRecipeJobs(jobs);

  if (path.basename(support.hermes.cxx) !== "icpx" || path.basename(support.openmp.cxx) !== "icpx") {
    throw new Error("Native support SDK tools must preserve Intel's icpx invocation alias");
  }

  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  canonicalBuildDirectory(paths.buildInputs, "LLVM/OpenSSL source archive cohort", true); regularFile(archive, "Pinned LLVM 21.1.8 source archive");
  invoke("offline LLVM/OpenSSL source archive verification", process.execPath, [fetcher, "verify-build", paths.buildInputs]);
  invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]);
  invoke("pinned Linux source preparation and re-admission", process.execPath, [__filename, "--prepare-linux-sources", root]);

  if (!canonicalBuildDirectory(paths.hermesSource, "Hermes source fetch", true)) {
    invoke("pinned Hermes source fetch", process.execPath,
      [fetcher, "hermes", support.hermes.git, paths.hermesSource, "--allow-network"]);
  }
  admitSupport("hermes-source", paths, invoke, dependencies);
  if (hermesReused) {admitSupport("hermes-sdk", paths, invoke, dependencies);} else {
    const command = ciSupport.hermesBuildCommand(/** @type {Parameters<typeof ciSupport.hermesBuildCommand>[0]} */ (hermes), support.hermes, String(jobs), linuxSysrootToolchain(env));
    invoke("static Hermes SDK producer", process.execPath, [path.join(repo, command.script), ...command.args]);
    admitSupport("hermes-sdk", paths, invoke, dependencies);
  }

  if (!canonicalBuildDirectory(paths.ioSource, "I/O source archive cohort", true)) {
    invoke("pinned I/O source archive fetch", process.execPath, [fetcher, "io", paths.ioSource, "--allow-network"]);
  }
  invoke("offline I/O source archive verification", process.execPath, [fetcher, "verify-io", paths.ioSource]);
  const ioPaths = {env, source: paths.buildInputs, buildInputs: paths.buildInputs, ioSource: paths.ioSource, ioSdk: paths.ioSdk,
      opensslSdk: paths.opensslSdk, opensslTools: paths.opensslTools},
    opensslConfig = {version: 1, jobs: supportJobs, tools: support.openssl},
    ioCommands = ciSupport.ioOpenSslBuildCommands(/** @type {Parameters<typeof ciSupport.ioOpenSslBuildCommands>[0]} */ (/** @type {unknown} */ (ioPaths)),
      support.cmake, support.ninja, support.openssl, String(supportJobs));
  const ioReused = canonicalBuildDirectory(paths.ioSdk, "Native I/O SDK", true);
  if (ioReused) {admitIo(ioPaths, {cmake: support.cmake, ninja: support.ninja, openssl: support.openssl});} else {
    const command = ioCommands[0];
    if (!command) {throw new Error("CI I/O SDK producer command is missing");}
    invoke("static I/O SDK producer", process.execPath, [path.join(repo, command.script), ...command.args]);
    admitIo(ioPaths, {cmake: support.cmake, ninja: support.ninja, openssl: support.openssl});
  }
  const opensslReused = canonicalBuildDirectory(paths.opensslSdk, "Static OpenSSL SDK", true);
  admitToolDescriptor(paths.opensslTools, opensslConfig, opensslReused);
  if (opensslReused) {admitOpenSsl(ioPaths, support.openssl);} else {
    const command = ioCommands[1];
    if (!command) {throw new Error("CI OpenSSL SDK producer command is missing");}
    invoke("static OpenSSL SDK producer", process.execPath, [path.join(repo, command.script), ...command.args]);
    admitOpenSsl(ioPaths, support.openssl);
  }
  admitSupport("io-openssl", paths, invoke, dependencies);

  if (!canonicalBuildDirectory(paths.zstdSource, "zstd source archive cohort", true)) {
    invoke("pinned zstd source archive fetch", process.execPath, [fetcher, "zstd", paths.zstdSource, "--allow-network"]);
  }
  invoke("offline zstd source archive verification", process.execPath, [fetcher, "verify-zstd", paths.zstdSource]);
  const zstdConfig = {version: 1, jobs: 4, tools: support.zstd}, zstdReused = canonicalBuildDirectory(paths.zstdSdk, "static zstd SDK", true);
  admitToolDescriptor(paths.zstdTools, zstdConfig, zstdReused);
  if (zstdReused) {admitSupport("zstd-sdk", paths, invoke, dependencies);} else {
    const state = /** @type {Parameters<typeof ciSupport.zstdBuildCommand>[0]} */ ({source: paths.zstdSource,
        archive: path.join(paths.zstdSource, "zstd-" + zstdPin.version + ".tar.gz"), sdkRoot: paths.zstdSdk}),
      command = ciSupport.zstdBuildCommand(state, paths.zstdTools);
    invoke("static zstd SDK producer", process.execPath, [path.join(repo, command.script), ...command.args]);
    admitSupport("zstd-sdk", paths, invoke, dependencies);
  }

  const openmpConfig = {version: 1, jobs: supportJobs, tools: support.openmp}, openmpReused = canonicalBuildDirectory(paths.openmpSdk, "static OpenMP SDK", true);
  admitToolDescriptor(paths.openmpTools, openmpConfig, openmpReused);
  const openmpState = /** @type {Parameters<typeof ciSupport.openmpBuildCommand>[0]} */ ({source: paths.buildInputs, archive,
    sdkRoot: paths.openmpSdk, toolsFile: paths.openmpTools});
  if (openmpReused) {admitSupport("openmp-sdk", paths, invoke, dependencies);} else {
    const command = ciSupport.openmpBuildCommand(openmpState);
    invoke("static OpenMP SDK producer", process.execPath, [path.join(repo, command.script), ...command.args]);
    admitSupport("openmp-sdk", paths, invoke, dependencies);
  }
  return {workspace: root, linuxSources: paths.linuxSources, hermesSdk: paths.hermesSdk, ioSdk: paths.ioSdk, opensslSdk: paths.opensslSdk,
    zstdSdk: paths.zstdSdk, openmpSdk: paths.openmpSdk, reused: {hermes: hermesReused, io: ioReused,
      openssl: opensslReused, zstd: zstdReused, openmp: openmpReused}};
}

/** Stage the local DPC++/CUDA compiler inputs after the source/LLVM slice is admitted.
 * Installed vendor roots are prerequisites; this command never invokes install-dev.sh.
 * @param {{workspace:string,compilerFetch:string,compilerSources:string,buildInputs:string,llvmSdk:string}} context
 * @param {{run?:typeof runCommand,tools?:ReturnType<typeof localTools>,resolveTools?:typeof localTools,
 *   versionOf?:typeof executableVersion,environment?:NodeJS.ProcessEnv,installedRoots?:{dpcpp:string,cuda:string}}} [dependencies] */
function buildLocalCompilerInputs(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local compiler-input staging currently requires Linux x64");}
  if (!context || typeof context.workspace !== "string") {throw new Error("Compiler-input staging needs an admitted source-LLVM context");}
  canonicalBuildDirectory(context.workspace, "Admitted native SDK workspace", true);
  const root = admitWorkspace(context.workspace, repo), paths = supportPaths(root);
  if (context.compilerFetch !== paths.compilerFetch || context.compilerSources !== paths.compilerSources ||
      context.buildInputs !== paths.buildInputs || context.llvmSdk !== paths.llvmSdk) {
    throw new Error("Compiler-input context must use this workspace's admitted compiler sources, LLVM inputs and SDK paths");
  }
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    runnerTools = dependencies.tools || (dependencies.resolveTools || localTools)(env, dependencies.versionOf || executableVersion),
    invoke = scopedInvoker(runnerTools, env, dependencies.run || runCommand), archive = path.join(paths.buildInputs, archiveName),
    hadInputs = canonicalBuildDirectory(paths.dpcppCudaRoot, "Local DPC++/CUDA inputs", true);
  if (!hadInputs) {localDpcppCudaRoots(dependencies.installedRoots);}
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  canonicalBuildDirectory(paths.buildInputs, "LLVM/OpenSSL source archive cohort", true); regularFile(archive, "Pinned LLVM 21.1.8 source archive");
  invoke("offline LLVM/OpenSSL source archive verification", process.execPath,
    [producer("fetch-native-sdk-sources.js"), "verify-build", paths.buildInputs]);
  invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]);
  invoke("local DPC++/CUDA input staging and source re-admission", process.execPath, [__filename, "--stage-local-compiler-inputs", root]);
  return {workspace: root, dpcppRoot: path.join(paths.dpcppCudaRoot, "dpcpp"), cudaRoot: path.join(paths.dpcppCudaRoot, "cuda"),
    manifest: path.join(paths.dpcppCudaRoot, "inputs.json"), reused: hadInputs};
}

/** Build or audit the shared GNU target-header/copyright handoff without installing packages.
 * @param {{workspace:string,compilerFetch:string,compilerSources:string,buildInputs:string,llvmSdk:string}} context
 * @param {{run?:typeof runCommand,tools?:ReturnType<typeof localTools>,resolveTools?:typeof localTools,
 *   versionOf?:typeof executableVersion,environment?:NodeJS.ProcessEnv}} [dependencies] */
function buildLocalGnuHeaders(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local GNU target-header staging currently requires Linux x64");}
  if (!context || typeof context.workspace !== "string") {throw new Error("GNU header staging needs an admitted source-LLVM context");}
  canonicalBuildDirectory(context.workspace, "Admitted native SDK workspace", true);
  const root = admitWorkspace(context.workspace, repo), paths = supportPaths(root);
  if (context.compilerFetch !== paths.compilerFetch || context.compilerSources !== paths.compilerSources ||
      context.buildInputs !== paths.buildInputs || context.llvmSdk !== paths.llvmSdk) {
    throw new Error("GNU header context must use this workspace's admitted compiler sources, LLVM inputs and SDK paths");
  }
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    runnerTools = dependencies.tools || (dependencies.resolveTools || localTools)(env, dependencies.versionOf || executableVersion),
    invoke = scopedInvoker(runnerTools, env, dependencies.run || runCommand), hadHeaders = canonicalBuildDirectory(paths.gnuHeadersRoot, "Local GNU target headers", true),
    target = "x86_64-unknown-linux-gnu", script = producer("prepare-native-gnu-headers.js");
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]);
  invoke(hadHeaders ? "GNU target-header handoff re-admission" : "GNU target-header handoff producer", process.execPath,
    [script, hadHeaders ? "audit" : "build", paths.gnuHeadersRoot, target]);
  return {workspace: root, root: paths.gnuHeadersRoot, target, reused: hadHeaders};
}

/** Pin the exact tools chosen by the caller while keeping Intel's icpx path spelling. @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools */
function localJitToolsReceipt(tools) {
  const rows = Object.entries(tools).sort(([left], [right]) => left.localeCompare(right));
  if (rows.map(([name]) => name).join(",") !== "cc,cmake,cxx,ninja" || rows.some(([, file]) => !path.isAbsolute(file) || path.resolve(file) !== file)) {
    throw new Error("Local JIT resource tools must be four canonical absolute executable paths");
  }
  return {version: 1, tools: Object.fromEntries(rows.map(([name, file]) => {
    const stat = fs.statSync(file);
    fs.accessSync(file, fs.constants.X_OK);
    if (!stat.isFile()) {throw new Error("Local JIT resource tool is not a regular executable: " + name);}
    return [name, {path: file, sha256: fileHash(file)}];
  }))};
}

/** Bind local cache reuse to the shared read-only auditor's complete fingerprint set.
 * @param {ReturnType<typeof localLinuxJitResourceState>} state @param {ReturnType<typeof localJitToolsReceipt>} tools
 * @param {ReturnType<typeof ciJit.auditLinuxJitResourceTree>} admission */
function localJitAdmissionReceipt(state, tools, admission) {
  const fingerprints = admission?.fingerprints;
  if (admission?.resources?.target !== "x86_64-unknown-linux-gnu" || admission.resources.directory !== state.resources ||
      !/^[a-f0-9]{64}$/.test(admission.resources.manifestHash || "") || !/^\d{8}$/.test(admission.sourceDate || "") ||
      !fingerprints || typeof fingerprints !== "object" || Array.isArray(fingerprints) || Object.keys(fingerprints).length < 1 ||
      Object.keys(fingerprints).length > 65536 ||
      Object.entries(fingerprints).some(([file, hash]) => !path.isAbsolute(file) || path.resolve(file) !== file ||
        typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash))) {
    throw new Error("Shared Linux JIT auditor returned an incomplete local admission");
  }
  const fingerprintRows = Object.entries(fingerprints).sort(([left], [right]) => left.localeCompare(right)),
    fingerprintSha256 = createHash("sha256").update(JSON.stringify(fingerprintRows)).digest("hex");
  return {version: 1, root: state.root, target: admission.resources.target, sourceDate: admission.sourceDate,
    manifestHash: admission.resources.manifestHash, tools, fingerprintCount: fingerprintRows.length, fingerprintSha256};
}

/** Invoke the shared builder/auditor only from the bounded internal command. @param {"build"|"audit"} mode
 * @param {string} workspace @param {{cmake:string,ninja:string,cc:string,cxx:string}} tools
 * @param {{buildTree?:typeof ciJit.buildLinuxJitResourceTree,auditTree?:typeof ciJit.auditLinuxJitResourceTree,
 *   validateTools?:(tools:{cmake:string,ninja:string,cc:string,cxx:string})=>unknown,verifyScope?:()=>unknown}} [dependencies] */
function runLocalJitResourceChild(mode, workspace, tools, dependencies = {}) {
  if (!new Set(["build", "audit"]).has(mode)) {throw new Error("Invalid local Linux JIT resource operation");}
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Linux JIT resource admission requires Linux x64");}
  (dependencies.verifyScope || verifyCurrentMemoryScope)();
  canonicalBuildDirectory(workspace, "Native SDK workspace", true);
  const root = admitWorkspace(workspace, repo), paths = supportPaths(root), state = localLinuxJitResourceState(paths),
    env = {...process.env, MOM_AOT_BUILD_ONLY: "1"};
  (dependencies.validateTools || (selected => {
    const expectedCc = path.join(oneapiCompilerRoot, "2026.2/bin/icx"), expectedCxx = path.join(oneapiCompilerRoot, "2026.2/bin/icpx");
    if (selected.cc !== expectedCc || selected.cxx !== expectedCxx || path.basename(selected.cxx) !== "icpx") {
      throw new Error("Local JIT resources require the pinned Intel 2026.2 icx/icpx compiler paths");
    }
    for (const file of [selected.cmake, selected.ninja, selected.cc, selected.cxx]) {
      if (!path.isAbsolute(file) || path.resolve(file) !== file || !fs.statSync(file).isFile()) {
        throw new Error("Local JIT resource tool is missing or not canonical: " + file);
      }
      fs.accessSync(file, fs.constants.X_OK);
    }
  }))(tools);
  const expectedTools = localJitToolsReceipt(tools), expectedToolFile = paths.linuxJitTools, admissionFile = paths.linuxJitAdmission,
    toolsStat = fs.lstatSync(expectedToolFile, {throwIfNoEntry: false}), receiptStat = fs.lstatSync(admissionFile, {throwIfNoEntry: false}),
    outputExists = canonicalBuildDirectory(state.root, "Local Linux JIT resources", true);
  if (!toolsStat?.isFile() || toolsStat.isSymbolicLink() || toolsStat.nlink !== 1 || fs.realpathSync(expectedToolFile) !== expectedToolFile ||
      !isDeepStrictEqual(readBuildJson(expectedToolFile), expectedTools)) {
    throw new Error("Local Linux JIT tool descriptor is missing or changed; preserve the cache and investigate");
  }
  if (mode === "build" && (outputExists || receiptStat) || mode === "audit" && (!outputExists || !receiptStat?.isFile() || receiptStat.isSymbolicLink())) {
    throw new Error("Local Linux JIT output and admission receipt are incomplete or out of order; preserve them and investigate");
  }
  const buildTree = dependencies.buildTree || ciJit.buildLinuxJitResourceTree,
    auditTree = dependencies.auditTree || ciJit.auditLinuxJitResourceTree;
  const admission = mode === "build" ? buildTree(state, tools, env) : auditTree(state, tools, env),
    expectedReceipt = localJitAdmissionReceipt(state, expectedTools, admission);
  if (mode === "build") {
    fs.writeFileSync(admissionFile, JSON.stringify(expectedReceipt, null, 2) + "\n", {flag: "wx", mode: 0o600});
  } else {
    const stat = fs.lstatSync(admissionFile);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(admissionFile) !== admissionFile ||
        !isDeepStrictEqual(readBuildJson(admissionFile), expectedReceipt)) {
      throw new Error("Local Linux JIT resource fingerprints changed; preserve the cache and investigate");
    }
  }
  return expectedReceipt;
}

/** Build or re-admit Linux CPU JIT resources from the already-maintained source, input, and GNU recipes.
 * @param {{workspace:string,compilerFetch:string,compilerSources:string,buildInputs:string,llvmSdk:string}} context
 * @param {{run?:typeof runCommand,tools?:ReturnType<typeof localTools>,resolveTools?:typeof localTools,versionOf?:typeof executableVersion,
 * environment?:NodeJS.ProcessEnv,installedRoots?:{dpcpp:string,cuda:string},
 * prepareInputs?:(context:any,dependencies:any)=>unknown,prepareHeaders?:(context:any,dependencies:any)=>unknown}} [dependencies] */
function buildLocalLinuxJitResources(context, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local Linux JIT resource bootstrap currently requires Linux x64");}
  if (!context || typeof context.workspace !== "string") {throw new Error("Linux JIT resource bootstrap needs an admitted source-LLVM context");}
  canonicalBuildDirectory(context.workspace, "Admitted native SDK workspace", true);
  const root = admitWorkspace(context.workspace, repo), paths = supportPaths(root);
  if (context.compilerFetch !== paths.compilerFetch || context.compilerSources !== paths.compilerSources ||
      context.buildInputs !== paths.buildInputs || context.llvmSdk !== paths.llvmSdk) {
    throw new Error("Linux JIT context must use this workspace's admitted compiler sources, LLVM inputs and SDK paths");
  }
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"},
    runnerTools = dependencies.tools || (dependencies.resolveTools || localTools)(env, dependencies.versionOf || executableVersion),
    tools = {cmake: runnerTools.cmake, ninja: runnerTools.ninja, cc: runnerTools.cc, cxx: runnerTools.cxx},
    invoke = scopedInvoker(runnerTools, env, dependencies.run || runCommand),
    prepareDependencies = {run: dependencies.run, tools: runnerTools, environment: env, installedRoots: dependencies.installedRoots};
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]);
  const inputs = (dependencies.prepareInputs || buildLocalCompilerInputs)(context, prepareDependencies),
    headers = (dependencies.prepareHeaders || buildLocalGnuHeaders)(context, prepareDependencies);
  if (!inputs || typeof inputs !== "object" || !headers || typeof headers !== "object") {
    throw new Error("Local Linux JIT prerequisites did not return admitted input records");
  }
  const inputRecord = /** @type {{workspace?:string,dpcppRoot?:string,cudaRoot?:string,manifest?:string}} */ (inputs),
    headerRecord = /** @type {{workspace?:string,root?:string,target?:string}} */ (headers);
  if (inputRecord.workspace !== root || inputRecord.dpcppRoot !== path.join(paths.dpcppCudaRoot, "dpcpp") ||
      inputRecord.cudaRoot !== path.join(paths.dpcppCudaRoot, "cuda") || inputRecord.manifest !== path.join(paths.dpcppCudaRoot, "inputs.json") ||
      headerRecord.workspace !== root || headerRecord.root !== paths.gnuHeadersRoot || headerRecord.target !== "x86_64-unknown-linux-gnu") {
    throw new Error("Local Linux JIT inputs were not admitted at their stable workspace paths");
  }
  const descriptor = localJitToolsReceipt(tools), hadResources = canonicalBuildDirectory(paths.linuxJitResources, "Local Linux JIT resources", true),
    admissionStat = fs.lstatSync(paths.linuxJitAdmission, {throwIfNoEntry: false});
  if (!hadResources && admissionStat) {throw new Error("Local JIT admission receipt exists without its resource output; preserve the workspace and investigate");}
  admitToolDescriptor(paths.linuxJitTools, descriptor, hadResources);
  invoke(hadResources ? "Linux JIT resource tree re-admission" : "Linux JIT resource tree producer", process.execPath,
    [__filename, "--local-jit-resources", hadResources ? "audit" : "build", root, tools.cmake, tools.ninja, tools.cc, tools.cxx]);
  return {workspace: root, root: paths.linuxJitResources, target: "x86_64-unknown-linux-gnu", reused: hadResources};
}

/** Execute existing producer verifiers from the bounded support-audit subprocess. @param {string} kind @param {string} workspace */
function auditSupportInProcess(kind, workspace) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Native support SDK admission currently requires Linux x64");}
  canonicalBuildDirectory(workspace, "Native SDK workspace", true);
  const marker = path.join(workspace, markerName);
  regularFile(marker, "Native SDK workspace marker");
  if (fs.readFileSync(marker, "utf8") !== JSON.stringify({version: 1, checkout: repo}) + "\n") {
    throw new Error("Native SDK workspace marker does not bind this checkout");
  }
  const paths = supportPaths(workspace), tools = localSupportTools(), env = {...process.env, MOM_AOT_BUILD_ONLY: "1"},
    hermes = /** @type {Parameters<typeof ciSupport.auditHermesSdk>[0]} */ ({source: paths.hermesSource, sdkRoot: paths.hermesSdk}),
    zstd = {source: paths.zstdSource, archive: path.join(paths.zstdSource, "zstd-" + zstdPin.version + ".tar.gz"),
      sdkRoot: paths.zstdSdk, toolsFile: paths.zstdTools},
    openmp = {source: paths.buildInputs, archive: path.join(paths.buildInputs, archiveName), sdkRoot: paths.openmpSdk, toolsFile: paths.openmpTools};
  switch (kind) {
    case "compiler-sources": verifyCompilerSources(path.join(paths.compilerSources, "sources.json"), paths.compilerSources, verifyPreparedSdkSources); break;
    case "llvm-sdk": verifyLinuxLlvmAdmission(paths.llvmSdk); break;
    case "hermes-source": ciSources.verifyHermesSource(/** @type {Parameters<typeof ciSources.verifyHermesSource>[0]} */ (/** @type {unknown} */ ({env})), hermes, tools.hermes.git); break;
    case "hermes-sdk": ciSources.verifyHermesSource(/** @type {Parameters<typeof ciSources.verifyHermesSource>[0]} */ (/** @type {unknown} */ ({env})), hermes, tools.hermes.git); ciSupport.auditHermesSdk(hermes, tools.hermes,
      String(recordedParallelJobs(readBuildJson(path.join(paths.hermesSdk, "before.json")).compile)), linuxSysrootToolchain(env)); break;
    case "io-sdk":
      verifyIoArchives(paths.ioSource);
      verifyIoSdk({ioSource: paths.ioSource, ioSdk: paths.ioSdk}, {cmake: tools.cmake, ninja: tools.ninja, openssl: tools.openssl});
      break;
    case "openssl-sdk": verifyOpenSslSdk(paths, tools.openssl); break;
    case "io-openssl": ciSupport.auditIoOpenSsl(/** @type {Parameters<typeof ciSupport.auditIoOpenSsl>[0]} */ (/** @type {unknown} */ ({env, source: paths.buildInputs,
      ioSource: paths.ioSource, ioSdk: paths.ioSdk, opensslSdk: paths.opensslSdk, opensslTools: paths.opensslTools}))); break;
    case "zstd-sdk": verifyZstdArchive(paths.zstdSource); ciSupport.auditZstdSdk(zstd, tools.zstd, paths.zstdTools); break;
    case "openmp-sdk": verifyBuildArchives(paths.buildInputs); ciSupport.auditOpenmpSdk(/** @type {Parameters<typeof ciSupport.auditOpenmpSdk>[0]} */ (/** @type {unknown} */ (openmp)), tools.openmp); break;
    default: throw new Error("Unknown local support SDK audit: " + kind);
  }
}

/** Parse the single stable-workspace option shared by the explicit follow-up commands.
 * @param {string[]} args @param {string} command */
function workspaceOption(args, command) {
  let workspace;
  for (let index = 0; index < args.length; index++) {
    if (args[index] !== "--workspace" || !args[index + 1]) {
      throw new Error("Unknown or incomplete " + command + " option: " + args[index]);
    }
    workspace = args[index + 1]; index++;
  }
  return workspace;
}

/** @param {string} file @param {string} output */
function readConfigureState(file, output) {
  if (!fs.lstatSync(file, {throwIfNoEntry: false})) {
    throw new Error("LLVM configure did not leave a public configured receipt; its partial output is preserved: " + path.dirname(file));
  }
  regularFile(file, "LLVM configure receipt");
  const stat = fs.statSync(file);
  if (stat.size > (1 << 20)) {throw new Error("LLVM configure receipt exceeds its size bound");}
  const receipt = /** @type {{version?: number, role?: string, stage?: string,
    plan?: {output?: string, prefix?: string, phases?: {name?: string, targets?: string[], artifacts?: string[]}[]}}} */
    (JSON.parse(fs.readFileSync(file, "utf8")));
  const phases = receipt?.plan?.phases;
  if (receipt?.version !== 1 || receipt.role !== "linux-llvm21-source-sdk" || receipt.stage !== "configured" ||
      receipt.plan?.output !== output || receipt.plan?.prefix !== path.join(output, "sdk") ||
      !Array.isArray(phases) || phases.length < 1 || phases.length > 64 || phases.some((row, index) => {
    const targets = row && Array.isArray(row.targets) ? /** @type {unknown[]} */ (row.targets) : [],
      artifacts = row && Array.isArray(row.artifacts) ? /** @type {unknown[]} */ (row.artifacts) : [];
    return !row || typeof row.name !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(row.name) ||
        !Array.isArray(targets) || targets.length < 1 || targets.length > 4096 ||
        targets.some(target => typeof target !== "string" || !/^[A-Za-z0-9_.+-]{1,160}$/.test(target)) ||
        new Set(targets).size !== targets.length || !Array.isArray(artifacts) || artifacts.length !== targets.length ||
        artifacts.some(artifact => typeof artifact !== "string" || !path.isAbsolute(artifact)) ||
        phases.findIndex(item => item?.name === row.name) !== index;
  })) {
    throw new Error("LLVM producer did not emit a valid generated build-phase list");
  }
  return {phases: phases.map(row => /** @type {string} */ (row.name)), prefix: /** @type {string} */ (receipt.plan.prefix)};
}

/** Resolve the current process into its mounted cgroup-v2 scope. @param {string} cgroup @param {string} mounts */
function cgroupMemoryPaths(cgroup, mounts) {
  const membership = cgroup.split(/\r?\n/).find(line => line.startsWith("0::"))?.slice(3),
    mount = mounts.split(/\r?\n/).map(line => {
      const parts = line.split(" - ");
      if (parts.length !== 2 || !parts[1]?.startsWith("cgroup2 ")) {return null;}
      const before = parts[0]?.split(" ");
      return before?.[3] && before[4] ? {root: before[3], point: before[4]} : null;
    }).find(Boolean);
  if (!membership?.startsWith("/") || !mount?.root?.startsWith("/") || !mount.point?.startsWith("/")) {
    throw new Error("Cannot resolve a unified cgroup-v2 memory scope");
  }
  /** @param {string} value */
  const decode = value => value.replace(/\\(040|011|012|134)/g, (/** @type {string} */ _match, /** @type {string} */ octal) => String.fromCharCode(parseInt(octal, 8))),
    root = decode(mount.root), point = decode(mount.point), member = membership;
  if ([root, point, member].some(value => path.posix.normalize(value) !== value)) {
    throw new Error("Current cgroup-v2 paths are not canonical");
  }
  if (member !== root && !(root === "/" ? member.startsWith("/") : member.startsWith(root + "/"))) {
    throw new Error("Current cgroup is outside the mounted cgroup-v2 subtree");
  }
  const relative = root === "/" ? member.slice(1) : path.posix.relative(root, member), scope = path.posix.resolve(point, relative),
    escaped = path.posix.relative(point, scope);
  if (path.posix.isAbsolute(escaped) || escaped === ".." || escaped.startsWith(".." + path.posix.sep)) {
    throw new Error("Current cgroup path escaped its mounted cgroup-v2 tree");
  }
  return {memory: path.posix.join(scope, "memory.max"), swap: path.posix.join(scope, "memory.swap.max")};
}

/** @param {string} memory @param {string} swap @param {number} maximum the selected scope cap in bytes */
function verifyMemoryLimits(memory, swap, maximum) {
  if (!/^\d+$/.test(memory) || BigInt(memory) < 1n || BigInt(memory) > BigInt(maximum) || swap !== "0") {
    throw new Error(`The active systemd scope did not enforce memory.max <= ${maximum / gib} GiB and memory.swap.max = 0`);
  }
}

/** Read the process's own scope limits so the transient systemd settings are proven, not assumed. */
function verifyCurrentMemoryScope() {
  const cgroup = fs.readFileSync("/proc/self/cgroup", "utf8"), mounts = fs.readFileSync("/proc/self/mountinfo", "utf8"),
    files = cgroupMemoryPaths(cgroup, mounts), memory = fs.readFileSync(files.memory, "utf8").trim(),
    swap = fs.readFileSync(files.swap, "utf8").trim();
  verifyMemoryLimits(memory, swap, localMemoryMax());
  console.log("[native-sdk] cgroup-v2 memory.max=" + memory + " memory.swap.max=" + swap);
}

/** @param {string} command @param {string[]} args @param {{label: string, env: NodeJS.ProcessEnv, cwd: string}} options */
function runCommand(command, args, options) {
  const result = spawnSync(command, args, {cwd: options.cwd, env: options.env, stdio: "inherit"});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error(`${options.label} failed (${result.status ?? "no exit code"}${result.signal ? ", " + result.signal : ""})` , {cause: result.error});
  }
  return result;
}

/** Keep acquisition, producers, and re-admission inside the same effective cgroup limit.
 * @param {{systemdRun: string}} tools @param {NodeJS.ProcessEnv} env @param {typeof runCommand} run
 * @param {typeof verifyCurrentMemoryScope} [verifyScope] */
function scopedInvoker(tools, env, run, verifyScope = verifyCurrentMemoryScope) {
  return (/** @type {string} */ label, /** @type {string} */ command, /** @type {string[]} */ args) => {
    // User-manager scopes can be siblings: keep the full command's children inside its
    // already enforced aggregate cap instead of giving each another independent one.
    if (env["MOM_LOCAL_NATIVE_SCOPE"] === "1") {
      verifyScope();
      return run(command, args, {label, env, cwd: repo});
    }
    const launch = ["--user", "--scope", "--quiet", "--property=MemoryMax=" + localMemoryMax(env),
      "--property=MemorySwapMax=0", "--", command, ...args];
    return run(tools.systemdRun, launch, {label, env, cwd: repo});
  };
}

/** Run the pinned source and LLVM producers for the first Linux bootstrap slice only.
 * @param {{workspace?: string}} [options]
 * @param {{run?: typeof runCommand, tools?: ReturnType<typeof localTools>, resolveTools?: typeof localTools,
 *   versionOf?: typeof executableVersion, environment?: NodeJS.ProcessEnv}} [dependencies] */
function buildLocalNativeSdk(options = {}, dependencies = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Local LLVM source bootstrap currently requires Linux x64");}
  const env = {...(dependencies.environment || process.env), MOM_AOT_BUILD_ONLY: "1"}, jobs = localBuildJobs(env),
    workspace = options.workspace || defaultWorkspace(), root = admitWorkspace(workspace, repo),
    tools = dependencies.tools || (dependencies.resolveTools || localTools)(env, dependencies.versionOf || executableVersion),
    invoke = scopedInvoker(tools, env, dependencies.run || runCommand);

  // A scope runs synchronously; its child must prove the effective cgroup-v2 limits before acquisition.
  invoke("systemd memory-limit preflight", process.execPath, [__filename, "--memory-probe"]);
  const compilerFetch = path.join(root, "compiler-fetch"), compilerSources = path.join(root, "compiler-sources"),
    buildInputs = path.join(root, "llvm-inputs"), llvmOutput = path.join(root, "llvm21"),
    sourceReceipt = path.join(compilerSources, "sources.json");
  let sourcesReused = false;
  if (canonicalBuildDirectory(compilerSources, "Prepared compiler sources", true)) {
    regularFile(sourceReceipt, "Prepared compiler source receipt");
    verifyCompilerSourceRoots(sourceReceipt, compilerSources);
    invoke("compiler source re-admission", process.execPath, [__filename, "--audit-support", "compiler-sources", root]);
    verifyCompilerSourceRoots(sourceReceipt, compilerSources); sourcesReused = true;
  } else {
    if (!canonicalBuildDirectory(compilerFetch, "Compiler source fetch", true)) {
      invoke("pinned compiler source fetch", process.execPath, [producer("fetch-native-sdk-sources.js"), "compiler", tools.git, compilerFetch, "--allow-network"]);
    }
    regularFile(path.join(compilerFetch, "roots.json"), "Fetched compiler source roots");
    invoke("offline compiler source preparation", process.execPath,
      [producer("prepare-native-sdk-sources.js"), "compiler", path.join(compilerFetch, "roots.json"), compilerSources]);
    regularFile(sourceReceipt, "Prepared compiler source receipt");
    verifyCompilerSourceRoots(sourceReceipt, compilerSources);
    invoke("compiler source re-admission", process.execPath, [__filename, "--audit-support", "compiler-sources", root]);
    verifyCompilerSourceRoots(sourceReceipt, compilerSources);
  }

  let llvmReused = false, llvmResumed = false;
  if (canonicalBuildDirectory(llvmOutput, "LLVM output", true)) {
    const result = path.join(llvmOutput, "result.json");
    if (fs.lstatSync(result, {throwIfNoEntry: false})) {
      regularFile(result, "Completed LLVM admission receipt");
      invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]); llvmReused = true;
    } else {
      const configured = readConfigureState(path.join(llvmOutput, "configured.json"), llvmOutput),
        buildComplete = path.join(llvmOutput, "build-complete.json"), installReceipt = path.join(llvmOutput, "install.json");
      let missing = false;
      const phasesToBuild = [];
      // Public phase commands re-admit the configure receipt and every preceding immutable phase.
      for (const phase of configured.phases) {
        const phaseFile = path.join(llvmOutput, "phase-" + phase + ".json"), exists = fs.lstatSync(phaseFile, {throwIfNoEntry: false});
        if (exists && missing) {throw new Error("LLVM interrupted build phases are not a contiguous prefix; partial output preserved: " + llvmOutput);}
        if (exists) {regularFile(phaseFile, "LLVM phase receipt");} else {missing = true; phasesToBuild.push(phase);}
      }
      if (missing && fs.lstatSync(buildComplete, {throwIfNoEntry: false})) {
        throw new Error("LLVM build-complete receipt exists with a missing phase; partial output preserved: " + llvmOutput);
      }
      for (const phase of phasesToBuild) {
        invoke("LLVM 21 build phase " + phase, process.execPath,
          [producer("build-native-llvm-sdk.js"), "build", llvmOutput, phase]);
      }
      if (!fs.lstatSync(buildComplete, {throwIfNoEntry: false})) {
        throw new Error("LLVM final phase receipt is incomplete; partial output preserved: " + llvmOutput);
      }
      regularFile(buildComplete, "LLVM build-complete receipt");
      if (fs.lstatSync(installReceipt, {throwIfNoEntry: false})) {regularFile(installReceipt, "LLVM install receipt");} else {
        if (fs.lstatSync(configured.prefix, {throwIfNoEntry: false})) {
          throw new Error("LLVM install prefix exists without its receipt; partial output preserved: " + llvmOutput);
        }
        invoke("LLVM 21 install", process.execPath, [producer("build-native-llvm-sdk.js"), "install", llvmOutput]);
      }
      invoke("LLVM 21 audit", process.execPath, [producer("build-native-llvm-sdk.js"), "audit", llvmOutput]);
      invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]); llvmResumed = true;
    }
  } else {
    if (!canonicalBuildDirectory(buildInputs, "LLVM source archive cohort", true)) {
      invoke("pinned LLVM source archive fetch", process.execPath, [producer("fetch-native-sdk-sources.js"), "build", buildInputs, "--allow-network"]);
    }
    invoke("offline LLVM source archive verification", process.execPath, [producer("fetch-native-sdk-sources.js"), "verify-build", buildInputs]);
    const archive = path.join(buildInputs, archiveName);
    regularFile(archive, "Pinned LLVM 21.1.8 source archive");
    invoke("LLVM 21 configure", process.execPath, [producer("build-native-llvm-sdk.js"), "configure", archive, llvmOutput,
      tools.cmake, tools.ninja, tools.cc, tools.cxx, String(jobs)]);
    for (const phase of readConfigureState(path.join(llvmOutput, "configured.json"), llvmOutput).phases) {
      invoke("LLVM 21 build phase " + phase, process.execPath, [producer("build-native-llvm-sdk.js"), "build", llvmOutput, phase]);
    }
    invoke("LLVM 21 install", process.execPath, [producer("build-native-llvm-sdk.js"), "install", llvmOutput]);
    invoke("LLVM 21 audit", process.execPath, [producer("build-native-llvm-sdk.js"), "audit", llvmOutput]);
    invoke("LLVM 21 source SDK re-admission", process.execPath, [__filename, "--audit-support", "llvm-sdk", root]);
  }
  return {workspace: root, compilerFetch, compilerSources, buildInputs, llvmSdk: llvmOutput, sourcesReused, llvmReused, llvmResumed, jobs, memoryMax: localMemoryMax(env)};
}

/** @param {string[]} argv @param {()=>unknown} [verifyScope] */
function main(argv, verifyScope = verifyCurrentMemoryScope) {
  const [command, ...args] = argv;
  if (command === "--memory-probe" && args.length === 0) {verifyScope(); return;}
  if (command === "--prepare-linux-sources") {
    if (args.length !== 1 || !args[0]) {throw new Error("Invalid internal Linux source preparation request");}
    verifyScope();
    prepareOrAdmitLinuxSources(args[0]); return;
  }
  if (command === "--stage-local-compiler-inputs") {
    if (args.length !== 1 || !args[0]) {throw new Error("Invalid internal compiler-input staging request");}
    verifyScope();
    stageLocalCompilerInputs(args[0]); return;
  }
  if (command === "--local-jit-resources") {
    if (args.length !== 6 || args.some(value => !value)) {throw new Error("Invalid internal Linux JIT resource request");}
    const [mode, workspace, cmake, ninja, cc, cxx] = args;
    runLocalJitResourceChild(/** @type {"build"|"audit"} */ (mode), workspace || "", {cmake: cmake || "", ninja: ninja || "", cc: cc || "", cxx: cxx || ""});
    return;
  }
  if (command === "--audit-support") {
    if (args.length !== 2 || !args[0] || !args[1]) {throw new Error("Invalid internal native support SDK audit request");}
    verifyScope();
    auditSupportInProcess(args[0], args[1]); return;
  }
  if (command === "--help" || command === "-h") {
    console.log("Usage: node scripts/build-local-native-sdk.js source-llvm [--workspace /absolute/path]\n" +
      "       node scripts/build-local-native-sdk.js support [--workspace /absolute/path]\n" +
      "       node scripts/build-local-native-sdk.js compiler-inputs [--workspace /absolute/path]\n" +
      "       node scripts/build-local-native-sdk.js gnu-headers [--workspace /absolute/path]\n" +
      "       node scripts/build-local-native-sdk.js jit-resources [--workspace /absolute/path]\n" +
      "Builds the explicit Linux source/LLVM slice, then the separate support slice (Hermes, I/O/OpenSSL, zstd, OpenMP).\n" +
      "compiler-inputs stages the installed DPC++/CUDA 12.6 roots after source-LLVM and source preparation.\n" +
      "gnu-headers stages installed Linux GNU target headers and package notices without installing packages.\n" +
      "jit-resources builds or re-admits the CPU-only source-JIT resource pack from those inputs.\n" +
      "No command builds all release SDKs or packages a release; follow-up slices require an admitted source-LLVM workspace.\n" +
      "MOM_BUILD_JOBS (1..16, default 2) selects build parallelism and sizes the memory scope.");
    return;
  }
  if (command === "support") {
    const root = workspaceOption(args, command) || defaultWorkspace(), paths = supportPaths(root),
      result = buildLocalNativeSupportSdk({workspace: root, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
        buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk});
    console.log(`[native-sdk] Linux static support SDKs admitted at ${result.workspace}; remaining release SDKs and packaging were not run.`);
    return;
  }
  if (command === "compiler-inputs") {
    const root = workspaceOption(args, command) || defaultWorkspace(), paths = supportPaths(root),
      result = buildLocalCompilerInputs({workspace: root, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
        buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk});
    console.log(`[native-sdk] Local DPC++/CUDA inputs admitted at ${result.workspace}; remaining release SDKs and packaging were not run.`);
    return;
  }
  if (command === "gnu-headers") {
    const root = workspaceOption(args, command) || defaultWorkspace(), paths = supportPaths(root),
      result = buildLocalGnuHeaders({workspace: root, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
        buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk});
    console.log(`[native-sdk] Local GNU target headers admitted at ${result.root}; remaining release SDKs and packaging were not run.`);
    return;
  }
  if (command === "jit-resources") {
    const root = workspaceOption(args, command) || defaultWorkspace(), paths = supportPaths(root),
      result = buildLocalLinuxJitResources({workspace: root, compilerFetch: paths.compilerFetch, compilerSources: paths.compilerSources,
        buildInputs: paths.buildInputs, llvmSdk: paths.llvmSdk});
    console.log(`[native-sdk] Linux CPU JIT resources admitted at ${result.root}; no JIT SDK, GPU, runtime or release package was built.`);
    return;
  }
  if (command !== "source-llvm") {throw new Error("Select the source-LLVM bootstrap slice; `npm run build:release` runs the full local release");}
  const workspace = workspaceOption(args, command), result = buildLocalNativeSdk(workspace ? {workspace} : {});
  console.log(`[native-sdk] Linux compiler sources and LLVM 21 SDK admitted at ${result.workspace}; remaining release SDKs and packaging were not run.`);
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {ensureLocalLinuxSysroot, buildLocalNativeSdk, buildLocalNativeSupportSdk, buildLocalCompilerInputs, buildLocalGnuHeaders, buildLocalLinuxJitResources,
  stageLocalCompilerInputs, prepareOrAdmitLinuxSources, prepareOrAdmitWindowsSources, localLinuxJitResourceState, runLocalJitResourceChild, localJitToolsReceipt,
  workspaceOption, main, localTools, localSupportTools, localDpcppCudaRoots, localDpcppCudaState, scopedInvoker, runCommand,
  admitWorkspace, readConfigureState, cgroupMemoryPaths,
  verifyMemoryLimits, verifyCurrentMemoryScope, trustedSystemdRun, defaultWorkspace, supportPaths,
  localBuildJobs, ciRecipeJobs, localMemoryMax, recordedParallelJobs};
