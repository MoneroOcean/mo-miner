"use strict";

// One locked WSL coordinator owns the release graph; Linux work follows MOM_BUILD_JOBS under its derived memory cap,
// while native Windows interop stages remain outside the Linux cgroup and retain their jobs=2 policy.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"),
  {createHash, randomUUID} = require("node:crypto");
const {prepareWindowsWslRelease, wslText} = require("./build-local-native-wsl");
const {requireBuildFields, hasBuildDirectoryEntries} = require("./build-source-inputs");
const repo = path.resolve(__dirname, ".."), nodeVersion = "v24.15.0", relayEnvironment = "ONEAPI_ROOT/w:MOM_NATIVE_PERL_ROOT/w",
  uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/;
/** @typedef {(command:string,args:string[],options:import("node:child_process").SpawnSyncOptions)=>{status:number|null,error?:Error,signal?:NodeJS.Signals|null,stdout?:Buffer|string}} LocalSpawn */

/** @param {unknown} value @returns {value is Record<string, any>} */
function record(value) {return Boolean(value && typeof value === "object" && !Array.isArray(value));}

/** @param {string} value */
function validWindowsPath(value) {
  return typeof value === "string" && value.length <= 4096 && path.win32.isAbsolute(value) &&
    path.win32.normalize(value).toLowerCase() === value.toLowerCase() && !/[\r\n\0]/.test(value);
}

/** Keep both native and WSL Node gates aligned with the maintained installer pin. */
function verifyNodeVersionPin() {
  const windowsInstaller = fs.readFileSync(path.join(repo, "scripts/install-dev.ps1"), "utf8"),
    windowsPins = [...windowsInstaller.matchAll(/^\$nodeVersion\s*=\s*'([^']+)'\s*$/gm)].map(match => match[1]),
    linuxInstaller = fs.readFileSync(path.join(repo, "scripts/install-dev.sh"), "utf8"),
    linuxPin = /^NODE_VERSION=\$\{MOM_NODE_VERSION-([^}]+)\}$/m.exec(linuxInstaller)?.[1];
  if (windowsPins.length !== 1 || windowsPins[0] !== nodeVersion.slice(1) || linuxPin !== nodeVersion.slice(1)) {
    throw new Error("Local Windows/WSL Node pin must match the maintained scripts/install-dev.ps1 version");
  }
}

/** @param {string} root @param {string} member */
function windowsContained(root, member) {
  const relative = path.win32.relative(root, member);
  return Boolean(relative) && !path.win32.isAbsolute(relative) && relative !== ".." && !relative.startsWith("..\\");
}

/** Stable, separate native cache roots derived from the literal Windows LocalApplicationData folder. @param {string} appData @param {string} checkout */
function localWindowsReleaseRoots(appData, checkout) {
  if (!validWindowsPath(appData) || !validWindowsPath(checkout)) {throw new Error("Windows release paths must be canonical absolute paths");}
  const checkoutKey = createHash("sha256").update(path.win32.normalize(checkout).toLowerCase()).digest("hex"),
    root = path.win32.join(appData, "MoM", "native-wsl-windows", checkoutKey);
  return {root, support: path.win32.join(root, "support"), exports: path.win32.join(root, "exports"), bridge: path.win32.join(root, "bridge")};
}

/** Require separate support/export leaves and a per-run bridge child under one checkout-keyed parent.
 * @param {{supportWorkspace:string,exportWorkspace:string,bridgeRunWindows:string}} input */
function validateWindowsNativeRoots(input) {
  const supportParent = path.win32.dirname(input["supportWorkspace"]), exportParent = path.win32.dirname(input["exportWorkspace"]),
    bridgeParent = path.win32.dirname(path.win32.dirname(input["bridgeRunWindows"])), nativeRoot = supportParent,
    same = (/** @type {string} */ left, /** @type {string} */ right) => left.toLowerCase() === right.toLowerCase();
  if (path.win32.basename(input["supportWorkspace"]).toLowerCase() !== "support" ||
      path.win32.basename(input["exportWorkspace"]).toLowerCase() !== "exports" || !same(supportParent, exportParent) ||
      path.win32.basename(path.win32.dirname(input["bridgeRunWindows"])).toLowerCase() !== "bridge" || !same(nativeRoot, bridgeParent) ||
      !/^[a-f0-9]{64}$/i.test(path.win32.basename(nativeRoot)) ||
      path.win32.basename(path.win32.dirname(nativeRoot)).toLowerCase() !== "native-wsl-windows" ||
      path.win32.basename(path.win32.dirname(path.win32.dirname(nativeRoot))).toLowerCase() !== "mom" ||
      !/^run-[a-f0-9]{8}-[a-f0-9]{4}-[1-8][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(path.win32.basename(input["bridgeRunWindows"]))) {
    throw new Error("Windows support, export and bridge directories must be separate checkout-keyed LocalApplicationData siblings");
  }
}

/** Create only these named native cache directories; never replace a link or unknown file. @param {ReturnType<typeof localWindowsReleaseRoots>} roots */
/** @param {ReturnType<typeof localWindowsReleaseRoots>} roots @param {{fs?:Pick<typeof fs,"lstatSync"|"mkdirSync"|"realpathSync">}} [dependencies] */
function ensureNativeWindowsReleaseRoots(roots, dependencies = {}) {
  const io = dependencies.fs || fs;
  const appData = path.win32.dirname(path.win32.dirname(path.win32.dirname(roots.root)));
  const directories = [path.win32.join(appData, "MoM"), path.win32.join(appData, "MoM", "native-wsl-windows"), roots.root,
    roots.bridge];
  for (const directory of directories) {
    const current = io.lstatSync(directory, {throwIfNoEntry: false});
    if (!current) {io.mkdirSync(directory, {mode: 0o700});}
    const selected = io.lstatSync(directory);
    if (!selected.isDirectory() || selected.isSymbolicLink() || io.realpathSync(directory).toLowerCase() !== directory.toLowerCase()) {
      throw new Error("Native Windows release cache directory is redirected or not a directory: " + directory);
    }
  }
  return roots;
}

/** @param {string} executable @param {string[]} args @param {LocalSpawn} spawn @param {NodeJS.ProcessEnv} environment @param {string} label */
function runChecked(executable, args, spawn, environment, label) {
  const result = spawn(executable, args, {env: environment, windowsHide: true, shell: false, timeout: 120000, maxBuffer: 2 << 20});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error(label + " failed (" + (result.status ?? "no exit code") + (result.signal ? ", " + result.signal : "") + ")", {cause: result.error});
  }
  return result;
}

/** Fixed preflight body, also exposed for real PowerShell parser checks without running SDK tools. */
function windowsPreflightSource() {
  const source = ["$ErrorActionPreference = 'Stop'", "$utf8 = [System.Text.UTF8Encoding]::new($false)",
    "[Console]::OutputEncoding = $utf8", ". $env:MOM_NATIVE_WINDOWS_TOOLS_SCRIPT",
    "[void](Get-MomNativeBuildNode)",
    "$io = Get-MomNativeWindowsBuildTools -Role io",
    "$compilerBin = Split-Path -Parent $io.cc; $compilerVersion = Split-Path -Parent $compilerBin",
    "$compilerRoot = Split-Path -Parent $compilerVersion; $oneApiRoot = Split-Path -Parent $compilerRoot",
    "$perlBin = Split-Path -Parent $io.openssl.perl; $perlHome = Split-Path -Parent $perlBin",
    "$nativePerlRoot = Split-Path -Parent $perlHome",
    "$oneApiRoot = [IO.Path]::GetFullPath($oneApiRoot); $nativePerlRoot = [IO.Path]::GetFullPath($nativePerlRoot)",
    "$payload = [ordered]@{ appData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)",
    "  oneApiRoot = $oneApiRoot; nativePerlRoot = $nativePerlRoot",
    "  tools = [ordered]@{ core = Get-MomNativeWindowsBuildTools -Role core;",
    "    io = [ordered]@{ cmake = $io.cmake; cc = $io.cc; ninja = $io.ninja }; openssl = $io.openssl;",
    "    hermes = Get-MomNativeWindowsBuildTools -Role hermes }", "}",
    "ConvertTo-Json -InputObject $payload -Depth 8 -Compress"];
  return source.join("\n");
}

/** Probe native SDK roles before building. The script path travels through the environment, never interpolation.
 * @param {NodeJS.ProcessEnv} environment @param {LocalSpawn} [spawn] */
function windowsToolPreflight(environment, spawn = spawnSync) {
  const systemRoot = environment["SystemRoot"] || "C:\\Windows", executable = path.win32.join(systemRoot, "System32/WindowsPowerShell/v1.0/powershell.exe"),
    canonicalExecutable = fs.realpathSync(executable), toolsScript = path.join(repo, "scripts/native-windows-tools.ps1");
  if (!validWindowsPath(executable) || canonicalExecutable.toLowerCase() !== executable.toLowerCase()) {
    throw new Error("Windows release requires the canonical system PowerShell executable");
  }
  const encoded = Buffer.from(windowsPreflightSource(), "utf16le").toString("base64"),
    result = runChecked(canonicalExecutable, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
      spawn, {...environment, MOM_NATIVE_WINDOWS_TOOLS_SCRIPT: toolsScript}, "Native Windows tool preflight"), output = wslText(result.stdout).trim();
  let payload;
  try {payload = JSON.parse(output);} catch (error) {throw new Error("Native Windows tool preflight returned invalid JSON", {cause: error});}
  requireBuildFields(payload, ["appData", "nativePerlRoot", "oneApiRoot", "tools"], "Native Windows tool preflight");
  if (!validWindowsPath(payload.appData) || !validWindowsPath(payload.oneApiRoot) || !validWindowsPath(payload.nativePerlRoot)) {
    throw new Error("PowerShell did not return canonical LocalApplicationData and selected native tool roots");
  }
  const appData = fs.realpathSync(payload.appData);
  if (appData.toLowerCase() !== payload.appData.toLowerCase()) {throw new Error("LocalApplicationData resolves through a path alias");}
  const tools = require("./build-local-native-windows-support").validateWindowsSupportTools(payload.tools, "win32");
  const compilerRelative = path.win32.relative(payload.oneApiRoot, tools.io.cc).split("\\"),
    perlExpected = path.win32.join(payload.nativePerlRoot, "perl", "bin", "perl.exe");
  if (compilerRelative.length !== 4 || compilerRelative[0]?.toLowerCase() !== "compiler" || !/^2026\.2(?:\.|$)/i.test(compilerRelative[1] || "") ||
      compilerRelative[2]?.toLowerCase() !== "bin" || compilerRelative[3]?.toLowerCase() !== "icx-cl.exe" ||
      path.win32.normalize(perlExpected).toLowerCase() !== tools.openssl.perl.toLowerCase()) {
    throw new Error("Native Windows relay roots do not contain the exact preflighted Intel compiler and Perl executables");
  }
  return {appData, tools, oneApiRoot: payload.oneApiRoot, nativePerlRoot: payload.nativePerlRoot, powershell: canonicalExecutable};
}

/** Open only a fresh UUID child of the owned bridge directory. @param {string} bridge @param {string} id */
function createBridgeRun(bridge, id) {
  if (!uuid.test(id || "")) {throw new Error("Windows bridge run needs a UUID");}
  if (fs.readdirSync(bridge).length !== 0) {throw new Error("Inspect the preserved Windows bridge run before retrying: " + bridge);}
  const root = path.win32.join(bridge, "run-" + id);
  if (fs.lstatSync(root, {throwIfNoEntry: false})) {throw new Error("Windows bridge run already exists; preserve it: " + root);}
  fs.mkdirSync(root, {mode: 0o700});
  const stat = fs.lstatSync(root);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("Windows bridge run is not a new directory");}
  return {root, dev: stat.dev, ino: stat.ino};
}

/** Remove just our successful transient bridge directory; retain support/export caches and failures.
 * @param {{root:string,dev:number,ino:number}} run */
function removeSuccessfulBridgeRun(run) {
  const current = fs.lstatSync(run.root, {throwIfNoEntry: false});
  if (!current?.isDirectory() || current.isSymbolicLink() || current.dev !== run.dev || current.ino !== run.ino) {return false;}
  const names = ["hermes-sources.tar", "support-input.json", "support-output.json"];
  if (!hasBuildDirectoryEntries(run.root, names)) {return false;}
  for (const name of names) {
    const file = path.win32.join(run.root, name), stat = fs.lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1) {return false;}
  }
  for (const name of names) {fs.unlinkSync(path.win32.join(run.root, name));}
  fs.rmdirSync(run.root);
  return true;
}

/** Windows frontdoor called by build-local-native-command. @param {NodeJS.ProcessEnv} [environment]
 * @param {LocalSpawn} [spawn]
 * @param {{platform?:NodeJS.Platform,arch?:string,nodeVersion?:string,checkout?:string,prepareWsl?:typeof prepareWindowsWslRelease,
 * preflightWindows?:typeof windowsToolPreflight,ensureRoots?:typeof ensureNativeWindowsReleaseRoots,
 * createRun?:(bridge:string,id:string)=>ReturnType<typeof createBridgeRun>,removeRun?:typeof removeSuccessfulBridgeRun,
 * createId?:typeof randomUUID,releaseVersion?:(environment:NodeJS.ProcessEnv)=>string}} [dependencies] */
function launchLocalNativeWindowsRelease(environment = process.env, spawn = spawnSync, dependencies = {}) {
  const platform = dependencies.platform || process.platform, arch = dependencies.arch || process.arch;
  if (platform !== "win32" || arch !== "x64") {throw new Error("The local Windows release frontdoor requires Windows x64");}
  const version = dependencies.nodeVersion || process.version;
  if (version !== nodeVersion) {throw new Error("The local Windows release frontdoor requires Node.js " + nodeVersion);}
  verifyNodeVersionPin();
  const checkout = dependencies.checkout || repo, releaseVersion = dependencies.releaseVersion || require("./build-release").releaseVersion;
  releaseVersion(environment);
  const jobs = require("./build-local-native-sdk").localBuildJobs(environment);
  const overrides = [];
  for (const name of ["MOM_RELEASE_VERSION", "MOM_FORCE_REBUILD", "MOM_NATIVE_CACHE"]) {
    const value = environment[name];
    if (value !== undefined) {
      if (value.length > 128 || /[\r\n\0]/.test(value)) {throw new Error("Invalid portable Windows-to-WSL setting: " + name);}
      overrides.push(name + "=" + value);
    }
  }
  const wsl = (dependencies.prepareWsl || prepareWindowsWslRelease)(checkout, environment, {platform, arch, spawn: /** @type {typeof spawnSync} */ (spawn)}),
    preflight = (dependencies.preflightWindows || windowsToolPreflight)(environment, spawn),
    validRelay = validWindowsPath(preflight.oneApiRoot) && validWindowsPath(preflight.nativePerlRoot);
  if (!validRelay) {throw new Error("Windows preflight did not admit native oneAPI and Perl roots for WSL interop");}
  const roots = (dependencies.ensureRoots || ensureNativeWindowsReleaseRoots)(localWindowsReleaseRoots(preflight.appData, checkout)),
    id = (dependencies.createId || randomUUID)(), run = (dependencies.createRun || createBridgeRun)(roots.bridge, id),
    envArgs = ["MOM_AOT_BUILD_ONLY=1", "MOM_BUILD_JOBS=" + jobs, "OMP_NUM_THREADS=" + jobs, "WSLENV=" + relayEnvironment,
      "ONEAPI_ROOT=" + preflight.oneApiRoot, "MOM_NATIVE_PERL_ROOT=" + preflight.nativePerlRoot, ...overrides];
  const script = path.posix.join(wsl.checkout, "scripts/build-local-native-windows-command.js"),
    args = [...wsl.prefix, "/usr/bin/env", ...envArgs, "node", script, "--wsl-launch", wsl.checkout,
      checkout, preflight.powershell, roots.support, roots.exports, run.root],
    result = spawn(wsl.executable, args, {env: {...environment, WSLENV: relayEnvironment}, cwd: checkout, windowsHide: true, shell: false, stdio: "inherit"});
  if (result.error) {throw result.error;}
  if (result.signal || result.status !== 0) {throw new Error("WSL local Windows release failed (" + (result.status ?? "no exit code") + (result.signal ? ", " + result.signal : "") + ")");}
  if (!(dependencies.removeRun || removeSuccessfulBridgeRun)(run)) {
    console.error("[native-windows] Preserved completed bridge run with unexpected contents: " + run.root);
  }
  return 0;
}

/** Run wslpath with argv only and require its path to round-trip to the native input. @param {string} option @param {string} input */
function mapWslPath(option, input, spawn = spawnSync) {
  const result = spawn("/usr/bin/wslpath", [option, input], {encoding: "utf8", timeout: 30000, maxBuffer: 1 << 20, shell: false});
  if (result.error || result.signal || result.status !== 0) {throw new Error("WSL path mapping failed for " + option, {cause: result.error});}
  const mapped = String(result.stdout || "").trim();
  if (/\r|\n|\0/.test(mapped)) {throw new Error("WSL path mapping returned multiple or invalid paths");}
  return mapped;
}

/** @param {string} windowsPath @param {typeof spawnSync} spawn */
function windowsPathToWsl(windowsPath, spawn = spawnSync) {
  if (!validWindowsPath(windowsPath)) {throw new Error("Expected a canonical Windows path to map into WSL");}
  const mapped = mapWslPath("-u", windowsPath, spawn), back = mapWslPath("-w", mapped, spawn);
  if (!path.posix.isAbsolute(mapped) || path.posix.normalize(mapped) !== mapped || !validWindowsPath(back) ||
      back.toLowerCase() !== windowsPath.toLowerCase()) {throw new Error("Windows path does not round-trip through WSL: " + windowsPath);}
  return mapped;
}

/** @param {string} posixPath @param {typeof spawnSync} spawn */
function wslPathToWindows(posixPath, spawn = spawnSync) {
  if (typeof posixPath !== "string" || !path.posix.isAbsolute(posixPath) || path.posix.normalize(posixPath) !== posixPath || /[\r\n\0]/.test(posixPath)) {
    throw new Error("Expected a canonical WSL path to map to Windows");
  }
  const mapped = mapWslPath("-w", posixPath, spawn), back = mapWslPath("-u", mapped, spawn);
  if (!validWindowsPath(mapped) || back !== posixPath) {throw new Error("WSL path does not round-trip to Windows: " + posixPath);}
  return mapped;
}

/** Read a bridge file after requiring a bounded, canonical regular-file snapshot. @param {string} file @param {number} maximum */
function readBoundedJson(file, maximum) {
  const before = fs.lstatSync(file, {bigint: true, throwIfNoEntry: false});
  if (!before?.isFile() || before.isSymbolicLink() || before.nlink !== 1n || before.size < 1n || before.size > BigInt(maximum) || fs.realpathSync(file) !== file) {
    throw new Error("Windows bridge JSON is missing, aliased or unbounded: " + file);
  }
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), chunk = Buffer.alloc(maximum + 1);
  try {
    const opened = fs.fstatSync(fd, {bigint: true});
    if (!sameBigSnapshot(before, opened)) {throw new Error("Windows bridge JSON changed while opening: " + file);}
    let bytes = 0;
    while (bytes < chunk.length) {
      const size = fs.readSync(fd, chunk, bytes, chunk.length - bytes, null);
      if (size === 0) {break;}
      bytes += size;
    }
    const after = fs.fstatSync(fd, {bigint: true}), current = fs.lstatSync(file, {bigint: true, throwIfNoEntry: false});
    if (bytes > maximum || !current?.isFile() || current.isSymbolicLink() || current.nlink !== 1n ||
        !sameBigSnapshot(before, after) || !sameBigSnapshot(before, current) || fs.realpathSync(file) !== file || BigInt(bytes) !== before.size) {
      throw new Error("Windows bridge JSON changed during its bounded read: " + file);
    }
    try {return JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(chunk.subarray(0, bytes)));} catch (error) {
      throw new Error("Windows bridge JSON is not valid UTF-8: " + file, {cause: error});
    }
  } finally {
    fs.closeSync(fd);
  }
}

/** @param {fs.BigIntStats} left @param {fs.BigIntStats} right */
function sameBigSnapshot(left, right) {
  return left.isFile() && right.isFile() && left.dev === right.dev && left.ino === right.ino && left.nlink === right.nlink &&
    left.size === right.size && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}

/** Preserve original receipts in a WSL-created tar and invoke the existing Windows support bridge.
 * @param {any} input @param {any} [dependencies] */
function runWindowsSupportBridge(input, dependencies = {}) {
  const writeFile = dependencies.writeFile || fs.writeFileSync, run = dependencies.spawn || spawnSync,
    inputPosix = input.inputPosix, outputPosix = input.outputPosix, environment = {...input.environment, WSLENV: relayEnvironment};
  if (!validWindowsPath(environment["ONEAPI_ROOT"]) || !validWindowsPath(environment["MOM_NATIVE_PERL_ROOT"])) {
    throw new Error("Windows support bridge requires the selected native oneAPI and Perl roots");
  }
  for (const file of [inputPosix, outputPosix]) {
    if (typeof file !== "string" || !path.posix.isAbsolute(file) || path.posix.normalize(file) !== file || /[\r\n\0]/.test(file)) {
      throw new Error("Windows support bridge state paths must be canonical WSL paths");
    }
  }
  if (fs.lstatSync(outputPosix, {throwIfNoEntry: false})) {throw new Error("Windows support bridge output already exists; preserve it");}
  writeFile(inputPosix, JSON.stringify(input.value) + "\n", {flag: "wx", mode: 0o600});
  const processPath = input.powershellPosix, scriptWin = input.scriptWindows;
  if (typeof processPath !== "string" || !path.posix.isAbsolute(processPath) || typeof scriptWin !== "string" || !validWindowsPath(scriptWin)) {
    throw new Error("Windows support bridge executable or script path is invalid");
  }
  const map = dependencies.windowsPathToWsl || windowsPathToWsl;
  if (map(input.inputWindows, run) !== inputPosix || map(input.outputWindows, run) !== outputPosix) {
    throw new Error("Windows support bridge file paths do not round-trip through WSL");
  }
  const result = run(processPath, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", scriptWin,
    "-InputFile", input.inputWindows, "-OutputFile", input.outputWindows], {env: environment, shell: false, stdio: "inherit"});
  if (result.error || result.signal || result.status !== 0) {
    throw new Error("Windows native support/Core bridge failed (" + (result.status ?? "no exit code") + (result.signal ? ", " + result.signal : "") + ")", {cause: result.error});
  }
  return readBoundedJson(outputPosix, 16 << 20);
}

/** Validate only the original Windows bridge result, then map its file roots without rewriting receipts.
 * @param {unknown} value @param {any} input @param {any} [dependencies] */
function admitWindowsBridgeResult(value, input, dependencies = {}) {
  const result = requireBuildFields(value, ["controllerSupport", "coreArchive", "coreReceipt", "exportRetirementCandidates", "supportRetirementCandidates"], "Windows support/Core bridge result"),
    controller = requireBuildFields(result["controllerSupport"], ["hermes", "io", "localBuildId"], "Windows controller-support references"),
    id = controller["localBuildId"], archive = result["coreArchive"], receipt = requireBuildFields(result["coreReceipt"], ["path", "sha256"], "Windows Core receipt reference");
  const retired = requireBuildFields(result["supportRetirementCandidates"], ["core", "hermes", "io", "openssl"], "Windows support retirement candidates");
  if (!uuid.test(id || "") || !validWindowsPath(archive) || !validWindowsPath(receipt["path"]) ||
      !/^[a-f0-9]{64}$/.test(receipt["sha256"] || "") || !windowsContained(input.supportWorkspace, archive) ||
      path.win32.dirname(archive).toLowerCase() !== path.win32.dirname(receipt["path"]).toLowerCase() ||
      path.win32.basename(receipt["path"]).toLowerCase() !== "inputs.json" ||
      !Object.values(retired).every(Array.isArray) || !Array.isArray(result["exportRetirementCandidates"])) {
    throw new Error("Windows support bridge returned invalid original Core or identity references");
  }
  /** @param {unknown} value @param {"io"|"hermes"} kind */
  const reference = (value, kind) => {
    const row = requireBuildFields(value, ["admitted", "handoffSha256", "root"], "Windows " + kind + " handoff reference"),
      admitted = requireBuildFields(row["admitted"], ["kind", "localBuildId"], "Windows " + kind + " handoff identity");
    if (admitted["kind"] !== kind || admitted["localBuildId"] !== id || !/^[a-f0-9]{64}$/.test(row["handoffSha256"] || "") ||
        !validWindowsPath(row["root"]) || !windowsContained(input.exportWorkspace, row["root"])) {
      throw new Error("Windows " + kind + " handoff does not retain the original export identity");
    }
    return {root: row["root"], handoffSha256: row["handoffSha256"], admitted: {kind, localBuildId: id}};
  };
  const map = dependencies.windowsPathToWsl || windowsPathToWsl, coreArchive = map(archive, dependencies.spawn || spawnSync),
    coreReceipt = map(receipt["path"], dependencies.spawn || spawnSync), io = reference(controller["io"], "io"),
    hermes = reference(controller["hermes"], "hermes");
  const ioRoot = map(io.root, dependencies.spawn || spawnSync), hermesRoot = map(hermes.root, dependencies.spawn || spawnSync),
    receiptStat = fs.lstatSync(coreReceipt, {throwIfNoEntry: false});
  const ioStat = fs.lstatSync(ioRoot, {throwIfNoEntry: false}), hermesStat = fs.lstatSync(hermesRoot, {throwIfNoEntry: false});
  if (!receiptStat?.isFile() || receiptStat.isSymbolicLink() || fs.realpathSync(coreReceipt) !== coreReceipt ||
      require("./build-source-inputs").fileHash(coreReceipt) !== receipt["sha256"] ||
      path.posix.dirname(coreArchive) !== path.posix.dirname(coreReceipt) || !ioStat?.isDirectory() || ioStat.isSymbolicLink() ||
      fs.realpathSync(ioRoot) !== ioRoot || !hermesStat?.isDirectory() || hermesStat.isSymbolicLink() || fs.realpathSync(hermesRoot) !== hermesRoot) {
    throw new Error("Mapped Windows support handoffs or original Core receipt failed WSL-side admission");
  }
  return {controllerSupport: {localBuildId: id, io: {...io, root: ioRoot}, hermes: {...hermes, root: hermesRoot}}, coreArchive,
    coreReceipt: {path: coreReceipt, sha256: receipt["sha256"]}};
}

/** Require the default native workspace to live on Linux ext4, never DrvFs. @param {string} workspace @param {Pick<typeof fs,"statfsSync">} [fileSystem] */
function verifyExt4WorkspaceParent(workspace, fileSystem = fs) {
  if (!path.isAbsolute(workspace) || path.resolve(workspace) !== workspace || workspace === "/mnt" || workspace.startsWith("/mnt/")) {
    throw new Error("WSL native build workspace must use the Linux filesystem, not a Windows mount");
  }
  const type = fileSystem.statfsSync(path.dirname(workspace)).type;
  if (type !== 0xef53) {throw new Error("WSL native build workspace parent must be ext4");}
}

/** Build, package and transactionally publish the WSL Windows candidate while the outer scope and flock remain held.
 * @param {{checkout:string,checkoutWindows:string,powershellWindows:string,supportWorkspace:string,exportWorkspace:string,bridgeRunWindows:string}} input
 * @param {any} [dependencies] */
async function runLocalWindowsWslRelease(input, dependencies = {}) {
  const verifyScope = dependencies.verifyScope || require("./build-local-native-sdk").verifyCurrentMemoryScope;
  verifyScope();
  if ((dependencies.platform || process.platform) !== "linux" || (dependencies.arch || process.arch) !== "x64") {
    throw new Error("The local Windows release worker requires Linux x64 under WSL2");
  }
  if ((dependencies.observedNodeVersion || process.version) !== nodeVersion) {throw new Error("WSL requires pinned build Node.js " + nodeVersion);}
  verifyNodeVersionPin();
  const workerEnvironment = {...process.env, ...(dependencies.environment || {})};
  if (workerEnvironment["MOM_NATIVE_BUILD_LOCK"] !== "1") {throw new Error("The WSL Windows release worker requires the held Engineering build lock");}
  if (!validWindowsPath(workerEnvironment["ONEAPI_ROOT"]) || !validWindowsPath(workerEnvironment["MOM_NATIVE_PERL_ROOT"])) {
    throw new Error("The WSL Windows release worker requires the selected native oneAPI and Perl roots");
  }
  const args = requireBuildFields(input, ["bridgeRunWindows", "checkout", "checkoutWindows", "exportWorkspace", "powershellWindows", "supportWorkspace"], "WSL Windows release worker input");
  if (args["checkout"] !== fs.realpathSync(repo) || !validWindowsPath(args["checkoutWindows"]) ||
      !validWindowsPath(args["powershellWindows"]) || !validWindowsPath(args["supportWorkspace"]) || !validWindowsPath(args["exportWorkspace"]) ||
      !validWindowsPath(args["bridgeRunWindows"])) {
    throw new Error("WSL Windows release worker paths are not canonical native roots for this checkout");
  }
  validateWindowsNativeRoots({supportWorkspace: args["supportWorkspace"], exportWorkspace: args["exportWorkspace"],
    bridgeRunWindows: args["bridgeRunWindows"]});
  const spawn = dependencies.spawn || spawnSync, toWsl = dependencies.windowsPathToWsl || windowsPathToWsl,
    toWindows = dependencies.wslPathToWindows || wslPathToWindows, mappedCheckout = toWindows(repo, spawn);
  if (mappedCheckout.toLowerCase() !== args["checkoutWindows"].toLowerCase()) {
    throw new Error("WSL checkout does not map back to the Windows-preflighted checkout");
  }
  const local = dependencies.local || require("./build-local-native-sdk"),
    jobs = local.localBuildJobs({...process.env, ...(dependencies.environment || {})}), environment = /** @type {NodeJS.ProcessEnv} */ ({
      ...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), OMP_NUM_THREADS: String(jobs),
      MOM_NATIVE_BUILD_LOCK: "1", MOM_LOCAL_NATIVE_SCOPE: "1", WSLENV: relayEnvironment}),
    validateReleaseVersion = dependencies.releaseVersion || require("./build-release").releaseVersion;
  validateReleaseVersion(environment);
  const linuxApi = dependencies.linuxApi || require("./build-local-native-linux-sdk"),
    tools = (dependencies.localTools || local.localTools)(environment),
    initialPreflight = (dependencies.preflight || linuxApi.preflightLocalLinuxSdk)({environment, tools}),
    workspace = (dependencies.defaultWorkspace || local.defaultWorkspace)();
  (dependencies.verifyExt4 || verifyExt4WorkspaceParent)(workspace);
  local.admitWorkspace(workspace, repo);
  const stage = path.join(workspace, "windows-release-work"), logger = dependencies.console || console;
  // A failed application occupies one slot. Refuse retry before dependency or archive work.
  if (fs.lstatSync(stage, {throwIfNoEntry: false})) {
    throw new Error("Inspect preserved local Windows release stage before retrying: " + stage);
  }
  const node = (dependencies.prepareNode || require("./build-local-native-wsl-node").prepareLocalWslNode)(workspace, {verifyScope, environment});
  if (!record(node) || typeof node["version"] !== "string" || !/^\d+\.\d+\.\d+$/.test(node["version"]) ||
      typeof node["binary"] !== "string" || !path.isAbsolute(node["binary"]) ||
      !/^[a-f0-9]{64}$/.test(node["sha256"] || "")) {throw new Error("WSL Node dependency admission did not return its pinned Linux esbuild binary");}
  environment["ESBUILD_BINARY_PATH"] = node["binary"];
  process.env["ESBUILD_BINARY_PATH"] = node["binary"];
  // Re-run lightweight host admission with the exact bundler environment used by every producer.
  const ready = (dependencies.preflight || linuxApi.preflightLocalLinuxSdk)({environment, tools});
  if (initialPreflight.tools && JSON.stringify(initialPreflight.tools) !== JSON.stringify(ready.tools)) {
    throw new Error("Linux tool selection changed during WSL node bootstrap");
  }
  const prerequisites = (dependencies.buildPrerequisites || linuxApi.buildLocalNativePrerequisites)({workspace}, {environment, tools});
  if (!prerequisites || prerequisites.workspace !== workspace || prerequisites.source?.workspace !== workspace || prerequisites.source?.jobs !== jobs ||
      prerequisites.tools && JSON.stringify(prerequisites.tools) !== JSON.stringify(tools) ||
      prerequisites.paths?.hermesSource !== local.supportPaths(workspace).hermesSource) {
    throw new Error("Local Linux prerequisite graph returned another workspace, tool cohort or Hermes source root");
  }

  const fsApi = dependencies.fs || fs, ci = dependencies.nativeCi || {...require("./native-ci-tar"), ...require("./native-ci-windows-controller")}, sourceRoot = prerequisites.paths.hermesSource,
    hermesSourceName = path.basename(sourceRoot), fetchFile = path.join(sourceRoot, "fetch.json"), rootsFile = path.join(sourceRoot, "roots.json"),
    roots = (dependencies.readJson || require("./build-source-inputs").readBuildJson)(rootsFile),
    originalRoot = roots?.hermes, archiveWindows = path.win32.join(args["bridgeRunWindows"], "hermes-sources.tar"),
    archivePosix = toWsl(archiveWindows, spawn),
    fetchSha256 = (dependencies.fileHash || require("./build-source-inputs").fileHash)(fetchFile),
    rootsSha256 = (dependencies.fileHash || require("./build-source-inputs").fileHash)(rootsFile);
  if (!ci.windowsHermesSourceName(hermesSourceName) || !ci.canonicalHermesOriginalRoot(originalRoot) ||
      !fsApi.lstatSync(path.join(sourceRoot, "roots/hermes/config"), {throwIfNoEntry: false})?.isFile()) {
    throw new Error("Local Hermes source cohort lacks its original canonical roots receipt");
  }
  (dependencies.packState || ci.packState)(archivePosix, workspace, [hermesSourceName], 5 * 60 * 1000);
  const memberNames = (dependencies.listState || ci.listState)(archivePosix, [hermesSourceName]);
  if (!memberNames.includes(hermesSourceName + "/fetch.json") || !memberNames.includes(hermesSourceName + "/roots.json") ||
      !memberNames.includes(hermesSourceName + "/roots/hermes/config") ||
      (dependencies.fileHash || require("./build-source-inputs").fileHash)(fetchFile) !== fetchSha256 ||
      (dependencies.fileHash || require("./build-source-inputs").fileHash)(rootsFile) !== rootsSha256) {
    throw new Error("Hermes transfer did not preserve the admitted original source receipts");
  }
  const archiveSha256 = (dependencies.fileHash || require("./build-source-inputs").fileHash)(archivePosix),
    bridgeRunPosix = path.posix.dirname(archivePosix), inputPosix = path.posix.join(bridgeRunPosix, "support-input.json"),
    outputPosix = path.posix.join(bridgeRunPosix, "support-output.json"), inputWindows = path.win32.join(args["bridgeRunWindows"], "support-input.json"),
    outputWindows = path.win32.join(args["bridgeRunWindows"], "support-output.json"),
    powershellPosix = toWsl(args["powershellWindows"], spawn),
    scriptPosix = path.join(repo, "scripts/build-local-native-windows-support.ps1"),
    scriptWindows = toWindows(scriptPosix, spawn);
  if (toWindows(archivePosix, spawn).toLowerCase() !== archiveWindows.toLowerCase() ||
      !fsApi.statSync(bridgeRunPosix).isDirectory() || fsApi.realpathSync(bridgeRunPosix) !== bridgeRunPosix) {
    throw new Error("Native bridge directory is not the exact WSL-mapped LocalApplicationData run root");
  }
  const bridgeValue = {workspace: args["supportWorkspace"], exportWorkspace: args["exportWorkspace"],
      hermesArchive: {path: archiveWindows, sha256: archiveSha256, rootName: hermesSourceName, originalRoot}},
    bridgeRaw = (dependencies.runWindowsBridge || runWindowsSupportBridge)({inputPosix, outputPosix, inputWindows, outputWindows,
      value: bridgeValue, powershellPosix, scriptWindows, environment}, {spawn: dependencies.spawn || spawnSync}),
    bridge = admitWindowsBridgeResult(bridgeRaw, {supportWorkspace: args["supportWorkspace"], exportWorkspace: args["exportWorkspace"]},
      {windowsPathToWsl: dependencies.windowsPathToWsl, spawn: dependencies.spawn || spawnSync}),
    stageStat = fs.lstatSync(stage, {throwIfNoEntry: false});
  if (stageStat) {throw new Error("Inspect preserved local Windows release stage before retrying: " + stage);}
  fs.mkdirSync(stage, {mode: 0o700});
  const ownedStage = fs.lstatSync(stage), sdkOutput = path.join(stage, "windows-sdk"), distributionRoot = path.join(stage, "release-work"),
    candidate = path.join(distributionRoot, "candidate");
  let published = false;
  try {
    const buildWindowsSdk = dependencies.buildWindowsSdk || require("./build-local-native-windows-sdk").buildLocalNativeWindowsSdk,
      sdk = await buildWindowsSdk({workspace, tools, linuxSdk: prerequisites.source, controllerSupport: bridge.controllerSupport,
        coreArchive: bridge.coreArchive, output: sdkOutput}, {verifyScope, environment});
    if (!sdk || sdk.workspace !== workspace || sdk.configFile !== path.join(sdkOutput, "native-windows-sdk.json") || sdk.tools &&
        JSON.stringify(sdk.tools) !== JSON.stringify(tools)) {throw new Error("Local Windows SDK producer returned an unexpected release descriptor");}
    fs.mkdirSync(distributionRoot, {mode: 0o700});
    const distribution = dependencies.buildDistribution || require("./build-local-native-distribution").buildLocalNativeDistribution,
      result = distribution(sdk, candidate, environment, {verifyScope}), publish = dependencies.publish || require("./build-local-native-command").publishLocalDistribution,
      output = publish(result, repo);
    published = true;
    logger.log("[native-windows] Built " + output.archive + "\n[native-windows] SHA256 " + output.sha256 + "; build audited, not runtime/publication qualified.");
    return output;
  } finally {
    if (published) {
      const current = fs.lstatSync(stage, {throwIfNoEntry: false});
      if (current?.isDirectory() && !current.isSymbolicLink() && current.dev === ownedStage.dev && current.ino === ownedStage.ino) {
        fs.rmSync(stage, {recursive: true});
      } else {logger.error("[native-windows] Preserved changed release stage: " + stage);}
    } else {
      logger.error("[native-windows] Preserved failed release stage: " + stage);
    }
  }
}

/** Launch one scope around the complete Linux graph; flock stays held while Windows interop builds Core.
 * @param {any} input @param {any} [dependencies] */
function launchWslWindowsWorker(input, dependencies = {}) {
  if ((dependencies.platform || process.platform) !== "linux" || (dependencies.arch || process.arch) !== "x64" ||
      (dependencies.observedNodeVersion || process.version) !== nodeVersion) {
    throw new Error("The local Windows release bootstrap requires Linux x64 and pinned build Node.js " + nodeVersion);
  }
  const local = dependencies.local || require("./build-local-native-sdk"),
    jobs = local.localBuildJobs({...process.env, ...(dependencies.environment || {})}), environment = /** @type {NodeJS.ProcessEnv} */ ({
      ...process.env, ...(dependencies.environment || {}), MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: String(jobs), OMP_NUM_THREADS: String(jobs),
      MOM_LOCAL_NATIVE_SCOPE: "1"}),
    tools = (dependencies.localTools || local.localTools)(environment),
    preflight = (dependencies.preflight || require("./build-local-native-linux-sdk").preflightLocalLinuxSdk)({environment, tools}),
    script = [
      // eslint-disable-next-line no-template-curly-in-string -- Established Engineering UID-scoped build lock.
      'if [ "${MOM_R_SH:-0}" != 1 ] && [ "${MOM_NATIVE_BUILD_LOCK:-0}" != 1 ]; then',
      "  lock_dir=$(git rev-parse --path-format=absolute --git-path mom-locks)",
      '  mkdir -p -m 700 "$lock_dir"',
      // eslint-disable-next-line no-template-curly-in-string -- Established Engineering UID-scoped build lock.
      '  exec 9>"$lock_dir/mom-r-sh-${UID}.build.lock"', "  flock 9", "fi", "export MOM_NATIVE_BUILD_LOCK=1", 'exec "$@"',
    ].join("\n"),
    command = [tools.systemdRun, "--user", "--scope", "--quiet", "--property=MemoryMax=" + local.localMemoryMax(environment),
      "--property=MemorySwapMax=0", "--", process.execPath, __filename, "--wsl-build", ...inputValues(input)],
    result = (dependencies.spawn || spawnSync)("/usr/bin/bash", ["-eu", "-c", script, "--", ...command],
      {cwd: repo, env: environment, stdio: "inherit", shell: false});
  if (preflight.tools && JSON.stringify(preflight.tools) !== JSON.stringify(tools)) {throw new Error("Linux tools changed during scope bootstrap");}
  if (result.error) {throw result.error;}
  return result.status ?? 1;
}

/** @param {Record<string, any>} input */
function inputValues(input) {
  return [input["checkout"], input["checkoutWindows"], input["powershellWindows"], input["supportWorkspace"], input["exportWorkspace"], input["bridgeRunWindows"]];
}

/** Linux entry before scope: no SDK builds or dependency installation happen here.
 * @param {any} input @param {any} [dependencies] */
function launchWslBootstrap(input, dependencies = {}) {
  if ((dependencies.platform || process.platform) !== "linux" || (dependencies.arch || process.arch) !== "x64") {
    throw new Error("The WSL bootstrap requires Linux x64");
  }
  const local = dependencies.local || require("./build-local-native-sdk"), environment = dependencies.environment || process.env,
    version = (dependencies.releaseVersion || require("./build-release").releaseVersion)(environment);
  if ((dependencies.observedNodeVersion || process.version) !== nodeVersion) {throw new Error("WSL requires pinned build Node.js " + nodeVersion);}
  verifyNodeVersionPin();
  const values = requireBuildFields(input, ["bridgeRunWindows", "checkout", "checkoutWindows", "exportWorkspace", "powershellWindows", "supportWorkspace"], "WSL Windows bootstrap input");
  if (values["checkout"] !== fs.realpathSync(repo)) {throw new Error("WSL checkout does not match the Windows-preflighted mapped repository");}
  const jobs = String(local.localBuildJobs(environment)),
    env = {...environment, MOM_AOT_BUILD_ONLY: "1", MOM_BUILD_JOBS: jobs, OMP_NUM_THREADS: jobs, MOM_RELEASE_VERSION: version};
  // This read-only tool admission fails before opening the WSL dependency cache or building source.
  const tools = (dependencies.localTools || local.localTools)(env), preflight = (dependencies.preflight || require("./build-local-native-linux-sdk").preflightLocalLinuxSdk)({environment: env, tools});
  return (dependencies.launchWorker || launchWslWindowsWorker)(values, {environment: env, local, localTools: () => tools,
    preflight: () => preflight, spawn: dependencies.spawn});
}

/** Internal CLI: Windows uses --wsl-launch; the scoped worker uses --wsl-build. */
/** @param {string[]} argv */
function main(argv) {
  const [mode, ...args] = argv;
  if (mode === "--wsl-launch" && args.length === 6) {
    const [checkout, checkoutWindows, powershellWindows, supportWorkspace, exportWorkspace, bridgeRunWindows] =
      /** @type {[string,string,string,string,string,string]} */ (args);
    const status = launchWslBootstrap({checkout, checkoutWindows, powershellWindows, supportWorkspace, exportWorkspace, bridgeRunWindows});
    if (status !== 0) {process.exitCode = status;}
    return;
  }
  if (mode === "--wsl-build" && args.length === 6) {
    const [checkout, checkoutWindows, powershellWindows, supportWorkspace, exportWorkspace, bridgeRunWindows] =
      /** @type {[string,string,string,string,string,string]} */ (args);
    void runLocalWindowsWslRelease({checkout, checkoutWindows, powershellWindows, supportWorkspace, exportWorkspace, bridgeRunWindows})
      .catch(error => {console.error("[native-windows] " + (error instanceof Error ? error.message : String(error))); process.exitCode = 1;});
    return;
  }
  throw new Error("Internal WSL release mode; use the existing native release entrypoint");
}

if (require.main === module) {
  try {main(process.argv.slice(2));} catch (error) {console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1;}
}

module.exports = {localWindowsReleaseRoots, ensureNativeWindowsReleaseRoots, windowsToolPreflight, launchLocalNativeWindowsRelease,
  windowsPreflightSource,
  mapWslPath, windowsPathToWsl, wslPathToWindows, readBoundedJson, runWindowsSupportBridge, admitWindowsBridgeResult,
  runLocalWindowsWslRelease, launchWslWindowsWorker, launchWslBootstrap, main, nodeVersion};
