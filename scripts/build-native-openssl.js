"use strict";

// Build-only original OpenSSL inputs for the native controller; sources are caller-supplied.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process"), {createHash} = require("node:crypto");
const {hasExactBuildFields, fileHash, sourceInputs, verifySourceInputs, readBuildJsonSnapshot} = require("./build-source-inputs");
const {runLoggedSdkStage} = require("./native-sdk-stage");
const {nativeSdkEnvironment} = require("./build-environment");
const {isStaticArchive} = require("./native-audit");
const {requireLinuxSysroot, linuxSysrootRecord, verifyLinuxSysrootRecord} = require("./native-linux-sysroot");
const repo = path.resolve(__dirname, "..");
const opensslPin = Object.freeze({version: "3.5.9", sha256: "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a",
  // Stable midnight UTC on the pinned source's VERSION.dat RELEASE_DATE (29 Sep 2026).
  sourceDateEpoch: 1790640000});
const configureOptions = Object.freeze(["no-shared", "no-module", "no-dso", "no-engine", "no-tests", "no-comp", "no-zlib", "no-zstd", "no-brotli"]);
const sourceLeaves = ["scripts/build-native-openssl.js", "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js",
  "scripts/native-linux-sysroot.js", "scripts/native-sdk-stage.js"];
const maxArchiveBytes = 128 << 20, maxInventoryBytes = 1 << 30, maxInventoryFiles = 100000;
const isHash = (/** @type {unknown} */ value) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);

/** @param {unknown} value @returns {value is string} */
function absolute(value) {
  return typeof value === "string" && value.length > 0 && value.length <= 4096 && path.isAbsolute(value) &&
    !/[;"'$%&|<>^!\r\n\0]/.test(value) && !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} platform */
function toolNames(platform) {return platform === "win32" ? ["ar", "cc", "link", "make", "perl", "rc", "tar"] : ["ar", "cc", "make", "perl", "ranlib", "tar"];}

/** @param {unknown} value @param {string} platform */
function validateTools(value, platform) {
  const names = toolNames(platform), tools = /** @type {Record<string, string>} */ (value);
  if (!hasExactBuildFields(tools, names) ||
      Object.values(tools).some(file => !absolute(file))) {throw new Error("OpenSSL SDK needs explicit absolute platform tools");}
  return tools;
}

/** OpenSSL's Windows templates quote raw tool paths; prequoting doubles the quotes.
 * GNU Make's tool variables instead need shell quotes for paths containing spaces.
 * @param {string} file @param {boolean} windows */
function makeTool(file, windows) {
  const value = windows ? file.replaceAll("\\", "/") : file;
  return windows ? value : /\s/.test(value) ? `'${value}'` : value;
}

/** Pure native build plan; NMake remains serial, while GNU Make uses the requested bounded jobs.
 * @param {string} output @param {unknown} tools @param {number} jobs @param {string} [platform] */
function opensslSdkPlan(output, tools, jobs, platform = process.platform) {
  if ((platform !== "linux" && platform !== "win32") || !absolute(output) || !Number.isInteger(jobs) || jobs < 1 || jobs > 16) {
    throw new Error("OpenSSL SDK needs Linux/Windows, an absolute output and jobs in 1..16");
  }
  const selected = validateTools(tools, platform), windows = platform === "win32";
  const tool = (/** @type {string} */ name) => {
    const file = selected[name]; if (!file) {throw new Error("OpenSSL SDK tool is missing: " + name);} return file;
  };
  const source = path.join(output, "source"), build = path.join(output, "build"), install = path.join(output, "install");
  // These are logical runtime paths, not build/staging destinations. Linux install_dev is
  // redirected into `install` with DESTDIR; Windows continues to stage from its build tree.
  const installPrefix = windows ? "C:/OpenSSL" : "/usr/local", openSslDir = windows ? "C:/OpenSSL/ssl" : "/etc/ssl";
  const stagedPrefix = windows ? null : path.join(install, "usr", "local");
  // OpenSSL's VC-WIN64A target uses /MT for static builds when no-shared is selected.
  const configure = [tool("perl"), path.join(source, "Configure").replaceAll("\\", "/"), windows ? "VC-WIN64A" : "linux-x86_64",
    "--prefix=" + installPrefix, "--libdir=lib", "--openssldir=" + openSslDir,
    ...configureOptions, ...(windows ? ["no-asm"] : [])];
  const make = windows ? [tool("make"), "/NOLOGO"] : [tool("make"), "-j", String(jobs)];
  // Separate wrapper goals launch recursive makes that race over the same object/depfile.
  // build_sw owns one dependency graph; the static configuration already omits modules/tests.
  // Windows install_dev unconditionally copies MSVC's ossl_static.pdb, which Intel's
  // CodeView objects do not produce. Stage its exact public header/library closure below.
  const buildCommand = [...make, "build_sw"], installCommand = windows ? [] : [...make, "DESTDIR=" + install, "install_dev"];
  const extension = windows ? ".lib" : ".a", executable = windows ? "openssl.exe" : "openssl";
  const toolEnvironment = windows ? {PERL: makeTool(tool("perl"), true), CC: makeTool(tool("cc"), true), AR: makeTool(tool("ar"), true),
    LD: makeTool(tool("link"), true), RC: makeTool(tool("rc"), true)} :
    {PERL: makeTool(tool("perl"), false), CC: makeTool(tool("cc"), false), AR: makeTool(tool("ar"), false),
      LD: makeTool(tool("cc"), false), RANLIB: makeTool(tool("ranlib"), false)};
  return {version: 1, platform, jobs, parallelJobs: windows ? 1 : jobs, source, build, install, stagedPrefix, installPrefix, openSslDir,
    sourceDateEpoch: opensslPin.sourceDateEpoch, sdk: output,
    configure, buildCommand, installCommand, toolEnvironment,
    libraries: {crypto: path.join(output, "lib", "libcrypto" + extension), ssl: path.join(output, "lib", "libssl" + extension)},
    headers: path.join(output, "include", "openssl"), executable: path.join(output, "bin", executable),
    buildExecutable: path.join(build, "apps", executable)};
}

/** Pin reproducibility inputs and remove caller-controlled make/install overrides.
 * DESTDIR is also set on the install command line so the staging root is explicit.
 * @param {NodeJS.ProcessEnv} environment @param {ReturnType<typeof opensslSdkPlan>} plan */
function opensslSdkEnvironment(environment, plan) {
  const env = nativeSdkEnvironment(environment), ignored = new Set([
    "MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS", "MAKEOVERRIDES", "MAKELEVEL", "MAKEFILES", "NMAKEFLAGS", "NMAKE",
    "PERL5OPT", "PERL5LIB", "AS", "ASFLAGS", "DESTDIR", "SOURCE_DATE_EPOCH",
  ]);
  for (const key of Object.keys(env)) {if (ignored.has(key.toUpperCase())) {delete env[key];}}
  env["SOURCE_DATE_EPOCH"] = String(plan.sourceDateEpoch);
  return Object.assign(env, plan.toolEnvironment);
}

/** Reject path aliases and symlinks before they can be read by a tool or copied into the SDK.
 * @param {string} file @param {string} label @param {boolean} [executable] */
function regularInput(file, label, executable = false) {
  if (!absolute(file)) {throw new Error("OpenSSL " + label + " must be an absolute path");}
  const stat = fs.lstatSync(file, {throwIfNoEntry: false});
  if (!stat?.isFile() || stat.isSymbolicLink() || !samePath(fs.realpathSync(file), file)) {throw new Error("OpenSSL " + label + " must be a canonical regular file");}
  if (executable && process.platform !== "win32") {fs.accessSync(file, fs.constants.X_OK);}
  return stat;
}

/** @param {string} left @param {string} right */
function samePath(left, right) {
  const a = path.resolve(left), b = path.resolve(right);
  return process.platform === "win32" ? a.toLowerCase() === b.toLowerCase() : a === b;
}

/** Only the checksum-pinned top-level regular-file/directory cohort is extractable. @param {string} listing @param {string} verbose */
function validateArchive(listing, verbose) {
  const lines = (/** @type {string} */ value) => {const rows = value.split(/\r?\n/); if (rows.at(-1) === "") {rows.pop();} return rows;};
  const names = lines(listing), types = lines(verbose), seen = new Set(), base = "openssl-" + opensslPin.version;
  if (!names.length || names.length > maxInventoryFiles || names.length !== types.length) {throw new Error("Invalid OpenSSL source archive listing");}
  for (const [index, name] of names.entries()) {
    const normalized = name.endsWith("/") ? name.slice(0, -1) : name, parts = normalized.split("/");
    if (!name || name.length > 4096 || /[\\\0\r\n]/.test(name) || Array.from(name).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127) ||
        parts[0] !== base || parts.some((/** @type {string} */ part) => !part || part === "." || part === "..") || seen.has(name) ||
        !["-", "d"].includes(types[index]?.[0] || "") || (parts.length === 1 && types[index]?.[0] !== "d")) {
      throw new Error("Unsafe or unexpected OpenSSL source archive member");
    }
    seen.add(name);
  }
  if (!seen.has(base + "/") || !seen.has(base + "/Configure") || !seen.has(base + "/LICENSE.txt")) {
    throw new Error("OpenSSL source archive lacks Configure or its license notice");
  }
  return names.length;
}

/** Hash only bounded regular files/directories; membership and bytes both affect the receipt.
 * @param {string} root */
function inventory(root) {
  const files = Object.create(null), digest = createHash("sha256"); let count = 0, bytes = 0;
  /** @param {string} directory */
  function visit(directory) {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("OpenSSL inventory contains a non-directory path");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(directory, entry.name), relative = path.relative(root, file).split(path.sep).join("/");
      if (!relative || relative.length > 4096 || /[\\\0\r\n]/.test(relative) || Array.from(relative).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) {
        throw new Error("OpenSSL inventory has an invalid path");
      }
      if (entry.isDirectory()) {visit(file); continue;}
      const child = fs.lstatSync(file);
      if (!entry.isFile() || child.isSymbolicLink() || !child.isFile() || ++count > maxInventoryFiles || (bytes += child.size) > maxInventoryBytes) {
        throw new Error("OpenSSL inventory is not bounded regular files");
      }
      const sha256 = fileHash(file); files[relative] = sha256; digest.update(relative + "\0" + sha256 + "\n");
    }
  }
  visit(root);
  if (!count) {throw new Error("OpenSSL inventory is empty");}
  return {files, count, bytes, sha256: digest.digest("hex")};
}

/** Require exact installed header/library/CLI membership; the build tree is evidence, not SDK input.
 * @param {string} root @param {string} platform */
function stagedFiles(root, platform) {
  const extension = platform === "win32" ? ".lib" : ".a", executable = platform === "win32" ? "openssl.exe" : "openssl";
  const expected = {include: ["openssl"], lib: ["libcrypto" + extension, "libssl" + extension], bin: [executable]}, files = Object.create(null);
  for (const [directory, names] of Object.entries(expected)) {
    const selected = path.join(root, directory), stat = fs.lstatSync(selected, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || stat.isSymbolicLink() || JSON.stringify(fs.readdirSync(selected).sort()) !== JSON.stringify(names.sort())) {
      throw new Error("Static OpenSSL SDK has unexpected staged members: " + directory);
    }
    const found = inventory(selected);
    for (const [relative, hash] of Object.entries(found.files)) {files[directory + "/" + relative] = hash;}
  }
  return files;
}

/** Admit a bounded native executable without running it. @param {string} file @param {string} platform */
function executableFormat(file, platform) {
  const stat = regularInput(file, "build CLI");
  if (stat.size < 64 || stat.size > (128 << 20)) {throw new Error("OpenSSL build CLI is not a bounded executable");}
  const fd = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NONBLOCK), header = Buffer.alloc(64);
  try {
    if (fs.readSync(fd, header, 0, header.length, 0) !== header.length) {throw new Error("OpenSSL build CLI has a truncated executable header");}
    if (platform === "linux" && header.toString("hex", 0, 4) !== "7f454c46") {throw new Error("OpenSSL build CLI is not ELF");}
    if (platform === "win32") {
      const offset = header.readUInt32LE(0x3c), signature = Buffer.alloc(4);
      if (header.toString("ascii", 0, 2) !== "MZ" || offset < header.length || offset > stat.size - 4 ||
          fs.readSync(fd, signature, 0, signature.length, offset) !== signature.length || signature.toString("hex") !== "50450000") {
        throw new Error("OpenSSL build CLI is not PE");
      }
    }
  } finally {fs.closeSync(fd);}
  if (platform === "linux" && !(stat.mode & 0o111)) {throw new Error("OpenSSL build CLI is not executable");}
  return stat;
}

/** Build from the exact original source archive; no download, host OpenSSL, CLI execution or release claim.
 * @param {string} archive @param {string} output @param {{version: number, jobs: number, tools: Record<string,string>}} config
 * @param {NodeJS.ProcessEnv} [environment] @param {string} [configSha256] */
function buildNativeOpenSsl(archive, output, config, environment = process.env, configSha256) {
  if (environment["MOM_AOT_BUILD_ONLY"] !== "1") {throw new Error("OpenSSL SDK production requires MOM_AOT_BUILD_ONLY=1");}
  if (!config || !hasExactBuildFields(config, ["jobs", "tools", "version"]) || config.version !== 1 ||
      !Number.isInteger(config.jobs) || config.jobs < 1 || config.jobs > 16) {throw new Error("Invalid OpenSSL SDK version-1 tool descriptor");}
  const platform = process.platform;
  if (platform !== "linux" && platform !== "win32") {throw new Error("Static OpenSSL production supports Linux and Windows only");}
  if (process.arch !== "x64") {throw new Error("Static OpenSSL production requires the x64 target");}
  const tools = validateTools(config.tools, platform), selectedOutput = path.resolve(output), parent = path.dirname(selectedOutput);
  const plan = opensslSdkPlan(selectedOutput, tools, config.jobs, platform), parentStat = fs.lstatSync(parent, {throwIfNoEntry: false});
  const relative = path.relative(repo, selectedOutput);
  if (!absolute(archive)) {throw new Error("OpenSSL source archive must be an absolute path");}
  if (!absolute(output) || output !== selectedOutput || !parentStat?.isDirectory() || parentStat.isSymbolicLink() ||
      !samePath(fs.realpathSync(parent), parent) || fs.lstatSync(selectedOutput, {throwIfNoEntry: false}) ||
      !relative || (!relative.startsWith(".." + path.sep) && !path.isAbsolute(relative))) {throw new Error("OpenSSL output must be new, canonical and outside the repository");}
  const archiveStat = regularInput(archive, "source archive");
  if (archiveStat.size > maxArchiveBytes) {throw new Error("OpenSSL source archive exceeds its size bound");}
  const toolHashes = /** @type {Record<string, {path: string, sha256: string}>} */ (Object.fromEntries(Object.entries(tools).map(([name, file]) => {
    regularInput(file, "tool", true); return [name, {path: file, sha256: fileHash(file)}];
  })));
  const archiveSha256 = fileHash(archive);
  if (archiveSha256 !== opensslPin.sha256) {throw new Error("OpenSSL source archive SHA256 differs from the pinned original 3.5.9 release");}
  const configHash = configSha256 || createHash("sha256").update(JSON.stringify(config)).digest("hex");
  if (!isHash(configHash)) {throw new Error("Invalid OpenSSL tool descriptor digest");}
  const recipe = sourceInputs(repo, [], sourceLeaves), env = opensslSdkEnvironment(environment, plan),
    sysroot = platform === "linux" ? requireLinuxSysroot(env) : null;
  const tar = tools["tar"] || "", listingResult = spawnSync(tar, ["-tzf", archive], {env, encoding: "utf8", timeout: 60000, maxBuffer: 16 << 20}),
    typesResult = spawnSync(tar, ["-tvzf", archive], {env, encoding: "utf8", timeout: 60000, maxBuffer: 16 << 20});
  for (const result of [listingResult, typesResult]) {
    if (result.error || result.signal || result.status !== 0 || result.stderr) {throw new Error("OpenSSL source archive listing failed", {cause: result.error});}
  }
  const sourceEntries = validateArchive(listingResult.stdout || "", typesResult.stdout || "");
  if (fileHash(archive) !== archiveSha256 || Object.values(toolHashes).some(row => fileHash(row.path) !== row.sha256)) {
    throw new Error("OpenSSL archive or tools changed during preflight");
  }
  fs.mkdirSync(selectedOutput, {mode: 0o700}); fs.mkdirSync(path.join(selectedOutput, "logs")); fs.mkdirSync(plan.source); fs.mkdirSync(plan.build);
  const runStage = (/** @type {string[]} */ command, /** @type {string} */ cwd, /** @type {string} */ stage) =>
    runLoggedSdkStage(command, cwd, env, selectedOutput, stage, "OpenSSL SDK", 45 * 60 * 1000);
  runStage([tar, "-xzf", archive, "-C", plan.source, "--strip-components=1"], selectedOutput, "extract");
  const sourceBefore = inventory(plan.source), license = path.join(plan.source, "LICENSE.txt");
  if (!fs.statSync(path.join(plan.source, "Configure")).isFile()) {throw new Error("Pinned OpenSSL source lacks Configure");}
  const stages = [plan.configure, plan.buildCommand, ...(plan.installCommand.length ? [plan.installCommand] : [])];
  for (const [index, command] of stages.entries()) {runStage(command, plan.build, ["configure", "build", "install-dev"][index] || "stage");}

  fs.mkdirSync(path.join(selectedOutput, "include"));
  if (platform === "win32") {stageWindowsHeaders(plan.source, plan.build, plan.headers);} else {
    copyTree(path.join(/** @type {string} */ (plan.stagedPrefix), "include", "openssl"), plan.headers);
  }
  fs.mkdirSync(path.dirname(plan.libraries.crypto));
  const libraryRoot = platform === "win32" ? plan.build : path.join(/** @type {string} */ (plan.stagedPrefix), "lib");
  const libraryRows = [{key: "crypto", installed: path.join(libraryRoot, "libcrypto" + (platform === "win32" ? ".lib" : ".a")), staged: plan.libraries.crypto},
    {key: "ssl", installed: path.join(libraryRoot, "libssl" + (platform === "win32" ? ".lib" : ".a")), staged: plan.libraries.ssl}];
  for (const {key, installed, staged} of libraryRows) {
    regularInput(installed, key + " static archive");
    if (!isStaticArchive(installed)) {throw new Error("OpenSSL output is not a static archive: " + key);}
    fs.copyFileSync(installed, staged, fs.constants.COPYFILE_EXCL);
    if (!isStaticArchive(staged) || fileHash(staged) !== fileHash(installed)) {throw new Error("Staged OpenSSL archive changed during copy: " + key);}
  }
  fs.mkdirSync(path.dirname(plan.executable));
  regularInput(plan.buildExecutable, "build CLI"); fs.copyFileSync(plan.buildExecutable, plan.executable, fs.constants.COPYFILE_EXCL);
  if (platform === "linux") {fs.chmodSync(plan.executable, fs.statSync(plan.buildExecutable).mode & 0o777);}
  executableFormat(plan.executable, platform);
  const sourceAfter = inventory(plan.source);
  if (sourceBefore.sha256 !== sourceAfter.sha256 || sourceBefore.count !== sourceAfter.count || sourceBefore.bytes !== sourceAfter.bytes) {
    throw new Error("OpenSSL extracted source changed during the out-of-tree build");
  }
  verifySourceInputs(recipe, sourceInputs(repo, [], Object.keys(recipe.graph)), "OpenSSL SDK producer");
  if (fileHash(archive) !== archiveSha256 || Object.values(toolHashes).some(row => fileHash(row.path) !== row.sha256)) {
    throw new Error("OpenSSL archive or tools changed during compilation");
  }
  const artifacts = stagedFiles(selectedOutput, platform), result = {version: 1, opensslVersion: opensslPin.version, platform, target: platform === "win32" ? "VC-WIN64A" : "linux-x86_64",
    archive: {path: archive, sha256: archiveSha256, bytes: archiveStat.size, members: sourceEntries},
    source: {files: sourceBefore.count, bytes: sourceBefore.bytes, sha256: sourceBefore.sha256,
      notice: {path: "source/LICENSE.txt", sha256: fileHash(license)}},
    config: {sha256: configHash}, tools: toolHashes, recipe, configure: plan.configure, sourceDateEpoch: plan.sourceDateEpoch, artifacts,
    ...(sysroot ? {sysroot: linuxSysrootRecord(sysroot)} : {}), buildOnly: true, executed: false, publicationQualified: false,
    scope: "Static OpenSSL 3.5.9 libraries, headers and build-only CLI; the CLI is not executed or published."};
  fs.writeFileSync(path.join(selectedOutput, "result.json"), JSON.stringify(result, null, 2) + "\n", {flag: "wx"});
  const inputs = nativeOpenSslInputs(selectedOutput, platform);
  console.log("[native-sdk] Built static OpenSSL 3.5.9; build CLI not executed or published.");
  return {root: selectedOutput, ...inputs};
}

/** Match upstream install_dev's public .h union; .in templates and optional PDBs are not
 * application inputs. Exclusive copies reject any unexpected source/generated collision.
 * @param {string} source @param {string} build @param {string} target */
function stageWindowsHeaders(source, build, target) {
  fs.mkdirSync(target);
  for (const root of [source, build]) {
    const directory = path.join(root, "include", "openssl"), stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("OpenSSL needs regular public headers");}
    for (const entry of fs.readdirSync(directory, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
      if (!entry.name.endsWith(".h")) {continue;}
      if (!entry.isFile()) {throw new Error("OpenSSL needs regular public headers");}
      fs.copyFileSync(path.join(directory, entry.name), path.join(target, entry.name), fs.constants.COPYFILE_EXCL);
    }
  }
}

/** Copy only regular files/directories from the generated OpenSSL header tree. @param {string} source @param {string} target */
function copyTree(source, target) {
  const stat = fs.lstatSync(source);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {throw new Error("OpenSSL headers must be a real directory");}
  fs.mkdirSync(target, {recursive: true});
  for (const entry of fs.readdirSync(source, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    const from = path.join(source, entry.name), to = path.join(target, entry.name);
    if (entry.isDirectory()) {copyTree(from, to); continue;}
    if (entry.isFile() && !fs.lstatSync(from).isSymbolicLink()) {fs.copyFileSync(from, to, fs.constants.COPYFILE_EXCL); continue;}
    throw new Error("OpenSSL staged headers cannot contain symlinks or special files");
  }
}

/** Revalidate the exact controller-consumed SDK closure before native compilation.
 * @param {string} root @param {string} [platform]
 * @returns {{include: string, headers: string, libraries: string[], executable: string, artifacts: Record<string, string>,
 * files: Record<string, string>, receipt: string, receiptSha256: string, notices: {component: string, path: string, sha256: string}[]}} */
function nativeOpenSslInputs(root, platform = process.platform) {
  if ((platform !== "linux" && platform !== "win32") || !absolute(root) || !samePath(fs.realpathSync(root), root)) {throw new Error("Select a canonical OpenSSL SDK root");}
  const receiptPath = path.join(root, "result.json"), {value: receipt, sha256: receiptSha256} = readBuildJsonSnapshot(receiptPath),
    extension = platform === "win32" ? ".lib" : ".a";
  if (receipt.version !== 1 || receipt.platform !== platform || receipt.target !== (platform === "win32" ? "VC-WIN64A" : "linux-x86_64") || receipt.opensslVersion !== opensslPin.version ||
      receipt.archive?.sha256 !== opensslPin.sha256 || !absolute(receipt.archive?.path) || !Number.isInteger(receipt.archive?.bytes) ||
      receipt.archive.bytes <= 0 || receipt.archive.bytes > maxArchiveBytes || !Number.isInteger(receipt.archive.members) ||
      receipt.archive.members < 1 || receipt.archive.members > maxInventoryFiles || receipt.sourceDateEpoch !== opensslPin.sourceDateEpoch ||
      receipt.buildOnly !== true || receipt.executed !== false || receipt.publicationQualified !== false || !isHash(receipt.config?.sha256) ||
      !receipt.source || !Number.isInteger(receipt.source.files) || receipt.source.files < 1 || !Number.isInteger(receipt.source.bytes) ||
      receipt.source.bytes < 1 || !isHash(receipt.source.sha256) || receipt.source.notice?.path !== "source/LICENSE.txt" || !isHash(receipt.source.notice.sha256)) {
    throw new Error("Invalid static OpenSSL SDK receipt");
  }
  if (platform === "linux") {verifyLinuxSysrootRecord(receipt.sysroot, "Static OpenSSL SDK");}
  const selectedTools = toolNames(platform), recordedTools = receipt.tools;
  if (!hasExactBuildFields(recordedTools, selectedTools) ||
      Object.values(recordedTools).some(row => !row || !absolute(row.path) || !isHash(row.sha256))) {throw new Error("Static OpenSSL SDK tool receipt is incomplete");}
  const expectedConfigure = opensslSdkPlan(root, Object.fromEntries(Object.entries(recordedTools).map(([name, row]) => [name, row.path])), 1, platform).configure;
  if (JSON.stringify(receipt.configure) !== JSON.stringify(expectedConfigure)) {throw new Error("Static OpenSSL SDK Configure plan mismatch");}
  const license = path.join(root, "source", "LICENSE.txt"); regularInput(license, "license notice");
  if (fileHash(license) !== receipt.source.notice.sha256) {throw new Error("Static OpenSSL source notice changed");}
  verifySourceInputs(receipt.recipe, sourceInputs(repo, [], sourceLeaves), "OpenSSL SDK producer");
  const artifacts = stagedFiles(root, platform), expected = receipt.artifacts;
  if (!expected || typeof expected !== "object" || Array.isArray(expected) || Object.keys(expected).length !== Object.keys(artifacts).length ||
      Object.entries(artifacts).some(([file, hash]) => expected[file] !== hash)) {throw new Error("Static OpenSSL SDK staged membership or bytes changed");}
  const headers = path.join(root, "include", "openssl"), libraries = [path.join(root, "lib", "libcrypto" + extension), path.join(root, "lib", "libssl" + extension)];
  for (const file of libraries) {if (!isStaticArchive(file)) {throw new Error("Static OpenSSL SDK library is not an archive");}}
  const executable = path.join(root, "bin", platform === "win32" ? "openssl.exe" : "openssl"); executableFormat(executable, platform);
  const versionHeader = fs.readFileSync(path.join(headers, "opensslv.h"), "utf8");
  if (!versionHeader.includes('OPENSSL_VERSION_STR "' + opensslPin.version + '"')) {throw new Error("Static OpenSSL headers have the wrong version");}
  const notices = [{component: "OpenSSL LICENSE", path: license, sha256: receipt.source.notice.sha256}];
  const files = /** @type {Record<string, string>} */ (Object.create(null));
  for (const [relative, sha256] of Object.entries(artifacts)) {files[path.join(root, relative)] = sha256;}
  files[license] = receipt.source.notice.sha256; files[receiptPath] = receiptSha256;
  for (const [file, sha256] of Object.entries(files)) {
    if (fileHash(file) !== sha256) {throw new Error("Static OpenSSL SDK input changed during admission: " + file);}
  }
  if (fileHash(receiptPath) !== receiptSha256) {throw new Error("Static OpenSSL SDK receipt changed during admission");}
  return {include: path.join(root, "include"), headers, libraries, executable, artifacts, files, receipt: receiptPath, receiptSha256, notices};
}

if (require.main === module) {
  try {
    const [archive, output, configFile, extra] = process.argv.slice(2);
    if (!archive || !output || !configFile || extra || !absolute(configFile)) {throw new Error("Usage: MOM_AOT_BUILD_ONLY=1 node scripts/build-native-openssl.js /pinned/openssl-3.5.9.tar.gz /new/sdk /absolute/tools.json");}
    regularInput(configFile, "tool descriptor");
    const {value: config, sha256} = readBuildJsonSnapshot(configFile);
    buildNativeOpenSsl(archive, output, config, process.env, sha256);
  } catch (error) {console.error(error instanceof Error ? error.message : "Static OpenSSL build failed"); process.exitCode = 1;}
}

module.exports = {opensslPin, configureOptions, toolNames, opensslSdkPlan, validateArchive, inventory, stagedFiles, opensslBuildRecipe: () => sourceInputs(repo, [], sourceLeaves),
  opensslSdkEnvironment, nativeOpenSslInputs, buildNativeOpenSsl, stageWindowsHeaders};
