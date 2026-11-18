"use strict";

// Read-only admission for Linux SDK receipts which do not yet have producer-side readers.
// This proves local receipt continuity only; it does not qualify a native release.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {isDeepStrictEqual} = require("node:util");
const {fileHash, readBuildJson, readBuildText, sourceInputs, verifySourceInputs} = require("./build-source-inputs");
const {isStaticArchive} = require("./native-audit");
const {hermesPins, hermesSourceFiles} = require("./build-native-hermes");
const {comgrSdkInputs} = require("./native-comgr-inputs");
const {admitSyclSource, rocmInputs} = require("./native-sycl-inputs");
const {elfReport} = require("./native-elf");

const repo = path.resolve(__dirname, "..");
const syclRevision = "da2463e45aa90aa36306c45abcfc05b87de51bc6";
const openmpContainerPath = "/openmp/libomp.a";
const openmpProviderFile = /^lib(?:omp|iomp5|gomp)(?:\.a|\.so(?:\.[0-9]+)*)?$/i;
const syclRecipe = ["scripts/build-native-sycl-sdk.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js", "scripts/native-jit-link.js",
  "tests/reference/native-sycl-hcf-repro.cpp",
  "tests/reference/native-jit-cache.cpp",
  "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-resources.js",
  "runtime/adaptivecpp-static.patch", "runtime/adaptivecpp-jit.patch"];
const hermesRecipe = ["scripts/build-native-hermes.js", "scripts/build-source-inputs.js", "scripts/build-environment.js",
  "scripts/native-audit.js", "runtime/hermes-host-rejections.patch"];
const rocmRecipe = ["scripts/build-native-rocm-host-runtime.js", "scripts/build-native-sycl-sdk.js", "scripts/build-native-comgr-sdk.js",
  "scripts/native-comgr-inputs.js", "scripts/prepare-native-sdk-sources.js", "scripts/build-native-release.js",
  "scripts/build-source-inputs.js", "scripts/build-environment.js", "scripts/native-audit.js", "scripts/native-sycl-inputs.js", "scripts/native-hcf.js"];
const supportLinks = new Set(["dl", "pthread", "m", "z", "zstd", "stdc++fs"]);
const digest = /^[0-9a-f]{64}$/;
/**
 * JSON fields are still validated at each producer boundary; the named keys allow checked
 * schema access while the open index handles the exact dynamic path/hash maps.
 * @typedef {Record<string, unknown> & {
 * graph?: any, files?: any, libraries?: any, roots?: {device?: any, lld?: any, [key: string]: any}, device?: any, lld?: any, revision?: any,
 * flags?: any, links?: any, source?: any, build?: any, cxx?: any, comgr?: any, receipt?: any,
 * version?: any, backend?: any, cudaVersion?: any, scope?: any, inputs?: any, buildOnly?: any,
 * publicationQualified?: any, executed?: any, recipe?: any, admitted?: any, commands?: any,
 * hostCompilers?: any, hostCompilerInputs?: any, compilerVersions?: any, fingerprints?: any,
 * artifacts?: any, target?: any, pins?: any, sources?: any, path?: any, sha256?: any, bytes?: any,
 * staged?: any, package?: any, architecture?: any, status?: any, stagedSha256?: any, osOrigins?: any,
 * consumerAdmission?: any, sdkInputs?: any, dependencies?: any, configured?: any, name?: any, prefix?: any,
 * sourceArchives?: any, archivesAfter?: any, archivesBefore?: any, stagedDependencies?: any,
 * cache?: any, database?: any, depfiles?: any, commandCount?: any, sdk?: any, needed?: any,
 * archives?: any, file?: any, constructorScript?: any, byteExactOracle?: any
 * }} JsonRecord
 */

/** @param {unknown} value @returns {value is JsonRecord} */
function record(value) {return value !== null && typeof value === "object" && !Array.isArray(value);}

/** Keep path admission independent of the release composer; it may import this reader. @param {unknown} value @returns {value is string} */
function absolutePath(value) {
  return typeof value === "string" && value.length <= 4096 && path.isAbsolute(value) &&
    !Array.from(value).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127);
}

/** @param {string} file @param {string} label @param {"file" | "directory"} [kind] */
function canonicalPath(file, label, kind = "file") {
  if (!absolutePath(file) || path.resolve(file) !== file) {throw new Error(`${label} must be an absolute canonical path`);}
  const stat = fs.lstatSync(file);
  if ((kind === "file" && !stat.isFile()) || (kind === "directory" && !stat.isDirectory()) || fs.realpathSync(file) !== file) {
    throw new Error(`${label} must be a canonical regular ${kind}`);
  }
  return file;
}

/** Fingerprinted inputs may be selected through producer-recorded tool/archive symlinks.
 * Keep their literal path, but bind the regular canonical target before hashing.
 * @param {string} file @param {string} label */
function canonicalInputPath(file, label) {
  if (!absolutePath(file) || path.resolve(file) !== file || !fs.statSync(file).isFile() || !path.isAbsolute(fs.realpathSync(file))) {
    throw new Error(`${label} must resolve to a canonical regular file`);
  }
  return file;
}

/** CMake may retain an SDK symlink path while the producer records its resolved archive target.
 * Admit only symlink spellings which resolve to that exact selected canonical file.
 * @param {string | undefined} cachePath @param {string} admittedPath */
function cachePathMatches(cachePath, admittedPath) {
  if (!cachePath || !absolutePath(cachePath) || path.resolve(cachePath) !== cachePath) {return false;}
  try {return fs.realpathSync(cachePath) === canonicalPath(admittedPath, "Admitted CUDA archive");} catch {return false;}
}

/** Existing proof executables must be little-endian ELF64 for the Linux x86-64 target.
 * ELF headers do not encode the full compiler triple. @param {string} file @param {string} label */
function verifyX64Elf(file, label) {
  const header = elfReport(file, "--file-header");
  if (!/^\s*Class:\s+ELF64\s*$/m.test(header) || !/^\s*Data:\s+2's complement, little endian\s*$/m.test(header) ||
      !/^\s*Machine:\s+Advanced Micro Devices X86-64\s*$/m.test(header)) {throw new Error(`${label} is not Linux ELF64 x86-64`);}
}

/** NVRTC records exact lib64 paths which can be producer-created symlinks into sibling
 * extracted packages. Admit only regular archive targets within that SDK input root.
 * @param {string} file @param {string} sdk */
function nvrtcArchive(file, sdk) {
  if (!absolutePath(file) || path.resolve(file) !== file) {throw new Error("NVRTC archive path is not canonical");}
  const target = fs.realpathSync(file), root = fs.realpathSync(path.dirname(sdk)), relative = path.relative(root, target);
  if (relative === ".." || relative.startsWith(".." + path.sep) || path.isAbsolute(relative) || !fs.statSync(target).isFile() || !isStaticArchive(target)) {
    throw new Error("NVRTC archive target escapes its selected input root or is not static");
  }
  return file;
}

/** Read a canonical bounded producer receipt and retain its hash across admission. @param {string} file */
function readReceipt(file) {
  canonicalPath(file, "SDK receipt");
  const sha256 = fileHash(file), value = readBuildJson(file);
  if (!record(value)) {throw new Error("SDK receipt must contain a JSON object");}
  return {file, sha256, value};
}

/** @param {unknown} value @param {string} label @param {number} [maximum] @returns {JsonRecord} */
function boundedRecord(value, label, maximum = 8192) {
  if (!record(value) || Object.keys(value).length > maximum) {throw new Error(`Invalid or excessive ${label}`);}
  return /** @type {Record<string, unknown>} */ (value);
}

/** Reject fields not written by the selected producer schema. @param {unknown} value @param {string[]} keys @param {string} label */
function exactRecord(value, keys, label) {
  const result = boundedRecord(value, label, keys.length);
  if (Object.keys(result).sort().join("\n") !== [...keys].sort().join("\n")) {throw new Error(`${label} membership differs`);}
  return result;
}

/** Bind declared hash maps to current canonical regular files. The known container OpenMP
 * path is remapped only to the admitted SDK archive; a different ambient file is never used.
 * @param {Array<Record<string, string>>} maps @param {string} [openmpArchive] */
function verifyHashMaps(maps, openmpArchive = undefined) {
  /** @type {Map<string, {sha256: string, paths: Set<string>}>} */ const selected = new Map();
  /** @type {Record<string, string>} */ const bindings = {};
  for (const hashes of maps) {
    if (Object.keys(hashes).length > 20000) {throw new Error("SDK receipt has too many input hashes");}
    for (const [declared, sha256] of Object.entries(hashes)) {
      if (!absolutePath(declared) || typeof sha256 !== "string" || !digest.test(sha256)) {throw new Error("Invalid SDK input hash row");}
      let file = declared;
      if (declared === openmpContainerPath && openmpArchive) {
        file = canonicalPath(openmpArchive, "Admitted OpenMP archive");
        const ambient = fs.lstatSync(declared, {throwIfNoEntry: false});
        if (ambient) {
          const sameFile = (() => {
            try {
              const ambientPath = fs.realpathSync(declared), admittedPath = fs.realpathSync(file), ambientStat = fs.statSync(declared), admittedStat = fs.statSync(file);
              return ambientPath === admittedPath || ambientStat.dev === admittedStat.dev && ambientStat.ino === admittedStat.ino;
            } catch {return false;}
          })();
          if (!sameFile) {throw new Error("Ambient OpenMP mount conflicts with the admitted SDK archive");}
        }
        bindings[declared] = file;
      }
      const realFile = canonicalInputPath(file, "SDK input"), realPath = fs.realpathSync(realFile), previous = selected.get(realPath);
      if (previous && previous.sha256 !== sha256) {throw new Error("Conflicting SDK hashes resolve to one file: " + file);}
      if (previous) {previous.paths.add(realFile);} else {selected.set(realPath, {sha256, paths: new Set([realFile])});}
    }
  }
  /** @type {Record<string, string>} */ const files = {};
  for (const [file, {sha256, paths}] of selected) {
    canonicalInputPath(file, "SDK input");
    if ([file, ...paths].some(name => name.endsWith(".a")) && !isStaticArchive(file)) {throw new Error("SDK input is not a regular static archive: " + file);}
    if (fileHash(file) !== sha256) {throw new Error("SDK input differs from its receipt: " + file);}
    for (const alias of paths) {
      if (fs.realpathSync(alias) !== file) {throw new Error("SDK input path changed while admitting: " + alias);}
      files[alias] = sha256;
    }
  }
  if (openmpArchive && !Object.hasOwn(bindings, openmpContainerPath) && !maps.some(hashes => Object.hasOwn(hashes, openmpArchive))) {
    throw new Error("Admitted OpenMP archive is absent from the AdaptiveCpp receipt");
  }
  return {files, bindings};
}

/** With an admitted archive, remap only the producer's exact container mount and reject other
 * OpenMP provider names in direct or linker-wrapped tokens.
 * @param {string[]} links @param {string | undefined} openmpArchive */
function bindOpenmpLinks(links, openmpArchive) {
  const archive = openmpArchive || "";
  if (!archive) {return links;}
  /** Match only the known alternate OpenMP providers; unrelated static/runtime libraries remain valid.
   * @param {string} token */
  const alternateProvider = token => {
    if (token.startsWith("-l")) {
      const requested = token.slice(2);
      return openmpProviderFile.test(requested.startsWith(":") ? requested.slice(1) : "lib" + requested);
    }
    return openmpProviderFile.test(path.basename(token));
  };
  /** @param {string} token */
  function bind(token) {
    if (token === openmpContainerPath || token === archive) {return archive;}
    if (token.startsWith("-Wl,")) {
      const parts = token.slice(4).split(",");
      if (parts.some((part, index) =>
        ((part.includes("/openmp") || alternateProvider(part)) && part !== archive && part !== openmpContainerPath) ||
        (part === "-l" && alternateProvider("-l" + (parts[index + 1] || ""))))) {
        throw new Error("AdaptiveCpp link bypasses the admitted OpenMP archive");
      }
      return "-Wl," + parts.map(part => part === openmpContainerPath || part === archive ? archive : part).join(",");
    }
    if (token.includes("/openmp") || alternateProvider(token)) {
      throw new Error("AdaptiveCpp link bypasses the admitted OpenMP archive");
    }
    return token;
  }
  const result = links.map(bind);
  for (let i = 0; i + 1 < links.length; i++) {
    if ((links[i] === "-L" && links[i + 1] === "/openmp") ||
        (links[i] === "-l" && alternateProvider("-l" + (links[i + 1] || "")))) {
      throw new Error("AdaptiveCpp link bypasses the admitted OpenMP archive");
    }
  }
  return result;
}

/** Require an exact maintained producer recipe, then compare its recorded bytes to this tree.
 * @param {unknown} value @param {string[]} files @param {string} label */
function verifyRecipe(value, files, label) {
  const recipe = boundedRecord(value, `${label} recipe`, 2), graph = boundedRecord(recipe.graph, `${label} recipe graph`, 32);
  const sourceFiles = boundedRecord(recipe.files, `${label} source inventory`, 1);
  if (Object.keys(sourceFiles).length || Object.keys(graph).sort().join("\n") !== [...files].sort().join("\n")) {
    throw new Error(`${label} recipe membership differs`);
  }
  verifySourceInputs(value, sourceInputs(repo, [], files), label + " recipe");
}

/** Invoke only Git's local, read-only metadata commands with bounded output. @param {string} command @param {string[]} args */
function gitOutput(command, args) {
  const env = {...process.env};
  for (const name of Object.keys(env)) {if (/^GIT_|^LD_(?:PRELOAD|AUDIT)$/.test(name)) {delete env[name];}}
  Object.assign(env, {GIT_OPTIONAL_LOCKS: "0", GIT_NO_REPLACE_OBJECTS: "1", GIT_TERMINAL_PROMPT: "0", GIT_NO_LAZY_FETCH: "1"});
  const result = spawnSync(command === "git" ? "/usr/bin/git" : command, args,
    {encoding: "utf8", env, timeout: 30000, maxBuffer: 8 << 20});
  if (result.error || result.signal || result.status !== 0 || result.stderr) {throw new Error("SDK source Git admission failed", {cause: result.error});}
  return result.stdout;
}

/** @param {string} text @param {string} key @param {string} expected */
function requireCacheValue(text, key, expected) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), matches = [...text.matchAll(new RegExp(`^${escaped}:[^=\\r\\n]*=([^\\r\\n]*)$`, "gm"))];
  if (matches.length !== 1 || matches[0]?.[1] !== expected) {throw new Error("Native SDK CMake cache differs: " + key);}
}

/** Validate the reviewed AdaptiveCpp static-GPU configure contract without invoking tools.
 * @param {string} build @param {string} source @param {string} backend @param {JsonRecord} inputs */
function verifySyclCache(build, source, backend, inputs) {
  const cache = readBuildText(path.join(build, "CMakeCache.txt"), 4 << 20), backends = backend.split("-");
  requireCacheValue(cache, "CMAKE_HOME_DIRECTORY", source);
  requireCacheValue(cache, "ACPP_STATIC_RUNTIME", "ON");
  for (const name of ["CUDA", "ROCM", "OPENCL", "LEVEL_ZERO", "VULKAN"]) {
    const key = "WITH_" + name + "_BACKEND", active = backends.includes(name === "ROCM" ? "hip" : name.toLowerCase());
    const match = [...cache.matchAll(new RegExp(`^${key}:(?:BOOL|UNINITIALIZED)=(ON|OFF)$`, "gm"))];
    if (match.length !== 1 || match[0]?.[1] !== (active ? "ON" : "OFF")) {throw new Error("Native SDK CMake backend differs: " + key);}
  }
  requireCacheValue(cache, "ACPP_CUSTOM_PROFILE_WITH_SSCP_COMPILER", "ON");
  requireCacheValue(cache, "ACPP_CUSTOM_PROFILE_WITH_STDPAR_COMPILER", "OFF");
  requireCacheValue(cache, "ACPP_CUSTOM_PROFILE_WITH_ACCELERATED_CPU", "OFF");
  const generator = path.join(repo, "scripts/native-resources.js");
  const generatorLine = [...cache.matchAll(/^ACPP_RESOURCE_GENERATOR:FILEPATH=([^\r\n]+)$/gm)];
  // Producer and consumer checkouts can differ; the recipe binds generator bytes, not its installation prefix.
  if (generatorLine.length !== 1 || fileHash(canonicalPath(generatorLine[0]?.[1] || "", "SDK resource generator")) !== fileHash(generator)) {
    throw new Error("Native SDK resource generator differs");
  }
  if (backends.includes("cuda")) {
    const libraries = /** @type {string[]} */ (inputs.libraries);
    const cuda = libraries.filter(file => ["libcudart_static.a", "libculibos.a"].includes(path.basename(file)));
    if (cuda.length !== 2) {throw new Error("Native SDK CUDA runtime pair is incomplete");}
    // Defer the producer helper so build-native-release can import this reader without a
    // module-initialization cycle through build-native-sycl-sdk.js.
    require("./build-native-sycl-sdk").admitCudaLibraries(cuda);
    const byName = /** @type {Record<string, string>} */ (Object.fromEntries(cuda.map(file => [path.basename(file), file])));
    for (const [key, name] of /** @type {Array<[string, string]>} */ ([
      ["CUDA_cudart_static_LIBRARY", "libcudart_static.a"], ["CUDA_culibos_LIBRARY", "libculibos.a"]])) {
      const value = [...cache.matchAll(new RegExp(`^${key}:[^=]+=([^\\r\\n]+)$`, "gm"))];
      if (value.length !== 1 || !cachePathMatches(value[0]?.[1], byName[name] || "")) {throw new Error("Native SDK CUDA cache differs: " + key);}
    }
  }
  if (backends.includes("hip")) {
    const comgr = /** @type {{roots?: {device?: string, lld?: string}}} */ (inputs.comgr);
    const hip = /** @type {string[]} */ (inputs.libraries).find(file => path.basename(file) === "libamdhip64.a");
    if (!hip || !comgr?.roots?.device || !comgr.roots.lld) {throw new Error("Native SDK HIP runtime roots are incomplete");}
    requireCacheValue(cache, "CMAKE_DISABLE_FIND_PACKAGE_HIP", "ON");
    requireCacheValue(cache, "ROCM_PATH", path.dirname(path.dirname(hip)));
    requireCacheValue(cache, "ROCM_DEVICE_LIBS_PATH", path.join(comgr.roots.device, "amdgcn/bitcode"));
    requireCacheValue(cache, "LLD_DIR", path.join(comgr.roots.lld, "lib/cmake/lld"));
  }
}

/** Resolve the producer's named static libraries and hash explicit path/link inputs.
 * OS support names remain tokens for the later final-link policy.
 * @param {string[]} tokens @param {Record<string,string>} files */
function resolveStaticLinks(tokens, files) {
  const roots = /** @type {string[]} */ ([]), resolved = /** @type {Record<string, string>} */ ({}), selected = {...files};
  /** @param {string} file */
  const add = file => {
    canonicalPath(file, "Native SDK link input");
    if (file.endsWith(".a") && !isStaticArchive(file)) {throw new Error("Native SDK link input is not static: " + file);}
    const sha256 = fileHash(file);
    if (selected[file] && selected[file] !== sha256) {throw new Error("Native SDK link hash conflicts with its receipt: " + file);}
    selected[file] = sha256;
  };
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i] || "";
    if (token === "-L") {
      const root = tokens[++i];
      canonicalPath(root || "", "Native SDK library root", "directory"); roots.push(root || "");
    } else if (token.startsWith("-L")) {
      const root = token.slice(2); canonicalPath(root, "Native SDK library root", "directory"); roots.push(root);
    }
  }
  for (const token of tokens) {
    if (token.startsWith("-l")) {
      const name = token.slice(2);
      if (!/^[A-Za-z0-9_+.-]{1,128}$/.test(name)) {throw new Error("Invalid native SDK library token");}
      if (supportLinks.has(name)) {continue;}
      const candidates = roots.map(root => path.join(root, "lib" + name + ".a")).filter(file => fs.existsSync(file));
      const candidate = candidates[0];
      if (candidates.length !== 1 || !candidate) {throw new Error(candidates.length ? "Ambiguous static native SDK library: " + name : "Missing static native SDK library: " + name);}
      add(candidate); resolved[token] = candidate;
    } else if (absolutePath(token) && !roots.includes(token)) {
      if (!selected[token]) {throw new Error("Direct SDK link is absent from producer fingerprints: " + token);}
      add(token);
    }
    if (token.startsWith("-Wl,")) {
      for (const part of token.slice(4).split(",")) {
        if (absolutePath(part)) {
          if (!selected[part]) {throw new Error("Linker path is absent from producer fingerprints: " + part);}
          add(part);
        }
      }
    }
    if (token.startsWith("--ld-path=")) {
      const linker = token.slice("--ld-path=".length);
      if (!selected[linker]) {throw new Error("Selected LLD is absent from producer fingerprints: " + linker);}
      add(linker);
    }
  }
  return {files: selected, roots, resolved};
}

/** Read/admit the exact static AdaptiveCpp closure emitted by build-native-sycl-sdk.js.
 * The only remappable producer path is `/openmp/libomp.a`, and it must be bound to an admitted
 * OpenMP SDK root rather than a caller-selected archive.
 * @param {string} receiptFile @param {{openmpSdkRoot?: string, gitRunner?: (command: string, args: string[]) => string,
 * readComgr?: typeof comgrSdkInputs, readOpenmp?: (root: string) => {receipt: string, receiptSha256: string, archive: string, files: Record<string,string>}}} [options] */
function nativeStaticRuntimeInputs(receiptFile, options = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("AdaptiveCpp SDK receipt is Linux x86-64 only");}
  if (!record(options) || Object.keys(options).some(key => !["openmpSdkRoot", "gitRunner", "readComgr", "readOpenmp"].includes(key)) ||
      options.openmpSdkRoot !== undefined && (!absolutePath(options.openmpSdkRoot) || path.resolve(options.openmpSdkRoot) !== options.openmpSdkRoot) ||
      options.gitRunner !== undefined && typeof options.gitRunner !== "function" ||
      options.readComgr !== undefined && typeof options.readComgr !== "function" ||
      options.readOpenmp !== undefined && (typeof options.readOpenmp !== "function" || options.openmpSdkRoot === undefined)) {
    throw new Error("Invalid AdaptiveCpp receipt options");
  }
  const {file, sha256: receiptSha256, value: r} = readReceipt(receiptFile), root = path.dirname(file);
  const inputs = boundedRecord(r.inputs, "AdaptiveCpp inputs", 16), libs = inputs.libraries;
  if (r.version !== 1 || !["cuda", "hip", "cuda-hip"].includes(r.backend) || r.revision !== syclRevision ||
      inputs.revision !== syclRevision || r.buildOnly !== true || r.publicationQualified !== false || r.executed === true ||
      r.cudaVersion !== (r.backend.includes("cuda") ? 12060 : undefined) || typeof r.scope !== "string" || r.scope.length > 4096 ||
      !Array.isArray(libs) || !libs.length || libs.length > 512 || libs.some(file => !absolutePath(file)) || new Set(libs).size !== libs.length ||
      !Array.isArray(inputs.flags) || inputs.flags.length > 128 || inputs.flags.some(flag => typeof flag !== "string" || flag.length > 4096 || flag.includes("\0")) ||
      !Array.isArray(inputs.links) || !inputs.links.length || inputs.links.length > 4096 || inputs.links.some(flag => typeof flag !== "string" || flag.length > 4096 || flag.includes("\0")) ||
      !absolutePath(inputs.source) || !absolutePath(inputs.build) || inputs.build !== path.join(root, "build") ||
      !absolutePath(inputs.cxx)) {throw new Error("Invalid static AdaptiveCpp SDK receipt");}
  canonicalPath(inputs.source, "AdaptiveCpp source", "directory"); canonicalPath(inputs.build, "AdaptiveCpp build", "directory");
  const admitted = admitSyclSource(inputs.source, root, true, options.gitRunner || gitOutput);
  if (!isDeepStrictEqual(admitted, {source: inputs.source, revision: syclRevision})) {throw new Error("AdaptiveCpp source proof differs from its receipt");}
  verifyRecipe(r.recipe, syclRecipe, "AdaptiveCpp SDK");
  const planPath = path.join(root, "plan.json"); canonicalPath(planPath, "AdaptiveCpp SDK plan");
  const planSha256 = fileHash(planPath);
  const plan = readBuildJson(planPath);
  if (!record(plan) || plan.version !== "21.1.8" ||
      !isDeepStrictEqual(plan.admitted, {source: inputs.source, revision: syclRevision}) ||
      !Array.isArray(plan.commands) || plan.commands.length !== 2 || plan.commands.some(command => !Array.isArray(command) || !command.length ||
        command.some(token => typeof token !== "string" || token.length > 4096 || token.includes("\0"))) ||
      !isDeepStrictEqual(plan.hostCompilers, r.hostCompilers) || !isDeepStrictEqual(plan.hostCompilerInputs, r.hostCompilerInputs) ||
      !isDeepStrictEqual(plan.compilerVersions, r.compilerVersions) || !isDeepStrictEqual(plan.fingerprints, r.fingerprints) ||
      !isDeepStrictEqual(plan.recipe, r.recipe)) {
    throw new Error("AdaptiveCpp SDK result differs from its producer plan");
  }
  const hostCompilers = boundedRecord(r.hostCompilers, "AdaptiveCpp host compilers", 4), compilerVersions = boundedRecord(r.compilerVersions, "AdaptiveCpp compiler versions", 4);
  if (Object.keys(hostCompilers).sort().join(",") !== "cc,cxx" || Object.values(hostCompilers).some(file => !absolutePath(file)) ||
      Object.keys(compilerVersions).sort().join("\n") !== Object.values(hostCompilers).sort().join("\n")) {throw new Error("AdaptiveCpp compiler identity is incomplete");}
  const hostInputs = boundedRecord(r.hostCompilerInputs, "AdaptiveCpp compiler fingerprints", 8);
  const selectedCompilers = /** @type {Record<string, string>} */ (hostCompilers);
  if (!Object.keys(hostCompilers).every(key => {
    const compiler = selectedCompilers[key];
    return compiler !== undefined && Object.hasOwn(hostInputs, compiler);
  })) {throw new Error("AdaptiveCpp compiler fingerprints omit a selected compiler");}
  const libraries = /** @type {Record<string, string>} */ (/** @type {unknown} */ (boundedRecord(r.libraries, "AdaptiveCpp library hashes", 512)));
  if (Object.keys(libraries).sort().join("\n") !== [...libs].sort().join("\n")) {throw new Error("AdaptiveCpp library membership differs from its receipt");}
  const fingerprints = /** @type {Record<string, string>} */ (/** @type {unknown} */ (boundedRecord(r.fingerprints, "AdaptiveCpp input fingerprints", 8192)));
  if (Object.entries(hostInputs).some(([file, sha]) => fingerprints[file] !== sha) ||
      !Object.hasOwn(fingerprints, inputs.cxx)) {
    throw new Error("AdaptiveCpp output/compiler fingerprints differ");
  }
  let openmp;
  if (options.openmpSdkRoot !== undefined) {
    canonicalPath(options.openmpSdkRoot, "OpenMP SDK root", "directory");
    const reader = options.readOpenmp || require("./build-native-openmp-sdk").nativeOpenmpInputs;
    openmp = reader(options.openmpSdkRoot);
    const expectedReceipt = path.join(options.openmpSdkRoot, "result.json"), expectedArchive = path.join(options.openmpSdkRoot, "sdk/libomp.a");
    canonicalPath(expectedReceipt, "OpenMP SDK receipt"); canonicalPath(expectedArchive, "OpenMP SDK archive");
    if (!record(openmp) || openmp.receipt !== expectedReceipt || openmp.archive !== expectedArchive ||
        !absolutePath(openmp.receipt) || !absolutePath(openmp.archive) || !digest.test(openmp.receiptSha256) || !record(openmp.files) ||
        openmp.files[expectedReceipt] !== openmp.receiptSha256 || typeof openmp.files[expectedArchive] !== "string" ||
        !digest.test(openmp.files[expectedArchive]) || fileHash(expectedReceipt) !== openmp.receiptSha256 ||
        fileHash(expectedArchive) !== openmp.files[expectedArchive] || !isStaticArchive(expectedArchive)) {
      throw new Error("AdaptiveCpp SDK does not bind the admitted OpenMP SDK receipt and archive");
    }
    const selectedOpenmp = libs.filter(file => file === openmpContainerPath || openmpProviderFile.test(path.basename(file)));
    if (selectedOpenmp.length !== 1 || ![openmpContainerPath, expectedArchive].includes(selectedOpenmp[0] || "") ||
        libraries[selectedOpenmp[0] || ""] !== openmp.files[expectedArchive]) {
      throw new Error("AdaptiveCpp SDK must select exactly the admitted OpenMP SDK archive");
    }
  }
  const verified = verifyHashMaps([fingerprints, /** @type {Record<string,string>} */ (libraries)], openmp?.archive);
  const boundLinks = bindOpenmpLinks(inputs.links, openmp?.archive);
  verifySyclCache(inputs.build, inputs.source, r.backend, inputs);
  const hip = r.backend.includes("hip");
  let comgr;
  if (hip) {
    const selected = boundedRecord(inputs.comgr, "AdaptiveCpp COMGR selection", 8);
    if (!absolutePath(selected.receipt)) {throw new Error("AdaptiveCpp HIP result omits its COMGR receipt");}
    comgr = (options.readComgr || comgrSdkInputs)(selected.receipt);
    if (!isDeepStrictEqual(selected, comgr)) {throw new Error("AdaptiveCpp COMGR selection differs from its source receipt");}
    for (const [file, hash] of Object.entries(comgr.files)) {
      if (fingerprints[file] !== hash) {throw new Error("AdaptiveCpp COMGR input is absent from SDK fingerprints: " + file);}
    }
  } else if (inputs.comgr !== undefined) {throw new Error("CUDA-only AdaptiveCpp receipt unexpectedly selects COMGR");}
  const closure = resolveStaticLinks(boundLinks, verified.files);
  if (fileHash(planPath) !== planSha256 || fileHash(file) !== receiptSha256) {throw new Error("AdaptiveCpp receipt changed during admission");}
  return {receipt: file, receiptSha256, backend: r.backend, revision: r.revision, cudaVersion: r.cudaVersion,
    source: inputs.source, build: inputs.build, cxx: inputs.cxx, libraries: libs.map(file => file === openmpContainerPath && verified.bindings[file] ? verified.bindings[file] : file),
    links: boundLinks, resolvedLinks: closure.resolved, files: closure.files, pathBindings: verified.bindings, ...(openmp ? {openmp} : {}), ...(comgr ? {comgr} : {})};
}

/** @param {string} receipt @param {{gitRunner?: (command: string, args: string[]) => string}} [options] */
function nativeHermesInputs(receipt, options = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("Hermes receipt is Linux x86-64 only");}
  if (!record(options) || Object.keys(options).some(key => key !== "gitRunner") ||
      options.gitRunner !== undefined && typeof options.gitRunner !== "function") {throw new Error("Invalid Hermes receipt options");}
  const {file, sha256: receiptSha256, value: r} = readReceipt(receipt), root = path.dirname(file);
  if (r.version !== 1 || r.target !== "linux" || !isDeepStrictEqual(r.pins, hermesPins) || r.buildOnly !== true ||
      r.publicationQualified !== false || r.executed === true || !absolutePath(r.source) || !absolutePath(r.build) ||
      r.source !== path.join(root, "source") || r.build !== path.join(root, "build")) {throw new Error("Invalid static Hermes SDK receipt");}
  canonicalPath(r.source, "Hermes source", "directory"); canonicalPath(r.build, "Hermes build", "directory");
  verifyRecipe(r.recipe, hermesRecipe, "Hermes SDK");
  const fingerprints = boundedRecord(r.fingerprints, "Hermes tool fingerprints", 16), sources = boundedRecord(r.sources, "Hermes source hashes", 100000);
  const patches = Object.keys(fingerprints).filter(file => path.basename(file) === "hermes-host-rejections.patch");
  // Retain the original path/hash evidence while requiring the exact maintained patch at any checkout prefix.
  if (Object.keys(fingerprints).length !== 6 || patches.length !== 1 ||
      fingerprints[patches[0] || ""] !== fileHash(path.join(repo, "runtime/hermes-host-rejections.patch")) ||
      !Object.keys(fingerprints).some(file => path.basename(file) === "git")) {throw new Error("Hermes tool/patch fingerprints are incomplete");}
  canonicalPath(patches[0] || "", "Hermes patch");
  const verified = verifyHashMaps([/** @type {Record<string,string>} */ (fingerprints)]);
  const git = Object.keys(fingerprints).find(file => path.basename(file) === "git");
  if (!git) {throw new Error("Hermes tool fingerprints omit Git");}
  const runGit = options.gitRunner || gitOutput;
  if (runGit(git, ["-C", r.source, "rev-parse", "HEAD"]).trim() !== hermesPins.revision ||
      runGit(git, ["-C", r.source, "diff", "--name-only", "HEAD"]).trim() !== "lib/InternalJavaScript/01-Promise.js" ||
      runGit(git, ["-C", r.source, "ls-files", "--others", "--directory", "--"]).length !== 0) {
    throw new Error("Hermes checkout differs from the pinned, patched source");
  }
  if (fs.existsSync(path.join(r.source, ".git/objects/info/alternates"))) {throw new Error("Hermes source may not borrow mutable Git objects");}
  const names = runGit(git, ["-C", r.source, "ls-files", "-z"]), actualSources = hermesSourceFiles(r.source, names);
  if (!isDeepStrictEqual(sources, actualSources) || sources["lib/InternalJavaScript/01-Promise.js"] !== hermesPins.promiseSha256) {
    throw new Error("Hermes source tree differs from its recorded patched inventory");
  }
  const cache = readBuildText(path.join(r.build, "CMakeCache.txt"), 4 << 20);
  for (const [key, value] of [["HERMESVM_INTERNAL_JAVASCRIPT_NATIVE", "ON"], ["HERMESVM_ALLOW_JIT", "0"],
    ["HERMES_ENABLE_NAPI", "OFF"], ["HERMES_BUILD_SHARED_JSI", "OFF"], ["CMAKE_BUILD_TYPE", "Release"]]) {
    const lines = [...cache.matchAll(new RegExp(`^${key}:[^=]+=([^\\r\\n]+)$`, "gm"))];
    if (lines.length !== 1 || lines[0]?.[1] !== value) {throw new Error("Hermes CMake cache differs: " + key);}
  }
  const artifactNames = ["compiler", "context", "jsi", "public", "runtime"], artifacts = boundedRecord(r.artifacts, "Hermes artifacts", 8);
  if (Object.keys(artifacts).sort().join(",") !== artifactNames.join(",")) {throw new Error("Hermes artifact membership differs");}
  const expected = /** @type {Record<string, string>} */ ({compiler: path.join(r.build, "bin/shermes"), context: path.join(r.build, "external/boost/boost_1_86_0/libs/context/libboost_context.a"),
    jsi: path.join(r.build, "jsi/libjsi.a"), public: path.join(r.build, "public/hermes/Public/libhermesPublic.a"), runtime: path.join(r.build, "lib/libhermesvmlean_a.a")});
  const artifactHashes = /** @type {Record<string, string>} */ ({});
  for (const name of artifactNames) {
    const item = boundedRecord(artifacts[name], "Hermes artifact", 4), target = expected[name];
    if (!target || item.path !== target || typeof item.sha256 !== "string" || !digest.test(item.sha256) || !Number.isSafeInteger(item.bytes) || item.bytes < 1) {
      throw new Error("Invalid Hermes artifact record: " + name);
    }
    canonicalPath(target, "Hermes artifact");
    if (fs.statSync(target).size !== item.bytes || fileHash(target) !== item.sha256 || name !== "compiler" && !isStaticArchive(target)) {
      throw new Error("Hermes artifact differs from its receipt: " + name);
    }
    if (name === "compiler") {verifyX64Elf(target, "Hermes source compiler");}
    artifactHashes[target] = item.sha256;
  }
  if (fileHash(file) !== receiptSha256) {throw new Error("Hermes receipt changed during admission");}
  return {receipt: file, receiptSha256, source: r.source, build: r.build, artifacts: expected, files: {...verified.files, ...artifactHashes}};
}

/** @param {string} receipt @param {{readComgr?: typeof comgrSdkInputs, readConsumer?: typeof rocmInputs}} [options] */
function nativeRocmHostInputs(receipt, options = {}) {
  // The producer depends on the release builder for its build helpers. Load it only after a
  // caller has finished requiring this reader, so the release builder may safely import us.
  const {pins: rocmPins, archiveNames: rocmArchiveNames} = require("./build-native-rocm-host-runtime");
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("ROCm host receipt is Linux x86-64 only");}
  if (!record(options) || Object.keys(options).some(key => !["readComgr", "readConsumer"].includes(key)) ||
      options.readComgr !== undefined && typeof options.readComgr !== "function" ||
      options.readConsumer !== undefined && typeof options.readConsumer !== "function") {throw new Error("Invalid ROCm receipt options");}
  const {file, sha256: receiptSha256, value: r} = readReceipt(receipt), root = path.dirname(file), prefix = path.join(root, "sdk");
  if (r.version !== 1 || !isDeepStrictEqual(r.pins, rocmPins) || r.prefix !== prefix || r.buildOnly !== true ||
      r.executed !== false || r.publicationQualified !== false) {throw new Error("Invalid pinned ROCm host runtime receipt");}
  canonicalPath(root, "ROCm output", "directory"); canonicalPath(prefix, "ROCm SDK prefix", "directory");
  verifyRecipe(r.recipe, rocmRecipe, "ROCm host runtime");
  const sources = boundedRecord(r.sources, "ROCm source snapshot", 2);
  verifySourceInputs(sources, sourceInputs(root, ["source/clr", "source/hip", "source/rocr"], []), "ROCm source snapshots");
  const sourceArchives = boundedRecord(r.sourceArchives, "ROCm source archives", 4);
  if (Object.keys(sourceArchives).sort().join(",") !== "clr,hip,rocr") {throw new Error("ROCm source archive membership differs");}
  const files = /** @type {Record<string, string>} */ ({});
  for (const name of ["clr", "hip", "rocr"]) {
    const item = exactRecord(sourceArchives[name], ["path", "bytes", "sha256"], "ROCm source archive"), expected = path.join(root, `source-${name}.tar`);
    if (item.path !== expected || !Number.isSafeInteger(item.bytes) || item.bytes < 1 || !digest.test(item.sha256)) {throw new Error("Invalid ROCm source archive: " + name);}
    canonicalPath(expected, "ROCm source archive");
    if (fs.statSync(expected).size !== item.bytes || fileHash(expected) !== item.sha256) {throw new Error("ROCm source archive differs: " + name);}
    files[expected] = item.sha256;
  }
  const archives = boundedRecord(r.archivesAfter, "ROCm runtime archives", 8), archiveBefore = boundedRecord(r.archivesBefore, "ROCm pre-install archives", 8);
  if (Object.keys(archives).sort().join(",") !== Object.keys(rocmArchiveNames).sort().join(",") ||
      Object.keys(archiveBefore).sort().join(",") !== Object.keys(rocmArchiveNames).sort().join(",")) {throw new Error("ROCm runtime archive membership differs");}
  for (const [name, basename] of Object.entries(rocmArchiveNames)) {
    const item = exactRecord(archives[name], ["path", "bytes", "sha256"], "ROCm runtime archive"), expected = path.join(prefix, "lib", basename);
    if (item.path !== expected || !Number.isSafeInteger(item.bytes) || item.bytes < 1 || !digest.test(item.sha256)) {throw new Error("Invalid staged ROCm archive: " + name);}
    canonicalPath(expected, "ROCm runtime archive");
    if (fs.statSync(expected).size !== item.bytes || !isStaticArchive(expected) || fileHash(expected) !== item.sha256) {throw new Error("Staged ROCm archive differs: " + name);}
    files[expected] = item.sha256;
  }
  const dependencies = boundedRecord(r.stagedDependencies, "ROCm staged dependencies", 8), owners = {drm: "libdrm-dev", drmAmdgpu: "libdrm-dev", elf: "libelf-dev", numa: "libnuma-dev"};
  if (Object.keys(dependencies).sort().join(",") !== Object.keys(owners).sort().join(",")) {throw new Error("ROCm dependency membership differs");}
  for (const [name, owner] of Object.entries(owners)) {
    const item = exactRecord(dependencies[name], ["path", "sha256", "package", "architecture", "version", "status", "staged", "stagedSha256"], "ROCm staged dependency");
    const expectedName = name === "drmAmdgpu" ? "libdrm_amdgpu.a" : `lib${name}.a`;
    const staged = path.join(prefix, "lib", expectedName);
    if (!absolutePath(item.path) || item.staged !== staged || item.package !== owner || item.architecture !== "amd64" || item.status !== "ii" ||
        typeof item.version !== "string" || !item.version || item.sha256 !== item.stagedSha256 || !digest.test(item.sha256)) {
      throw new Error("Invalid ROCm staged package dependency: " + name);
    }
    const origin = exactRecord(r.osOrigins?.[name], ["path", "sha256", "package", "architecture", "version", "status"], "ROCm dependency origin");
    if (!isDeepStrictEqual(origin, {path: item.path, sha256: item.sha256, package: item.package, architecture: item.architecture,
      version: item.version, status: item.status})) {throw new Error("ROCm staged dependency owner differs: " + name);}
    canonicalInputPath(item.path, "ROCm package input"); canonicalPath(staged, "ROCm staged dependency");
    if (!isStaticArchive(staged) || fileHash(staged) !== item.stagedSha256) {throw new Error("ROCm staged dependency differs: " + name);}
    files[staged] = item.stagedSha256;
  }
  const readComgr = options.readComgr || comgrSdkInputs, comgr = readComgr(r.comgr?.receipt || "");
  exactRecord(r.comgr, ["receipt", "archive", "llvmVersion", "roots"], "ROCm COMGR selection");
  if (r.comgr.archive !== comgr.archive || r.comgr.llvmVersion !== comgr.llvmVersion ||
      !isDeepStrictEqual(r.comgr.roots, comgr.roots)) {throw new Error("ROCm COMGR selection differs from its source receipt");}
  const admission = (options.readConsumer || rocmInputs)(prefix, r.comgr.receipt);
  if (!isDeepStrictEqual(r.consumerAdmission, admission)) {throw new Error("ROCm consumer link admission differs from its source receipt");}
  const inputHashes = boundedRecord(r.sdkInputs, "ROCm SDK inputs", 8192), dependencyHashes = boundedRecord(r.dependencies, "ROCm build dependencies", 8192);
  const hostCompilers = boundedRecord(r.hostCompilers, "ROCm host compilers", 4), hostCompilerInputs = boundedRecord(r.hostCompilerInputs, "ROCm compiler hashes", 4);
  if (Object.keys(hostCompilers).sort().join(",") !== "cc,cxx" || Object.values(hostCompilers).some(file => !absolutePath(file)) ||
      Object.keys(hostCompilerInputs).sort().join("\n") !== Object.values(hostCompilers).sort().join("\n")) {throw new Error("ROCm compiler fingerprints are incomplete");}
  const verified = verifyHashMaps([/** @type {Record<string,string>} */ (inputHashes), /** @type {Record<string,string>} */ (dependencyHashes)]);
  if (Object.entries(hostCompilerInputs).some(([file, hash]) => inputHashes[file] !== hash) ||
      Object.entries(comgr.files).some(([file, hash]) => verified.files[file] !== hash) ||
      Object.entries(dependencies).some(([_name, item]) => verified.files[item.path] !== item.sha256)) {
    throw new Error("ROCm compiler or COMGR source fingerprints are absent from the admitted SDK inputs");
  }
  if (!Array.isArray(r.configured) || r.configured.length !== 2) {throw new Error("ROCm configured build membership differs");}
  for (const c of r.configured) {
    if (!record(c) || !["rocr", "clr"].includes(c.name) || !absolutePath(c.cache) || !absolutePath(c.database) ||
        !Array.isArray(c.depfiles) || c.depfiles.length > 4096 || c.depfiles.some(file => !absolutePath(file)) ||
        !Number.isSafeInteger(c.commandCount) || c.commandCount < 1) {throw new Error("Invalid ROCm configured build record");}
  }
  for (const [pathName, hash] of Object.entries(files)) {
    if (verified.files[pathName] && verified.files[pathName] !== hash) {throw new Error("ROCm output hash conflicts with its input record: " + pathName);}
  }
  for (const [fileName, hash] of Object.entries(verified.files)) {files[fileName] = hash;}
  if (fileHash(file) !== receiptSha256) {throw new Error("ROCm receipt changed during admission");}
  const numa = /** @type {{path: string, staged: string}} */ (dependencies["numa"]);
  return {receipt: file, receiptSha256, prefix, libraries: admission.libraries, links: admission.links, files, comgr,
    numa: {original: numa.path, staged: numa.staged}};
}

/** Read the CPU-only static NVRTC probe receipt and bind the exact three CUDA archive inputs.
 * Legacy proofs remain reviewable; source-composed release callers must request vendor proof.
 * @param {string} receipt @param {{requireVendorProvenance?: boolean}} [options] */
function nativeNvrtcInputs(receipt, options = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("NVRTC receipt is Linux x86-64 only");}
  if (!options || typeof options !== "object" || Array.isArray(options) ||
      Object.keys(options).some(key => key !== "requireVendorProvenance") ||
      typeof options.requireVendorProvenance !== "undefined" && typeof options.requireVendorProvenance !== "boolean" ||
      Object.keys(options).length > 1) {
    throw new Error("Invalid NVRTC receipt options");
  }
  const {file, sha256: receiptSha256, value: r} = readReceipt(receipt), output = path.dirname(file),
    selectedSdk = canonicalPath(r.sdk, "NVRTC SDK", "directory");
  const receiptKeys = ["sdk", "bytes", "needed", "archives", "files", "source", "lld", "scope"], hasVendor = Object.hasOwn(r, "vendor");
  exactRecord(r, hasVendor ? [...receiptKeys, "vendor"] : receiptKeys, "NVRTC receipt");
  if (options.requireVendorProvenance === true && !hasVendor) {
    throw new Error("Source-composed native releases require NVRTC vendor provenance");
  }
  exactRecord(r.source, ["file", "sha256"], "NVRTC source proof");
  exactRecord(r.lld, ["bytes", "constructorScript", "byteExactOracle"], "NVRTC LLD proof");
  const nvrtcScope = "Maintained CN/GPU source compiled statically, byte-exact shared compiler oracle; CPU only, no GPU/runtime/performance qualification";
  if (!Number.isSafeInteger(r.bytes) || r.bytes < 1 || !Array.isArray(r.needed) || r.needed.length > 64 ||
      r.needed.some(name => typeof name !== "string" || !/^[A-Za-z0-9_.+-]{1,256}$/.test(name)) || new Set(r.needed).size !== r.needed.length ||
      !Array.isArray(r.archives) || r.archives.length !== 3 || r.source.file !== "sycl/cn-gpu/cuda-expand-kernel.inc" || !digest.test(r.source.sha256) ||
      !Number.isSafeInteger(r.lld.bytes) || r.lld.bytes < 1 || r.lld.byteExactOracle !== true || !digest.test(r.lld.constructorScript) ||
      r.scope !== nvrtcScope) {throw new Error("Invalid static NVRTC probe receipt");}
  const source = path.join(repo, r.source.file), constructorScript = path.join(repo, "runtime/static-constructors.ld");
  if (fileHash(source) !== r.source.sha256 || fileHash(constructorScript) !== r.lld.constructorScript) {throw new Error("NVRTC source or constructor script changed");}
  const names = ["libnvrtc_static.a", "libnvrtc-builtins_static.a", "libnvptxcompiler_static.a"],
    archives = /** @type {string[]} */ ([]), files = /** @type {Record<string, string>} */ ({});
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    if (!name) {throw new Error("NVRTC archive list is incomplete");}
    const item = exactRecord(r.archives[i], ["file", "sha256"], "NVRTC archive proof"), expected = path.join(selectedSdk, "lib64", name);
    if (item.file !== expected || !digest.test(item.sha256)) {throw new Error("Invalid NVRTC archive binding: " + name);}
    nvrtcArchive(expected, selectedSdk);
    if (fileHash(expected) !== item.sha256) {throw new Error("NVRTC archive differs: " + name);}
    archives.push(expected); files[expected] = item.sha256;
  }
  if (hasVendor) {
    const proof = /** @type {{receiptSha256: string}} */ (exactRecord(r["vendor"], ["receiptSha256"], "NVRTC vendor SDK binding"));
    if (!digest.test(proof.receiptSha256)) {throw new Error("Invalid NVRTC vendor SDK receipt hash");}
    const assembler = /** @type {{nvrtcSdkInputs: (platform: "linux", root: string) => {receiptSha256: string, files: Record<string, string>}}} */ (
      require("./prepare-native-nvrtc-sdk"));
    const vendor = assembler.nvrtcSdkInputs("linux", selectedSdk);
    if (proof.receiptSha256 !== vendor.receiptSha256) {throw new Error("NVRTC vendor SDK receipt differs from compiler proof");}
    if (!record(vendor.files) || Object.keys(vendor.files).length < 1 || Object.keys(vendor.files).length > 64) {
      throw new Error("Invalid admitted NVRTC vendor SDK input map");
    }
    for (const [vendorFile, sha256] of Object.entries(vendor.files)) {
      if (!absolutePath(vendorFile) || !digest.test(sha256) || files[vendorFile] !== undefined && files[vendorFile] !== sha256) {
        throw new Error("Invalid or conflicting NVRTC vendor SDK input: " + vendorFile);
      }
      files[vendorFile] = sha256;
    }
  }
  const rows = r.files;
  if (!Array.isArray(rows) || rows.length !== 4 || rows.some(row => {
    exactRecord(row, ["name", "bytes", "sha256"], "NVRTC output row");
    return typeof row.name !== "string" || !/^[A-Za-z0-9_.-]{1,128}$/.test(row.name) ||
      !Number.isSafeInteger(row.bytes) || row.bytes < 1 || !digest.test(row.sha256);
  }) || new Set(rows.map(row => row.name)).size !== 4) {
    throw new Error("Invalid NVRTC compiled-output receipt");
  }
  for (const directory of ["actual", "expected", "lld"]) {
    const dir = canonicalPath(path.join(output, directory), "NVRTC output directory", "directory"), namesOnDisk = fs.readdirSync(dir).sort();
    if (!isDeepStrictEqual(namesOnDisk, rows.map(row => row.name).sort())) {throw new Error("NVRTC output membership differs: " + directory);}
    for (const row of rows) {
      const pathName = path.join(dir, row.name); canonicalPath(pathName, "NVRTC compiled output");
      if (fs.statSync(pathName).size !== row.bytes || fileHash(pathName) !== row.sha256) {throw new Error("NVRTC compiled output differs: " + row.name);}
    }
  }
  const executable = canonicalPath(path.join(output, "nvrtc-native"), "NVRTC static proof executable");
  if (fs.statSync(executable).size !== r.bytes) {throw new Error("NVRTC proof executable size differs");}
  verifyX64Elf(executable, "NVRTC static proof executable");
  const lldExe = canonicalPath(path.join(output, "nvrtc-lld"), "NVRTC LLD proof executable");
  if (fs.statSync(lldExe).size !== r.lld.bytes) {throw new Error("NVRTC LLD executable size differs");}
  verifyX64Elf(lldExe, "NVRTC LLD proof executable");
  if (fileHash(file) !== receiptSha256) {throw new Error("NVRTC receipt changed during admission");}
  return {receipt: file, receiptSha256, sdk: selectedSdk, archives, files};
}

/** Share the exact static-SDK recipe with transported Windows consumers, which cannot
 * reopen all original Linux compiler paths but must still reject stale producer code. */
function currentStaticSyclRecipe() {return sourceInputs(repo, [], syclRecipe);}

module.exports = {nativeStaticRuntimeInputs, nativeHermesInputs, nativeRocmHostInputs, nativeNvrtcInputs, gitOutput, currentStaticSyclRecipe};
