#!/usr/bin/env node
"use strict";

// Make's timestamps cannot see backdated edits. Keep content receipts and invalidate outputs,
// never input mtimes. Only a completed build may advance the receipt.
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");
const {execFileSync} = require("node:child_process");
/** @typedef {Record<string, string>} Hashes */
/** @typedef {{name: string, deps: string[] | null, depfile?: string}} ObjectDependencies */
/** @typedef {{content: string, dependencies: string | null}} ObjectHashes */
/**
 * @typedef {object} BuildState
 * @property {number} schema
 * @property {number} startedAt
 * @property {string} configuration
 * @property {Hashes} files
 * @property {Hashes} graph
 * @property {Record<string, string | null>} [external]
 * @property {Record<string, ObjectHashes>} [objects]
 * @property {Hashes} [generated]
 * @property {Hashes} [links]
 * @property {string} [output]
 */
const [operation, buildArg, identity = "", ...tools] = process.argv.slice(2);
if (operation === "merge-deps") {
  const names = [buildArg, identity, ...tools];
  if (names.length < 2 || names.some(name => !name)) {throw new Error("Missing SYCL dependency inputs");}
  const rules = names.map(name => {
    if (!name || !fs.lstatSync(name).isFile()) {throw new Error("Invalid SYCL dependency leaf");}
    const text = fs.readFileSync(name, "utf8").replace(/\\\r?\n/g, " ").trim();
    // Compiler -MD emits one literal rule. Refuse assignments, extra rules and Make expansion.
    const rule = text.match(/^((?:\\.|[^:\n])+):\s*((?:\\.|[^:\n])+)$/);
    if (!rule || /[$#]/.test(text)) {throw new Error("Malformed SYCL dependency rule");}
    return {target: rule[1], tokens: rule[2]?.match(/(?:\\.|[^\s])+/g) || []};
  });
  if (!rules[0] || rules.some(rule => rule.target !== rules[0]?.target || !rule.tokens.length)) {
    throw new Error("Inconsistent SYCL dependency targets");
  }
  const output = names[0];
  if (!output) {throw new Error("Missing SYCL dependency output");}
  fs.writeFileSync(output, rules[0].target + ": " + [...new Set(rules.flatMap(rule => rule.tokens))].join(" ") + "\n");
  for (const name of names.slice(1)) {if (name) {fs.unlinkSync(name);}}
  process.exit(0);
}
const root = process.cwd();
const gypConfigDir = (process.env["GYP_CONFIG_DIR"] || path.join(os.homedir(), ".gyp"))
  .replace(/^~(?=\/|$)/, () => os.homedir());
const customGypInputs = ["config.gypi", "common.gypi", path.join(gypConfigDir, "include.gypi")];
const customNodeHeaders = Object.entries(process.env).some(([name, value]) => value &&
  /^(npm_config_|npm_package_config_node_gyp_)(nodedir|dist_?url|devdir)$/i.test(name));
const build = path.resolve(buildArg || "build");
if (build !== path.join(root, "build") || (fs.existsSync(build) && fs.lstatSync(build).isSymbolicLink())) {
  throw new Error("Linux content state requires the local real build directory");
}
const receipt = path.join(build, ".mom-content-state.json");
const pending = path.join(build, ".mom-content-pending.json");
/** @param {import("node:crypto").BinaryLike} value */
const hash = value => crypto.createHash("sha256").update(value).digest("hex");
/** @param {string} name */
const fileHash = name => hash(fs.readFileSync(name));
/** @param {string} dir @param {(name: string) => void} visit */
function walk(dir, visit) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    const name = path.join(dir, entry.name);
    if (entry.isDirectory()) {walk(name, visit);} else {visit(name);}
  }
}
function assertOutputTree() {
  for (const leaf of ["Release", ".deps"]) {
    const dir = path.join(build, leaf);
    if (!fs.existsSync(dir)) {continue;}
    if (fs.lstatSync(dir).isSymbolicLink()) {throw new Error("Symlinked Linux output directory");}
    walk(dir, name => {
      if (fs.lstatSync(name).isSymbolicLink()) {throw new Error("Symlinked Linux output leaf");}
    });
  }
}
assertOutputTree();
function inputs() {
  /** @type {Hashes} */
  const files = {};
  for (const dir of ["native", "sycl", "xmrig"]) {walk(dir, name => { files[name] = fileHash(name); });}
  /** @type {Hashes} */
  const graph = {};
  for (const name of ["binding.gyp", "scripts/linux-build-state.js", "scripts/build-helpers.sh", "scripts/build-environment.js", "scripts/cpu-cflags.sh",
    "scripts/cpu-optflags.sh", "scripts/cpu-feature.sh", "scripts/cxx-combined.sh",
    "scripts/cxx-adaptivecpp.sh", "scripts/cxx-oneapi.sh", "scripts/combined-build.sh", "scripts/adaptivecpp-entrypoint.sh",
    "scripts/multicompiler-entrypoint.sh"]) {graph[name] = fileHash(name);}
  for (const [name, value] of Object.entries(files)) {if (/\.gypi?$/.test(name)) {graph[name] = value;}}
  return {files, graph};
}
/** @param {string} name @param {boolean} [filesOnly] @returns {string[] | null} */
function optionalSdkMetadata(name, filesOnly = false) {
  try {
    const stat = fs.statSync(name, {bigint: true});
    return filesOnly && !stat.isFile() ? null : [String(stat.size), String(stat.ctimeNs), String(stat.mtimeNs)];
  } catch (error) {
    if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") {throw error;}
    const leaf = fs.lstatSync(name, {bigint: true});
    // Optional SDK aliases may be dangling. Selected tools and consumed headers do not use
    // this fallback; binding the link itself and missing state makes repair invalidate caches.
    if (!leaf.isSymbolicLink()) {throw error;}
    return ["symlink", fs.readlinkSync(name), String(leaf.ctimeNs), "missing"];
  }
}
/** @param {string} command */
function toolIdentity(command) {
  const selected = command.includes("/") ? command : execFileSync("sh", ["-c", 'command -v "$1"', "sh", command], {encoding: "utf8"}).trim();
  const real = fs.realpathSync(selected);
  const prefix = path.dirname(path.dirname(real));
  const selectedPrefix = path.dirname(path.dirname(path.resolve(selected)));
  /** @type {Record<string, string | string[]>} */
  const sdk = {};
  for (const dir of new Set([prefix, selectedPrefix])) {
    for (const name of [".mom-build-identity", ".mom-toolchain-sha256"]) {
      const marker = path.join(dir, name);
      if (fs.existsSync(marker)) {sdk[marker] = fileHash(marker);}
    }
  }
  for (const name of ["sycl-post-link", "sycl-link", "llvm-spirv", "ld.lld",
    "clang-offload-wrapper", "clang-offload-bundler", "ptxas", "nvlink"]) {
    const tool = path.join(path.dirname(selected), name);
    if (fs.existsSync(tool)) {sdk[name] = fileHash(tool);}
  }
  // Bind selected SDK link inputs without recursively scanning toolchain installations. ctime
  // cannot be restored by an installer preserving old mtimes; dependency headers use content.
  for (const dir of [path.join(prefix, "lib"), path.join(prefix, "lib64"),
    ...(process.env["CMPLR_ROOT"] ? [path.join(process.env["CMPLR_ROOT"], "lib")] : [])]) {
    if (!fs.existsSync(dir)) {continue;}
    for (const name of fs.readdirSync(dir).sort()) {
      if (/\.(a|so)(\.|$)/.test(name)) {
        const full = path.join(dir, name);
        const metadata = optionalSdkMetadata(full, true);
        if (metadata) {sdk[full] = metadata;}
      }
    }
  }
  for (const dir of ["etc/AdaptiveCpp", "etc/hipSYCL"]) {
    const full = path.join(prefix, dir);
    if (fs.existsSync(full)) {
      walk(full, name => {
        sdk[name] = fileHash(name);
        if (name.endsWith(".json")) {
          const config = /** @type {Record<string, unknown>} */ (JSON.parse(fs.readFileSync(name, "utf8")));
          for (const key of ["default-clang", "default-cpu-cxx"]) {
            const driver = config[key];
            if (typeof driver === "string" && fs.existsSync(driver) && fs.realpathSync(driver) !== real) {
              sdk["driver:" + driver] = hash(JSON.stringify(toolIdentity(driver)));
            }
          }
        }
      });
    }
  }
  // Dependencies hash consumed headers. Inventory this SDK's include subtree too: a newly added
  // header can change include resolution without changing any previously consumed file.
  const include = path.join(prefix, "include");
  if (prefix.startsWith("/opt/") && fs.existsSync(include)) {
    walk(include, name => {
      const metadata = optionalSdkMetadata(name);
      if (metadata) {sdk[name] = metadata;}
    });
  }
  return {selected, real, sha256: fileHash(real), sdk};
}
function configuration() {
  const names = ["PATH", "CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS", "AR", "CC", "CXX", "LINK",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
    "MOM_CPU_MARCH", "MOM_PORTABLE_BUILD", "MOM_LTO", "MOM_DPCPP_ROOT", "MOM_ICPX",
    "MOM_DPCPP_IMPL", "MOM_COMBINED_TARGETS", "MOM_INTEL_AOT_DEVICE", "ACPP_TARGETS",
    "MOM_ADAPTIVE_BACKEND", "MOM_ADAPTIVE_HOST_CC", "MOM_ADAPTIVE_HOST_CXX", "MOM_ROCM_ROOT", "ROCM_PATH",
    "CMPLR_ROOT", "CUDA_PATH", "CUDA_HOME", "GYP_DEFINES", "GYP_GENERATOR_FLAGS", "GYP_CONFIG_DIR",
    "NODE_GYP_FORCE_PYTHON", "PYTHON", "PYTHONPATH"];
  const env = Object.fromEntries(names.map(name => [name, process.env[name] || ""]));
  const npmConfig = Object.fromEntries(Object.keys(process.env).sort()
    .filter(name => /^(npm_config_|npm_package_config_node_gyp_)/i.test(name))
    .map(name => [name, process.env[name] || ""]));
  const nodeGyp = toolIdentity("node-gyp");
  const nodeGypRoot = path.dirname(path.dirname(nodeGyp.real));
  const deviceTools = [];
  /** @type {{root: string, real: string, libdevice: Hashes} | null} */
  let cudaSdk = null;
  if (process.env["MOM_INTEL_AOT_DEVICE"]) {deviceTools.push(toolIdentity("/usr/bin/ocloc"));}
  if ((process.env["MOM_COMBINED_TARGETS"] || "").includes("nvidia") || process.env["MOM_ADAPTIVE_BACKEND"] === "cuda") {
    for (const command of ["ptxas", "nvlink"]) {
      try { deviceTools.push(toolIdentity(command)); } catch { deviceTools.push({command, missing: true}); }
    }
    const selectedRoot = process.env["CUDA_PATH"] || process.env["CUDA_HOME"] || "/usr/local/cuda";
    const libdevice = path.join(selectedRoot, "nvvm/libdevice");
    if (!fs.existsSync(libdevice)) {throw new Error("Selected CUDA SDK is missing nvvm/libdevice");}
    /** @type {Hashes} */
    const bitcode = {};
    // Device math bitcode is consumed implicitly by Clang, not recorded by -MD. Hash only
    // immediate libdevice files in the selected SDK; do not scan unrelated CUDA installations.
    for (const name of fs.readdirSync(libdevice).sort()) {
      if (/^libdevice.*\.bc$/.test(name)) {bitcode[name] = fileHash(path.join(libdevice, name));}
    }
    if (!Object.keys(bitcode).length) {throw new Error("Selected CUDA SDK has no libdevice bitcode");}
    cudaSdk = {root: selectedRoot, real: fs.realpathSync(selectedRoot), libdevice: bitcode};
  }
  return hash(JSON.stringify({identity, node: process.versions, arch: process.arch, env, npmConfig,
    nodeGyp, nodeGypVersion: execFileSync("node-gyp", ["--version"], {encoding: "utf8"}).trim(),
    gypInputs: [...customGypInputs, path.join(nodeGypRoot, "addon.gypi"), path.join(nodeGypRoot, "package.json")]
      .map(name => fs.existsSync(name) ? fileHash(name) : null),
    deviceTools, cudaSdk,
    nativeCpu: fs.existsSync("/proc/cpuinfo") ? (fs.readFileSync("/proc/cpuinfo", "utf8").split("\n\n")[0] || "")
      .split("\n").filter(line => /^(vendor_id|cpu family|model\s|model name|stepping|flags)\s*:/.test(line)) : [],
    nodeGraph: ["/usr/local/include/node/common.gypi", "/usr/local/include/node/config.gypi",
      "/usr/local/include/node/node_version.h"].map(name => fs.existsSync(name) ? fileHash(name) : null),
    cpu: execFileSync("bash", ["scripts/cpu-cflags.sh"], {encoding: "utf8"}).trim(),
    opt: ["cflags", "ldflags"].map(mode =>
      execFileSync("bash", ["scripts/cpu-optflags.sh", mode], {encoding: "utf8"}).trim()),
    features: ["x86_64", "arm", "arm64", "aes", "msr", "sse2", "ssse3", "sse4_1", "xop", "avx2", "avx512f", "vaes"]
      .map(feature => { try { execFileSync("bash", ["scripts/cpu-feature.sh", feature], {stdio: "ignore"}); return true; } catch { return false; } }),
    tools: tools.map(toolIdentity)}));
}
/** @returns {ObjectDependencies[]} */
function dependencies() {
  /** @type {ObjectDependencies[]} */
  const objects = [];
  if (!fs.existsSync(build)) {return objects;}
  walk(build, name => {
    if (!name.endsWith(".o")) {return;}
    const relative = path.relative(build, name).split(path.sep).join("/");
    const candidates = [path.join(build, "Release/.deps", relative + ".d"),
      path.join(build, ".deps", relative + ".d"), name + ".d"];
    const depfile = candidates.find(candidate => fs.existsSync(candidate));
    if (!depfile) { objects.push({name, deps: null}); return; }
    const text = fs.readFileSync(depfile, "utf8").replace(/\\\r?\n/g, " ");
    const match = text.match(/^deps_[^\n]*?\s*:=([^\n]*)/m) || text.match(/^[^:\n]+:(?!=)([^\n]*)/m);
    const dependencyText = match?.[1];
    if (!dependencyText || /[$]/.test(dependencyText)) { objects.push({name, deps: null}); return; }
    // Gyp escapes spaces in dependency names; preserve them when tokenizing.
    const tokens = dependencyText.match(/(?:\\.|[^\s])+/g) || [];
    const deps = tokens.map(token => path.resolve(build, token.replace(/\\(.)/g, "$1")));
    objects.push({name, deps, depfile});
  });
  return objects;
}
/** @param {ObjectDependencies[]} objects */
function externalHashes(objects) {
  /** @type {Record<string, string | null>} */
  const external = {};
  for (const object of objects) {
    for (const name of object.deps || []) {
      if (!name.startsWith(root + path.sep) && !Object.hasOwn(external, name)) {
        external[name] = fs.existsSync(name) ? fileHash(name) : null;
      }
    }
  }
  return external;
}
/** @param {ObjectDependencies[]} objects */
function objectHashes(objects) {
  return Object.fromEntries(objects.map(object => [path.relative(build, object.name),
    {content: fileHash(object.name), dependencies: object.depfile ? fileHash(object.depfile) : null}]));
}
function generatedGraph() {
  /** @type {Hashes} */
  const graph = {};
  if (fs.existsSync(build)) {
    for (const name of fs.readdirSync(build).sort()) {
      if (name === "Makefile" || name.endsWith(".mk") || name === "config.gypi") {
        const full = path.join(build, name);
        if (!fs.lstatSync(full).isFile()) {throw new Error("Invalid Linux generated graph");}
        graph[name] = fileHash(full);
      }
    }
  }
  return graph;
}
function removeLinks() {
  if (!fs.existsSync(build)) {return;}
  walk(build, name => { if (/\.(node|a)$/.test(name)) {fs.unlinkSync(name);} });
}
function linkHashes() {
  /** @type {Hashes} */
  const links = {};
  if (fs.existsSync(build)) {
    walk(build, name => {
      if (/\.(node|a)$/.test(name)) {links[path.relative(build, name)] = fileHash(name);}
    });
  }
  return links;
}
/** @param {string} name @param {string} boundary */
function pruneEmptyParents(name, boundary) {
  for (let dir = path.dirname(name); dir !== boundary && dir.startsWith(boundary + path.sep); dir = path.dirname(dir)) {
    try {fs.rmdirSync(dir);} catch (error) {
      const code = /** @type {NodeJS.ErrnoException} */ (error).code;
      if (code === "ENOTEMPTY" || code === "EEXIST") {break;}
      if (code !== "ENOENT") {throw error;}
    }
  }
}
// Run only after success: dependency intermediates may still belong to a retried live build.
// Derive object names from Gyp's reserved output grammar, never from receipt/dependency contents.
function pruneOrphans() {
  for (const leaf of ["Release/.deps", ".deps"]) {
    const dir = path.join(build, leaf);
    if (!fs.existsSync(dir)) {continue;}
    walk(dir, name => {
      const relative = path.relative(dir, name);
      const match = relative.match(/^(Release\/obj(?:\.target|\.host)?)\/[^\r\n]+\.o\.d(\.raw(?:\.device-[a-zA-Z0-9_-]+)?)?$/);
      if (!match?.[1]) {return;}
      const object = path.join(build, relative.replace(/\.d(?:\.raw(?:\.device-[a-zA-Z0-9_-]+)?)?$/, ""));
      if (match[2] || !fs.existsSync(object)) {
        fs.unlinkSync(name);
        pruneEmptyParents(name, path.join(dir, match[1]));
        pruneEmptyParents(object, path.join(build, match[1]));
      }
    });
  }
  for (const name of fs.readdirSync(build)) {
    if (name !== ".node-version" && !/^\.mom-content-(state|pending)\.json\.[a-f0-9]{24}\.tmp$/.test(name)) {continue;}
    const full = path.join(build, name);
    if (!fs.lstatSync(full).isFile()) {throw new Error("Invalid Linux generated cache leaf");}
    fs.unlinkSync(full);
  }
}
/** @param {string} name @returns {BuildState | null} */
function read(name) {
  try {
    if (!fs.lstatSync(name).isFile() || fs.lstatSync(name).isSymbolicLink()) {return null;}
    const state = /** @type {BuildState} */ (JSON.parse(fs.readFileSync(name, "utf8")));
    if (state.schema !== 1 || typeof state.configuration !== "string" ||
      typeof state.startedAt !== "number" || !Number.isFinite(state.startedAt) ||
      !state.files || !state.graph || ![state.files, state.graph].every(map =>
      Object.values(map).every(value => typeof value === "string"))) {return null;}
    return state;
  } catch { return null; }
}
/** @param {string} name @param {BuildState} state */
function write(name, state) {
  fs.mkdirSync(build, {recursive: true});
  const temporary = name + "." + crypto.randomBytes(12).toString("hex") + ".tmp";
  fs.writeFileSync(temporary, JSON.stringify(state), {flag: "wx", mode: 0o600});
  fs.renameSync(temporary, name);
}
if (operation === "prepare") {
  /** @type {BuildState} */
  const current = {schema: 1, startedAt: Date.now(), ...inputs(), configuration: configuration()};
  const unfinished = read(pending);
  // A failed build retains its invalidation checkpoint and previously verified objects. New
  // objects have no completion proof: a killed compiler can leave a partial .o beside an old .d.
  const prior = unfinished || read(receipt);
  let clean = process.env["MOM_FORCE_REBUILD"] === "1" || !prior || prior.schema !== 1 ||
    // User Gyp configs can include arbitrary files outside our source inventory. Reconfigure
    // these uncommon builds conservatively instead of inventing a second recursive Gyp parser.
    Boolean(process.env["GYP_CONFIG_DIR"]) || customNodeHeaders || customGypInputs.some(name => fs.existsSync(name)) ||
    prior.configuration !== current.configuration ||
    JSON.stringify(prior.graph) !== JSON.stringify(current.graph) ||
    Boolean(prior.generated && JSON.stringify(prior.generated) !== JSON.stringify(generatedGraph())) ||
    !fs.existsSync(path.join(build, "Makefile"));
  const objects = dependencies();
  current.external = externalHashes(objects);
  if (!clean && objects.some(object => !object.deps || !object.deps.length)) {clean = true;}
  if (clean || !prior) {
    fs.rmSync(build, {recursive: true, force: true});
  } else {
    if (unfinished || !fs.existsSync(path.join(build, "Release/mom.node")) ||
      (!unfinished && (prior.output !== fileHash(path.join(build, "Release/mom.node")) ||
        JSON.stringify(prior.links) !== JSON.stringify(linkHashes())))) {removeLinks();}
    const changed = new Set();
    for (const name of new Set([...Object.keys(prior.files), ...Object.keys(current.files)])) {
      if (prior.files[name] !== current.files[name]) {changed.add(path.resolve(name));}
    }
    const external = current.external;
    for (const name of new Set([...Object.keys(prior.external || {}), ...Object.keys(external)])) {
      if ((prior.external || {})[name] !== external[name]) {changed.add(name);}
    }
    const membershipChanged = Object.keys(prior.files).join("\n") !== Object.keys(current.files).join("\n");
    let invalidated = false;
    const hashes = objectHashes(objects);
    for (const object of objects) {
      const key = path.relative(build, object.name);
      const expected = prior.objects && prior.objects[key];
      const altered = JSON.stringify(expected) !== JSON.stringify(hashes[key]);
      if (membershipChanged || altered || object.deps?.some(name => changed.has(name))) {
        fs.unlinkSync(object.name);
        invalidated = true;
      }
    }
    if (invalidated) {removeLinks();}
    // Adding/removing a source can change gyp's generated graph without editing binding.gyp.
    if (membershipChanged) {fs.rmSync(path.join(build, "Makefile"), {force: true});}
  }
  current.objects = objectHashes(dependencies());
  write(pending, current);
  process.stdout.write(clean ? "clean\n" : "incremental\n");
} else if (operation === "finish") {
  const state = read(pending);
  if (!state || !fs.existsSync(path.join(build, "Release/mom.node")) ||
    fs.statSync(path.join(build, "Release/mom.node")).size === 0) {throw new Error("Incomplete Linux build");}
  if (JSON.stringify(inputs()) !== JSON.stringify({files: state.files, graph: state.graph})) {
    throw new Error("Linux build inputs changed during compilation");
  }
  if (configuration() !== state.configuration) {throw new Error("Linux toolchain changed during compilation");}
  const builtObjects = dependencies();
  if (builtObjects.some(object => !object.deps?.length)) {throw new Error("Incomplete Linux dependency graph");}
  const currentExternal = externalHashes(builtObjects);
  for (const [name, value] of Object.entries(state.external || {})) {
    const current = Object.hasOwn(currentExternal, name) ? currentExternal[name] :
      (fs.existsSync(name) ? fileHash(name) : null);
    if (current !== value) {throw new Error("Linux dependency changed during compilation");}
  }
  const knownExternal = state.external || {};
  state.external = currentExternal;
  for (const name of Object.keys(state.external)) {
    if (!(name in knownExternal) && fs.existsSync(name) &&
      fs.statSync(name).ctimeMs > state.startedAt) {throw new Error("New Linux dependency changed during compilation");}
  }
  state.objects = objectHashes(builtObjects);
  state.generated = generatedGraph();
  state.output = fileHash(path.join(build, "Release/mom.node"));
  state.links = linkHashes();
  pruneOrphans();
  write(receipt, state);
  fs.unlinkSync(pending);
} else {throw new Error("Expected prepare or finish");}
