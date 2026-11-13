"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const pathInternals = Object.freeze(["internal/constants", "internal/validators", "internal/util", "internal/deps/minimatch/index", "internal/fs/glob"]);

/** Return the exact built-in path source and internal imports used by the controller aliases.
 * @returns {{source: string, internals: readonly string[]}} */
function nativePathSource() {
  const binding = /** @type {(name: string) => object} */ (Reflect.get(process, "binding"));
  const source = Reflect.get(binding("natives"), "path");
  if (typeof source !== "string") {throw new Error("Build Node does not expose the path source");}
  const imports = [...source.matchAll(/require\(['"]([^'"]+)['"]\)/g)].map((match) => match[1]);
  if (imports.some((name) => !name || !pathInternals.includes(name))) {
    throw new Error("Build Node path has an unreviewed internal dependency");
  }
  return {source, internals: pathInternals};
}

/** Stable virtual application root used for built-in resources, independent of checkout location.
 * @param {string} platform */
function nativeApplicationRoot(platform) {
  if (platform === "linux") {return "/__mom_builtin__";}
  if (platform === "win32") {return "C:\\__mom_builtin__";}
  throw new Error("Native application platform must be linux or win32");
}

/** Build-only aliases. Preserve Node's path source/license and record its exact provenance;
 * never evaluate builtin source in the production process. The final SDK must pin build Node.
 * Read-only verification must not rewrite a previously qualified build's generated inputs.
 * @param {string} output @param {boolean} [readOnly] */
function nativeAliases(output, readOnly = false) {
  const repo = path.resolve(__dirname, "..");
  const {source, internals} = nativePathSource();
  const shim = path.join(repo, "runtime", "path-internals.js");
  const generated = path.join(output, "node-path.cjs");
  const code = `const {primordials} = require(${JSON.stringify(shim)});\n${source}`;
  const receipt = JSON.stringify({
    node: process.version, sha256: crypto.createHash("sha256").update(source).digest("hex"),
    license: "MIT (included in generated source)", scope: "Node path algorithms; matchesGlob explicitly unsupported",
  }, null, 2) + "\n";
  const provenance = path.join(output, "node-path-source.json");
  if (readOnly) {
    if (fs.readFileSync(generated, "utf8") !== code || fs.readFileSync(provenance, "utf8") !== receipt) {
      throw new Error("Native bundle path source/provenance differs from the selected build Node");
    }
  } else {
    fs.writeFileSync(generated, code);
    fs.writeFileSync(provenance, receipt);
  }
  return {
    "node:path": generated,
    ...Object.fromEntries(internals.map((name) => [name, shim])),
    "node:buffer": require.resolve("buffer/"), buffer: require.resolve("buffer/"),
    "node:events": require.resolve("events/"),
    "node:string_decoder": require.resolve("string_decoder/"),
    "node:net": path.join(repo, "runtime", "net.js"),
    "node:tls": path.join(repo, "runtime", "tls.js"),
    "node:child_process": path.join(repo, "runtime", "child-process.js"),
    ...Object.fromEntries(["fs", "os", "crypto", "zlib", "cluster"].map((name) =>
      [`node:${name}`, path.join(repo, "runtime", `${name}.js`)])),
  };
}

/** Canonical source bundling for both compilation and read-only parity checks. esbuild's
 * generated module names/comments depend on cwd even when every input path is absolute.
 * @param {string} entry @param {string} bundle @param {boolean} [readOnly] @param {string} [platform]
 * @returns {import("esbuild").BuildOptions} */
function controllerBundleOptions(entry, bundle, readOnly = false, platform = process.platform) {
  if (!path.isAbsolute(entry) || !path.isAbsolute(bundle)) {throw new Error("Native bundle needs absolute input/output paths");}
  const repo = path.resolve(__dirname, "..");
  const applicationRoot = nativeApplicationRoot(platform);
  return {absWorkingDir: repo, entryPoints: [entry], outfile: bundle, write: !readOnly,
    bundle: true, metafile: true, platform: "neutral", target: "es2020", format: "iife", external: ["node:*"],
    // esbuild's unminified CommonJS wrappers embed generated modules' temporary paths.
    // Drop those debug labels, but retain function names and avoid syntax optimizations.
    minifyIdentifiers: true, minifyWhitespace: true, keepNames: true,
    inject: [path.join(repo, "runtime/bootstrap.mjs")], alias: nativeAliases(path.dirname(bundle), readOnly),
    // Native fs supplies immutable policy here; no application file is extracted at runtime.
    define: {__dirname: JSON.stringify(applicationRoot)}};
}

/** Hermes records its input filename in the generated native data. Use stable basenames
 * in the private output directory, not a random absolute build path. Both platforms share
 * these semantic flags; Engineering Mode does not invoke AOT.
 * @param {string} command @param {string} bundle @param {string} cSource */
function controllerAotCommand(command, bundle, cSource) {
  if (![command, bundle, cSource].every(file => path.isAbsolute(file) && !/[\0\r\n]/.test(file)) ||
      path.dirname(bundle) !== path.dirname(cSource)) {throw new Error("Controller AOT requires an absolute tool and colocated absolute inputs/outputs");}
  return {command, cwd: path.dirname(bundle), args: ["-Xenable-tdz", "-Xes6-block-scoping",
    "-exported-unit=mom_entry", "-emit-c", "-o", "./" + path.basename(cSource), "./" + path.basename(bundle)]};
}

/** Compare bytes without assert's unbounded Buffer inspection/Myers diff on a mismatch.
 * A changed 661-KB bundle previously consumed 49 GiB merely formatting that diagnostic.
 * @param {Uint8Array} actual @param {Uint8Array} expected */
function verifyBundle(actual, expected) {
  if (!(actual instanceof Uint8Array) || !(expected instanceof Uint8Array)) {throw new TypeError("Native bundle comparison needs byte arrays");}
  const a = Buffer.from(actual.buffer, actual.byteOffset, actual.byteLength);
  const b = Buffer.from(expected.buffer, expected.byteOffset, expected.byteLength);
  const hash = (/** @type {Uint8Array} */ bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
  if (!a.equals(b)) {throw new Error(`Native bundle mismatch: actual ${a.length} bytes SHA256 ${hash(a)}; expected ${b.length} bytes SHA256 ${hash(b)}`);}
  return {bytes: a.length, sha256: hash(a)};
}

module.exports = Object.assign(nativeAliases, {nativePathSource, nativeApplicationRoot, controllerBundleOptions, controllerAotCommand, verifyBundle});
