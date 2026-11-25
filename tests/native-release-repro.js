"use strict";

const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {randomUUID, createHash} = require("node:crypto");
const {buildNativeArtifact} = require("../scripts/build-release");
const {packageNativeRelease} = require("../scripts/package-native-release");
const {fileHash, readBuildJson} = require("../scripts/build-source-inputs");
const {tempDir} = require("./common/temp-dir");
const repo = path.resolve(__dirname, "..");
const enabled = process.env["MOM_TEST_NATIVE_REPRO"] === "1";
const keys = /** @type {[string, string, string, string, string]} */ (["MOM_TEST_NATIVE_REPRO_CONFIG", "MOM_TEST_NATIVE_REPRO_UPX", "MOM_TEST_NATIVE_REPRO_UPX_SHA256",
  "MOM_TEST_NATIVE_REPRO_CMAKE", "MOM_TEST_NATIVE_REPRO_ROOT"]);
const lateNoticeKeys = /** @type {[string, string, string, string]} */ (["MOM_TEST_NATIVE_REPRO_NOTICES", "MOM_TEST_NATIVE_REPRO_NOTICES_SHA256",
  "MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW", "MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW_SHA256"]);
const markerName = ".native-repro-test-owner.json", summaryName = "repro-summary.json";

/** @typedef {{notices: string, noticesSha256: string, review: string, reviewSha256: string}} LateNoticeSettings */
/** @typedef {{notices: ReturnType<typeof regularInput>, review: ReturnType<typeof regularInput>, reviewRecipe: ReturnType<typeof regularInput>}} LateNoticePins */
/** @typedef {{config: string, upx: string, upxSha256: string, cmake: string, root: string, keep: boolean, lateNotice?: LateNoticeSettings,
 * cacheMode: "disabled" | "warm"}} Settings */
/** @typedef {{label: string, output: string, unpacked: string, raw: {bytes: number, sha256: string}, packed: {bytes: number, sha256: string},
 * archive: {bytes: number, sha256: string}, configHash: string, recipe: unknown, stableComponents: {map: Record<string, string>, producerReceiptSha256: string},
 * inputEvidence: {sha256: string, bytes: number, fields: string[]}, objectCache: {cacheHits: number, cacheMisses: number, rows: number},
 * packing: {toolSha256: string, flags: string[], recipe: unknown}, packageOutput?: string}} PassResult */
/** @typedef {{sha256A: string, sha256B: string, bytesA: number, bytesB: number, identical: boolean}} Comparison */
/** @typedef {{version: number, scope: string, buildOnly: boolean, publicationQualified: boolean, freshRuntimeRequired: boolean,
 * cacheDisabled: boolean, cacheMode: "disabled" | "warm", inputs: {config: ReturnType<typeof regularInput>, upx: ReturnType<typeof regularInput>, cmake: ReturnType<typeof regularInput>, packageRecipe: ReturnType<typeof regularInput>,
 * lateNotice?: {notices: ReturnType<typeof regularInput>, review: ReturnType<typeof regularInput>, reviewRecipe: ReturnType<typeof regularInput>}},
 * passes: PassResult[], comparisons: Record<string, Comparison>, pairByteIdentical: boolean, stableSourceConfigToolInputs?: boolean, failure?: string}} ReproSummary */

/** Pure admission helper; opt-out never stats a tool, SDK or output.
 * @param {NodeJS.ProcessEnv} env @returns {Settings | null} */
function settingsFromEnvironment(env) {
  if (env["MOM_TEST_NATIVE_REPRO"] !== "1") {return null;}
  for (const key of keys) {if (!env[key]) {throw new Error("Missing " + key);}}
  const config = env[keys[0]] || "", upx = env[keys[1]] || "", upxSha256 = env[keys[2]] || "",
    cmake = env[keys[3]] || "", root = env[keys[4]] || "";
  const noticeValues = lateNoticeKeys.map(key => env[key]);
  /** @type {LateNoticeSettings | undefined} */ let lateNotice;
  if (noticeValues.some(value => value !== undefined)) {
    if (noticeValues.some(value => !value)) {
      throw new Error("Late-notice A/B requires a notice body and review manifest, each with an explicit SHA-256 pin");
    }
    const notices = env[lateNoticeKeys[0]] || "", noticesSha256 = env[lateNoticeKeys[1]] || "",
      review = env[lateNoticeKeys[2]] || "", reviewSha256 = env[lateNoticeKeys[3]] || "";
    for (const [key, value] of /** @type {[string, string][]} */ ([[lateNoticeKeys[1], noticesSha256], [lateNoticeKeys[3], reviewSha256]])) {
      if (!/^[a-f0-9]{64}$/.test(value)) {throw new Error(key + " must be a lowercase SHA-256");}
    }
    lateNotice = {notices, noticesSha256, review, reviewSha256};
  }
  const cacheMode = env["MOM_TEST_NATIVE_REPRO_CACHE_MODE"] || "disabled";
  if (cacheMode !== "disabled" && cacheMode !== "warm") {throw new Error("MOM_TEST_NATIVE_REPRO_CACHE_MODE must be disabled or warm");}
  const paths = /** @type {[string, string][]} */ ([[keys[0], config], [keys[1], upx], [keys[3], cmake], [keys[4], root],
    ...(lateNotice ? [[lateNoticeKeys[0], lateNotice.notices], [lateNoticeKeys[2], lateNotice.review]] : [])]);
  for (const [key, value] of paths) {
    if (!path.isAbsolute(value) || value.length > 4096 || /[\0\r\n]/.test(value)) {throw new Error(key + " must be an absolute path");}
  }
  if (!/^[a-f0-9]{64}$/.test(upxSha256)) {throw new Error(keys[2] + " must be a lowercase SHA-256");}
  const keep = env["MOM_TEST_NATIVE_REPRO_KEEP"];
  if (keep !== undefined && keep !== "0" && keep !== "1") {throw new Error("MOM_TEST_NATIVE_REPRO_KEEP must be 0 or 1");}
  return {config, upx, upxSha256, cmake, root: path.resolve(root), keep: keep === "1", ...(lateNotice ? {lateNotice} : {}), cacheMode};
}

/** @param {string} root @param {string} candidate */
function inside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}

/** Warm qualification may use only a caller-provided empty overlay; never clear a developer cache.
 * @param {string} directory */
function requireEmptyCacheOverlay(directory) {
  const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
  assert.ok(stat?.isDirectory() && !stat.isSymbolicLink(), "Warm-cache mode requires a real cache overlay directory");
  const real = fs.realpathSync(directory), expected = process.platform === "win32" ? directory.toLowerCase() : directory;
  assert.equal(process.platform === "win32" ? real.toLowerCase() : real, expected, "Warm-cache overlay must be canonical, not a symlink path");
  assert.deepEqual(fs.readdirSync(directory), [], "Warm-cache mode requires a fresh empty cache overlay");
}
/** @param {string} root @returns {[string, string]} */
function outputDirectories(root) {
  return [path.join(root, "pass-a", "release"), path.join(root, "pass-b", "deep", "release")];
}

/** Canonicalize JSON-like evidence without changing array order.
 * @param {unknown} value @returns {unknown} */
function canonical(value) {
  if (Array.isArray(value)) {return value.map(canonical);}
  if (!value || typeof value !== "object") {return value;}
  return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => [key, canonical(child)]));
}

/** @param {unknown} value */
function stableJson(value) {return JSON.stringify(canonical(value));}

/** @param {unknown} value @returns {Record<string, string>} */
function sortedHashMap(value) {
  assert.ok(value && typeof value === "object" && !Array.isArray(value));
  const record = /** @type {Record<string, unknown>} */ (value);
  const rows = Object.entries(record).sort(([a], [b]) => a.localeCompare(b));
  for (const [name, hash] of rows) {assert.equal(typeof name, "string"); assert.match(String(hash), /^[a-f0-9]{64}$/);}
  return Object.fromEntries(rows.map(([name, hash]) => [name, /** @type {string} */ (hash)]));
}

/** Exclude only each invocation's own public producer receipt; all admitted components remain exact. */
/** @param {unknown} entryValue @param {string} producerPath */
function stableComponentHashes(entryValue, producerPath) {
  assert.ok(entryValue && typeof entryValue === "object" && !Array.isArray(entryValue));
  const entry = /** @type {Record<string, unknown>} */ (entryValue), map = sortedHashMap(entry["componentHashes"]);
  const receiptHash = map[producerPath];
  assert.match(receiptHash || "", /^[a-f0-9]{64}$/);
  delete map[producerPath];
  return {map, producerReceiptSha256: /** @type {string} */ (receiptHash)};
}

/** Stable hashes of public receipt inputs, excluding only declared pass-local object outputs.
 * @param {unknown} producerValue @param {string} rawRoot @returns {{sha256: string, bytes: number, fields: string[]}} */
function stableInputEvidence(producerValue, rawRoot) {
  assert.ok(producerValue && typeof producerValue === "object" && !Array.isArray(producerValue));
  const producer = /** @type {Record<string, unknown>} */ (producerValue), application = producer["application"];
  const app = application && typeof application === "object" && !Array.isArray(application) ? /** @type {Record<string, unknown>} */ (application) : {};
  const inputs = app["applicationInputs"];
  const appInputs = inputs && typeof inputs === "object" && !Array.isArray(inputs) ? /** @type {Record<string, unknown>} */ (inputs) : {};
  const inputHashes = producer["inputHashes"];
  const cacheValue = producer["objectCache"] || app["objectCache"], cache = cacheValue && typeof cacheValue === "object" && !Array.isArray(cacheValue) ?
    /** @type {Record<string, unknown>} */ (cacheValue) : {};
  const rows = Array.isArray(cache["objects"]) ? cache["objects"] : Array.isArray(cache["batches"]) ?
    cache["batches"].flatMap(batch => batch && typeof batch === "object" && Array.isArray(batch.objects) ? batch.objects : []) : [];
  const objects = new Map(rows.filter(row => row && typeof row === "object" && typeof row.object === "string" &&
    path.isAbsolute(row.object) && inside(rawRoot, row.object) && /\.(?:obj|o)$/.test(row.object)).map(row => [row.object, row.objectSha256]));
  /** @param {unknown} value */
  function stableExternal(value) {
    const map = sortedHashMap(value), selected = /** @type {Record<string, string>} */ ({});
    for (const [file, hash] of Object.entries(map)) {
      assert.ok(path.isAbsolute(file), "Public build-input receipt paths must be absolute");
      if (objects.has(file)) {
        assert.equal(hash, objects.get(file), "Generated object differs from its declared build input");
      } else {
        // Retain other generated inputs, normalizing only the intentionally varying output root.
        selected[inside(rawRoot, file) ? "$output/" + path.relative(rawRoot, file).split(path.sep).join("/") : file] = hash;
      }
    }
    return selected;
  }
  const evidence = {
    recipe: producer["recipe"] || null,
    sourceHashes: producer["sourceHashes"] && typeof producer["sourceHashes"] === "object" ? producer["sourceHashes"] : null,
    windowsInputs: app["files"] && typeof app["files"] === "object" ? sortedHashMap(app["files"]) : null,
    windowsGraphs: app["graphs"] || null,
    linuxApplicationFiles: appInputs["files"] && typeof appInputs["files"] === "object" ? sortedHashMap(appInputs["files"]) : null,
    linuxApplicationGraph: appInputs["graph"] && typeof appInputs["graph"] === "object" ? sortedHashMap(appInputs["graph"]) : null,
    sdk: producer["sdkInputs"] && typeof producer["sdkInputs"] === "object" ?
      sortedHashMap(/** @type {Record<string, unknown>} */ (producer["sdkInputs"])["hashes"]) : null,
    components: producer["components"] && typeof producer["components"] === "object" ?
      sortedHashMap(/** @type {Record<string, unknown>} */ (producer["components"])["hashes"]) : null,
    dependencies: producer["dependencyHashes"] ? stableExternal(producer["dependencyHashes"]) : null,
    externalBuildInputs: inputHashes && typeof inputHashes === "object" ? stableExternal(inputHashes) : null,
  };
  assert.ok(evidence.windowsInputs || evidence.linuxApplicationFiles, "Producer receipt is missing admitted application inputs");
  const serialized = stableJson(evidence);
  return {sha256: createHash("sha256").update(serialized).digest("hex"), bytes: Buffer.byteLength(serialized), fields: Object.keys(evidence)};
}

/** @param {string} file @param {string} label */
function regularInput(file, label) {
  const stat = fs.lstatSync(file);
  assert.ok(stat.isFile() && !stat.isSymbolicLink(), label + " must be a regular non-symlink file");
  return {path: file, realpath: fs.realpathSync(file), bytes: stat.size, sha256: fileHash(file)};
}

/** @param {ReturnType<typeof regularInput>} pin @param {string} label */
function verifyInputPin(pin, label) {assert.deepEqual(regularInput(pin.path, label), pin, label + " changed during A/B build");}

/** @param {string} file @param {string} expected @param {string} label */
function artifactRecord(file, expected, label) {
  const input = regularInput(file, label);
  assert.equal(input.sha256, expected, label + " hash differs from its public receipt");
  return {bytes: input.bytes, sha256: input.sha256};
}

/** Check the same reviewed body and dependency snapshot as maintained packaging.
 * @param {LateNoticePins} pins */
function verifyLateNoticeReview(pins) {
  const review = require("../scripts/check-native-notices").checkNativeNoticeReview(pins.review.path, pins.notices.path);
  assert.deepEqual(review.notice, {sha256: pins.notices.sha256, bytes: pins.notices.bytes},
    "Pinned review manifest must admit the exact selected notice body");
}

/** Bind caller-selected notice and review files before any build starts.
 * @param {LateNoticeSettings} settings @returns {LateNoticePins} */
function pinLateNoticeInputs(settings) {
  const pins = {
    notices: regularInput(settings.notices, "Late notice body"),
    review: regularInput(settings.review, "Late notice review manifest"),
    reviewRecipe: regularInput(require.resolve("../scripts/check-native-notices"), "Notice review recipe"),
  };
  assert.equal(pins.notices.sha256, settings.noticesSha256, "Late notice body differs from its explicit SHA-256");
  assert.equal(pins.review.sha256, settings.reviewSha256, "Late notice review manifest differs from its explicit SHA-256");
  verifyLateNoticeReview(pins);
  return pins;
}

/** @param {LateNoticePins} pins */
function verifyLateNoticeInputs(pins) {
  verifyInputPin(pins.notices, "Late notice body");
  verifyInputPin(pins.review, "Late notice review manifest");
  verifyInputPin(pins.reviewRecipe, "Notice review recipe");
  verifyLateNoticeReview(pins);
}

/** @param {string} root @param {string} token */
function removeOwnedRoot(root, token) {
  const marker = path.join(root, markerName), stat = fs.lstatSync(marker);
  assert.ok(stat.isFile() && !stat.isSymbolicLink());
  assert.equal(fs.readFileSync(marker, "utf8"), token);
  const realRoot = fs.realpathSync(root);
  assert.equal(process.platform === "win32" ? realRoot.toLowerCase() : realRoot, process.platform === "win32" ? root.toLowerCase() : root);
  fs.rmSync(root, {recursive: true});
}

/** @param {string} text */
function bounded(text) {return text.length > 2048 ? text.slice(0, 2048) + "…" : text;}

test.describe("Native build reproducibility opt-in admission", () => {
  test("default invocation opts out without resolving any SDK, tool, or output path", () => {
    assert.equal(settingsFromEnvironment({}), null);
    assert.equal(settingsFromEnvironment({MOM_TEST_NATIVE_REPRO: "0", MOM_TEST_NATIVE_REPRO_CONFIG: "/must/not/be-read"}), null);
    assert.equal(settingsFromEnvironment({MOM_TEST_NATIVE_REPRO: "0", MOM_TEST_NATIVE_REPRO_NOTICES: "relative.txt"}), null);
  });
  test("opt-in requires every explicit immutable input and a new output root", () => {
    assert.throws(() => settingsFromEnvironment({MOM_TEST_NATIVE_REPRO: "1"}), /MOM_TEST_NATIVE_REPRO_CONFIG/);
    const env = validEnvironment();
    assert.equal(settingsFromEnvironment(env)?.root, path.resolve(env["MOM_TEST_NATIVE_REPRO_ROOT"] || ""));
    for (const key of keys) {const missing = {...env}; delete missing[key]; assert.throws(() => settingsFromEnvironment(missing), new RegExp(key));}
  });
  test("cache mode defaults to disabled and rejects unsupported values", () => {
    const env = validEnvironment();
    assert.equal(settingsFromEnvironment(env)?.cacheMode, "disabled");
    env["MOM_TEST_NATIVE_REPRO_CACHE_MODE"] = "warm";
    assert.equal(settingsFromEnvironment(env)?.cacheMode, "warm");
    env["MOM_TEST_NATIVE_REPRO_CACHE_MODE"] = "force";
    assert.throws(() => settingsFromEnvironment(env), /CACHE_MODE/);
  });
  test("warm-cache admission refuses nonempty, missing, and symlink overlays without cleanup", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-repro-cache-overlay-"));
    t.after(() => fs.rmSync(root, {recursive: true}));
    const cache = path.join(root, "cache"); fs.mkdirSync(cache);
    assert.doesNotThrow(() => requireEmptyCacheOverlay(cache));
    const sentinel = path.join(cache, "existing"); fs.writeFileSync(sentinel, "preserve");
    assert.throws(() => requireEmptyCacheOverlay(cache), /fresh empty cache overlay/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "preserve");
    assert.throws(() => requireEmptyCacheOverlay(path.join(root, "missing")), /real cache overlay directory/);
    if (process.platform !== "win32") {
      const link = path.join(root, "cache-link"); fs.symlinkSync(cache, link, "dir");
      assert.throws(() => requireEmptyCacheOverlay(link), /real cache overlay directory/);
    }
  });
  test("relative paths, malformed hashes, and ambiguous KEEP values fail closed", () => {
    const env = validEnvironment(); env["MOM_TEST_NATIVE_REPRO_CONFIG"] = "config.json";
    assert.throws(() => settingsFromEnvironment(env), /absolute path/);
    env["MOM_TEST_NATIVE_REPRO_CONFIG"] = path.join(os.tmpdir(), "native-repro-config.json"); env["MOM_TEST_NATIVE_REPRO_UPX_SHA256"] = "bad";
    assert.throws(() => settingsFromEnvironment(env), /SHA-256/);
    env["MOM_TEST_NATIVE_REPRO_UPX_SHA256"] = "0".repeat(64); env["MOM_TEST_NATIVE_REPRO_KEEP"] = "yes";
    assert.throws(() => settingsFromEnvironment(env), /KEEP/);
  });
  test("late notice body and review manifest are opt-in as a complete hash-pinned pair", () => {
    const base = validEnvironment();
    assert.equal(settingsFromEnvironment(base)?.lateNotice, undefined);
    const late = validLateNoticeEnvironment(), settings = settingsFromEnvironment(late);
    assert.deepEqual(settings?.lateNotice, {
      notices: late[lateNoticeKeys[0]], noticesSha256: late[lateNoticeKeys[1]],
      review: late[lateNoticeKeys[2]], reviewSha256: late[lateNoticeKeys[3]],
    });
    for (const key of lateNoticeKeys) {
      const missing = {...late}; delete missing[key];
      assert.throws(() => settingsFromEnvironment(missing), /Late-notice A\/B requires/);
    }
  });
  test("late notice and review paths and both pins are validated without reading the files", () => {
    for (const key of [lateNoticeKeys[0], lateNoticeKeys[2]]) {
      const env = validLateNoticeEnvironment(); env[key] = "relative-input";
      assert.throws(() => settingsFromEnvironment(env), /absolute path/);
    }
    for (const key of [lateNoticeKeys[1], lateNoticeKeys[3]]) {
      const env = validLateNoticeEnvironment(); env[key] = "bad";
      assert.throws(() => settingsFromEnvironment(env), /lowercase SHA-256/);
    }
  });
  // The reviewed Node builtin-source pin belongs to the Linux AOT/packing producer (the CI-pinned
  // Node), including when it emits Windows artifacts. Other hosts and Node versions still test
  // the schema/hash guards above.
  const reviewedBuildNode = require("../runtime/native-notice-review.json").dependencies.buildNode.version;
  const skip = process.platform !== "linux" || process.version !== reviewedBuildNode;
  test("late notice body and review manifest hashes are checked against maintained review", {skip}, t => {
    const root = tempDir(t, "native-repro-notice-review-");
    const env = validReviewedLateNoticeEnvironment(), late = settingsFromEnvironment(env)?.lateNotice; assert.ok(late);
    const pins = pinLateNoticeInputs(late);
    assert.equal(pins.notices.sha256, late.noticesSha256);
    assert.equal(pins.review.sha256, late.reviewSha256);
    assert.throws(() => pinLateNoticeInputs({...late, noticesSha256: "0".repeat(64)}), /explicit SHA-256/);
    assert.throws(() => pinLateNoticeInputs({...late, reviewSha256: "0".repeat(64)}), /explicit SHA-256/);

    const altered = path.join(root, "native-notice-review.json"), review = JSON.parse(fs.readFileSync(late.review, "utf8"));
    review.notice.sha256 = "0".repeat(64); fs.writeFileSync(altered, JSON.stringify(review));
    assert.throws(() => pinLateNoticeInputs({...late, review: altered, reviewSha256: fileHash(altered)}), /tracked third-party notices differ/i);
  });
  test("control characters and overlong paths fail before accessing any input", () => {
    for (const key of [keys[0], keys[1], keys[3], keys[4]]) {
      for (const suffix of ["\0", "\r", "\n", "x".repeat(4097)]) {
        const env = validEnvironment(); env[key] += suffix;
        assert.throws(() => settingsFromEnvironment(env), /absolute path/);
      }
    }
    for (const key of [lateNoticeKeys[0], lateNoticeKeys[2]]) {
      for (const suffix of ["\0", "\r", "\n", "x".repeat(4097)]) {
        const env = validLateNoticeEnvironment(); env[key] += suffix;
        assert.throws(() => settingsFromEnvironment(env), /absolute path/);
      }
    }
  });
  test("pass roots are fresh, distinct, and intentionally different-depth", () => {
    const root = path.join(os.tmpdir(), "native-repro-root"), [a, b] = outputDirectories(root);
    assert.notEqual(a, b); assert.equal(path.relative(root, a), path.join("pass-a", "release"));
    assert.equal(path.relative(root, b), path.join("pass-b", "deep", "release"));
    assert.ok(inside(root, a) && inside(root, b));
    assert.equal(inside(root, path.join(os.tmpdir(), "native-repro-root-escape", "pass-a")), false);
  });
  test("only the exact pass producer receipt is excluded from component equality", () => {
    const maps = [
      stableComponentHashes({componentHashes: {"/a/raw/result.json": "a".repeat(64), "/sdk/lib.a": "b".repeat(64)}}, "/a/raw/result.json"),
      stableComponentHashes({componentHashes: {"/b/raw/result.json": "c".repeat(64), "/sdk/lib.a": "b".repeat(64)}}, "/b/raw/result.json"),
    ];
    assert.deepEqual(maps[0]?.map, maps[1]?.map);
    assert.notEqual(maps[0]?.producerReceiptSha256, maps[1]?.producerReceiptSha256);
    assert.throws(() => stableComponentHashes({componentHashes: {"/sdk/lib.a": "b".repeat(64)}}, "/missing/result.json"));
  });
  test("admitted component drift remains visible", () => {
    const a = stableComponentHashes({componentHashes: {"/a/result.json": "a".repeat(64), "/sdk/lib.a": "b".repeat(64)}}, "/a/result.json");
    const b = stableComponentHashes({componentHashes: {"/b/result.json": "c".repeat(64), "/sdk/lib.a": "d".repeat(64)}}, "/b/result.json");
    assert.notDeepEqual(a.map, b.map);
  });
  test("only pass-local generated object paths are omitted from stable input evidence", () => {
    const base = path.join(os.tmpdir(), "native-repro-evidence"), hash = "a".repeat(64), make = (/** @type {string} */ output, /** @type {string} */ toolHash) => stableInputEvidence({
      application: {files: {"/sdk/lib.a": hash}, graphs: ["fixed graph"]},
      objectCache: {objects: [{object: path.join(output, "controller.obj"), objectSha256: hash}]},
      inputHashes: {[path.join(output, "controller.obj")]: hash, "/tool/compiler": toolHash},
    }, output);
    assert.equal(make(path.join(base, "a"), hash).sha256, make(path.join(base, "b"), hash).sha256);
    assert.notEqual(make(path.join(base, "a"), hash).sha256, make(path.join(base, "b"), "b".repeat(64)).sha256);
  });
  test("Windows controller source drift is not hidden by unchanged native inputs", () => {
    const raw = path.join(os.tmpdir(), "native-repro-source"), hash = "a".repeat(64), producer = {
        application: {files: {"/sdk/lib.a": hash}}, sourceHashes: {before: {"mom.js": hash}, after: {"mom.js": hash}},
      }, before = stableInputEvidence(producer, raw);
    producer.sourceHashes.before["mom.js"] = "b".repeat(64);
    producer.sourceHashes.after["mom.js"] = "b".repeat(64);
    assert.notEqual(stableInputEvidence(producer, raw).sha256, before.sha256);
  });
  test("transitive dependency drift remains visible beyond application source files", () => {
    const raw = path.join(os.tmpdir(), "native-repro-dependencies"), hash = "a".repeat(64), producer = {
        application: {files: {"/source/controller.js": hash}}, dependencyHashes: {"/sdk/hip/device.h": hash},
      }, before = stableInputEvidence(producer, raw);
    producer.dependencyHashes["/sdk/hip/device.h"] = "b".repeat(64);
    assert.notEqual(stableInputEvidence(producer, raw).sha256, before.sha256);
  });
  test("canonical evidence ignores object key order but preserves array order", () => {
    assert.equal(stableJson({z: 1, a: {y: 2, x: 3}}), stableJson({a: {x: 3, y: 2}, z: 1}));
    assert.notEqual(stableJson({rows: [1, 2]}), stableJson({rows: [2, 1]}));
  });
  test("pass-local non-object inputs remain hash-bound despite fresh output paths", () => {
    const base = path.join(os.tmpdir(), "native-repro-local-input"), hash = "a".repeat(64);
    const make = (/** @type {string} */ root, /** @type {string} */ inputHash) => stableInputEvidence({
      application: {files: {"/sdk/lib.a": hash}}, inputHashes: {[path.join(root, "policy.json")]: inputHash},
    }, root);
    assert.equal(make(path.join(base, "a"), hash).sha256, make(path.join(base, "b"), hash).sha256);
    assert.notEqual(make(path.join(base, "a"), hash).sha256, make(path.join(base, "b"), "b".repeat(64)).sha256);
  });
  test("an undeclared object extension cannot hide a changed input", () => {
    const root = path.join(os.tmpdir(), "native-repro-undeclared"), hash = "a".repeat(64), make = (/** @type {string} */ inputHash) => stableInputEvidence({
      application: {files: {"/sdk/lib.a": hash}}, inputHashes: {[path.join(root, "not-generated.obj")]: inputHash},
    }, root);
    assert.notEqual(make(hash).sha256, make("b".repeat(64)).sha256);
  });
  test("declared generated objects must match their public input hashes", () => {
    const root = path.join(os.tmpdir(), "native-repro-object-binding"), file = path.join(root, "controller.obj"), hash = "a".repeat(64);
    assert.throws(() => stableInputEvidence({application: {files: {"/sdk/lib.a": hash}},
      objectCache: {batches: [{objects: [{object: file, objectSha256: "b".repeat(64)}]}]}, inputHashes: {[file]: hash}}, root), /Generated object differs/);
  });
  test("artifact comparisons rehash actual output rather than trusting earlier receipts", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-repro-artifact-")), file = path.join(root, "mom");
    t.after(() => fs.rmSync(root, {recursive: true}));
    fs.writeFileSync(file, "original"); const hash = fileHash(file);
    assert.equal(artifactRecord(file, hash, "Artifact").bytes, 8);
    fs.writeFileSync(file, "modified");
    assert.throws(() => artifactRecord(file, hash, "Artifact"), /hash differs/);
  });
});

test.describe("Native build reproducibility full A/B", () => {
  // The builder APIs are synchronous. The invoking host/container must enforce the resource cap
  // and deadline; node:test's timeout cannot preempt a synchronous compiler or packer.
  test("two fresh public native builds, packing passes, and archives are byte-identical", {
    skip: !enabled && "Set MOM_TEST_NATIVE_REPRO=1 and explicit CONFIG, UPX, UPX_SHA256, CMAKE, and ROOT; optional late notices require NOTICES, NOTICES_SHA256, NOTICE_REVIEW, NOTICE_REVIEW_SHA256",
  }, () => {
    assert.equal(enabled, true);
    assert.equal(process.arch, "x64");
    const settings = settingsFromEnvironment(process.env);
    assert.ok(settings);
    const root = settings.root, parent = path.dirname(root);
    const canonicalParent = fs.realpathSync(parent);
    assert.equal(process.platform === "win32" ? canonicalParent.toLowerCase() : canonicalParent, process.platform === "win32" ? parent.toLowerCase() : parent,
      "The caller-selected existing output parent must be canonical");
    assert.equal(inside(repo, root), false, "Output cannot enter the maintained repository");
    assert.equal(fs.lstatSync(root, {throwIfNoEntry: false}), undefined, "Select a new output root; existing data is never overwritten");
    const configPin = regularInput(settings.config, "Native SDK descriptor"), upxPin = regularInput(settings.upx, "Pinned UPX"),
      cmakePin = regularInput(settings.cmake, "Pinned CMake"), packageRecipePin = regularInput(require.resolve("../scripts/package-native-release"), "Archive recipe");
    assert.equal(upxPin.sha256, settings.upxSha256, "Selected UPX differs from its explicit SHA-256");
    const lateNoticePins = settings.lateNotice ? pinLateNoticeInputs(settings.lateNotice) : undefined;
    if (process.platform !== "win32") {fs.accessSync(settings.upx, fs.constants.X_OK); fs.accessSync(settings.cmake, fs.constants.X_OK);}
    const descriptor = readBuildJson(settings.config);
    assert.ok(descriptor && typeof descriptor === "object" && !Array.isArray(descriptor) && descriptor.version === 1,
      "Configuration must be an explicit version-1 native descriptor");
    const {nativeSdkConfig} = require("../scripts/build-native-release");
    if (Object.hasOwn(descriptor, "windowsSdk")) {
      assert.ok(descriptor.application && !descriptor.nativeSources, "Pair testing requires a pre-admitted Windows application descriptor");
      require("../scripts/build-native-windows-controller").validateWindowsControllerConfig(descriptor);
    } else {
      assert.ok(descriptor.components && !descriptor.nativeSources, "Pair testing requires explicit pre-admitted Linux components");
      nativeSdkConfig(descriptor, repo);
    }
    if (settings.cacheMode === "warm") {
      assert.equal(process.platform, "linux", "Warm-cache qualification currently targets the Linux application cache");
      assert.equal(Object.hasOwn(descriptor, "windowsSdk"), false, "Warm-cache qualification requires the Linux SDK descriptor");
      requireEmptyCacheOverlay(path.join(repo, "build", "cache"));
    }

    const token = randomUUID(), rawOutputs = outputDirectories(root),
      /** @type {PassResult[]} */ passes = [], summaryPath = path.join(root, summaryName);
    let owned = false, success = false;
    const summary = /** @type {ReproSummary} */ ({version: 1, scope: "Two fresh public builds from one admitted fixed-SDK descriptor, public in-memory UPX and archive APIs; no miner execution or runtime/performance/publication claim. A full cold SDK-chain reproducibility claim is not made.",
      buildOnly: true, publicationQualified: false, freshRuntimeRequired: true, cacheDisabled: settings.cacheMode === "disabled", cacheMode: settings.cacheMode,
      inputs: {config: configPin, upx: upxPin, cmake: cmakePin, packageRecipe: packageRecipePin,
        ...(lateNoticePins ? {lateNotice: lateNoticePins} : {})}, passes, comparisons: {}, pairByteIdentical: false});
    try {
      fs.mkdirSync(root, {mode: 0o700}); owned = true;
      fs.writeFileSync(path.join(root, markerName), token, {flag: "wx", mode: 0o600});
      const environment = /** @type {NodeJS.ProcessEnv} */ ({...process.env,
        MOM_NATIVE_CACHE: settings.cacheMode === "warm" ? "1" : "0", TZ: "UTC", LC_ALL: "C"});
      if (settings.cacheMode === "warm") {delete environment["MOM_FORCE_REBUILD"];}
      const version = require("../package.json").version;
      for (const [index, output] of rawOutputs.entries()) {
        fs.mkdirSync(path.dirname(output), {recursive: true});
        assert.equal(fs.existsSync(output), false); assert.equal(fs.existsSync(output + ".unpacked"), false);
        const packageOutput = /** @type {string | undefined} */ (settings.lateNotice ? output + ".late-notice" : undefined);
        if (packageOutput) {assert.equal(fs.existsSync(packageOutput), false, "Late-notice package output must be fresh for each pass");}
        const built = buildNativeArtifact(settings.config, output, {upx: settings.upx, upxSha256: settings.upxSha256}, environment);
        assert.equal(built.buildOnly, true); assert.equal(built.publicationQualified, false);
        assert.ok("source" in built && built.source && built.artifact);
        const unpacked = output + ".unpacked", entryPath = path.join(output, "entrypoint.json"), producerPath = path.join(unpacked, "result.json");
        const entry = readBuildJson(entryPath), producer = readBuildJson(producerPath);
        if (Object.hasOwn(descriptor, "windowsSdk")) {
          assert.ok(producer.dependencyHashes && Object.keys(sortedHashMap(producer.dependencyHashes)).length > 0,
            "Windows producer is missing transitive dependency hashes");
        }
        assert.equal(entry.buildOnly, true); assert.equal(entry.publicationQualified, false);
        assert.equal(entry.configHash, configPin.sha256); assert.equal(entry.generatedDescriptor, undefined,
          "The pair must consume the same pre-admitted SDK components, not produce new SDKs");
        const raw = artifactRecord(built.source.path, built.source.sha256, "Unpacked executable");
        assert.equal(built.source.bytes, raw.bytes);
        const packed = artifactRecord(built.artifact.path, built.artifact.sha256, "Packed executable");
        assert.equal(entry.source.sha256, raw.sha256); assert.equal(entry.artifact.sha256, packed.sha256);
        assert.equal(producer.buildOnly, true); assert.equal(producer.publicationQualified, false);
        const rawReceiptHash = producer.artifact?.sha256 || producer.sha256;
        assert.equal(rawReceiptHash, raw.sha256, "Public producer receipt does not bind the raw executable");
        const cache = producer.objectCache || producer.application?.objectCache;
        assert.ok(cache, "Public producer must report application object-cache evidence");
        const cacheRows = /** @type {{cacheHit: boolean}[]} */ (Array.isArray(cache.objects) ? cache.objects : Array.isArray(cache.batches) ?
          cache.batches.flatMap((/** @type {{objects?: {cacheHit: boolean}[]}} */ batch) => batch.objects || []) : []);
        assert.ok(cacheRows.length > 0, "Every application pass must report its complete object batch");
        if (settings.cacheMode === "disabled") {
          assert.ok(cache.cacheHits === 0 && cache.cacheMisses === cacheRows.length && cacheRows.every(row => row.cacheHit === false),
            "MOM_NATIVE_CACHE=0 must build every application object fresh");
        } else if (index === 0) {
          assert.equal(cacheRows.length, 13, "Current Linux application cohort must expose all 13 cached objects");
          assert.ok(cache.cacheHits === 0 && cache.cacheMisses === 13 && cacheRows.every(row => row.cacheHit === false),
            "Cold pass must compile all 13 application objects into the new overlay");
        } else {
          assert.equal(cacheRows.length, 13, "Warm pass must retain all 13 application objects");
          assert.ok(cache.cacheMisses === 0 && cache.cacheHits === 13 && cacheRows.every(row => row.cacheHit === true),
            "Warm pass must validate and reuse all 13 application objects from the same overlay");
          assert.equal(cacheRows.length, passes[0]?.objectCache.rows, "Warm/cold application object membership changed");
        }

        if (lateNoticePins) {
          assert.ok(settings.lateNotice);
          verifyLateNoticeInputs(lateNoticePins);
        }
        const packaged = /** @type {ReturnType<typeof packageNativeRelease>} */ (settings.lateNotice && packageOutput ?
          packageNativeRelease(entryPath, settings.cmake, version, undefined, {output: packageOutput, notices: settings.lateNotice.notices}) :
          packageNativeRelease(entryPath, settings.cmake, version));
        assert.equal(packaged.buildOnly, true); assert.equal(packaged.publicationQualified, false); assert.equal(packaged.exactOriginalBytes, true);
        const actualPackageOutput = packageOutput || output, packageReceipt = readBuildJson(path.join(actualPackageOutput, "native-package.json"));
        assert.equal(packageReceipt.tool.path, cmakePin.path); assert.equal(packageReceipt.tool.sha256, cmakePin.sha256);
        if (lateNoticePins && settings.lateNotice && packageOutput) {
          assert.equal(packageReceipt.version, 2);
          assert.deepEqual(packageReceipt.notices, {path: path.join(packageOutput, "THIRD-PARTY-NOTICES.txt"),
            sha256: lateNoticePins.notices.sha256, bytes: lateNoticePins.notices.bytes});
          if (!entry.noticeAssembly) {
            assert.equal(packageReceipt.noticeAssembly, undefined, "Late notice review metadata must not be fabricated into an assembler receipt");
            assert.equal(packaged.noticeAssembly, undefined);
          }
        } else {assert.equal(packageReceipt.version, 1);}
        const archive = artifactRecord(packaged.archive.path, packaged.archive.sha256, "Release archive");
        assert.equal(packageReceipt.archive.sha256, archive.sha256);
        const stableComponents = stableComponentHashes(entry, producerPath);
        assert.equal(stableComponents.producerReceiptSha256, fileHash(producerPath), "Entrypoint component map does not bind this pass's producer receipt");
        const inputEvidence = stableInputEvidence(producer, unpacked);
        verifyInputPin(configPin, "Native SDK descriptor"); verifyInputPin(upxPin, "Pinned UPX"); verifyInputPin(cmakePin, "Pinned CMake");
        verifyInputPin(packageRecipePin, "Archive recipe");
        if (lateNoticePins && settings.lateNotice) {verifyLateNoticeInputs(lateNoticePins);}
        const packCommand = built.commands?.find(command => command.args?.includes("--best"));
        assert.ok(packCommand && packCommand.command === settings.upx);
        const packingFlags = packCommand.args.filter(arg => !path.isAbsolute(arg) || !inside(root, arg));
        passes.push({label: index === 0 ? "pass-a" : "pass-b", output, unpacked, ...(packageOutput ? {packageOutput} : {}),
          raw: {bytes: raw.bytes, sha256: raw.sha256}, packed: {bytes: packed.bytes, sha256: packed.sha256},
          archive: {bytes: archive.bytes, sha256: archive.sha256}, configHash: entry.configHash,
          recipe: entry.recipe, stableComponents, inputEvidence, objectCache: {cacheHits: cache.cacheHits, cacheMisses: cache.cacheMisses, rows: cacheRows.length},
          packing: {toolSha256: upxPin.sha256, flags: packingFlags, recipe: built.recipe}});
      }
      const a = passes[0], b = passes[1]; assert.ok(a && b);
      // Recheck both outputs after all tools have finished, including the first pass.
      const windows = Object.hasOwn(descriptor, "windowsSdk"), name = windows ? "mom.exe" : "mom";
      for (const pass of passes) {
        artifactRecord(path.join(pass.unpacked, name), pass.raw.sha256, "Final raw executable");
        artifactRecord(path.join(pass.output, name), pass.packed.sha256, "Final packed executable");
        artifactRecord(path.join(pass.packageOutput || pass.output, `mom-v${version}-${windows ? "win.zip" : "lin.tgz"}`), pass.archive.sha256, "Final release archive");
      }
      const comparisons = /** @type {Record<string, Comparison>} */ (Object.fromEntries(["raw", "packed", "archive"].map(name => {
        const left = a[/** @type {"raw" | "packed" | "archive"} */ (name)], right = b[/** @type {"raw" | "packed" | "archive"} */ (name)];
        return [name, {sha256A: left.sha256, sha256B: right.sha256, bytesA: left.bytes, bytesB: right.bytes,
          identical: left.sha256 === right.sha256 && left.bytes === right.bytes}];
      })));
      summary.comparisons = comparisons;
      const stable = a.configHash === b.configHash && stableJson(a.recipe) === stableJson(b.recipe) &&
        stableJson(a.stableComponents.map) === stableJson(b.stableComponents.map) && stableJson(a.inputEvidence) === stableJson(b.inputEvidence) &&
        stableJson(a.packing) === stableJson(b.packing);
      summary.pairByteIdentical = stable && Object.values(comparisons).every(value => value.identical);
      summary.stableSourceConfigToolInputs = stable;
      assert.equal(summary.pairByteIdentical, true, "A/B raw executable, packed executable, archive, or selected source/tool inputs differ; both passes were retained");
      fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2) + "\n", {flag: "wx", mode: 0o600});
      success = true;
      console.log(JSON.stringify({pairByteIdentical: summary.pairByteIdentical, comparisons: summary.comparisons, stableSourceConfigToolInputs: stable,
        evidence: settings.keep ? summaryPath : "printed only; set MOM_TEST_NATIVE_REPRO_KEEP=1 to retain successful outputs"}));
    } catch (error) {
      summary.failure = bounded(error instanceof Error ? error.stack || error.message : String(error));
      if (owned && !fs.existsSync(summaryPath)) {fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2) + "\n", {flag: "wx", mode: 0o600});}
      throw error;
    } finally {
      if (success && owned && !settings.keep) {removeOwnedRoot(root, token);}
    }
  });
});

/** @returns {NodeJS.ProcessEnv} */
function validEnvironment() {
  const base = path.join(os.tmpdir(), "native-repro-fixture");
  return {MOM_TEST_NATIVE_REPRO: "1", MOM_TEST_NATIVE_REPRO_CONFIG: path.join(base, "config.json"),
    MOM_TEST_NATIVE_REPRO_UPX: path.join(base, "upx"), MOM_TEST_NATIVE_REPRO_UPX_SHA256: "0".repeat(64),
    MOM_TEST_NATIVE_REPRO_CMAKE: path.join(base, "cmake"), MOM_TEST_NATIVE_REPRO_ROOT: path.join(base, "fresh-output")};
}

/** @returns {NodeJS.ProcessEnv} */
function validLateNoticeEnvironment() {
  const env = validEnvironment(), base = path.dirname(env["MOM_TEST_NATIVE_REPRO_ROOT"] || os.tmpdir());
  env[lateNoticeKeys[0]] = path.join(base, "THIRD-PARTY-NOTICES.txt");
  env[lateNoticeKeys[1]] = "a".repeat(64);
  env[lateNoticeKeys[2]] = path.join(base, "native-notice-review.json");
  env[lateNoticeKeys[3]] = "b".repeat(64);
  return env;
}

/** @returns {NodeJS.ProcessEnv} */
function validReviewedLateNoticeEnvironment() {
  const env = validEnvironment(), notices = path.join(repo, "THIRD-PARTY-NOTICES.txt"),
    review = path.join(repo, "runtime/native-notice-review.json");
  env[lateNoticeKeys[0]] = notices;
  env[lateNoticeKeys[1]] = fileHash(notices);
  env[lateNoticeKeys[2]] = review;
  env[lateNoticeKeys[3]] = fileHash(review);
  return env;
}
