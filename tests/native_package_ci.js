"use strict";
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), assert = require("node:assert/strict"),
  test = require("node:test"), {runInNewContext, Script} = require("node:vm"),
  {spawnSync} = require("node:child_process"),
  {isDeepStrictEqual} = require("node:util"), {fileHash} = require("../scripts/build-source-inputs"),
  {packState, listState} = require("../scripts/prepare-native-ci");
const workflow = fs.readFileSync(path.join(__dirname, "../.github/workflows/package-native-application.yml"), "utf8"),
  bodies = [...workflow.matchAll(/node <<'NODE'\n([\s\S]*?)\n {10}NODE/g)].map(match =>
    (match[1] || "").split("\n").map(line => line.slice(10)).join("\n")),
  cmakeStep = workflow.split("      - name: Resolve CMake before artifact downloads\n")[1]?.split("\n      - name: ")[0] || "",
  cmakeRun = cmakeStep.match(/\x20{8}run: \|\n((?:\x20{10}.*(?:\n|$))+)/),
  cmakeBody = (cmakeRun?.[1] || "").split("\n").map(line => line.slice(10)).join("\n");

/** Exercise the late packaging job with same-run artifact-shaped inputs and no candidate execution.
 * The two fixed /tmp roots are virtualized into this fixture's private temp directory.
 * @param {import('node:test').TestContext} t @param {'linux'|'win32'} [target] */
function fixture(t, target = "linux") {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-package-ci-")), suffix = "123456789-1",
    cmakeDirectory = path.join(temp, "cmake-bin"), cmakePath = path.join(cmakeDirectory, "cmake"),
    platform = target === "win32" ? "windows" : "linux", binary = target === "win32" ? "mom.exe" : "mom",
    roots = {packed: `native-${platform}-packed-build-${suffix}`, original: `native-${platform}-application-build-${suffix}`},
    restored = {packed: path.join(temp, "restored-packed"), original: path.join(temp, "restored-original")},
    source = {packed: path.join(temp, "archive-packed"), original: path.join(temp, "archive-original")},
    repoRoot = path.join(temp, "repo"), noticeBody = path.join(repoRoot, "THIRD-PARTY-NOTICES.txt"),
    noticeManifest = path.join(repoRoot, "runtime/native-notice-review.json"),
    inputRoot = path.join(temp, "native-package-inputs"), inputDirs = {
      raw: path.join(inputRoot, "raw"), "packed-executable": path.join(inputRoot, "packed-executable"),
      "packed-build": path.join(inputRoot, "packed-build"), "original-build": path.join(inputRoot, "original-build"),
    }, calls = /** @type {Array<{stage:string, [key:string]:any}>} */ ([]);
  let tarOverride = /** @type {{kind:string, members:string[]}|undefined} */ (undefined),
    expandedOverride = /** @type {number|undefined} */ (undefined), apiOverride = /** @type {string|undefined} */ (undefined);
  for (const directory of [...Object.values(inputDirs), ...Object.values(source)]) {fs.mkdirSync(directory, {recursive: true});}
  fs.mkdirSync(cmakeDirectory);
  fs.writeFileSync(cmakePath, "#!/bin/sh\nprintf 'cmake version 3.99.0\\n'\n", {mode: 0o755});
  fs.mkdirSync(path.dirname(noticeManifest), {recursive: true});
  const noticeBytes = Buffer.from("Fixture reviewed notices\n"), noticeDependencies = {fixture: {revision: "a".repeat(40)}},
    noticeSnapshot = {version: 1, notice: {bytes: noticeBytes.length, sha256: require("node:crypto").createHash("sha256").update(noticeBytes).digest("hex")},
      dependencies: noticeDependencies};
  fs.writeFileSync(noticeBody, noticeBytes);
  fs.writeFileSync(noticeManifest, JSON.stringify({version: 1, notice: noticeSnapshot.notice, dependencies: noticeDependencies}, null, 2) + "\n");
  const pinnedNoticeHash = fileHash(noticeBody), pinnedManifestHash = fileHash(noticeManifest);
  for (const file of ["outputs", "summary"]) {fs.writeFileSync(path.join(temp, file), "");}
  const mapFixed = (/** @type {string} */ file) => {
    for (const key of /** @type {Array<keyof typeof roots>} */ (["packed", "original"])) {
      const root = path.join("/tmp", roots[key]);
      if (file === root || file.startsWith(root + path.sep)) {return path.join(restored[key], path.relative(root, file));}
    }
    return file;
  };
  const virtualFs = new Proxy(fs, {get(object, key) {
    const value = Reflect.get(object, key); if (typeof value !== "function") {return value;}
    if (key === "realpathSync") {
      return (/** @type {string} */ file) => {
        const actual = mapFixed(file), real = value.call(object, actual);
        return actual !== file && real === actual ? file : real;
      };
    }
    if (key === "copyFileSync") {
      return (/** @type {string} */ from, /** @type {string} */ to, /** @type {number} */ flags) =>
        value.call(object, mapFixed(from), mapFixed(to), flags);
    }
    if (key === "mkdirSync") {return (/** @type {string} */ file, /** @type {any[]} */ ...args) => value.call(object, mapFixed(file), ...args);}
    return (/** @type {string} */ file, /** @type {any[]} */ ...args) => value.call(object, typeof file === "string" ? mapFixed(file) : file, ...args);
  }});
  const rawBytes = Buffer.alloc(96, target === "linux" ? 0x41 : 0x42); rawBytes.write(target === "linux" ? "\x7fELF" : "MZ");
  const packedBytes = Buffer.concat([Buffer.from(target === "linux" ? "\x7fELF" : "MZ"), Buffer.alloc(92, 0x50)]),
    rawPath = path.join(inputDirs.raw, binary), packedDownload = path.join(inputDirs["packed-executable"], binary);
  fs.writeFileSync(rawPath, rawBytes, {mode: 0o755}); fs.writeFileSync(packedDownload, packedBytes, {mode: 0o755});
  const packedTar = path.join(inputDirs["packed-build"], roots.packed + ".tar"),
    originalTar = path.join(inputDirs["original-build"], roots.original + ".tar");
  for (const file of [packedTar, originalTar]) {fs.writeFileSync(file, "fixture transport");}
  const runTempConfig = path.join(temp, target === "linux" ? `native-linux-assembly-config-${suffix}.json` : `native-windows-application-config-${suffix}.json`),
    configBytes = JSON.stringify({version: 1, buildOnly: true}), configHash = require("node:crypto").createHash("sha256").update(configBytes).digest("hex");
  const metadataTree = (/** @type {string} */ directory, /** @type {string} */ name, /** @type {unknown} */ value) =>
    fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + "\n");
  const originalSourcePath = path.join(target === "linux" ? "/tmp" : temp,
      target === "linux" ? `mom-native-linux-application-${suffix}` : `native-windows-application-output-${suffix}`, binary),
    originalProducerPath = path.join(path.dirname(originalSourcePath), "result.json"),
    originalTree = source.original;
  fs.writeFileSync(path.join(originalTree, "application-config.json"), configBytes);
  const producerRecord = target === "win32" ? {target: "x86_64-pc-windows-msvc", nativeRuntime: true, executed: false,
    artifact: {path: originalSourcePath, bytes: rawBytes.length, sha256: fileHash(rawPath)}, configHash, buildOnly: true, publicationQualified: false} :
    {binary: originalSourcePath, bytes: rawBytes.length, sha256: fileHash(rawPath), configHash, buildOnly: true, publicationQualified: false};
  metadataTree(originalTree, "result.json", producerRecord);
  const producerHash = fileHash(path.join(originalTree, "result.json")), originEntry = {
    version: 1, target, configFile: runTempConfig, configHash,
    source: {path: originalSourcePath, bytes: rawBytes.length, sha256: fileHash(rawPath)},
    artifact: {path: originalSourcePath, bytes: rawBytes.length, sha256: fileHash(rawPath)},
    recipe: {}, componentHashes: {[originalProducerPath]: producerHash},
    producerReceipt: {path: originalProducerPath, sha256: producerHash}, buildOnly: true, publicationQualified: false,
  };
  metadataTree(originalTree, "entrypoint.json", originEntry);
  const originEntryHash = fileHash(path.join(originalTree, "entrypoint.json")),
    handoff = {version: 1, format: `mom-${platform}-application-build-handoff-v1`, runAttempt: suffix,
      target: target === "win32" ? "x86_64-pc-windows-msvc" : "x86_64-unknown-linux-gnu",
      artifact: {file: binary, bytes: rawBytes.length, sha256: fileHash(rawPath)},
      config: {file: "application-config.json", sha256: fileHash(path.join(originalTree, "application-config.json"))},
      producerReceipt: {file: "result.json", sha256: producerHash}, entrypointReceipt: {file: "entrypoint.json", sha256: originEntryHash},
      buildOnly: true, executed: false, publicationQualified: false};
  metadataTree(originalTree, "handoff.json", handoff);
  const originalMembers = [roots.original, ...["application-config.json", "entrypoint.json", "handoff.json", "result.json"].map(name => `${roots.original}/${name}`)];
  const packedTree = source.packed, packedCandidate = path.join(packedTree, binary),
    packedSha = (fs.writeFileSync(packedCandidate, packedBytes, {mode: 0o755}), fileHash(packedCandidate)),
    packLogs = /** @type {Record<string, {path:string, sha256:string}>} */ ({});
  for (const stage of ["pack", "integrity", "restore"]) {
    const file = path.join(packedTree, stage + ".log"); fs.writeFileSync(file, stage + " log\n");
    packLogs[stage] = {path: path.join("/tmp", roots.packed, stage + ".log"), sha256: fileHash(file)};
  }
  const candidateProducerPath = path.join(temp, "native-raw", "result.json"), packedReceipt = {
    source: {path: path.join(temp, "native-raw", binary), bytes: rawBytes.length, sha256: fileHash(rawPath)},
    artifact: {path: path.join("/tmp", roots.packed, binary), bytes: packedBytes.length, sha256: packedSha},
    buildOnly: true, publicationQualified: false, logs: Object.fromEntries(Object.entries(packLogs).map(([stage, row]) => [stage, row])),
  };
  metadataTree(packedTree, "result.json", packedReceipt);
  const packHash = fileHash(path.join(packedTree, "result.json")), candidateComponents = {[candidateProducerPath]: producerHash},
    candidate = {version: 1, target, configFile: runTempConfig, configHash, recipe: {}, componentHashes: candidateComponents,
      source: {path: path.join(temp, "native-raw", binary), bytes: rawBytes.length, sha256: fileHash(rawPath)},
      artifact: {path: path.join("/tmp", roots.packed, binary), bytes: packedBytes.length, sha256: packedSha},
      sourceEntrypoint: {path: path.join("/tmp", roots.original, "entrypoint.json"), sha256: originEntryHash},
      producerReceipt: {path: candidateProducerPath, sha256: producerHash},
      packingReceipt: {path: path.join("/tmp", roots.packed, "result.json"), sha256: packHash},
      buildOnly: true, publicationQualified: false};
  metadataTree(packedTree, "entrypoint.json", candidate);
  const packedMembers = [roots.packed, ...[binary, "entrypoint.json", "result.json", "pack.log", "integrity.log", "restore.log"].map(name => `${roots.packed}/${name}`)];
  const packedExpected = packedMembers.map(name => name === roots.packed ? name + "/" : name),
    originalExpected = originalMembers.map(name => name === roots.original ? name + "/" : name);
  const archiveMembers = (/** @type {string} */ archive) => archive === packedTar ? packedExpected : archive === originalTar ? originalExpected :
    [path.basename(archive, ".tar") + "/", `${path.basename(archive, ".tar")}/build-audit.json`,
      `${path.basename(archive, ".tar")}/native-package.json`, `${path.basename(archive, ".tar")}/native-notice-review.json`];
  const ci = {
    listState: (/** @type {string} */ archive) => {
      const kind = archive === packedTar ? "packed" : archive === originalTar ? "original" : "metadata";
      return tarOverride?.kind === kind ? tarOverride.members : archiveMembers(archive);
    },
    tarExpandedBytes: () => expandedOverride ?? 4096,
    extractState: (/** @type {string} */ archive, /** @type {string} */ base, /** @type {string[]} */ _expected, /** @type {string[]} */ restoreRoots) => {
      const kind = archive === packedTar ? "packed" : "original";
      calls.push({stage: "extract", kind});
      const destination = mapFixed(path.join(base, restoreRoots[0] || ""));
      fs.mkdirSync(destination, {recursive: false}); fs.cpSync(source[kind], destination, {recursive: true});
    },
    packState: (/** @type {string} */ archive, /** @type {string} */ base, /** @type {string[]} */ selected) => {
      // Use the real archive boundary and tiny metadata tar, not a permissive byte-writing stub.
      calls.push({stage: "pack-metadata", selected}); return packState(archive, base, selected, 60000);
    },
  };
  const env = {RUNNER_TEMP: temp, GITHUB_RUN_ID: "123456789", GITHUB_RUN_ATTEMPT: "1", NATIVE_PLATFORM: target,
    PACKED_BUILD_BUNDLE_SHA256: fileHash(packedTar), RAW_EXECUTABLE_SHA256: fileHash(rawPath),
    ORIGINAL_BUILD_BUNDLE_SHA256: fileHash(originalTar), ORIGINAL_HANDOFF_SHA256: fileHash(path.join(originalTree, "handoff.json")),
    GITHUB_WORKSPACE: repoRoot,
    MOM_CMAKE_PATH: cmakePath,
    GITHUB_OUTPUT: path.join(temp, "outputs"), GITHUB_STEP_SUMMARY: path.join(temp, "summary")};
  const hash = (/** @type {string} */ file) => fileHash(mapFixed(file));
  const execute = (/** @type {number} */ index) => runInNewContext(bodies[index] || "", {
    process: {env, cwd: () => repoRoot}, console: {log: (/** @type {unknown[]} */ ...args) => {calls.push({stage: "console", args});}},
    require: (/** @type {string} */ name) => {
      if (name === "node:fs") {return virtualFs;}
      if (name === "node:path") {return path;}
      if (name === "node:child_process") {
        return {spawnSync: (/** @type {string} */ command, /** @type {string[]} */ args) => {
          calls.push({stage: "tar-list", command, args});
          const names = archiveMembers(args[args.indexOf("--file") + 1] || "");
          const stdout = names.map(member => `${member.endsWith("/") ? "drwxr-xr-x" : "-rw-r--r--"} 0/0 0 2026-01-01 00:00:00 ${member}`).join("\n") + "\n";
          return {status: 0, stdout};
        }};
      }
      if (name === "./scripts/prepare-native-ci") {return ci;}
      if (name === "./scripts/build-source-inputs") {
        return {fileHash: hash,
          readBuildJson: (/** @type {string} */ file) => JSON.parse(fs.readFileSync(mapFixed(file), "utf8"))};
      }
      if (name === "./scripts/check-native-release") {
        return {auditNativeRelease: (/** @type {string} */ entrypoint, /** @type {any} */ options) => {
          calls.push({stage: options?.archive ? "archive-audit" : "build-audit", entrypoint, options});
          if (apiOverride === "pre-audit" && !options?.archive) {throw new Error("fixture pre-audit rejected");}
          if (apiOverride === "post-audit" && options?.archive) {throw new Error("fixture archive audit rejected");}
          const answer = {scope: "build-audit", target, sourceSha256: fileHash(rawPath), sha256: packedSha, bytes: packedBytes.length,
            buildEvidencePassed: true, runtimeTested: false, publicationQualified: false};
          return options?.archive ? {...answer, archiveChecked: true, archiveSha256: fileHash(options.archive.path)} : answer;
        }};
      }
      if (name === "./scripts/check-native-notices") {
        return {checkNativeNoticeReview: (/** @type {string|undefined} */ manifestPath, /** @type {string|undefined} */ noticePath) => {
          calls.push({stage: "notice-review", manifestPath, noticePath});
          const selectedManifest = manifestPath || noticeManifest, selectedNotice = noticePath || path.join(repoRoot, "THIRD-PARTY-NOTICES.txt"),
            selected = {version: 1, notice: {bytes: fs.statSync(selectedNotice).size, sha256: fileHash(selectedNotice)},
              dependencies: noticeDependencies};
          if (selectedManifest !== noticeManifest || selectedNotice !== noticeBody ||
              fileHash(selectedManifest) !== pinnedManifestHash || selected.notice.sha256 !== pinnedNoticeHash ||
              !isDeepStrictEqual(JSON.parse(fs.readFileSync(selectedManifest, "utf8")),
                {version: 1, notice: noticeSnapshot.notice, dependencies: noticeDependencies})) {
            throw new Error("fixture native notice review drift");
          }
          return selected;
        }};
      }
      if (name === "./scripts/package-native-release") {
        return {packageNativeRelease: (/** @type {string} */ entrypoint, /** @type {string} */ tool,
        /** @type {string} */ version, /** @type {unknown} */ spawn, /** @type {any} */ lateNotice) => {
          calls.push({stage: "package", entrypoint, tool, version, spawn, lateNotice});
          if (apiOverride === "package") {throw new Error("fixture package rejected");}
          fs.mkdirSync(lateNotice.output); fs.copyFileSync(mapFixed(lateNotice.notices), path.join(lateNotice.output, "THIRD-PARTY-NOTICES.txt"));
          if (apiOverride === "mutate-notice-body") {fs.appendFileSync(noticeBody, "changed during package\n");}
          if (apiOverride === "mutate-notice-manifest") {fs.appendFileSync(noticeManifest, " ");}
          const archivePath = path.join(lateNotice.output, `mom-v${version}-${target === "win32" ? "win.zip" : "lin.tgz"}`);
          fs.writeFileSync(archivePath, "fixture archive"); const archiveSha = fileHash(archivePath), artifactPath = path.join("/tmp", roots.packed, binary);
          const receiptPath = path.join(lateNotice.output, "native-package.json"); fs.writeFileSync(receiptPath, "fixture package receipt\n");
          return {archive: {path: archivePath, bytes: fs.statSync(archivePath).size, sha256: archiveSha},
            executable: {path: artifactPath, bytes: packedBytes.length, sha256: packedSha}, buildOnly: true,
            publicationQualified: false, members: apiOverride === "extra-members" ? [binary, "THIRD-PARTY-NOTICES.txt", "extra"] : [binary, "THIRD-PARTY-NOTICES.txt"]};
        }};
      }
      if (name === "./package.json") {return require("../package.json");}
      return require(name);
    },
  }, {timeout: 10000});
  t.after(() => fs.rmSync(temp, {recursive: true, force: true}));
  return {env, calls, execute, temp, roots, inputDirs, rawPath, packedTar, originalTar, originalTree, source, noticeBody, noticeManifest,
    cmakeDirectory, cmakePath,
    setTar: (/** @type {string[]} */ value, /** @type {'packed'|'original'} */ kind) => {tarOverride = {kind, members: value};},
    setExpanded: (/** @type {number} */ value) => {expandedOverride = value;},
    setApi: (/** @type {string} */ value) => {apiOverride = value;}};
}

test.describe("SDK-free late native package workflow", {skip: process.platform !== "linux" && "Linux workflow harness"}, () => {
  test("job is bounded, pinned, and packages without rebuilding or running the candidate", () => {
    assert.equal(bodies.length, 2); bodies.forEach(body => new Script(body)); assert.ok(cmakeBody);
    assert.match(workflow, /timeout-minutes: 30/); assert.match(workflow, /timeout-minutes: 25/);
    assert.match(workflow, /node-version: 24\.15\.0/); assert.match(workflow, /packageNativeRelease\(entrypoint, cmake/);
    assert.match(workflow, /auditNativeRelease\(entrypoint, \{archive:/);
    assert.match(cmakeBody, /command -v cmake/); assert.match(cmakeBody, /realpath -e/);
    assert.match(workflow, /MOM_CMAKE_PATH: \$\{\{ steps\.cmake\.outputs\.path \}\}/);
    const resolveCmake = workflow.indexOf("- name: Resolve CMake before artifact downloads"),
      firstDownload = workflow.indexOf("- name: Download the original raw executable"),
      packageBody = bodies[1] || "", cmakeGuard = packageBody.indexOf("if (!canonicalExecutable(cmake))"),
      firstArtifactAccess = packageBody.indexOf("for (const directory of Object.values(inputDirs))");
    assert.ok(resolveCmake >= 0 && resolveCmake < firstDownload);
    assert.ok(cmakeGuard >= 0 && cmakeGuard < firstArtifactAccess);
    assert.doesNotMatch(workflow, /cmake\s*=\s*["']\/usr\/bin\/cmake/);
    assert.doesNotMatch(workflow, /actions\/cache@|npm run build|npm ci|self-hosted|checkNativeExecutable|packNativeEntrypoint/);
    for (const match of workflow.matchAll(/uses: (actions\/[\w-]+)@([^\s#]+)/g)) {assert.match(match[2] || "", /^[a-f0-9]{40}$/);}
    assert.match(workflow, /archive: true/); assert.match(workflow, /retention-days: 1/);
    assert.match(bodies[0] || "", /checkNativeNoticeReview\(\)/);
    assert.match(bodies[1] || "", /checkNativeNoticeReview\(undefined, noticeBody\)/);
    assert.match(workflow, /lateNotice = \{output: outputRoot, notices: noticeBody\}/);
    assert.match(workflow, /native-notice-review\.json/);
    assert.doesNotMatch(workflow, /NOTICES_BUNDLE|NOTICES_RECEIPT|native-release-notices|native-notices\.json|notice-source bundle/);
    const platformExpression = ["$", "{{ inputs.platform == 'win32' && 'windows' || 'linux' }}"].join(""),
      downloadNames = workflow.split("\n").filter(line => line.trimStart().startsWith("name: native-${{ inputs.platform"));
    assert.equal(downloadNames.length, 4);
    for (const suffix of ["-executable-", "-packed-executable-", "-packed-build-", "-application-build-"]) {
      assert.ok(downloadNames.some(line => line.includes(platformExpression + suffix)), `missing platform mapping for ${suffix}`);
    }
  });

  test("resolves canonical PATH CMake before downloads and reuses it for package and archive audit", t => {
    const f = fixture(t), realTool = path.join(f.cmakeDirectory, "cmake-real");
    fs.renameSync(f.cmakePath, realTool); fs.symlinkSync(realTool, f.cmakePath);
    const result = spawnSync("/bin/bash", ["-c", cmakeBody], {encoding: "utf8",
      env: {...process.env, PATH: [f.cmakeDirectory, process.env["PATH"] || "/usr/bin:/bin"].join(path.delimiter), GITHUB_OUTPUT: f.env.GITHUB_OUTPUT}});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
    const output = fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), resolved = output.match(/^path=(.*)$/m)?.[1];
    assert.equal(resolved, fs.realpathSync(realTool)); assert.notEqual(resolved, "/usr/bin/cmake");
    f.env.MOM_CMAKE_PATH = resolved;
    f.execute(0); f.execute(1);
    const packaged = f.calls.find(row => row.stage === "package"), audit = f.calls.find(row => row.stage === "archive-audit");
    assert.equal(packaged?.["tool"], resolved); assert.equal(audit?.["options"]?.archive?.tool, resolved);
  });

  test("missing or invalid PATH CMake fails before artifact downloads and package APIs", t => {
    const missing = fixture(t), emptyBin = path.join(missing.temp, "empty-bin"); fs.mkdirSync(emptyBin);
    fs.writeFileSync(missing.env.GITHUB_OUTPUT, "");
    const missingResult = spawnSync("/bin/bash", ["-c", cmakeBody], {encoding: "utf8",
      env: {...process.env, PATH: emptyBin, GITHUB_OUTPUT: missing.env.GITHUB_OUTPUT}});
    assert.notEqual(missingResult.status, 0); assert.match(missingResult.stderr || "", /CMake is required/);
    assert.equal(fs.readFileSync(missing.env.GITHUB_OUTPUT, "utf8"), ""); assert.equal(missing.calls.length, 0);

    const invalid = fixture(t), invalidBin = path.join(invalid.temp, "invalid-bin"), invalidTool = path.join(invalidBin, "cmake");
    fs.mkdirSync(invalidBin); fs.writeFileSync(invalidTool, "#!/bin/sh\nprintf 'cmake version 3.bad\\n'\n", {mode: 0o755});
    fs.writeFileSync(invalid.env.GITHUB_OUTPUT, "");
    const invalidResult = spawnSync("/bin/bash", ["-c", cmakeBody], {encoding: "utf8",
      env: {...process.env, PATH: [invalidBin, process.env["PATH"] || "/usr/bin:/bin"].join(path.delimiter), GITHUB_OUTPUT: invalid.env.GITHUB_OUTPUT}});
    assert.notEqual(invalidResult.status, 0); assert.match(invalidResult.stderr || "", /invalid version banner/);
    assert.equal(fs.readFileSync(invalid.env.GITHUB_OUTPUT, "utf8"), ""); assert.equal(invalid.calls.length, 0);
  });

  test("package-body CMake guard has no missing-path or symlink bypass", t => {
    const missing = fixture(t); missing.execute(0); Reflect.deleteProperty(missing.env, "MOM_CMAKE_PATH");
    assert.throws(() => missing.execute(1), /Resolved CMake path is missing/);
    assert.equal(missing.calls.some(row => ["extract", "package", "archive-audit"].includes(row.stage)), false);

    const linked = fixture(t), linkPath = path.join(linked.cmakeDirectory, "cmake-link"); linked.execute(0);
    fs.symlinkSync(linked.cmakePath, linkPath); linked.env.MOM_CMAKE_PATH = linkPath;
    assert.throws(() => linked.execute(1), /Resolved CMake path is missing/);
    assert.equal(linked.calls.some(row => ["extract", "package", "archive-audit"].includes(row.stage)), false);
  });

  for (const target of ["linux", "win32"]) {
    test(`${target} restores the exact pinned provenance and uses only the existing package/audit APIs`, t => {
      const f = fixture(t, /** @type {'linux'|'win32'} */ (target)); f.execute(0); f.execute(1);
      const stages = f.calls.map(row => row.stage);
      assert.deepEqual(stages.filter(stage => stage === "extract"), ["extract", "extract"]);
      assert.deepEqual(stages.filter(stage => stage.endsWith("audit")), ["build-audit", "archive-audit"]);
      const packaged = f.calls.find(row => row.stage === "package"); assert.ok(packaged);
      assert.equal(packaged["entrypoint"], path.join("/tmp", f.roots.packed, "entrypoint.json"));
      assert.equal(packaged["tool"], f.cmakePath); assert.equal(packaged["spawn"], undefined);
      assert.equal(f.calls.find(row => row.stage === "archive-audit")?.["options"]?.archive?.tool, f.cmakePath);
      assert.deepEqual(Object.keys(packaged["lateNotice"]).sort(), ["notices", "output"]);
      assert.equal(packaged["lateNotice"].notices, f.noticeBody);
      const reviews = f.calls.filter(row => row.stage === "notice-review");
      assert.deepEqual(reviews.map(row => [row["manifestPath"], row["noticePath"]]), [[undefined, undefined], [undefined, f.noticeBody]]);
      const metadataName = "native-" + (target === "win32" ? "windows" : "linux") + "-package-metadata-123456789-1";
      const reviewAudit = JSON.parse(fs.readFileSync(path.join(packaged["lateNotice"].output, metadataName,
        "native-notice-review.json"), "utf8"));
      assert.deepEqual(listState(path.join(f.temp, metadataName + ".tar"), [metadataName], 60000)
        .map(name => name.replace(/\/$/, "")).sort(), [metadataName, ...["build-audit.json", "native-package.json",
        "native-notice-review.json"].map(name => metadataName + "/" + name)].sort());
      assert.equal(reviewAudit.manifest.sha256, fileHash(f.noticeManifest));
      assert.deepEqual(reviewAudit.snapshot, {version: 1, notice: {bytes: fs.statSync(f.noticeBody).size, sha256: fileHash(f.noticeBody)},
        dependencies: {fixture: {revision: "a".repeat(40)}}});
      assert.match(fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), /archive_sha256=[a-f0-9]{64}/);
      assert.match(fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), /metadata_artifact_name=native-.*-package-metadata-123456789-1\.tar/);
      assert.match(fs.readFileSync(f.env.GITHUB_STEP_SUMMARY, "utf8"), /no candidate execution or publication qualification/);
    });
  }

  test("rejects another run attempt before unpacking the same-run build artifact", t => {
    const f = fixture(t); Reflect.set(f.env, "GITHUB_RUN_ATTEMPT", "2");
    f.execute(0);
    assert.throws(() => f.execute(1), /packed-build bundle contents/);
    assert.equal(f.calls.filter(row => row.stage === "extract").length, 0);
  });
  test("rejects malformed platform and missing external pins before artifact access", t => {
    const f = fixture(t); Reflect.set(f.env, "NATIVE_PLATFORM", "darwin");
    assert.throws(() => f.execute(0), /exact platform, workspace and build SHA-256 pins/); Reflect.set(f.env, "NATIVE_PLATFORM", "linux");
    Reflect.set(f.env, "ORIGINAL_HANDOFF_SHA256", ""); assert.throws(() => f.execute(0), /exact platform, workspace and build SHA-256 pins/);
    assert.equal(f.calls.length, 0);
  });
  test("rejects the windows alias in both guards while retaining win32 as the canonical Windows input", t => {
    const f = fixture(t, "win32"); Reflect.set(f.env, "NATIVE_PLATFORM", "windows");
    assert.throws(() => f.execute(0), /exact platform, workspace and build SHA-256 pins/);
    assert.throws(() => f.execute(1), /Late native package paths, pins, platform or run identity/);
    assert.equal(f.calls.length, 0);
    const g = fixture(t, "win32"); g.execute(0);
    assert.deepEqual(g.calls.map(row => row.stage), ["notice-review"]);
  });
  test("rejects an unexpected raw download member and raw byte tampering", t => {
    const f = fixture(t); fs.writeFileSync(path.join(f.inputDirs["raw"], "extra"), "not raw"); f.execute(0);
    assert.throws(() => f.execute(1), /raw executable artifact contents/);
    const g = fixture(t); fs.appendFileSync(g.rawPath, "changed"); g.execute(0);
    assert.throws(() => g.execute(1), /raw executable/); assert.equal(g.calls.filter(row => row.stage === "extract").length, 0);
  });
  test("rejects extra packed-build/original members and oversized packed expansion before extraction", t => {
    const f = fixture(t); f.setTar(["native-linux-packed-build-123456789-1/", "unexpected"], "packed"); f.execute(0);
    assert.throws(() => f.execute(1), /missing or unexpected members/); assert.equal(f.calls.filter(row => row.stage === "extract").length, 0);
    const g = fixture(t); g.setTar(["native-linux-application-build-123456789-1/", "bad"], "original"); g.execute(0);
    assert.throws(() => g.execute(1), /missing or unexpected members/); assert.equal(g.calls.filter(row => row.stage === "extract").length, 0);
    const h = fixture(t); h.setExpanded(129 << 20); h.execute(0);
    assert.throws(() => h.execute(1), /bounded expansion limit/); assert.equal(h.calls.filter(row => row.stage === "extract").length, 0);
  });
  test("rejects a false original-to-packed source chain", t => {
    const f = fixture(t); f.execute(0);
    const entry = path.join(f.source.packed, "entrypoint.json"), candidate = JSON.parse(fs.readFileSync(entry, "utf8"));
    candidate.sourceEntrypoint.sha256 = "0".repeat(64); fs.writeFileSync(entry, JSON.stringify(candidate, null, 2) + "\n");
    assert.throws(() => f.execute(1), /Packed artifact, original handoff, raw executable or retained packing evidence/);
  });
  test("pre-package audit failure blocks packaging and archive upload outputs", t => {
    const f = fixture(t); f.setApi("pre-audit"); f.execute(0); assert.throws(() => f.execute(1), /fixture pre-audit rejected/);
    assert.equal(f.calls.some(row => row.stage === "package"), false); assert.equal(fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), "");
  });
  test("package result cannot broaden its member set or skip archive audit", t => {
    const f = fixture(t); f.setApi("extra-members"); f.execute(0); assert.throws(() => f.execute(1), /different bytes or members/);
    assert.equal(f.calls.some(row => row.stage === "archive-audit"), false);
    const g = fixture(t); g.setApi("post-audit"); g.execute(0); assert.throws(() => g.execute(1), /fixture archive audit rejected/);
    assert.equal(fs.readFileSync(g.env.GITHUB_OUTPUT, "utf8"), "");
  });
  test("notice review drift blocks before downloads and again at the tracked body path", t => {
    const f = fixture(t); fs.appendFileSync(f.noticeManifest, " ");
    assert.throws(() => f.execute(0), /fixture native notice review drift/);
    assert.deepEqual(f.calls.map(row => row.stage), ["notice-review"]);
    const g = fixture(t); g.execute(0); fs.appendFileSync(g.noticeBody, "changed after preflight\n");
    assert.throws(() => g.execute(1), /fixture native notice review drift/);
    assert.equal(g.calls.some(row => row.stage === "package"), false);
    assert.equal(g.calls.filter(row => row.stage === "extract").length, 0);
  });
  test("body and review manifest mutation during package generation blocks archive admission", t => {
    for (const mutation of ["mutate-notice-body", "mutate-notice-manifest"]) {
      const f = fixture(t); f.setApi(mutation); f.execute(0);
      assert.throws(() => f.execute(1), /changed during packaging/);
      assert.equal(f.calls.some(row => row.stage === "archive-audit"), false);
      assert.equal(fs.readFileSync(f.env.GITHUB_OUTPUT, "utf8"), "");
    }
  });
});
