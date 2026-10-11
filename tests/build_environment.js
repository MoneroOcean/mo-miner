"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");
const {spawnSync} = require("node:child_process");
const test = require("node:test");
const {normalizeBuildEnvironment} = require("../scripts/build-environment");

/** @param {string} original @param {NodeJS.Platform} [platform] */
function npmEnvironment(original, platform = "linux") {
  const windows = platform === "win32";
  const api = windows ? path.win32 : path.posix;
  const root = windows ? "C:\\source\\project" : "/source/project";
  const npmCli = windows ? "C:\\tools\\npm\\bin\\npm-cli.js" : "/tools/npm/bin/npm-cli.js";
  const prefix = [];
  let directory = root;
  while (true) {
    prefix.push(api.resolve(directory, "node_modules", ".bin"));
    const parent = api.dirname(directory);
    if (parent === directory) {
      break;
    }
    directory = parent;
  }
  prefix.push(api.resolve(api.dirname(npmCli), "../node_modules/@npmcli/run-script/lib/node-gyp-bin"));
  return {[windows ? "Path" : "PATH"]: [...prefix, original].join(windows ? ";" : ":"),
    npm_package_json: api.join(root, "package.json"), npm_execpath: npmCli,
    npm_lifecycle_event: "build:release", npm_config_jobs: "4"};
}

test.describe("Build environment normalization", () => {
  test("actual npm CLI resolves shared distro run-script metadata without executing it", () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-npm-layout-"));
    try {
      const cli = path.join(temp, "node_modules/npm/bin/npm-cli.js");
      const runScript = path.join(temp, "node_modules/@npmcli/run-script");
      fs.mkdirSync(path.dirname(cli), {recursive: true});
      fs.mkdirSync(runScript, {recursive: true});
      fs.writeFileSync(cli, "throw new Error('npm CLI must not execute');\n");
      fs.writeFileSync(path.join(runScript, "package.json"), JSON.stringify({name: "@npmcli/run-script", main: "index.js"}));
      fs.writeFileSync(path.join(runScript, "index.js"), "throw new Error('run-script must not execute');\n");
      const env = npmEnvironment(process.platform === "win32" ? "C:\\caller\\bin;C:\\Windows" : "/caller/bin:/usr/bin", process.platform);
      const oldCli = env.npm_execpath;
      const api = process.platform === "win32" ? path.win32 : path.posix;
      const key = process.platform === "win32" ? "Path" : "PATH";
      assert.ok(oldCli && env[key]);
      env.npm_execpath = cli;
      env[key] = env[key].replace(api.resolve(api.dirname(oldCli), "../node_modules/@npmcli/run-script/lib/node-gyp-bin"),
        path.join(runScript, "lib/node-gyp-bin"));
      const normalized = normalizeBuildEnvironment(env);
      assert.equal(normalized[key], process.platform === "win32" ? "C:\\caller\\bin;C:\\Windows" : "/caller/bin:/usr/bin");
      assert.equal(normalized["npm_package_json"], undefined);
      // Existing caller scaffolds cannot be removed merely because metadata resolves.
      const shadowed = {...env, [key]: "caller-tools" + path.delimiter + env[key]};
      assert.deepEqual(normalizeBuildEnvironment(shadowed), shadowed);
    } finally {
      fs.rmSync(temp, {recursive: true, force: true});
    }
  });

  test("npm build environment removes one exact scaffold and only consumes its anchor", () => {
    const original = "/source/project/node_modules/.bin:/custom tools/bin:/usr/bin:";
    const env = npmEnvironment(original);
    const result = normalizeBuildEnvironment(env, "linux");
    assert.equal(result["PATH"], original);
    assert.equal(result["npm_package_json"], undefined);
    assert.equal(result["npm_config_jobs"], "4");
    assert.equal(result["npm_lifecycle_event"], "build:release");
    assert.ok(env["npm_package_json"]);
    assert.deepEqual(normalizeBuildEnvironment(result, "linux"), result);
  });

  test("direct, malformed and shadowed npm paths preserve the complete environment", () => {
    const good = npmEnvironment("/usr/bin");
    assert.ok(good["PATH"]);
    for (const env of [{PATH: "/usr/bin"}, {...good, npm_package_json: "package.json"},
      {...good, npm_execpath: "/tools/custom.js"}, {...good, npm_lifecycle_event: ""},
      {...good, PATH: "/deliberate/shadow/bin:" + good["PATH"]},
      {...good, PATH: good["PATH"].replace("/source/node_modules/.bin:", "")},
      {...good, PATH: good["PATH"] + "\n"}]) {
      assert.deepEqual(normalizeBuildEnvironment(env, "linux"), env);
    }
  });

  test("ambiguous repeated scaffold stays intact; nested normalized boundaries consume fresh anchors", () => {
    const original = "/caller/bin:/usr/bin";
    const outer = npmEnvironment(original);
    assert.ok(outer["PATH"]);
    const ambiguous = npmEnvironment(outer["PATH"]);
    assert.deepEqual(normalizeBuildEnvironment(ambiguous, "linux"), ambiguous);
    const outerChild = normalizeBuildEnvironment(outer, "linux");
    assert.ok(outerChild["PATH"]);
    const inner = npmEnvironment(outerChild["PATH"]);
    assert.equal(normalizeBuildEnvironment(inner, "linux")["PATH"], original);
    // Deliberate identical scaffold in the caller remains protected after consumption.
    const callerScaffold = npmEnvironment(original)["PATH"];
    const child = {...outerChild, PATH: callerScaffold};
    assert.deepEqual(normalizeBuildEnvironment(child, "linux"), child);
  });

  test("Windows PATH casing, caller shadow entries and Unicode are retained exactly", () => {
    const original = "C:\\source\\project\\node_modules\\.bin;C:\\工具\\bin;C:\\Windows;";
    const env = npmEnvironment(original, "win32");
    const result = normalizeBuildEnvironment({...env, PATH: env["Path"]}, "win32");
    assert.equal(result["Path"], original);
    assert.equal(result["PATH"], original);
    assert.equal(result["npm_package_json"], undefined);
    const inconsistent = {...env, PATH: "C:\\shadow"};
    assert.deepEqual(normalizeBuildEnvironment(inconsistent, "win32"), inconsistent);
  });

  test("build environment CLI has unambiguous consumed, unchanged and ASCII JSON results", () => {
    const helper = path.join(__dirname, "../scripts/build-environment.js");
    const fixture = npmEnvironment("/工具/bin:/usr/bin", process.platform);
    const env = {...process.env};
    // npm merges case-insensitive PATH keys; this fixture supplies one fresh key.
    for (const key of Object.keys(env)) {
      if (/^path$/i.test(key)) {
        delete env[key];
      }
    }
    Object.assign(env, fixture);
    const plain = spawnSync(process.execPath, [helper, "--path"], {env, encoding: "utf8", timeout: 5000});
    assert.equal(plain.status, 0);
    const expected = normalizeBuildEnvironment(env);
    const key = Object.keys(expected).find(name => process.platform === "win32" ? /^path$/i.test(name) : name === "PATH");
    assert.ok(key);
    assert.equal(plain.stdout, expected[key]);
    const json = spawnSync(process.execPath, [helper, "--json"], {env, encoding: "utf8", timeout: 5000});
    assert.equal(json.status, 0);
    assert.equal([...json.stdout].some(char => char.charCodeAt(0) > 127), false);
    assert.equal(JSON.parse(json.stdout), plain.stdout);
    const unchanged = spawnSync(process.execPath, [helper, "--path"], {env: expected, encoding: "utf8", timeout: 5000});
    assert.equal(unchanged.status, 3);
    assert.equal(unchanged.stdout, "");
  });
});
