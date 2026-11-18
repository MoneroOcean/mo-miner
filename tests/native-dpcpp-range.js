"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs");
const {dpcppGpuFlags} = require("../scripts/native-gpu-build");
const {deviceLinkArgs} = require("../scripts/build-native-gpu");

const repo = path.resolve(__dirname, "..");
const gyp = fs.readFileSync(path.join(repo, "binding.gyp"), "utf8");
const rangeFlag = "-fno-sycl-id-queries-fit-in-int";
const sdkRoot = process.env["MOM_TEST_DPCPP_SDK"];

/**
 * The compiler-backed check is deliberately opt-in. It never executes a generated image or
 * opens a SYCL runtime; the device-link driver's saved post-link property is the assertion.
 */
test.describe("Native Intel ID-query metadata", () => {
  test("maintained Intel device linking emits size_t ID-query metadata", {skip: process.platform !== "linux" ?
    "DPC++ compiler regression is Linux-only" : !sdkRoot ? "Set MOM_TEST_DPCPP_SDK to run the compiler-backed check" : false}, t => {
    assert.ok(sdkRoot && path.isAbsolute(sdkRoot), "MOM_TEST_DPCPP_SDK must be an absolute SDK path");
    assert.doesNotMatch(sdkRoot, /[\0\r\n]/, "MOM_TEST_DPCPP_SDK must be a literal path");
    assert.ok(fs.statSync(sdkRoot).isDirectory(), "MOM_TEST_DPCPP_SDK must name a directory");

    const auditEnv = process.env["MOM_TEST_DPCPP_AUDIT"];
    let output = "";
    if (auditEnv !== undefined) {
      assert.ok(path.isAbsolute(auditEnv), "MOM_TEST_DPCPP_AUDIT must be absolute");
      assert.doesNotMatch(auditEnv, /[\0\r\n]/, "MOM_TEST_DPCPP_AUDIT must be a literal path");
      fs.mkdirSync(auditEnv, {recursive: true});
      assert.deepEqual(fs.readdirSync(auditEnv), [], "refusing to overwrite an existing regression audit");
      output = auditEnv;
    } else {
      output = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dpcpp-range-"));
      t.after(() => fs.rmSync(output, {recursive: true, force: true}));
    }

    const compiler = path.join(sdkRoot, "bin/clang++");
    for (const tool of [compiler, path.join(sdkRoot, "bin/clang-offload-bundler"), path.join(sdkRoot, "bin/llvm-spirv")]) {
      assert.ok(fs.statSync(tool).isFile(), "missing selected DPC++ tool: " + tool);
    }
    const source = path.join(output, "range.cpp");
    fs.writeFileSync(source, "#include <sycl/sycl.hpp>\n\nint main() {\n  sycl::queue queue;\n  queue.single_task([=] {});\n  queue.wait();\n  return 0;\n}\n", {flag: "wx"});
    fs.writeFileSync(path.join(output, "scope.json"), JSON.stringify({version: 1,
      covers: ["maintained Intel dpcppGpuFlags compilation", "maintained deviceLinkArgs driver path", "compiler-generated default versus alias post-link .prop metadata"],
      doesNotCover: ["GPU discovery or runtime launch", "B580 correctness, throughput, shape/rank qualification", "full application graph or release admission"],
      source, sdk: sdkRoot, rangeFlag, productionQualified: false}, null, 2) + "\n", {flag: "wx"});

    /** @param {string} label @param {string} command @param {string[]} args */
    function run(label, command, args) {
      fs.writeFileSync(path.join(output, label + ".command.json"), JSON.stringify({command, args}) + "\n", {flag: "wx"});
      const result = spawnSync(command, args, {cwd: output, encoding: "utf8", timeout: 120000, maxBuffer: 16 << 20});
      fs.writeFileSync(path.join(output, label + ".log"), (result.stdout || "") + (result.stderr || ""), {flag: "wx"});
      assert.equal(result.error, undefined, label + ": " + (result.error?.message || "spawn failed"));
      assert.equal(result.signal, null, label + ": terminated by " + result.signal);
      assert.equal(result.status, 0, label + ": " + (result.stderr || result.stdout));
    }

    const defaultGyp = gyp.replaceAll(rangeFlag, "");
    const defaultCompile = [...dpcppGpuFlags(defaultGyp, "oneapi"), "-fsycl-targets=spir64", "-fPIC"];
    const aliasCompile = [...dpcppGpuFlags(gyp, "oneapi"), "-fsycl-targets=spir64", "-fPIC"];
    const defaultObject = path.join(output, "default.o"), aliasObject = path.join(output, "alias.o");
    run("compile-default", compiler, [...defaultCompile, "-c", source, "-o", defaultObject]);
    run("compile-alias", compiler, [...aliasCompile, "-c", source, "-o", aliasObject]);

    const defaultDevice = path.join(output, "default-device.o"), aliasDevice = path.join(output, "alias-device.o");
    const defaultLink = deviceLinkArgs(defaultGyp, undefined, [defaultObject], defaultDevice, "oneapi");
    const aliasLink = deviceLinkArgs(gyp, undefined, [aliasObject], aliasDevice, "oneapi");
    assert.equal(defaultLink.filter(flag => flag === rangeFlag).length, 0, "default device link unexpectedly carries the alias");
    assert.equal(aliasLink.filter(flag => flag === rangeFlag).length, 1, "maintained device link dropped or duplicated the alias");
    run("device-link-default", compiler, ["-save-temps", ...defaultLink]);
    run("device-link-alias", compiler, ["-save-temps", ...aliasLink]);

    const properties = (/** @type {string} */ prefix) => {
      const files = fs.readdirSync(output).filter(name => name.startsWith(prefix + "-spir64_") && name.endsWith(".prop"));
      assert.equal(files.length, 1, prefix + " compiler did not retain exactly one post-link property file");
      return path.join(output, files[0] || "");
    };
    const defaultProperty = properties("default"), aliasProperty = properties("alias");
    const defaultText = fs.readFileSync(defaultProperty, "utf8"), aliasText = fs.readFileSync(aliasProperty, "utf8");
    assert.doesNotMatch(defaultText, /^idQueriesRange=/m, "default device-link metadata unexpectedly selects size_t IDs");
    assert.equal((aliasText.match(/^idQueriesRange=1\|2$/gm) || []).length, 1, "alias device-link metadata is not mode2");
    assert.equal((aliasText.match(/^idQueriesRange=/gm) || []).length, 1, "alias device-link metadata is ambiguous");

    fs.writeFileSync(path.join(output, "result.json"), JSON.stringify({version: 1, productionQualified: false,
      defaultProperty: {file: defaultProperty, sha256: fileHash(defaultProperty), text: defaultText},
      aliasProperty: {file: aliasProperty, sha256: fileHash(aliasProperty), text: aliasText},
      defaultObjectSha256: fileHash(defaultObject), aliasObjectSha256: fileHash(aliasObject),
      defaultDeviceSha256: fileHash(defaultDevice), aliasDeviceSha256: fileHash(aliasDevice),
      defaultLink, aliasLink}, null, 2) + "\n", {flag: "wx"});
  });
});
