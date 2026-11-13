"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {compileNativeObjects} = require("../scripts/native-object-cache");
const {linuxCompileEnvironment} = require("../scripts/probe-native-controller");

test.describe("Native controller object reuse", () => {
  test("fresh application manifests do not invalidate Linux object hits across fresh outputs", t => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-controller-cache-env-"));
    t.after(() => fs.rmSync(repo, {recursive: true, force: true}));
    const names = ["MOM_AOT_GPU_INPUTS", "MOM_AOT_CPU_CORE", "MOM_AOT_IO_SDK", "MOM_AOT_NVML_HEADER", "MOM_AOT_OPENSSL_ROOT", "MOM_AOT_BUILD_ONLY"];
    const previous = new Map(names.map(name => [name, process.env[name]]));
    t.after(() => {for (const name of names) {const value = previous.get(name); if (value === undefined) {delete process.env[name];} else {process.env[name] = value;}}});
    const source = path.join(repo, "main.cpp"), count = path.join(repo, "compile-count");
    fs.writeFileSync(source, "int main() { return 0; }\n"); fs.writeFileSync(count, "0");
    /** @type {(output: string) => string} */ const object = output => path.join(output, "main.o");
    /** @type {(output: string) => string} */ const depfile = output => path.join(output, "main.d");
    const script = `const fs=require("node:fs"),a=process.argv.slice(1),value=k=>a[a.indexOf(k)+1],source=value("-c"),depfile=value("-MF"),object=value("-o");if(!source||!depfile||!object)process.exit(41);const count=${JSON.stringify(count)};fs.writeFileSync(count,String(Number(fs.readFileSync(count,"utf8"))+1));fs.writeFileSync(object,fs.readFileSync(source));fs.writeFileSync(depfile,object+": "+source+"\\n");`;
    /** @type {(output: string) => {command: string, args: string[], source: string, object: string, depfile: string}} */
    const command = output => ({command: process.execPath, args: ["-e", script, "--", "-MF", depfile(output), "-c", source, "-o", object(output)],
      source, object: object(output), depfile: depfile(output)});
    const firstOutput = path.join(repo, "candidate-one"), secondOutput = path.join(repo, "candidate-two");
    for (const output of [firstOutput, secondOutput]) {fs.mkdirSync(output);}
    process.env["MOM_AOT_GPU_INPUTS"] = path.join(repo, "gpu-first.json");
    process.env["MOM_AOT_CPU_CORE"] = path.join(repo, "core-first.a");
    const firstEnvironment = linuxCompileEnvironment();
    for (const name of names) {assert.equal(Object.hasOwn(firstEnvironment, name), false, name);}
    const first = compileNativeObjects({repo, output: firstOutput, namespace: "linux/controller/app-test", commands: [command(firstOutput)],
      environment: firstEnvironment, jobs: 1, finalize: true, liveObjects: [object(firstOutput)]});
    assert.equal(first.objects[0]?.status, 0); assert.equal(first.objects[0]?.cacheHit, false);

    process.env["MOM_AOT_GPU_INPUTS"] = path.join(repo, "gpu-second.json");
    process.env["MOM_AOT_CPU_CORE"] = path.join(repo, "core-second.a");
    const secondEnvironment = linuxCompileEnvironment();
    const second = compileNativeObjects({repo, output: secondOutput, namespace: "linux/controller/app-test", commands: [command(secondOutput)],
      environment: secondEnvironment, jobs: 1, finalize: true, liveObjects: [object(secondOutput)]});
    assert.equal(second.objects[0]?.status, 0); assert.equal(second.objects[0]?.cacheHit, true);
    assert.deepEqual(fs.readFileSync(object(secondOutput)), fs.readFileSync(object(firstOutput)));
    assert.equal(fs.readFileSync(count, "utf8"), "1");
  });
});
