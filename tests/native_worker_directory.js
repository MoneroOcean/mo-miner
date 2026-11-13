"use strict";

const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), test = require("node:test"), assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process"), {extractCppBlock} = require("./common/cpp_source");
const source = fs.readFileSync(path.join(__dirname, "../runtime/process.cpp"), "utf8");
const block = extractCppBlock(source, "const char *prepareWorkerDirectory(");
let root = "", posix = "", windows = "";
/** @param {string} command @param {string[]} args @param {import('node:child_process').SpawnSyncOptionsWithStringEncoding} [options] */
function run(command, args, options = {encoding: "utf8"}) {
  const result = spawnSync(command, args, {timeout: 30000, ...options});
  assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
}

test.describe("Native worker data paths and safe loader directory", {skip: process.platform !== "linux"}, () => {
  test.before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-worker-directory-")); posix = path.join(root, "posix"); windows = path.join(root, "windows");
    const probe = "#include <algorithm>\n#include <string>\n#include <vector>\n#include <stdexcept>\n#include <iostream>\n" +
      "#ifndef _WIN32\n#include <filesystem>\n#include <sys/stat.h>\n#endif\n" + block +
      "int main(int argc,char**argv){std::vector<std::string> env;for(int i=1;i<argc;++i)env.emplace_back(argv[i]);" +
      'const char* directory=prepareWorkerDirectory(env);std::cout<<(directory?directory:"<inherit>")<<"\\n";' +
      'for(const auto& entry:env)std::cout<<entry<<"\\n";}\n';
    const file = path.join(root, "probe.cpp"); fs.writeFileSync(file, probe);
    /** @type {Array<[string, string[]]>} */
    const profiles = [[posix, []], [windows, ["-D_WIN32"]]];
    for (const [output, flags] of profiles) {
      run(process.env["CXX"] || "c++", ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", ...flags, file, "-o", output]);
    }
  });
  test.after(() => {if (root) {fs.rmSync(root, {recursive: true, force: true});}});
  /** @param {string[]} env @param {string} [executable] */
  function prepare(env, executable = posix) {
    const lines = run(executable, env, {encoding: "utf8", cwd: root}).trimEnd().split("\n");
    return {cwd: lines.shift(), env: Object.fromEntries(lines.map(line => {const index = line.indexOf("="); return [line.slice(0, index), line.slice(index + 1)];}))};
  }
  const atRoot = (/** @type {string} */ name) => root + "/" + name;
  test("non-workers retain their directory and original environment", () => {
    assert.deepEqual(prepare(["HOME=relative", "MOM_CLUSTER_WORKER=0"]), {cwd: "<inherit>", env: {HOME: "relative", MOM_CLUSTER_WORKER: "0"}});
  });
  test("only the exact worker marker selects a neutral directory", () => {
    assert.equal(prepare(["MOM_CLUSTER_WORKER=1extra"]).cwd, "<inherit>");
    assert.equal(prepare(["MOM_CLUSTER_WORKER=1"]).cwd, "/");
  });
  for (const name of ["HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "MOM_JIT_CACHE_DIR", "MOM_VERTHASH_DATA", "ACPP_APPDB_DIR", "HIPSYCL_APPDB_DIR", "SYCL_CACHE_DIR", "CUDA_CACHE_PATH"]) {
    test(`${name} retains caller-relative and absolute locations`, () => {
      assert.equal(prepare(["MOM_CLUSTER_WORKER=1", name + "=relative-é-💎"]).env[name], atRoot("relative-é-💎"));
      assert.equal(prepare(["MOM_CLUSTER_WORKER=1", name + "=/absolute/path"]).env[name], "/absolute/path");
    });
  }
  test("the default or empty Verthash dataset remains in the caller's directory", () => {
    for (const value of [[], ["MOM_VERTHASH_DATA="]]) {assert.equal(prepare(["MOM_CLUSTER_WORKER=1", ...value]).env["MOM_VERTHASH_DATA"], root + "/verthash.dat");}
  });
  test("empty caches, unrelated values and symlink-sensitive dot components are preserved", () => {
    const prepared = prepare(["MOM_CLUSTER_WORKER=1", "MOM_JIT_CACHE_DIR=", "UNKNOWN=relative", "SYCL_CACHE_DIR=link/../cache"]);
    assert.equal(prepared.env["MOM_JIT_CACHE_DIR"], ""); assert.equal(prepared.env["UNKNOWN"], "relative");
    assert.equal(prepared.env["SYCL_CACHE_DIR"], root + "/link/../cache");
  });
  test("Windows keeps its existing behavior pending its separate loader policy", () => {
    assert.deepEqual(prepare(["MOM_CLUSTER_WORKER=1", "HOME=relative"], windows), {cwd: "<inherit>", env: {HOME: "relative", MOM_CLUSTER_WORKER: "1"}});
  });
  test("spawn sets cwd before exec without invalidating environment pointers or changing JS APIs", () => {
    const loop = fs.readFileSync(path.join(__dirname, "../runtime/event-loop.cpp"), "utf8");
    const prepare = loop.indexOf("prepareWorkerDirectory(env)"), pointers = loop.indexOf("environmentPointers.push_back(entry.data())");
    assert.ok(prepare >= 0 && pointers > prepare);
    assert.match(loop, /options\.cwd = directory/);
    assert.match(block, /root\.st_uid != 0 \|\| \(root\.st_mode & 0022\)/);
  });
  test("an empty driver RUNPATH imports a cwd library before, but not after, neutral cwd", () => {
    const cwd = path.join(root, "untrusted"), driver = path.join(root, "driver"), link = path.join(root, "link");
    for (const directory of [cwd, driver, link]) {fs.mkdirSync(directory);}
    const needed = path.join(root, "needed.cpp"), library = path.join(root, "driver.cpp"), loader = path.join(root, "loader.cpp");
    fs.writeFileSync(needed, 'extern "C" int mom_answer(){return 42;}\n');
    fs.writeFileSync(library, 'extern "C" int mom_answer(); extern "C" int mom_probe(){return mom_answer();}\n');
    fs.writeFileSync(loader, "#include <dlfcn.h>\n#include <cstdlib>\nint main(int argc,char**argv){if(argc!=3)return 85;" +
      "void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)return std::atoi(argv[2])==0?0:86;" +
      'auto p=reinterpret_cast<int(*)()>(dlsym(h,"mom_probe"));int result=p?p():-1;dlclose(h);return result==std::atoi(argv[2])?0:87;}\n');
    const cxx = process.env["CXX"] || "c++", name = "libmom-cwd-dependency.so", executable = path.join(root, "loader");
    run(cxx, ["-shared", "-fPIC", needed, "-Wl,-soname," + name, "-o", path.join(link, name)]);
    fs.copyFileSync(path.join(link, name), path.join(cwd, name));
    const image = path.join(driver, "libmom-test-driver.so");
    run(cxx, ["-shared", "-fPIC", library, "-L" + link, "-l:" + name, "-Wl,-rpath,$ORIGIN/missing:", "-o", image]);
    run(cxx, [loader, "-ldl", "-o", executable]);
    const env = {...process.env}; delete env["LD_LIBRARY_PATH"]; delete env["LD_PRELOAD"]; delete env["LD_AUDIT"];
    run(executable, [image, "42"], {encoding: "utf8", cwd, env}); // The deliberately writable cwd supplies executable code.
    assert.equal(prepare(["MOM_CLUSTER_WORKER=1"]).cwd, "/");
    run(executable, [image, "0"], {encoding: "utf8", cwd: "/", env}); // Missing OS dependency now fails closed.
  });
});
