"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const crypto = require("node:crypto");
const vm = require("node:vm");
const {execFileSync, spawnSync} = require("node:child_process");
const helper = path.resolve(__dirname, "../scripts/linux-build-state.js");

/** @param {import("node:test").TestContext} t */
function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-linux-incremental-test-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  /** @param {string} name @param {string | Buffer} text */
  const put = (name, text) => {
    fs.mkdirSync(path.dirname(path.join(root, name)), {recursive: true});
    fs.writeFileSync(path.join(root, name), text);
  };
  for (const dir of ["native", "sycl", "xmrig", "scripts"]) {fs.mkdirSync(path.join(root, dir));}
  put("scripts/linux-build-state.js", fs.readFileSync(helper));
  put("scripts/build-helpers.sh", fs.readFileSync(path.resolve(__dirname, "../scripts/build-helpers.sh")));
  put("scripts/build-environment.js", fs.readFileSync(path.resolve(__dirname, "../scripts/build-environment.js")));
  for (const name of ["cpu-cflags.sh", "cpu-optflags.sh", "cpu-feature.sh", "cxx-combined.sh",
    "cxx-adaptivecpp.sh", "cxx-oneapi.sh", "combined-build.sh", "adaptivecpp-entrypoint.sh", "multicompiler-entrypoint.sh"]) {
    put("scripts/" + name, "#!/bin/sh\nexit 0\n");
  }
  for (const name of ["cpu-cflags.sh", "cpu-optflags.sh", "cpu-feature.sh"]) {
    const source = path.resolve(__dirname, "../scripts", name);
    put("scripts/" + name, fs.readFileSync(source));
    fs.chmodSync(path.join(root, "scripts", name), fs.statSync(source).mode & 0o777);
  }
  put("binding.gyp", "{}");
  put("native/a.h", "#define A 1\n");
  put("native/a.c", '#include "a.h"\nint a(void) { return A; }\n');
  put("native/b.c", "int b(void) { return 2; }\n");
  // Source-only tests do not require global node-gyp. This strict identity probe cannot configure
  // or build; native qualification exercises the actual installed command separately.
  put("fixture-gyp/bin/node-gyp", '#!/bin/sh\n[ "$#" -eq 1 ] && [ "$1" = --version ] || exit 2\nprintf "v12.2.0\\n"\n');
  fs.chmodSync(path.join(root, "fixture-gyp/bin/node-gyp"), 0o755);
  put("fixture-gyp/package.json", '{"name":"node-gyp","version":"12.2.0"}');
  put("fixture-gyp/addon.gypi", "{}");
  /** @type {NodeJS.ProcessEnv} */
  const env = {...process.env, MOM_FORCE_REBUILD: "0", CFLAGS: "", CXXFLAGS: "", CPPFLAGS: "",
    PATH: path.join(root, "fixture-gyp/bin") + path.delimiter + (process.env["PATH"] || "")};
  const dependencyArgument = execFileSync("bash", ["-ec", 'source scripts/build-helpers.sh; printf "%s" "$mom_make_dependency_args"'],
    {cwd: root, env, encoding: "utf8"});
  /** @param {string} operation */
  const state = operation => execFileSync(process.execPath,
    ["scripts/linux-build-state.js", operation, "build", "fixture", "cc"], {cwd: root, env, encoding: "utf8", stdio: "pipe"}).trim();
  const configure = () => put("build/Makefile", `DEPFLAGS = -MMD -MF $(depfile).raw
depfile = Release/.deps/$@.d
OBJECTS := $(patsubst ../native/%.c,Release/obj.target/mom/native/%.o,$(wildcard ../native/*.c))
all: Release/mom.node
Release/mom.node: $(OBJECTS)
\tcc -shared $^ -o $@
\t@echo link >> ../events
Release/obj.target/mom/native/%.o: ../native/%.c
\t@mkdir -p $(dir $@) Release/.deps/$(dir $@)
\tcc -fPIC $(DEPFLAGS) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
\tmv $(depfile).raw $(depfile)
\t@echo $* >> ../events
-include Release/.deps/Release/obj.target/mom/native/*.o.d
`);
  /** @param {() => void} [afterCompile] */
  const build = (afterCompile) => {
    const mode = state("prepare");
    if (!fs.existsSync(path.join(root, "build/Makefile"))) {configure();}
    execFileSync("make", ["-s", "-C", "build", dependencyArgument], {cwd: root, env, stdio: "pipe"});
    afterCompile?.();
    state("finish");
    return mode;
  };
  const events = () => fs.readFileSync(path.join(root, "events"), "utf8").trim().split("\n");
  return {root, put, env, state, build, events};
}

test.describe("Linux content-based incremental build", {skip: process.platform !== "linux"}, () => {
  test("Linux compiler boundary consumes npm scaffolding once and preserves caller tool paths", t => {
    const f = fixture(t);
    f.put("package.json", JSON.stringify({scripts: {probe: "bash scripts/probe.sh"}}));
    f.put("scripts/probe.sh", [
      "set -eu",
      "source scripts/build-helpers.sh",
      "source scripts/build-helpers.sh",
      'node -e \'console.log("FIXTURE_PATH=" + JSON.stringify({path:process.env.PATH,anchor:!!process.env.npm_package_json}))\'',
    ].join("\n"));
    const npmCli = process.env["npm_execpath"];
    const command = npmCli ? process.execPath : "npm";
    const args = [...(npmCli ? [npmCli] : []), "run", "probe"];
    /** @type {NodeJS.ProcessEnv} */
    const env = {...process.env, npm_package_json: "", npm_config_update_notifier: "false"};
    for (const callerPath of [env["PATH"] || "",
      path.join(f.root, "node_modules/.bin") + path.delimiter + (env["PATH"] || "")]) {
      for (let invocation = 0; invocation < 2; invocation++) {
        const result = spawnSync(command, args, {cwd: f.root, env: {...env, PATH: callerPath},
          encoding: "utf8", timeout: 20000});
        assert.equal(result.error, undefined);
        assert.equal(result.status, 0);
        const marker = result.stdout.split(/\r?\n/).find(line => line.startsWith("FIXTURE_PATH="));
        assert.ok(marker);
        assert.deepEqual(JSON.parse(marker.slice("FIXTURE_PATH=".length)), {path: callerPath, anchor: false});
      }
    }
  });

  test("optional SDK include aliases bind missing, retargeted and repaired states without masking other errors", t => {
    const f = fixture(t);
    // Exercise the exact shared production function; native preflight covers /opt SDK inventory
    // admission, without making source-only fixtures require privileged writes under /opt.
    const source = fs.readFileSync(helper, "utf8").match(/^function optionalSdkMetadata[^]*?\n}/m)?.[0];
    assert.ok(source);
    const metadata = /** @type {(name: string, filesOnly?: boolean) => string[] | null} */ (
      vm.runInNewContext("(" + source + ")", {fs, Error}));
    fs.mkdirSync(path.join(f.root, "sdk/include"), {recursive: true});
    const alias = path.join(f.root, "sdk/include/llvm");
    fs.symlinkSync("llvm-one", alias);
    const missing = metadata(alias);
    assert.equal(missing?.[0], "symlink");
    assert.equal(missing?.[3], "missing");
    fs.unlinkSync(alias);
    fs.symlinkSync("llvm-two", alias);
    const retargeted = metadata(alias);
    assert.notEqual(JSON.stringify(retargeted), JSON.stringify(missing));
    f.put("sdk/include/llvm-two/header.h", "repaired");
    assert.notEqual(JSON.stringify(metadata(alias)), JSON.stringify(retargeted));
    assert.equal(metadata(alias)?.length, 3);
    assert.equal(metadata(alias, true), null);
    fs.unlinkSync(alias);
    fs.symlinkSync("llvm", alias);
    assert.throws(() => metadata(alias), {code: "ELOOP"});
    assert.throws(() => metadata(path.join(f.root, "absent")), {code: "ENOENT"});
  });

  test("SYCL compile wrapper merges complete per-target graphs and refuses missing or malformed leaves", t => {
    const f = fixture(t);
    f.put("fake-compiler", `#!/usr/bin/env bash
set -euo pipefail
host= previous= target=
for arg in "$@"; do
  if [ "$previous" = -MF ]; then host=$arg; fi
  case "$arg" in
    -Xsycl-target-frontend=*) target=\${arg#*=} ;;
    -MF?*)
      if [ "\${PROBE_MISSING:-}" = "$target" ]; then continue; fi
      if [ "\${PROBE_MALFORMED:-}" = "$target" ]; then printf 'invalid\\n' >"\${arg#-MF}";
      else printf 'result.o: ../sycl/source.cpp ../sycl/%s.inc\\n' "$target" >"\${arg#-MF}"; fi ;;
  esac
  previous=$arg
done
printf 'result.o: ../sycl/source.cpp ../native/host.h\\n' >"$host"
printf 'compile\\n' >> events
`);
    fs.chmodSync(path.join(f.root, "fake-compiler"), 0o755);
    /** @param {NodeJS.ProcessEnv} [extra] */
    const compile = (extra = {}) => execFileSync("bash", ["-ec",
      "source scripts/build-helpers.sh; mom_compile_sycl_dependencies spir64,nvidia_gpu_sm_80 scripts/linux-build-state.js ./fake-compiler -MD -MF raw.d -c source.cpp"],
    {cwd: f.root, env: {...f.env, ...extra}, stdio: "pipe"});
    compile();
    const merged = fs.readFileSync(path.join(f.root, "raw.d"), "utf8");
    assert.match(merged, /host\.h/);
    assert.match(merged, /spir64\.inc/);
    assert.match(merged, /nvptx64-nvidia-cuda\.inc/);
    assert.equal(merged.split("\n").filter(Boolean).length, 1);
    assert.deepEqual(f.events(), ["compile"]);
    assert.throws(() => compile({PROBE_MISSING: "spir64"}));
    assert.throws(() => compile({PROBE_MALFORMED: "nvptx64-nvidia-cuda"}));
    f.put("outside", "keep");
    fs.unlinkSync(path.join(f.root, "raw.d"));
    fs.symlinkSync(path.join(f.root, "outside"), path.join(f.root, "raw.d"));
    assert.throws(() => compile());
    assert.equal(fs.readFileSync(path.join(f.root, "outside"), "utf8"), "keep");
  });

  test("merged device-only graph invalidates a backdated include without rebuilding unrelated objects", t => {
    const f = fixture(t);
    f.put("sycl/device.inc", "device one\n");
    const depfile = "build/Release/.deps/Release/obj.target/mom/native/a.o.d";
    f.build(() => {
      f.put("device.d", "Release/obj.target/mom/native/a.o: ../sycl/device.inc\n");
      execFileSync(process.execPath, [helper, "merge-deps", depfile, "device.d"], {cwd: f.root});
    });
    const header = path.join(f.root, "sycl/device.inc");
    const stat = fs.statSync(header);
    f.put("sycl/device.inc", "device two\n");
    fs.utimesSync(header, stat.atime, stat.mtime);
    assert.equal(f.state("prepare"), "incremental");
    assert.equal(fs.existsSync(path.join(f.root, "build/Release/obj.target/mom/native/a.o")), false);
    assert.equal(fs.existsSync(path.join(f.root, "build/Release/obj.target/mom/native/b.o")), true);
  });

  test("cache parent and fixed worker symlinks are refused before parking or deleting outputs", t => {
    const f = fixture(t);
    f.put("build/keep.txt", "keep");
    f.put("outside/keep.txt", "keep");
    const admit = () => execFileSync("bash", ["-ec",
      "source scripts/build-helpers.sh; mom_assert_linux_cache_paths build/cache; mv build build-platforms-hold"],
    {cwd: f.root, stdio: "pipe"});
    const cache = path.join(f.root, "build/cache");
    fs.symlinkSync(path.join(f.root, "outside"), cache);
    assert.throws(admit);
    fs.unlinkSync(cache);
    fs.mkdirSync(cache);
    for (const key of ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"]) {
      const worker = path.join(cache, key);
      fs.symlinkSync(path.join(f.root, "outside"), worker);
      assert.throws(admit);
      fs.unlinkSync(worker);
    }
    assert.equal(fs.existsSync(path.join(f.root, "build-platforms-hold")), false);
    assert.equal(fs.readFileSync(path.join(f.root, "outside/keep.txt"), "utf8"), "keep");
    assert.equal(fs.readFileSync(path.join(f.root, "build/keep.txt"), "utf8"), "keep");
    fs.symlinkSync(path.join(f.root, "outside"), path.join(cache, "custom"));
    admit();
    assert.equal(fs.lstatSync(path.join(f.root, "build-platforms-hold/cache/custom")).isSymbolicLink(), true);
    const entry = fs.readFileSync(path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
    assert.ok(entry.indexOf("mom_assert_linux_cache_paths build/cache") < entry.indexOf("if [ -d build ]; then mv build"));
    const adaptive = fs.readFileSync(path.resolve(__dirname, "../scripts/adaptivecpp-entrypoint.sh"), "utf8");
    assert.ok(adaptive.indexOf('mom_assert_linux_cache_paths "$build_root_path/cache"') < adaptive.indexOf("mkdir -m 0700"));
  });

  test("missing oneAPI fails before host cache locking or build-tree mutations", t => {
    const f = fixture(t);
    const entry = fs.readFileSync(path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
    const start = entry.indexOf("if [ ! -r /opt/intel/oneapi/setvars.sh ]; then");
    const end = entry.indexOf("\nset +u", start);
    assert.ok(start >= 0 && end > start);
    assert.ok(start < entry.indexOf("lock_dir=$(git rev-parse"));
    assert.ok(start < entry.indexOf("if [ -d build ]; then mv build"));
    f.put("build/keep.txt", "keep");
    // Exercise the production guard and lock branch with an absent fixture SDK, never /opt.
    const admission = entry.slice(start, end).replace("/opt/intel/oneapi/setvars.sh", '"$PWD/missing-setvars.sh"');
    const probe = [
      'unexpected() { printf "%s\\n" "$1" >> unexpected-events; return 77; }',
      ...["git", "mkdir", "flock", "mv", "node-gyp"].map(command => `${command}() { unexpected ${command}; }`),
      admission,
      "mv build build-platforms-hold",
      "node-gyp build",
    ].join("\n");
    const result = spawnSync("bash", ["-euc", probe], {cwd: f.root,
      env: {...f.env, MOM_DEV_HOST: "1"}, encoding: "utf8", timeout: 10000});
    assert.equal(result.error, undefined);
    assert.equal(result.status, 2);
    assert.match(result.stderr, /Missing oneAPI development tools.*scripts\/install-dev\.sh/);
    assert.equal(fs.existsSync(path.join(f.root, "unexpected-events")), false);
    assert.equal(fs.existsSync(path.join(f.root, "build-platforms-hold")), false);
    assert.equal(fs.readFileSync(path.join(f.root, "build/keep.txt"), "utf8"), "keep");
  });

  test("native cache admission rejects nonwritable selected paths before parking but ignores unselected caches", t => {
    const entry = fs.readFileSync(path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
    const start = entry.indexOf("mom_assert_linux_cache_paths build/cache\n");
    const end = entry.indexOf("\ncache_root=", start);
    assert.ok(start >= 0 && end > start);
    for (const blocked of ["build", "build/cache", "build/cache/oneapi", "build/cache/dpcpp"]) {
      const f = fixture(t);
      for (const key of ["oneapi", "dpcpp"]) {fs.mkdirSync(path.join(f.root, "build/cache", key), {recursive: true});}
      // Mock only the permission predicate: root-running fixtures must also reject unwritable caches.
      const probe = [
        "source scripts/build-helpers.sh",
        "build_compilers=(oneapi); platforms_hold=build-platforms-hold",
        '[() { if [[ $# = 4 && $1 = ! && $2 = -w ]]; then [[ $3 = "$UNWRITABLE" ]]; else builtin [ "$@"; fi; }',
        'mv() { printf "park\\n" > parking-event; }',
        entry.slice(start, end),
      ].join("\n");
      const result = spawnSync("bash", ["-euc", probe], {cwd: f.root,
        env: {...f.env, MOM_DEV_HOST: "1", UNWRITABLE: blocked}, encoding: "utf8", timeout: 10000});
      assert.equal(result.error, undefined);
      const selected = blocked !== "build/cache/dpcpp";
      assert.equal(result.status, selected ? 2 : 0);
      assert.equal(fs.existsSync(path.join(f.root, "parking-event")), !selected);
      if (selected) {
        assert.ok(result.stderr.includes(`Native development requires a writable cache: ${blocked}.`));
        assert.match(result.stderr, /Repair its ownership before retrying/);
      }
    }
  });

  test("cache ownership return is root-only and precedes platform-tree restoration", () => {
    const entry = fs.readFileSync(path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
    assert.match(entry, /^if \[ "\$EUID" -eq 0 \]; then chown -R --reference="\$PWD" "\$cache_root"; fi\nrestore_platform_tree$/m);
  });

  test("content edits with preserved/backdated mtimes invalidate only dependent objects and skip no-op relink", t => {
    const f = fixture(t);
    assert.equal(f.build(), "clean");
    assert.deepEqual(f.events(), ["a", "b", "link"]);
    assert.equal(f.build(), "incremental");
    assert.deepEqual(f.events(), ["a", "b", "link"]);
    const source = path.join(f.root, "native/a.c");
    const old = new Date("2000-01-01T00:00:00Z");
    f.put("native/a.c", '#include "a.h"\nint a(void) { return A + 1; }\n');
    fs.utimesSync(source, old, old);
    f.build();
    assert.equal(fs.statSync(source).mtimeMs, old.getTime());
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
    f.put("native/a.h", "#define A 3\n");
    fs.utimesSync(path.join(f.root, "native/a.h"), old, old);
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link", "a", "link"]);
  });

  test("graph, effective flags, compiler content and clean override invalidate all objects", t => {
    const f = fixture(t);
    f.build();
    f.put("binding.gyp", '{"changed":true}');
    fs.utimesSync(path.join(f.root, "binding.gyp"), new Date(0), new Date(0));
    assert.equal(f.build(), "clean");
    f.env["CXXFLAGS"] = "-DFIXTURE_CHANGED";
    assert.equal(f.build(), "clean");
    f.env["MOM_FORCE_REBUILD"] = "1";
    assert.equal(f.build(), "clean");
    assert.equal(f.events().filter(event => event === "b").length, 4);
  });

  test("new/deleted headers invalidate conservatively and force graph refresh", t => {
    const f = fixture(t);
    f.build();
    f.put("native/new.h", "#define NEW 1\n");
    assert.equal(f.build(), "incremental");
    fs.unlinkSync(path.join(f.root, "native/new.h"));
    assert.equal(f.build(), "incremental");
    assert.equal(f.events().filter(event => event === "b").length, 3);
  });

  test("successful builds prune renamed/deleted-source dependencies and reserved residue, not custom data", t => {
    const f = fixture(t);
    f.build();
    const deps = "build/Release/.deps/Release/obj.target/mom/native/";
    const current = fs.readFileSync(path.join(f.root, deps + "b.o.d"));
    f.put(deps + "a.o.d.raw", "interrupted compiler dependency output");
    f.put("build/.deps/Release/obj.target/mom/native/deleted.o.d", "obsolete dependency output");
    f.put("build/.mom-content-pending.json." + "a".repeat(24) + ".tmp", "interrupted checkpoint");
    f.put("build/.mom-content-state.json." + "b".repeat(24) + ".tmp", "interrupted receipt");
    f.put("build/.node-version", "obsolete version marker");
    const custom = ["build/.mom-content-state.json.short.tmp", "build/custom/notes.o.d",
      "build/Release/.deps/custom.o.d", "build/other-platform/keep.txt", "build/custom-cache/keep.txt"];
    for (const name of custom) {f.put(name, "keep");}
    fs.renameSync(path.join(f.root, "native/a.c"), path.join(f.root, "native/renamed.c"));
    f.build();
    for (const name of [deps + "a.o.d", deps + "a.o.d.raw", "build/.node-version",
      "build/.deps/Release/obj.target/mom/native/deleted.o.d",
      "build/.mom-content-pending.json." + "a".repeat(24) + ".tmp",
      "build/.mom-content-state.json." + "b".repeat(24) + ".tmp"]) {
      assert.equal(fs.existsSync(path.join(f.root, name)), false, name);
    }
    assert.ok(fs.existsSync(path.join(f.root, deps + "renamed.o.d")));
    assert.deepEqual(fs.readFileSync(path.join(f.root, deps + "b.o.d")), current);
    for (const name of custom) {assert.equal(fs.readFileSync(path.join(f.root, name), "utf8"), "keep");}
    fs.unlinkSync(path.join(f.root, "native/renamed.c"));
    f.build();
    assert.equal(fs.existsSync(path.join(f.root, deps + "renamed.o.d")), false);
    assert.ok(fs.existsSync(path.join(f.root, deps + "b.o.d")));
  });

  test("successful cleanup bounds abandoned dependency intermediates and empty generated directory chains", t => {
    const f = fixture(t);
    f.build();
    const deps = "build/Release/.deps/Release/obj.target/";
    const current = deps + "mom/native/b.o.d";
    const content = fs.readFileSync(path.join(f.root, current));
    const obsolete = deps + "retired/nested/deleted.o.d";
    const intermediates = [current + ".raw", current + ".raw.device-spir64",
      obsolete + ".raw.device-nvptx64-nvidia-cuda"];
    f.put(obsolete, "obsolete dependency");
    for (const name of intermediates) {f.put(name, "interrupted dependency");}
    fs.mkdirSync(path.join(f.root, "build/Release/obj.target/retired/nested"), {recursive: true});
    const unrelated = deps + "custom/keep.o.d.raw.device-invalid.target";
    f.put(unrelated, "keep");
    fs.mkdirSync(path.join(f.root, deps + "unrelated-empty"));
    f.state("prepare");
    for (const name of intermediates) {assert.ok(fs.existsSync(path.join(f.root, name)));}
    f.build();
    for (const name of [obsolete, ...intermediates, deps + "retired", "build/Release/obj.target/retired"]) {
      assert.equal(fs.existsSync(path.join(f.root, name)), false, name);
    }
    assert.deepEqual(fs.readFileSync(path.join(f.root, current)), content);
    assert.equal(fs.readFileSync(path.join(f.root, unrelated), "utf8"), "keep");
    assert.ok(fs.existsSync(path.join(f.root, deps + "unrelated-empty")));
    assert.equal(f.build(), "incremental");
    assert.deepEqual(f.events(), ["a", "b", "link", "link"]);
  });

  test("orphan cleanup waits for success and refuses reserved symlinks", t => {
    const f = fixture(t);
    f.build();
    const orphan = "build/Release/.deps/Release/obj.target/mom/native/removed.o.d.raw";
    f.put(orphan, "interrupted output");
    f.state("prepare");
    assert.equal(fs.existsSync(path.join(f.root, orphan)), true);
    const sentinel = path.join(f.root, "sentinel");
    f.put("sentinel", "keep");
    const temporary = path.join(f.root, "build/.mom-content-state.json." + "c".repeat(24) + ".tmp");
    fs.symlinkSync(sentinel, temporary);
    assert.throws(() => f.state("finish"));
    assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
    assert.equal(fs.lstatSync(temporary).isSymbolicLink(), true);
    fs.unlinkSync(temporary);
    fs.symlinkSync(sentinel, path.join(f.root, "build/.node-version"));
    assert.throws(() => f.state("finish"));
    assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
  });

  test("shared dependency flags preserve user flags and capture backdated system-header edits selectively", t => {
    const f = fixture(t);
    const flags = execFileSync("bash", ["-ec", 'source scripts/build-helpers.sh; source scripts/build-helpers.sh; printf "%s\\n%s" "$CFLAGS" "$CXXFLAGS"'],
      {cwd: f.root, env: {...f.env, CFLAGS: "-DFIXTURE_USER_C", CXXFLAGS: "-DFIXTURE_USER_CXX"}, encoding: "utf8"});
    assert.equal(flags, "-DFIXTURE_USER_C\n-DFIXTURE_USER_CXX");
    const sdk = fs.mkdtempSync(path.join(os.tmpdir(), "mom-system-header-test-"));
    t.after(() => fs.rmSync(sdk, {recursive: true, force: true}));
    const header = path.join(sdk, "fixture-system.h");
    fs.writeFileSync(header, "#define SYSTEM_A 5\n");
    f.env["CPPFLAGS"] = "-isystem " + sdk;
    f.put("native/a.c", "#include <fixture-system.h>\nint a(void) { return SYSTEM_A; }\n");
    f.build();
    const dep = fs.readFileSync(path.join(f.root, "build/Release/.deps/Release/obj.target/mom/native/a.o.d"), "utf8");
    assert.ok(dep.includes(header));
    fs.writeFileSync(header, "#define SYSTEM_A 7\n");
    fs.utimesSync(header, new Date(0), new Date(0));
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
  });

  test("fixture-local node-gyp has a strict probe contract and identity changes invalidate", t => {
    const f = fixture(t);
    const probe = path.join(f.root, "fixture-gyp/bin/node-gyp");
    assert.throws(() => execFileSync(probe, ["configure"], {stdio: "pipe"}));
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.put("fixture-gyp/package.json", '{"name":"node-gyp","version":"12.2.1"}');
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.put("fixture-gyp/addon.gypi", '{"defines":["CHANGED"]}');
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    fs.mkdirSync(path.join(f.root, "extra-tools"));
    f.env["PATH"] = path.join(f.root, "extra-tools") + path.delimiter + f.env["PATH"];
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
  });

  test("normalizer-only edits invalidate Linux recipes and immutable worker admission", t => {
    const f = fixture(t);
    const entrypoint = fs.readFileSync(path.resolve(__dirname, "../scripts/multicompiler-entrypoint.sh"), "utf8");
    const fingerprint = entrypoint.match(/^source_fingerprint\(\) \{[^]*?^\}/m)?.[0];
    const reuse = entrypoint.match(/^case "\$\{MOM_REUSE_BUILT_WORKER:-0\}" in[^]*?^esac$/m)?.[0];
    assert.ok(fingerprint && reuse);
    const shell = {cwd: f.root, env: {...f.env, MOM_REUSE_BUILT_WORKER: "1", backend: "intel", default: "oneapi"},
      encoding: /** @type {const} */ ("utf8"), timeout: 10000};
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.put("build/lin/Release/oneapi/mom.node", "fixture worker");
    const receipt = execFileSync("bash", ["-ec", fingerprint + "\nsource_fingerprint oneapi"], shell);
    f.put("build/lin/Release/oneapi/mom.node.sources.sha256", receipt);
    const admission = fingerprint + "\n" + reuse;
    const unchanged = spawnSync("bash", ["-ec", admission, "fixture-reuse", "true"], shell);
    assert.equal(unchanged.error, undefined);
    assert.equal(unchanged.status, 0);
    const normalizer = path.join(f.root, "scripts/build-environment.js");
    fs.appendFileSync(normalizer, "\n// Fixture recipe change.\n");
    fs.utimesSync(normalizer, new Date(0), new Date(0));
    const changed = spawnSync("bash", ["-ec", admission, "fixture-reuse", "true"], shell);
    assert.equal(changed.error, undefined);
    assert.equal(changed.status, 2);
    assert.match(changed.stderr, /source fingerprint differs/);
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
  });

  test("Gyp environment changes invalidate; custom nested include configurations rebuild conservatively", t => {
    const f = fixture(t);
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.env["GYP_DEFINES"] = "mom_cuda_arch=nvidia_gpu_sm_90";
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.env["npm_package_config_node_gyp_arch"] = "arm64";
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.env["npm_config_nodedir"] = path.join(f.root, "alternate-node-sdk");
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "clean");
    delete f.env["npm_config_nodedir"];
    f.put("common.gypi", '{"includes":["custom/nested.gypi"]}');
    f.put("custom/nested.gypi", '{"defines":["INITIAL"]}');
    assert.equal(f.build(), "clean");
    f.put("custom/nested.gypi", '{"defines":["CHANGED"]}');
    fs.utimesSync(path.join(f.root, "custom/nested.gypi"), new Date(0), new Date(0));
    assert.equal(f.build(), "clean");
  });

  test("interrupted same-input build retains valid partial objects; stale/missing outputs relink", t => {
    const f = fixture(t);
    f.build();
    f.put("native/a.c", '#include "a.h"\nint a(void) { return 8; }\n');
    assert.equal(f.state("prepare"), "incremental");
    assert.equal(fs.existsSync(path.join(f.root, "build/Release/obj.target/mom/native/b.o")), true);
    // A killed compiler can recreate a dirty .o without replacing its old complete .d file.
    f.put("build/Release/obj.target/mom/native/a.o", "interrupted compile");
    f.put("build/Release/obj.target/mom.node", "interrupted link");
    f.put("build/Release/mom.node", "interrupted copy");
    assert.equal(f.build(), "incremental");
    assert.equal(f.events().filter(event => event === "b").length, 1);
    fs.unlinkSync(path.join(f.root, "build/Release/mom.node"));
    f.build();
    f.put("build/Release/mom.node", "tampered output");
    f.build();
    assert.equal(f.events().filter(event => event === "b").length, 1);
    assert.equal(f.events().filter(event => event === "link").length, 4);
  });

  test("object or dependency-receipt corruption invalidates only that object before relinking", t => {
    const f = fixture(t);
    f.build();
    f.put("build/Release/obj.target/mom/native/a.o", "corrupt object");
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
    f.put("build/Release/.deps/Release/obj.target/mom/native/a.o.d",
      "deps_Release/obj.target/mom/native/a.o := ../native/a.c\n");
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link", "a", "link"]);
  });

  test("missing dependency coverage and malformed receipt fail closed; finish rejects concurrent source changes", t => {
    const f = fixture(t);
    f.build();
    fs.unlinkSync(path.join(f.root, "build/Release/.deps/Release/obj.target/mom/native/b.o.d"));
    assert.equal(f.build(), "clean");
    f.put("build/.mom-content-state.json", '{"schema":1}');
    assert.equal(f.build(), "clean");
    f.state("prepare");
    f.put("native/a.h", "#define A 19\n");
    assert.throws(() => f.state("finish"));
    assert.equal(fs.existsSync(path.join(f.root, "build/.mom-content-pending.json")), true);
  });

  test("SDK recipe and compiler bytes bind the selected tool without scanning unrelated installations", t => {
    const f = fixture(t);
    f.put("sdk/bin/compiler", "#!/bin/sh\nexit 0\n");
    fs.chmodSync(path.join(f.root, "sdk/bin/compiler"), 0o755);
    f.put("sdk/.mom-build-identity", "recipe-one");
    const alias = path.join(f.root, "compiler-alias");
    fs.symlinkSync(path.join(f.root, "sdk/bin/compiler"), alias);
    /** @param {string} operation */
    const selected = operation => execFileSync(process.execPath,
      ["scripts/linux-build-state.js", operation, "build", "fixture", alias],
      {cwd: f.root, env: f.env, encoding: "utf8"}).trim();
    assert.equal(selected("prepare"), "clean");
    // No output is published in this fixture. A changed tool or recipe must still clear the cache.
    f.put("build/Makefile", "# fixture");
    f.put("unselected-sdk/.mom-build-identity", "unrelated-recipe");
    assert.equal(selected("prepare"), "incremental");
    f.put("sdk/.mom-build-identity", "recipe-two");
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    f.put("sdk/bin/compiler", "#!/bin/sh\nexit 1\n");
    assert.equal(selected("prepare"), "clean");
    const driver = path.join(f.root, "llvm20/bin/clang++");
    f.put("llvm20/bin/clang++", "#!/bin/sh\nexit 0\n");
    f.put("sdk/etc/AdaptiveCpp/acpp-core.json", JSON.stringify({"default-clang": driver}));
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    f.put("llvm20/bin/clang++", "#!/bin/sh\nexit 1\n");
    assert.equal(selected("prepare"), "clean");
    fs.unlinkSync(path.join(f.root, "sdk/.mom-build-identity"));
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    assert.equal(selected("prepare"), "incremental");
  });

  test("optional dangling SDK library aliases are fingerprinted; selected missing tools still fail", t => {
    const f = fixture(t);
    f.put("sdk/bin/compiler", "#!/bin/sh\nexit 0\n");
    fs.mkdirSync(path.join(f.root, "sdk/lib"));
    const library = path.join(f.root, "sdk/lib/libLLVM-21.so");
    fs.symlinkSync("libLLVM.so.21.1", library);
    const compiler = path.join(f.root, "sdk/bin/compiler");
    /** @param {string} operation */
    const selected = operation => execFileSync(process.execPath,
      ["scripts/linux-build-state.js", operation, "build", "fixture", compiler],
      {cwd: f.root, env: f.env, encoding: "utf8", stdio: "pipe"}).trim();
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    assert.equal(selected("prepare"), "incremental");
    fs.unlinkSync(library);
    fs.symlinkSync("libLLVM.so.21.2", library);
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    f.put("sdk/lib/libLLVM.so.21.2", "repaired library");
    assert.equal(selected("prepare"), "clean");
    f.put("build/Makefile", "# fixture");
    assert.equal(selected("prepare"), "incremental");
    fs.unlinkSync(compiler);
    fs.symlinkSync("missing-compiler", compiler);
    assert.throws(() => selected("prepare"));
  });

  test("selected CUDA libdevice content changes invalidate and absent bitcode fails closed", t => {
    const f = fixture(t);
    f.env["MOM_COMBINED_TARGETS"] = "nvidia_gpu_sm_80";
    f.env["CUDA_PATH"] = path.join(f.root, "sdk");
    f.put("sdk/nvvm/libdevice/libdevice.10.bc", "device-math-one");
    assert.equal(f.build(), "clean");
    assert.equal(f.build(), "incremental");
    f.put("sdk/nvvm/libdevice/libdevice.10.bc", "device-math-two");
    fs.utimesSync(path.join(f.root, "sdk/nvvm/libdevice/libdevice.10.bc"), new Date(0), new Date(0));
    assert.equal(f.build(), "clean");
    fs.unlinkSync(path.join(f.root, "sdk/nvvm/libdevice/libdevice.10.bc"));
    assert.throws(() => f.state("prepare"));
  });

  test("external dependency contents invalidate their object even with backdated mtimes", t => {
    const f = fixture(t);
    const external = fs.mkdtempSync(path.join(os.tmpdir(), "mom-external-header-test-"));
    t.after(() => fs.rmSync(external, {recursive: true, force: true}));
    const header = path.join(external, "sdk.h");
    fs.writeFileSync(header, "#define SDK_VALUE 1\n");
    f.put("native/a.c", `#include "${header}"\nint a(void) { return SDK_VALUE; }\n`);
    f.build();
    fs.writeFileSync(header, "#define SDK_VALUE 2\n");
    fs.utimesSync(header, new Date(0), new Date(0));
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
  });

  test("generated graph corruption cleans and symlinked outputs cannot be signed", t => {
    const f = fixture(t);
    f.build();
    f.put("build/Makefile", "# corrupt generated graph");
    assert.equal(f.build(), "clean");
    const output = path.join(f.root, "build/Release/mom.node");
    fs.unlinkSync(output);
    fs.symlinkSync(path.join(f.root, "native/a.c"), output);
    assert.throws(() => f.state("prepare"));
  });

  test("modern Gyp command-first dependency files do not confuse := commands with : dependency rules", t => {
    const f = fixture(t);
    f.build();
    const dep = "build/Release/.deps/Release/obj.target/mom/native/a.o.d";
    f.put(dep, ["cmd_Release/obj.target/mom/native/a.o := cc -c ../native/a.c -o Release/obj.target/mom/native/a.o",
      "Release/obj.target/mom/native/a.o: ../native/a.c " + String.fromCharCode(92),
      " ../native/a.h", "../native/a.h:", ""].join("\n"));
    const receipt = path.join(f.root, "build/.mom-content-state.json");
    const state = JSON.parse(fs.readFileSync(receipt, "utf8"));
    state.objects["Release/obj.target/mom/native/a.o"].dependencies = crypto.createHash("sha256")
      .update(fs.readFileSync(path.join(f.root, dep))).digest("hex");
    fs.writeFileSync(receipt, JSON.stringify(state));
    f.put("native/a.h", "#define A 11\n");
    fs.utimesSync(path.join(f.root, "native/a.h"), new Date(0), new Date(0));
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
  });

  test("Gyp deps-variable format with continued absolute source-root headers remains selective", t => {
    const f = fixture(t);
    f.build();
    const dep = "build/Release/.deps/Release/obj.target/mom/native/a.o.d";
    f.put(dep, ["deps_Release/obj.target/mom/native/a.o := ../native/a.c " + String.fromCharCode(92),
      " " + path.join(f.root, "native/a.h"), "",
      "Release/obj.target/mom/native/a.o: $(deps_Release/obj.target/mom/native/a.o)", ""].join("\n"));
    const receipt = path.join(f.root, "build/.mom-content-state.json");
    const state = JSON.parse(fs.readFileSync(receipt, "utf8"));
    state.objects["Release/obj.target/mom/native/a.o"].dependencies = crypto.createHash("sha256")
      .update(fs.readFileSync(path.join(f.root, dep))).digest("hex");
    fs.writeFileSync(receipt, JSON.stringify(state));
    f.put("native/a.h", "#define A 17\n");
    fs.utimesSync(path.join(f.root, "native/a.h"), new Date(0), new Date(0));
    f.build();
    assert.deepEqual(f.events(), ["a", "b", "link", "a", "link"]);
  });
});
