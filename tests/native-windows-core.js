"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {windowsCoreIncludeMembership} = require("../scripts/build-native-windows-core");

const root = path.resolve(__dirname, "..");
const binding = fs.readFileSync(path.join(root, "binding.gyp"), "utf8");
const coreSource = fs.readFileSync(path.join(root, "native/core.cpp"), "utf8");
const gpuHeader = fs.readFileSync(path.join(root, "sycl/lib.h"), "utf8");
const ethashSource = fs.readFileSync(path.join(root, "xmrig/3rdparty/libethash/ethash_internal.c"), "utf8");
// Make graph fixtures are never compiled; keep the add-on hook synthetic because node-gyp may be under Program Files.
const fixtureNodeGypDir = "node-gyp";
const windowsSources = [
  "xmrig/crypto/common/VirtualMemory_win.cpp",
  "xmrig/backend/cpu/platform/BasicCpuInfo.cpp",
  "xmrig/hw/msr/Msr.cpp",
  "xmrig/hw/msr/Msr_win.cpp",
  "xmrig/crypto/rx/RxFix_win.cpp",
  "xmrig/crypto/randomx/jit_compiler_x86.cpp",
  "xmrig/crypto/randomx/jit_compiler_x86_static.asm",
  "xmrig/crypto/cn/r/CryptonightR_gen.cpp",
  "xmrig/crypto/cn/asm/win64/cn_main_loop.asm",
  "xmrig/crypto/cn/asm/win64/CryptonightR_template.asm",
  "xmrig/3rdparty/argon2/arch/generic/lib/argon2-arch.c",
];

/** @param {string} command @returns {string | undefined} */
function commandPath(command) {
  const result = spawnSync(process.platform === "win32" ? "where.exe" : "which", [command], {encoding: "utf8", timeout: 5000});
  return result.status === 0 && !result.error ? result.stdout.trim().split(/\r?\n/)[0] : undefined;
}

/** @returns {string | undefined} */
function findGyp() {
  const candidates = [];
  if (process.env["MOM_GYP"]) {candidates.push(process.env["MOM_GYP"]);}
  const nodeGyp = commandPath("node-gyp");
  if (nodeGyp) {candidates.push(path.resolve(path.dirname(nodeGyp), "..", "gyp", "gyp_main.py"));}
  const globalNpmRoot = spawnSync("npm", ["root", "-g"], {encoding: "utf8", timeout: 5000});
  if (globalNpmRoot.status === 0 && !globalNpmRoot.error) {
    const npmRoot = globalNpmRoot.stdout.trim();
    candidates.push(path.join(npmRoot, "node-gyp", "gyp", "gyp_main.py"));
    candidates.push(path.join(npmRoot, "npm", "node_modules", "node-gyp", "gyp", "gyp_main.py"));
  }
  candidates.push(
    path.join(path.dirname(process.execPath), "node_modules", "npm", "node_modules", "node-gyp", "gyp", "gyp_main.py"),
    "/usr/share/nodejs/node-gyp/gyp/gyp_main.py",
    "/usr/local/lib/node_modules/npm/node_modules/node-gyp/gyp/gyp_main.py",
  );
  return candidates.find((candidate) => candidate && fs.statSync(candidate, {throwIfNoEntry: false})?.isFile());
}

const gyp = findGyp();
/** Windows "App execution aliases" put python.exe stubs on PATH that only print a Store hint, so
 * accept a candidate only if it actually runs. @param {string | undefined} command */
const runnablePython = command => command &&
  spawnSync(command, ["--version"], {encoding: "utf8", timeout: 5000}).status === 0 ? command : undefined;
const python = process.env["PYTHON"] || runnablePython(commandPath("python3") ? "python3" : undefined) ||
  runnablePython(commandPath("python"));

/** @param {string} gypFile @returns {string} */
function gypRoot(gypFile) {return path.dirname(path.dirname(gypFile));}

/** @param {import("node:test").TestContext} context @param {0 | 1} nativeCore @param {string} osName @returns {{mom: string, core: string, directory: string} | undefined} */
function generate(context, nativeCore, osName = "win") {
  if (!gyp || !python) {
    context.skip("Python GYP is unavailable for the disposable graph generation");
    return undefined;
  }
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-windows-core-gyp-"));
  context.after(() => fs.rmSync(output, {recursive: true, force: true}));
  const packageRoot = gypRoot(gyp);
  const args = [gyp, path.join(root, "binding.gyp"), "-f", "make", "--generator-output", output,
    "-Goutput_dir=.", `-DOS=${osName}`, "-Dtarget_arch=x64", `-Dmom_native_core=${nativeCore}`, "-Dmom_native_uv_include=/sdk/libuv/include", "-Dlibrary=shared_library",
    "-Dvisibility=default", "-Dnode_root_dir=/tmp/node", `-Dnode_gyp_dir=${fixtureNodeGypDir}`,
    `-Dmodule_root_dir=${root}`, "-Dnode_engine=v8", "-Dnode_lib_file=/tmp/node/node.lib",
    "-Dstandalone_static_library=1", "-I", path.join(packageRoot, "addon.gypi"), `--depth=${root}`];
  const result = spawnSync(python, args, {cwd: root, encoding: "utf8", timeout: 30000, maxBuffer: 16 << 20});
  assert.equal(result.status, 0, `GYP generation failed:\n${result.stdout}\n${result.stderr}`);
  return {
    mom: fs.readFileSync(path.join(output, "mom.target.mk"), "utf8"),
    core: fs.readFileSync(path.join(output, "mom_native_core.target.mk"), "utf8"),
    directory: output,
  };
}

/** @param {import("node:test").TestContext} context @param {boolean} [nodeDefaults] @param {boolean} [intel]
 * @returns {{project: string, addonProject: string, directory: string} | undefined} */
function generateFakeMsvsProject(context, nodeDefaults = false, intel = false) {
  if (!gyp || !python) {
    context.skip("Python GYP is unavailable for configure-only MSVS project generation");
    return undefined;
  }
  const sourceDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-windows-core-msvs-"));
  context.after(() => fs.rmSync(sourceDirectory, {recursive: true, force: true}));
  fs.writeFileSync(path.join(sourceDirectory, "binding.gyp"), binding);
  const output = path.join(sourceDirectory, "generated");
  const fakeVsPath = path.join(sourceDirectory, "fake-vs-descriptor");
  const common = path.join(sourceDirectory, "node-common.gypi");
  // Real downloaded Node headers add these defaults after addon.gypi. A bare GYP
  // graph misses them and cannot prove a standalone Core is free of addon/LTCG policy.
  if (nodeDefaults) {
    fs.writeFileSync(common, JSON.stringify({target_defaults: {
      include_dirs: ["<(node_root_dir)/deps/uv/include"], defines: ["_HAS_EXCEPTIONS=0"],
      configurations: Object.fromEntries(["Debug", "Release"].map(name => [name, {msvs_settings: {
        VCCLCompilerTool: {WholeProgramOptimization: "true", AdditionalOptions: ["/Zc:__cplusplus", "-std:c++20", "/Zm2000"]},
        VCLibrarianTool: {AdditionalOptions: ["/LTCG:INCREMENTAL"]},
        VCLinkerTool: {AdditionalOptions: ["/LTCG:INCREMENTAL"]},
      }}])),
    }}));
  }
  const packageRoot = gypRoot(gyp);
  // Python modules sit beside gyp_main.py, one level below addon.gypi's package root.
  const wrapper = [
    "import os, sys",
    `sys.path.insert(0, ${JSON.stringify(path.join(path.dirname(gyp), "pylib"))})`,
    "import gyp",
    "from gyp import MSVSVersion",
    "from gyp.MSVSVersion import VisualStudioVersion",
    "import gyp.easy_xml",
    "",
    "def write_xml(content, filename, encoding='utf-8', pretty=False, win32=False):",
    "    dirname = os.path.dirname(filename)",
    "    if dirname: os.makedirs(dirname, exist_ok=True)",
    "    xml = gyp.easy_xml.XmlToString(content, encoding, pretty)",
    "    if isinstance(xml, bytes): xml = xml.decode(encoding)",
    "    with open(filename, 'w', encoding=encoding, newline='') as stream: stream.write(xml)",
    "gyp.easy_xml.WriteXmlIfChanged = write_xml",
    `fake = VisualStudioVersion('2022', 'Fake Visual Studio 2022', '12.00', '17.0', False, True, ${JSON.stringify(fakeVsPath)}, False, default_toolset='v143', compatible_sdks=[])`,
    "MSVSVersion.SelectVisualStudioVersion = lambda version=None, e=None: fake",
    "sys.argv = ['gyp_main.py'] + sys.argv[1:]",
    "sys.exit(gyp.script_main())",
  ].join("\n");
  const args = ["binding.gyp", "-f", "msvs", "--generator-output", output, "-Goutput_dir=.", "-DOS=win",
    "-Dmom_native_core=1", `-Dmom_native_intel=${Number(intel)}`, "-Dmom_native_uv_include=/sdk/libuv/include", "-Dlibrary=shared_library", "-Dvisibility=default", "-Dnode_root_dir=/tmp/node",
    `-Dnode_gyp_dir=${packageRoot}`, `-Dmodule_root_dir=${root}`, "-Dnode_engine=v8",
    "-Dnode_lib_file=/tmp/node/node.lib", "-Dstandalone_static_library=1", "-I", path.join(packageRoot, "addon.gypi"),
    ...(nodeDefaults ? ["-I", common] : []), `--depth=${root}`];
  const result = spawnSync(python, ["-c", wrapper, ...args], {cwd: sourceDirectory, encoding: "utf8", timeout: 30000, maxBuffer: 16 << 20});
  assert.equal(result.status, 0, ["configure-only MSVS generation failed:", result.stdout, result.stderr].join("\n"));
  const project = path.join(output, "mom_native_core.vcxproj");
  assert.ok(fs.statSync(project).isFile(), "configure-only GYP run must emit the Core project");
  return {project: fs.readFileSync(project, "utf8"), addonProject: fs.readFileSync(path.join(output, "mom.vcxproj"), "utf8"), directory: sourceDirectory};
}

/** @param {string} makefile @returns {Set<string>} */
function objectGraph(makefile) {
  const body = makefile.match(/\r?\nOBJS := \\\r?\n([\s\S]*?)\r?\n\r?\n# Add/)?.[1];
  assert.ok(body, "generated target must contain an object graph");
  return new Set(body.split(/\r?\n/).map((line) => line.trim().replace(/\\$/, "").trim())
    .filter((line) => line.startsWith("$(obj).target/$(TARGET)/") && !line.includes("win_delay_load_hook.o")));
}

/** @param {string} makefile @returns {string} */
function semanticMakefile(makefile) {
  // GYP orders generic suffix rules by the union of all generated targets. Adding the opt-in
  // static target can therefore reorder those recipes without changing mom's sources, flags,
  // dependencies, or link rule.
  return makefile.replace(/\n\$\(obj\)\.\$\(TOOLSET\)\/\$\(TARGET\)\/%.o: [^\n]+\n\t@\$\(call do_cmd,[^\n]+\n/g, "\n");
}

/** @returns {string} */
function nativeTarget() {
  const start = binding.indexOf('"target_name": "mom_native_core"');
  const end = binding.indexOf('\n    {\n      "target_name": "hoohash_host"', start);
  assert.ok(start >= 0 && end > start, "native Core target block must be present");
  return binding.slice(start, end);
}

test.describe("Windows opt-in native Core GYP graph", () => {
  test("tracks include-tree file contents as well as membership", (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-native-windows-core-includes-"));
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const header = path.join(root, "sdk.h"); fs.writeFileSync(header, "alpha");
    const first = windowsCoreIncludeMembership([root]), same = windowsCoreIncludeMembership([root]);
    assert.equal(first.complete, true);
    assert.equal(same.complete, true);
    assert.equal(same.sha256, first.sha256);
    fs.writeFileSync(header, "omega");
    const changed = windowsCoreIncludeMembership([root]);
    assert.equal(changed.complete, true);
    assert.deepEqual(changed.roots, first.roots);
    assert.notEqual(changed.sha256, first.sha256);
  });

  test("generates the default and opt-in graphs in disposable directories", (t) => {
    const baseline = generate(t, 0), native = generate(t, 1);
    if (!baseline || !native) {return;}
    assert.notEqual(baseline.directory, native.directory);
    assert.ok(fs.statSync(baseline.directory).isDirectory());
    assert.ok(fs.statSync(native.directory).isDirectory());
  });

  test("keeps the Engineering mom graph semantics when Core is enabled", (t) => {
    const baseline = generate(t, 0), native = generate(t, 1);
    if (!baseline || !native) {return;}
    assert.equal(semanticMakefile(native.mom), semanticMakefile(baseline.mom));
  });

  test("keeps the add-on delay-load hook on a relative fixture path", (t) => {
    const baseline = generate(t, 0);
    if (!baseline) {return;}
    const makefile = baseline.mom.replace(/\\/g, "/");
    assert.match(makefile, new RegExp(`${fixtureNodeGypDir}/src/win_delay_load_hook\\.o`));
    assert.doesNotMatch(makefile, /node_modules\/node-gyp\/src\/win_delay_load_hook\.o/);
  });

  test("fails closed by leaving the default Core target without an archive", (t) => {
    const baseline = generate(t, 0), native = generate(t, 1);
    if (!baseline || !native) {return;}
    assert.doesNotMatch(baseline.core, /\$\(obj\)\.target\/mom_native_core\.a/);
    assert.match(native.core, /\$\(obj\)\.target\/mom_native_core\.a/);
    assert.match(native.core, /TARGET := mom_native_core/);
  });

  test("reuses the complete Engineering Windows CPU object graph", (t) => {
    const baseline = generate(t, 0), native = generate(t, 1);
    if (!baseline || !native) {return;}
    const engineeringObjects = objectGraph(baseline.mom), nativeObjects = objectGraph(native.core);
    assert.deepEqual([...nativeObjects].sort(), [...engineeringObjects].sort());
    assert.equal(nativeObjects.size, 70);
    assert.ok([...nativeObjects].some((object) => object.includes("xmrig/crypto/randomx/aes_hash.o")));
    assert.ok([...nativeObjects].some((object) => object.includes("xmrig/3rdparty/argon2/lib/argon2.o")));
  });

  test("removes node-gyp ABI inputs and the Windows delay-load hook from Core", (t) => {
    const native = generate(t, 1);
    if (!native) {return;}
    for (const forbidden of ["win_delay_load_hook", "NODE_GYP_MODULE_NAME", "USING_UV_SHARED", "USING_V8_SHARED",
      "V8_DEPRECATION_WARNINGS", "BUILDING_NODE_EXTENSION", "HOST_BINARY", "node.lib", "/tmp/node/include/node",
      "/tmp/node/src", "/tmp/node/deps/openssl", "/tmp/node/deps/uv/include", "/tmp/node/deps/zlib", "/tmp/node/deps/v8/include"]) {
      assert.doesNotMatch(native.core, new RegExp(forbidden.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), forbidden);
    }
    assert.match(native.core, /-I\/sdk\/libuv\/include/);
  });

  test("keeps the Core archive independent of SYCL while mom retains its dependency", (t) => {
    const baseline = generate(t, 0), native = generate(t, 1);
    if (!baseline || !native) {return;}
    assert.match(baseline.mom, /sycl\.(?:so|dll)/);
    assert.doesNotMatch(native.core, /\$\(builddir\)\/sycl|sycl\.(?:so|dll)/i);
  });

  test("emits MOM_NATIVE_RUNTIME and only the maintained Windows CPU defines", (t) => {
    const native = generate(t, 1);
    if (!native) {return;}
    for (const define of ["MOM_NATIVE_RUNTIME", "NOMINMAX", "WIN32_LEAN_AND_MEAN", "_CRT_SECURE_NO_WARNINGS", "XMRIG_FEATURE_ASM"]) {
      assert.match(native.core, new RegExp(`-D${define.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    }
    assert.doesNotMatch(native.core, /MOM_SYCL_/);
  });

  test("retains every Windows assembly and compiler source in one shared inventory", () => {
    const sourceInventory = binding.slice(binding.indexOf('"mom_core_sources"'), binding.indexOf('"targets"'));
    for (const source of windowsSources) {
      const escaped = source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      assert.equal((sourceInventory.match(new RegExp(`"${escaped}"`, "g")) || []).length, 1, source);
    }
    for (const source of ["native/core.cpp", "xmrig/crypto/randomx/reciprocal.c", "xmrig/crypto/ghostrider/ghostrider.cpp", "xmrig/3rdparty/fmt/format.cc"]) {
      const escaped = source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      assert.equal((sourceInventory.match(new RegExp(`"${escaped}"`, "g")) || []).length, 1, source);
    }
  });

  test("gates the static archive to Windows and an explicit mom_native_core value", () => {
    const target = nativeTarget();
    assert.match(target, /"type": "none"/);
    assert.match(target, /OS=='win' and mom_native_core==1/);
    assert.match(target, /"type": "static_library"/);
    assert.match(target, /"sources": \[ "<@\(mom_core_sources\)", "<@\(mom_core_windows_sources\)" \]/);
    assert.doesNotMatch(target, /"dependencies"/);
  });

  test("pins the Core recipe to MSVC v143, C++20, and static CRT", () => {
    const target = nativeTarget();
    assert.equal((target.match(/"msbuild_toolset": "v143"/g) || []).length, 2);
    assert.equal((target.match(/"RuntimeLibrary": 0/g) || []).length, 2);
    assert.match(target, /"LanguageStandard": "stdcpp20"/);
    assert.match(target, /"AdditionalOptions": \[ "\/O2", "\/fp:strict" \]/);
  });

  test("Intel Core enables the existing AES baseline without MSVC's multiprocess wrapper", t => {
    const baseline = generateFakeMsvsProject(t, true), intel = generateFakeMsvsProject(t, true, true);
    if (!baseline || !intel) {return;}
    for (const flag of ["-maes", "-fgnuc-version=4.2.1", "-Wno-error=incompatible-pointer-types", "-Wno-error=incompatible-function-pointer-types",
      "-Wno-error=int-conversion", "-Wno-error=implicit-int", "-Wno-error=implicit-function-declaration"]) {
      assert.ok(!baseline.project.includes(flag), flag + " must not reach MSVC");
      assert.ok(intel.project.includes(flag), flag + " must reach native Intel");
      assert.ok(intel.addonProject.includes(flag), flag + " must match Engineering Intel");
    }
    assert.match(intel.project, /<MultiProcessorCompilation>false<\/MultiProcessorCompilation>/);
    assert.doesNotMatch(intel.project, /\/arch:AVX|march=native|QxHost/);
  });

  test("clears addon defaults without clearing the maintained project includes", () => {
    const target = nativeTarget();
    for (const property of ["defines!", "include_dirs!", "libraries!", "win_delay_load_hook"]) {
      assert.match(target, new RegExp(`"${property.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    }
    assert.match(target, /"include_dirs": \[ "<@\(mom_core_include_dirs\)", "<\(mom_native_uv_include\)" \]/);
    for (const include of [".", "xmrig", "xmrig/3rdparty/argon2/include", "xmrig/3rdparty/argon2/lib"]) {
      const escaped = include.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      assert.match(binding, new RegExp(`"${escaped}"`));
    }
  });

  test("keeps the N-API bridge source behind MOM_NATIVE_RUNTIME", () => {
    assert.match(coreSource, /#ifndef MOM_NATIVE_RUNTIME[\s\S]*#include ["<]node-worker\.h[">][\s\S]*#endif/);
    assert.match(nativeTarget(), /"MOM_NATIVE_RUNTIME"/);
  });

  test("Ethash's C prefetch passes an address with the intrinsic's declared pointer type", t => {
    const compiler = ["cc", "clang", "gcc"].find(commandPath);
    if (!compiler) {t.skip("C syntax compiler unavailable"); return;}
    const macro = ethashSource.match(/^\s*#define kp_prefetch\(x\)[^\r\n]+/m)?.[0]; assert.ok(macro);
    const fixture = "void _mm_prefetch(const char *, int);\n#define _MM_HINT_T0 0\n" + macro +
      "\nunion node { unsigned words[16]; };\nvoid check(const union node *p) { kp_prefetch(p); }\n";
    const result = spawnSync(compiler, ["-std=c11", "-Wall", "-Wextra", "-Werror", "-x", "c", "-fsyntax-only", "-"],
      {input: fixture, encoding: "utf8", timeout: process.platform === "win32" ? 30000 : 10000, maxBuffer: 1 << 20});
    assert.equal(result.status, 0, result.error?.message || result.stderr);
  });

  test("uses direct GPU declarations only for native Windows, preserving the Engineering DLL ABI", (t) => {
    const compiler = [process.env["CXX"], "clang++", "c++"].find(name => name && commandPath(name));
    if (!compiler) {t.skip("C++ preprocessor unavailable"); return;}
    const visibility = gpuHeader.match(/^#if defined\(_WIN32\) && defined\(MOM_NATIVE_RUNTIME\)[\s\S]*?^#endif/m)?.[0];
    assert.ok(visibility, "Extract the actual public declaration visibility policy");
    /** @type {Array<[string[], string]>} */
    const cases = [
      [[], ""], [["_WIN32"], "__declspec(dllimport)"],
      [["_WIN32", "MOM_SYCL_BUILD"], "__declspec(dllexport)"],
      [["_WIN32", "MOM_NATIVE_RUNTIME"], ""],
      [["_WIN32", "MOM_NATIVE_RUNTIME", "MOM_SYCL_BUILD"], ""],
    ];
    for (const [defines, expected] of cases) {
      /** @type {import("node:child_process").SpawnSyncReturns<string>} */
      const result = spawnSync(compiler, ["-E", "-dM", "-x", "c++", "-U_WIN32", ...defines.map(name => "-D" + name), "-"], {
        input: visibility, encoding: "utf8", timeout: process.platform === "win32" ? 30000 : 10000, maxBuffer: 1 << 20,
      });
      assert.equal(result.status, 0, result.error?.message || result.stderr);
      assert.equal(result.stdout.match(/^#define MOM_SYCL_API([^\n]*)/m)?.[1]?.trim(), expected);
    }
  });

  test("keeps Linux defaults on the Unix CPU condition", {skip: process.platform !== "linux"}, (t) => {
    const linux = generate(t, 0, "linux");
    if (!linux) {return;}
    const objects = objectGraph(linux.mom);
    assert.ok([...objects].some((object) => object.includes("VirtualMemory_unix.o")));
    assert.ok([...objects].some((object) => object.includes("BasicCpuInfo.o")));
    assert.ok([...objects].some((object) => object.includes("jit_compiler_x86_static.o")));
    assert.ok([...objects].every((object) => !object.includes("VirtualMemory_win.o")));
    assert.ok([...objects].every((object) => !object.includes("cn_main_loop.asm.o")));
    assert.match(linux.mom, /sycl\.a/);
  });

  test("emits the Core MSVS project shape with an explicit fake descriptor (configure only)", (t) => {
    const generated = generateFakeMsvsProject(t);
    if (!generated) {return;}
    const project = generated.project;
    assert.match(project, /<ConfigurationType>StaticLibrary<\/ConfigurationType>/);
    assert.equal((project.match(/<PlatformToolset>v143<\/PlatformToolset>/g) || []).length, 1);
    assert.equal((project.match(/<RuntimeLibrary>MultiThreaded<\/RuntimeLibrary>/g) || []).length, 2);
    assert.equal((project.match(/<ExceptionHandling>Sync<\/ExceptionHandling>/g) || []).length, 2);
    for (const source of ["jit_compiler_x86_static.asm", "cn_main_loop.asm", "CryptonightR_template.asm"]) {
      assert.match(project, new RegExp(source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }
    assert.match(project, /MOM_NATIVE_RUNTIME/);
    assert.doesNotMatch(project, /<ProjectReference\b|sycl\.vcxproj|win_delay_load_hook|node\.lib|NODE_GYP_MODULE_NAME|HOST_BINARY/);
  });

  test("removes real Node header defaults that disable exceptions or produce opaque MSVC LTCG objects", t => {
    const generated = generateFakeMsvsProject(t, true);
    if (!generated) {return;}
    assert.doesNotMatch(generated.project, /_HAS_EXCEPTIONS=0|\/LTCG|<WholeProgramOptimization>true<\/WholeProgramOptimization>/i);
    assert.doesNotMatch(generated.project, /node[\\/]deps[\\/]uv[\\/]include/);
    assert.match(generated.project, /sdk[\\/]libuv[\\/]include/);
    assert.match(generated.project, /<ExceptionHandling>Sync<\/ExceptionHandling>/);
    assert.match(generated.project, /<WholeProgramOptimization>false<\/WholeProgramOptimization>/);
    assert.doesNotMatch(generated.project, /-std:c\+\+20/);
    assert.match(nativeTarget(), /"LanguageStandard": "stdcpp20"/);
    assert.match(generated.project, /\/Zc:__cplusplus/);
  });

});

