"use strict";

const fs = require("node:fs"), path = require("node:path"), {createHash} = require("node:crypto"), test = require("node:test"), assert = require("node:assert/strict");
const {dpcppGpuFlags, dpcppUniquePrefixFlags, dpcppSourceFlags, dpcppDeviceLinkFlags, syclSources, portableOpenclTranslatorFlags, combinedGpuTargets} = require("../scripts/native-gpu-build");
const {spawnSync} = require("node:child_process");
const {runInNewContext} = require("node:vm");
const {buildInputs, oneapiCompilerPath, admitOneapiCompiler, verifyOneapiCompiler, sourceCompiler, deviceLinkArgs, cudaDeviceLinkSetup,
  dpcppCompileCommands, deviceRegistrationArgs, deviceRegistrationImports} = require("../scripts/build-native-gpu");
const {fileHash, buildFileInputs} = require("../scripts/build-source-inputs");
const {nativeGpuLinkArgs} = require("../scripts/build-native-application");
const os = require("node:os");
const {tempDir} = require("./common/temp-dir");
const gyp = fs.readFileSync(path.join(__dirname, "../binding.gyp"), "utf8");
const flags = dpcppGpuFlags(gyp), sources = syclSources(gyp);

test.describe("Native portable OpenCL maintained compilation policy", () => {
  const flags = dpcppGpuFlags(gyp, "dpcpp-opencl");
  const wrapper = fs.readFileSync(path.join(__dirname, "../scripts/cxx-combined.sh"), "utf8");
  const translator = portableOpenclTranslatorFlags(wrapper);
  for (const flag of ["-DMOM_SYCL_PORTABLE_OPENCL", "-DMOM_NEXAPOW_PORTABLE_FIELD32", "-fno-sycl-rdc",
    "-fsycl-device-code-split=per_kernel", "-fno-sycl-instrument-device-code", "-ffp-contract=off"]) {
    test(`retains the original ${flag} control`, () => {assert.ok(flags.includes(flag));});
  }
  test("the portable image cannot acquire vendor-only compilation definitions", () => {
    assert.doesNotMatch(flags.join(" "), /ESIMD|HAS_CUDA|HAS_HIP|SYCL_NATIVE_FIELD|nvidia_gpu/);
  });
  test("OpenCL does not depend on a CUDA architecture default", () => {
    assert.deepEqual(dpcppGpuFlags(gyp.replace('"mom_cuda_arch%"', '"unrelated"'), "dpcpp-opencl"), flags);
  });
  test("the actual wrapper supplies the same translator flags, without executing shell", () => {
    assert.deepEqual(translator, ["-Xspirv-translator=spir64", "--spirv-max-version=1.3", "-Xspirv-translator=spir64",
      "--spirv-ext=-SPV_INTEL_memory_access_aliasing,-SPV_INTEL_subgroups,-SPV_KHR_expect_assume,-SPV_KHR_linkonce_odr"]);
  });
  test("intentional maintained translator changes propagate instead of a second fixed flag list", () => {
    assert.ok(portableOpenclTranslatorFlags(wrapper.replace("--spirv-max-version=1.3", "--spirv-max-version=1.4")).includes("--spirv-max-version=1.4"));
  });
  test("a missing or duplicated portable block cannot borrow another compiler's settings", () => {
    assert.throws(() => portableOpenclTranslatorFlags(wrapper.replace(" = dpcpp-opencl ]; then", " = other ]; then")), /translator policy/);
    assert.throws(() => portableOpenclTranslatorFlags(wrapper + "\n" + wrapper), /translator policy/);
  });
  test("both translator restrictions are mandatory, not optional empty array defaults", () => {
    for (const value of ["--spirv-max-version=1.3", translator[3] || ""]) {
      assert.throws(() => portableOpenclTranslatorFlags(wrapper.replace(value, "")), /translator policy/);
    }
  });
  test("duplicate restriction keys cannot masquerade as the complete policy", () => {
    assert.throws(() => portableOpenclTranslatorFlags(wrapper.replace(translator[3] || "", "--spirv-max-version=1.3")), /translator policy/);
  });
  test("unresolved shell expressions and unrelated options are never evaluated or forwarded", () => {
    for (const value of ["$(touch /tmp/forbidden)", "$SDK", "--plugin=unrelated", '"--spirv-max-version=1.3"']) {
      assert.throws(() => portableOpenclTranslatorFlags(wrapper.replace("--spirv-max-version=1.3", value)), /translator policy/);
    }
  });
  test("portable native builds need only the pinned SDK and output, never CUDA headers", () => {
    const env = {MOM_AOT_DPCPP_PROFILE: "dpcpp-opencl"};
    assert.deepEqual(buildInputs(env, [path.resolve("/sdk"), path.resolve("/output")]),
      {sdk: path.resolve("/sdk"), output: path.resolve("/output"), cuda: undefined, nvrtc: undefined, headers: undefined});
  });
  test("final linking preserves no-RDC and translator controls without compile-only IR embedding", () => {
    const args = deviceLinkArgs(gyp, undefined, [path.resolve("/object.o")], path.resolve("/device.o"), "dpcpp-opencl", wrapper);
    for (const flag of ["-fno-sycl-rdc", "-fno-sycl-instrument-device-code", "-fsycl-device-code-split=per_kernel", ...translator]) {assert.ok(args.includes(flag));}
    assert.doesNotMatch(args.join(" "), /embed-ir|(?:^| )-D|cuda|nvidia/);
    assert.throws(() => deviceLinkArgs(gyp, undefined, [path.resolve("/object.o")], path.resolve("/device.o"), "dpcpp-opencl"), /translator policy/);
  });
});

/** A minimal source graph preserves gyp's actual field layout, not executable Python.
 * @param {unknown} [values] @param {string} [architecture] */
function fixture(values = ["-fsycl -DMOM_SYCL_HAS_CUDA -fsycl-targets=<(mom_cuda_arch)"], architecture = "nvidia_gpu_sm_80") {
  return JSON.stringify({variables: {"mom_cuda_arch%": architecture}, targets: [{target_name: "sycl", sources,
    conditions: [["mom_sycl_impl=='dpcpp-cuda'", {"cflags+": values}]]}]});
}

test.describe("Native DPC++ maintained algorithm build profile", () => {
  test("the complete maintained source graph is reused", () => {
    assert.equal(sources.length, 21); assert.ok(sources.includes("sycl/c30/c30.cpp")); assert.ok(sources.includes("sycl/pearlhash/pearlhash.cpp"));
  });
  test("the actual NVIDIA profile retains SYCL device compilation", () => {assert.ok(flags.includes("-fsycl"));});
  test("the actual NVIDIA profile preserves strict floating-point behavior", () => {assert.ok(flags.includes("-ffp-contract=off"));});
  test("embedded device IR needed by source JIT is not dropped", () => {assert.ok(flags.includes("-fsycl-embed-ir"));});
  test("native field and tensor paths are not replaced with compatibility definitions", () => {
    assert.ok(flags.includes("-DMOM_NEXAPOW_SYCL_NATIVE_FIELD")); assert.ok(flags.includes("-DMOM_OCTOPUS_HAS_SYCL_NATIVE"));
    assert.doesNotMatch(flags.join(" "), /ADAPTIVECPP|PORTABLE_OPENCL|HAS_HIP/);
  });
  test("one forward-compatible target is derived, never a duplicated architecture list", () => {
    assert.equal(flags.filter((flag) => flag.startsWith("-fsycl-targets=")).length, 1);
    const architecture = "nvidia_gpu_sm_90";
    assert.deepEqual(dpcppGpuFlags(fixture(undefined, architecture), "dpcpp",
      `export MOM_COMBINED_TARGETS="\${MOM_COMBINED_TARGETS:-spir64,spir64_gen,${architecture}}"`),
    ["-fsycl", "-DMOM_SYCL_HAS_CUDA", "-fsycl-targets=spir64,nvidia_gpu_sm_90"]);
  });
  test("the combined graph retains portable Intel images for the maintained ZHash policy", () => {
    const target = flags.find(flag => flag.startsWith("-fsycl-targets="));
    assert.match(target || "", /^-fsycl-targets=spir64,nvidia_gpu_sm_\d+$/);
    assert.doesNotMatch(target || "", /spir64_gen|pvc|bmg|dg2/);
  });
  test("unrelated earlier gyp targets cannot supply the SYCL compiler flags", () => {
    const decoy = '"target_name":"cpu", "conditions":[["mom_sycl_impl==\'dpcpp-cuda\'",{"cflags+":["-DWRONG_TARGET"]}]],';
    assert.deepEqual(dpcppGpuFlags(fixture().replace('"targets":[{', '"targets":[{' + decoy + '"sources":[]},{')), dpcppGpuFlags(fixture()));
  });
  test("changed maintained optimization options propagate without a second flag registry", () => {
    assert.ok(dpcppGpuFlags(fixture(["-fsycl -DMOM_SYCL_HAS_CUDA -O2 -fsycl-targets=<(mom_cuda_arch)"])).includes("-O2"));
  });
  test("missing or duplicate source targets fail closed", () => {
    assert.throws(() => dpcppGpuFlags(fixture().replace('"sycl"', '"unknown"')), /SYCL source graph/);
    assert.throws(() => dpcppGpuFlags(gyp + gyp), /SYCL source graph/);
  });
  test("missing or duplicated architecture defaults cannot be guessed", () => {
    assert.throws(() => dpcppGpuFlags(fixture().replace("mom_cuda_arch%", "missing")), /compiler profile/);
    assert.throws(() => dpcppGpuFlags(fixture() + '"mom_cuda_arch%":"nvidia_gpu_sm_90"'), /compiler profile/);
  });
  test("unknown or multiple target architectures are rejected", () => {
    for (const architecture of ["auto", "sm_80", "nvidia_gpu_sm_80,nvidia_gpu_sm_90", "nvidia_gpu_sm_80;other"]) {
      assert.throws(() => dpcppGpuFlags(fixture(undefined, architecture)), /compiler profile/);
    }
  });
  test("the selected NVIDIA condition must be present exactly once", () => {
    assert.throws(() => dpcppGpuFlags(fixture().replace("dpcpp-cuda", "other")), /compiler profile/);
    assert.throws(() => dpcppGpuFlags(fixture() + '\n"mom_sycl_impl==\'dpcpp-cuda\'", {"cflags+":["-fsycl -DMOM_SYCL_HAS_CUDA"]}'), /compiler profile/);
  });
  test("a missing NVIDIA field cannot borrow a later condition's valid flags", () => {
    const missing = fixture().replace('"cflags+":', '"unrelated":') + '\n"other", {"cflags+":["-fsycl -DMOM_SYCL_HAS_CUDA"]}';
    assert.throws(() => dpcppGpuFlags(missing), /compiler profile/);
  });
  test("flag arrays are typed and cannot silently become empty options", () => {
    for (const values of [null, "-fsycl", [], [0], [""]]) {assert.throws(() => dpcppGpuFlags(fixture(values)), /compiler (?:flags|profile)/);}
  });
  test("both SYCL compilation and CUDA feature admission are required", () => {
    for (const values of [["-fsycl"], ["-DMOM_SYCL_HAS_CUDA"]]) {assert.throws(() => dpcppGpuFlags(fixture(values)), /Incomplete/);}
  });
  test("unresolved gyp or shell placeholders never reach the compiler", () => {
    for (const extra of ["-I<(unknown)", "-I$(path)", "-I$SDK"]) {
      assert.throws(() => dpcppGpuFlags(fixture(["-fsycl -DMOM_SYCL_HAS_CUDA " + extra])), /unresolved/);
    }
  });
});

test.describe("Native combined GPU target policy", () => {
  const declaration = (/** @type {string} */ targets) => `export MOM_COMBINED_TARGETS="\${MOM_COMBINED_TARGETS:-${targets}}"`;
  test("the maintained default retains Intel JIT and one NVIDIA target without host-specific AOT", () => {
    assert.equal(combinedGpuTargets(fs.readFileSync(path.join(__dirname, "../scripts/combined-build.sh"), "utf8")), "spir64,nvidia_gpu_sm_80");
  });
  for (const targets of ["", "spir64", "nvidia_gpu_sm_80", "spir64,spir64", "spir64,nvidia_gpu_sm_80,nvidia_gpu_sm_90",
    "spir64,nvidia_gpu_sm_80,unknown", "spir64,nvidia_gpu_sm_80,$SDK", "spir64,nvidia_gpu_sm_80\n", "spir64,nvidia_gpu_sm_80\t"]) {
    test(`invalid target policy ${JSON.stringify(targets)} fails closed`, () => {
      assert.throws(() => combinedGpuTargets(declaration(targets)), /combined GPU targets/);
    });
  }
  test("missing and duplicated assignments cannot borrow another build policy", () => {
    const valid = declaration("spir64,nvidia_gpu_sm_80");
    assert.throws(() => combinedGpuTargets(""), /combined GPU targets/);
    assert.throws(() => combinedGpuTargets(valid + "\n" + valid), /combined GPU targets/);
  });
  test("gyp and combined-worker architecture drift is rejected", () => {
    assert.throws(() => dpcppGpuFlags(gyp, "dpcpp", declaration("spir64,nvidia_gpu_sm_90")), /architecture differs/);
  });
  test("compilation and device linking carry the same target exactly once", () => {
    const compile = dpcppGpuFlags(gyp).filter(flag => flag.startsWith("-fsycl-targets="));
    const link = deviceLinkArgs(gyp, "/cuda", ["/object.o"], "/image.o").filter(flag => flag.startsWith("-fsycl-targets="));
    assert.deepEqual(link, compile); assert.equal(link.length, 1);
  });
  test("combined target policy participates in GPU content receipts", () => {
    const {nativeSourceInputs, fileHash} = require("../scripts/build-source-inputs");
    assert.equal(nativeSourceInputs(path.resolve(__dirname, ".."), "gpu").graph["scripts/combined-build.sh"],
      fileHash(path.join(__dirname, "../scripts/combined-build.sh")));
  });
});

test.describe("Native DPC++ source-specific policy", () => {
  const wrapper = fs.readFileSync(path.join(__dirname, "../scripts/cxx-combined.sh"), "utf8");
  test("reuses only the maintained ZHash source flag for the native mixed graph", () => {
    assert.deepEqual(dpcppSourceFlags(wrapper, "sycl/zhash/zhash.cpp"), ["-DMOM_ZHASH_INTEL_LATE_BUCKETS"]);
    assert.deepEqual(dpcppSourceFlags(wrapper, "sycl/c30/c30.cpp"), []);
  });
  test("oneAPI and portable OpenCL profiles do not acquire the mixed-graph source override", () => {
    assert.deepEqual(dpcppSourceFlags(wrapper, "sycl/zhash/zhash.cpp", "oneapi"), []);
    assert.deepEqual(dpcppSourceFlags(wrapper, "sycl/zhash/zhash.cpp", "dpcpp-opencl"), []);
  });
  test("missing or malformed maintained source blocks fail closed", () => {
    for (const source of [
      wrapper.replace(/\n {2}source_args=\(\)\n {2}if \[ -n "\$intel_aot_device" \][\s\S]*?\n {2}fi/, ""),
      wrapper.replace("-DMOM_ZHASH_INTEL_LATE_BUCKETS", "-DMOM_ZHASH_OTHER_LAYOUT"),
      wrapper + wrapper,
      wrapper + wrapper.replace("source_args=(-DMOM_ZHASH_INTEL_LATE_BUCKETS)", "source_args=()"),
    ]) {
      assert.throws(() => dpcppSourceFlags(source, "sycl/zhash/zhash.cpp"), /source flag policy/);
    }
  });
});

test.describe("Deterministic DPC++ SYCL translation-unit identities", () => {
  const wrapper = fs.readFileSync(path.join(__dirname, "../scripts/cxx-combined.sh"), "utf8");
  const files = ["sycl/kawpow/kawpow.cpp", "sycl/c29/c29.cpp", "sycl/c30/c30.cpp"];
  for (const profile of ["oneapi", "dpcpp", "dpcpp-opencl"]) {
    test(`${profile} IDs are stable, unique per TU and independent of checkout/output roots`, () => {
      const prefixes = files.map(file => dpcppUniquePrefixFlags(file, profile));
      const values = prefixes.map(flags => flags[1] || "");
      assert.equal(new Set(values).size, files.length);
      assert.ok(values.every(value => /^-fsycl-unique-prefix=uid[0-9a-f]{16}$/.test(value)));
      for (const [index, file] of files.entries()) {
        const expected = "uid" + createHash("sha256").update("mom-sycl-unique-prefix-v1\n" + profile + "\n" + file + "\n").digest("hex").slice(0, 16);
        assert.equal(values[index], "-fsycl-unique-prefix=" + expected);
      }
      assert.deepEqual(dpcppUniquePrefixFlags(files[0] || "", profile), prefixes[0]);
      assert.notDeepEqual(dpcppUniquePrefixFlags(files[0] || "", profile === "oneapi" ? "dpcpp" : "oneapi"), prefixes[0]);
      assert.deepEqual(dpcppUniquePrefixFlags("sycl/blake2/blake2b.c", profile), []);
      assert.deepEqual(dpcppUniquePrefixFlags("/checkout/sycl/c29/c29.cpp", profile), []);
      assert.deepEqual(dpcppUniquePrefixFlags(files[0] || "", "acpp-cuda"), []);
      assert.deepEqual(dpcppUniquePrefixFlags(files[0] || "", "acpp-hip"), []);

      /** @param {string} repo @param {string} output */
      const plan = (repo, output) => dpcppCompileCommands({sources: [files[0] || "", "sycl/blake2/blake2b.c"], repo, output,
        sdk: "/sdk", compiler: "/sdk/bin/clang++", flags: ["-fsycl"], wrapper, profile});
      const first = plan("/checkout/first", "/build/first"), moved = plan("/checkout/moved", "/build/nested/moved");
      assert.deepEqual(first[0]?.sourceFlags, moved[0]?.sourceFlags);
      const prefixOption = "-Xclang", prefixArgs = [prefixOption, values[0] || ""];
      const command = first[0]?.args || [], prefixIndex = command.indexOf(prefixOption);
      assert.ok(command.includes("-fsycl"));
      assert.deepEqual(command.slice(prefixIndex, prefixIndex + prefixArgs.length), prefixArgs,
        "the compile driver receives one adjacent cc1-forwarding pair for host and device");
      assert.equal(prefixIndex + prefixArgs.length, command.indexOf("-MD"), "the explicit prefix is the last compile flag and wins the cc1 ordering");
      assert.ok(!(first[1]?.args || []).some(flag => flag.startsWith("-fsycl-unique-prefix=")), "plain C remains unchanged");
    });
  }
  test("retains the exact oneAPI namespace/hash used by the Windows Intel path", () => {
    assert.deepEqual(dpcppUniquePrefixFlags("sycl/kawpow/kawpow.cpp", "oneapi"),
      ["-Xclang", "-fsycl-unique-prefix=uide1aae4a6a5850362"]);
  });
});

test.describe("Static DPC++ image registration lifetime", {skip: process.platform !== "linux"}, () => {
  const sections = "  [10] .init_array.1 INIT_ARRAY 00000000\n  [12] .fini_array.1 FINI_ARRAY 00000000\n";
  test("only the pinned wrapper's registration section names are retimed", () => {
    assert.deepEqual(deviceRegistrationArgs(sections).slice(0, 4), ["--rename-section", ".init_array.1=.init_array", "--rename-section", ".fini_array.1=.fini_array"]);
  });
  for (const profile of ["oneapi", "dpcpp", "dpcpp-opencl"]) {
    test(`${profile} binds static registration to its own deferred image gates`, () => {
      const key = profile.replaceAll("-", "_");
      assert.deepEqual(deviceRegistrationArgs(sections, profile).slice(4), ["--redefine-sym", "__sycl_register_lib=mom_native_register_" + key,
        "--redefine-sym", "__sycl_unregister_lib=mom_native_unregister_" + key]);
    });
  }
  test("unknown profile names cannot select or inject an image callback", () => {
    for (const profile of ["", "acpp-hip", "oneapi;other", "oneapi\nother"]) {assert.throws(() => deviceRegistrationArgs(sections, profile), /registration profile/);}
  });
  test("only the two original C imports admit the pinned merged image wrapper", () => {
    const symbols = "__sycl_register_lib U\n__sycl_unregister_lib U\nmemcpy U\n";
    assert.doesNotThrow(() => deviceRegistrationImports(symbols));
    for (const value of ["", symbols + "__sycl_register_lib U\n", symbols.replace("__sycl_register_lib U", "__sycl_register_lib T 0"),
      symbols.replace("__sycl_unregister_lib", "__sycl_unregister_lib_other")]) {assert.throws(() => deviceRegistrationImports(value), /registration imports/);}
  });
  test("each remapped graph must import exactly its own two deferred callbacks", () => {
    for (const profile of ["oneapi", "dpcpp", "dpcpp-opencl"]) {
      const key = profile.replaceAll("-", "_");
      const symbols = `mom_native_register_${key} U\nmom_native_unregister_${key} U\n`;
      assert.doesNotThrow(() => deviceRegistrationImports(symbols, profile));
      for (const value of ["", symbols + symbols, symbols.replace("unregister", "register"),
        "__sycl_register_lib U\n__sycl_unregister_lib U\n"]) {
        assert.throws(() => deviceRegistrationImports(value, profile), /registration imports/);
      }
    }
    assert.throws(() => deviceRegistrationImports("", "acpp-hip"), /registration profile/);
  });
  test("real object remapping binds each profile's callbacks and leaves unrelated imports intact", (t) => {
    const directory = tempDir(t, "mom-static-remap-");
    const source = path.join(directory, "image.cpp"), object = path.join(directory, "image.o");
    fs.writeFileSync(source, 'extern "C" void __sycl_register_lib(void*);\nextern "C" void __sycl_unregister_lib(void*);\n' +
      'extern "C" void other_callback(void*);\n' +
      'extern "C" void callbacks(void* p){__sycl_register_lib(p);__sycl_unregister_lib(p);other_callback(p);}\n' +
      '__attribute__((section(".init_array.1"),used)) static void(*init)(void*)=__sycl_register_lib;\n' +
      '__attribute__((section(".fini_array.1"),used)) static void(*fini)(void*)=__sycl_unregister_lib;\n');
    /** @param {string} command @param {string[]} args */
    const run = (command, args) => {
      const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
    };
    run(process.env["CXX"] || "c++", ["-O2", "-fPIC", "-c", source, "-o", object]);
    deviceRegistrationImports(run("nm", ["-P", "-u", object]));
    for (const profile of ["oneapi", "dpcpp", "dpcpp-opencl"]) {
      const output = path.join(directory, profile + ".o");
      run("objcopy", [...deviceRegistrationArgs(run("readelf", ["--sections", "--wide", object]), profile), object, output]);
      const symbols = run("nm", ["-P", "-u", output]);
      deviceRegistrationImports(symbols, profile);
      assert.doesNotMatch(symbols, /^__sycl_(?:un)?register_lib\s/m);
      assert.match(symbols, /^other_callback U(?:\s|$)/m);
    }
  });
  for (const name of ["init", "fini"]) {
    test(`missing ${name} lifetime metadata fails closed`, () => {
      assert.throws(() => deviceRegistrationArgs(sections.split("\n").filter((line) => !line.includes("." + name + "_array")).join("\n")), /registration lifetime/);
    });
    test(`duplicated ${name} lifetime metadata is not silently merged`, () => {
      assert.throws(() => deviceRegistrationArgs(sections + sections.split("\n").find((line) => line.includes("." + name + "_array"))), /registration lifetime/);
    });
  }
  test("a different compiler priority needs requalification", () => {
    assert.throws(() => deviceRegistrationArgs(sections.replaceAll("_array.1 ", "_array.2 ")), /registration lifetime/);
  });
  test("arbitrary sections cannot pretend to be constructor/destructor arrays", () => {
    assert.throws(() => deviceRegistrationArgs(sections.replaceAll("INIT_ARRAY", "PROGBITS")), /registration lifetime/);
  });
  test("retiming does not remove sections, edit code or disable relocation protection", () => {
    assert.doesNotMatch(deviceRegistrationArgs(sections).join(" "), /remove|strip|update|-no-pie|set-section/);
  });
  test("full Level Zero context constructors survive archive search without forcing unrelated SDKs", () => {
    for (const key of ["libraries", "links"]) {
      const args = nativeGpuLinkArgs({flags: [], libraries: [], links: [], objects: [], [key]: ["/sdk/libze_loader.a", "/sdk/libur_loader.a"]});
      assert.deepEqual(args, ["-Wl,--start-group", "-Wl,--push-state,--whole-archive", "/sdk/libze_loader.a",
        "-Wl,--pop-state", "/sdk/libur_loader.a", "-Wl,--end-group"]);
    }
  });
  test("real static startup/shutdown fails before, and passes after, lifetime normalization", (t) => {
    const directory = tempDir(t, "mom-static-registration-");
    fs.writeFileSync(path.join(directory, "runtime.cpp"), "#include <cstdlib>\nint* state=new int(7);\n" +
      'extern "C" void register_image(){if(!state)std::_Exit(86);++*state;}\n' +
      'extern "C" void unregister_image(){if(!state)std::_Exit(87);--*state;}\n' +
      "__attribute__((destructor(110))) static void shutdown(){if(!state||*state!=7)std::_Exit(88);delete state;state=nullptr;}\n" +
      "int main(){return state&&*state==8?0:89;}\n");
    fs.writeFileSync(path.join(directory, "image.cpp"), 'extern "C" void register_image();extern "C" void unregister_image();\n' +
      '__attribute__((section(".init_array.1"),used)) static void(*image_init)()=register_image;\n' +
      '__attribute__((section(".fini_array.1"),used)) static void(*image_fini)()=unregister_image;\n');
    /** @param {string} command @param {string[]} args */
    function run(command, args) {
      const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000});
      assert.equal(result.status, 0, result.error?.message || result.stderr); return result.stdout;
    }
    const cxx = process.env["CXX"] || "c++", runtime = path.join(directory, "runtime.o"), image = path.join(directory, "image.o");
    for (const [source, object] of [["runtime.cpp", runtime], ["image.cpp", image]]) {
      run(cxx, ["-std=c++17", "-O2", "-fPIC", "-c", path.join(directory, source || ""), "-o", object || ""]);
    }
    const before = path.join(directory, "before"), after = path.join(directory, "after"), fixed = path.join(directory, "fixed.o");
    run(cxx, ["-pie", runtime, image, "-o", before]);
    const baseline = spawnSync(before, [], {encoding: "utf8", timeout: 5000}); assert.equal(baseline.status, 86, baseline.stderr);
    run("objcopy", [...deviceRegistrationArgs(run("readelf", ["--sections", "--wide", image])), image, fixed]);
    run(cxx, ["-pie", runtime, fixed, "-o", after]);run(after, []);
    // Even normalized images fail if a static runtime's constructors follow registration.
    // Exercise the shared application's real link recipe, not just a source-text assertion.
    const archive = path.join(directory, "runtime.a"), reversed = path.join(directory, "reversed"), linked = path.join(directory, "linked");
    run("ar", ["crs", archive, runtime]);
    const inputs = {flags: [], libraries: [], links: ["-Wl,--whole-archive", archive, "-Wl,--no-whole-archive"], objects: [{object: fixed}]};
    run(cxx, ["-pie", fixed, ...inputs.links, "-o", reversed]);
    assert.equal(spawnSync(reversed, [], {encoding: "utf8", timeout: 5000}).status, 86);
    run(cxx, ["-pie", ...nativeGpuLinkArgs(inputs), "-o", linked]); run(linked, []);
  });
  test("the maintained builder normalizes registration before isolating the whole graph", () => {
    const source = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    assert.match(source, /deviceRegistrationArgs\(sections\.stdout, profile\)/);
    assert.match(source, /\.\.\.hostObjects, registeredDevice/);
    assert.doesNotMatch(source, /extraHostObjects|cudaImageCompileCommand|imageCommand/);
  });
});

test.describe("Native DPC++ selected CUDA device tools", {skip: process.platform !== "linux"}, () => {
  test("device linking finds hashed selected-SDK ptxas before an ambient tool", t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-device-tools-")), cuda = path.join(root, "cuda"),
      selectedBin = path.join(cuda, "bin"), ambient = path.join(root, "ambient");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.mkdirSync(selectedBin, {recursive: true}); fs.mkdirSync(ambient);
    for (const name of ["ptxas", "fatbinary"]) {fs.writeFileSync(path.join(selectedBin, name), "#!/bin/sh\nexit 0\n", {mode: 0o755});}
    fs.writeFileSync(path.join(ambient, "ptxas"), "#!/bin/sh\nexit 77\n", {mode: 0o755});
    const setup = cudaDeviceLinkSetup(cuda, {PATH: ambient});
    const ptxas = path.join(selectedBin, "ptxas");
    assert.deepEqual(setup.files, [ptxas, path.join(selectedBin, "fatbinary")]);
    assert.equal(setup.inputs[ptxas], fileHash(ptxas));
    assert.equal(buildFileInputs(setup.files)[ptxas], setup.inputs[ptxas]);
    assert.equal(setup.environment["PATH"]?.split(path.delimiter)[0], selectedBin);
    const resolved = spawnSync("/bin/sh", ["-c", "command -v ptxas"], {encoding: "utf8", env: setup.environment});
    assert.equal(resolved.status, 0, resolved.stderr); assert.equal(resolved.stdout.trim(), ptxas);
    fs.unlinkSync(ptxas);
    assert.throws(() => cudaDeviceLinkSetup(cuda, {PATH: ambient}), /executable ptxas/);
    const source = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    assert.match(source, /\.\.\.\(cudaLink\?\.files \|\| \[\]\)/);
    assert.match(source, /"device-link", 900000,\s*cudaLink\?\.environment/);
  });
});

test.describe("Native DPC++ source build boundary", () => {
  const args = ["sdk", "cuda", "output"].map((name) => path.resolve("/build", name));
  const env = {MOM_AOT_NVRTC_SDK: path.resolve("/build/nvrtc"), MOM_AOT_NVRTC_HEADERS: path.resolve("/build/headers.inc")};
  test("all SDKs, immutable headers and output are explicit absolute inputs", () => {
    assert.deepEqual(buildInputs(env, args), {sdk: args[0], cuda: args[1], output: args[2], nvrtc: env.MOM_AOT_NVRTC_SDK, headers: env.MOM_AOT_NVRTC_HEADERS});
  });
  for (const field of ["MOM_AOT_NVRTC_SDK", "MOM_AOT_NVRTC_HEADERS"]) {
    test(`missing ${field} cannot silently use a system installation`, () => {assert.throws(() => buildInputs({...env, [field]: undefined}, args), /absolute/);});
  }
  test("every positional input rejects a relative path", () => {
    for (let index = 0; index < args.length; ++index) {
      const values = [...args]; values[index] = "relative"; assert.throws(() => buildInputs(env, values), /absolute/);
    }
  });
  test("compiler/header inputs cannot contain control characters", () => {
    for (const value of [path.resolve("/build/bad") + "\n", path.resolve("/build/bad") + "\0"]) {
      assert.throws(() => buildInputs({...env, MOM_AOT_NVRTC_HEADERS: value}, args), /absolute/);
    }
  });
  test("missing and surplus arguments are rejected before output creation", () => {
    for (const values of [args.slice(1), [...args, path.resolve("/extra")]]) {assert.throws(() => buildInputs(env, values), /absolute/);}
  });
  test("build-only admission precedes all SDK, compiler and output access", () => {
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/build-native-gpu.js")], {
      encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"},
    });
    assert.equal(result.status, 1); assert.match(result.stderr, /MOM_AOT_BUILD_ONLY=1/); assert.doesNotMatch(result.stderr, /ENOENT|spawn/);
  });
  test("the complete original kernel graph keeps uniform static compiler layout", () => {
    const code = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    assert.match(code, /engineeringFlags = dpcppGpuFlags\(gyp, profile\)/);
    assert.match(code, /"-DMOM_STATIC_NVRTC", "-DMOM_EMBEDDED_NVRTC_HEADERS"/);
    assert.match(code, /"-DMOM_EMBEDDED_SYCL_JIT"/);
    assert.match(code, /"-MD", "-MF"/);
  });
  test("build tools never become normal Engineering or release commands implicitly", () => {
    for (const file of ["scripts/dev.js", "scripts/build-release.js"]) {
      assert.doesNotMatch(fs.readFileSync(path.join(__dirname, "..", file), "utf8"), /build-native-gpu/);
    }
  });
  test("ordinary compile failures retain all algorithm results; uncertain lifecycle stops", () => {
    const code = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    assert.match(code, /results\.some\(\(result\) => result\.status !== 0\)/);
    assert.match(code, /result\.error \|\| result\.signal \|\| result\.status === null/);
    assert.doesNotMatch(code, /nvidia-smi|\/dev\/nvidia|spawnSync\(.*controller-native/);
  });
});

test.describe("Native oneAPI mixed frontend admission", () => {
  const upstream = path.resolve("/build/dpcpp/bin/clang++"), vendor = path.resolve("/vendor/oneapi/bin/icpx");
  const mixedSources = ["sycl/beamhash3/beamhash3.cpp", "sycl/c29/c29.cpp", "sycl/c30/c30.cpp", "sycl/etchash/etchash.cpp",
    "sycl/kawpow/kawpow.cpp", "sycl/nexapow/nexapow.cpp", "sycl/nexapow/validation.cpp", "sycl/octopus/octopus.cpp", "sycl/pearlhash/pearlhash.cpp",
    "sycl/xelishashv3/xelishashv3.cpp", "sycl/zelhash/zelhash.cpp"];
  test("selects the vendor frontend for exactly the qualified TUs including A770 correctness fixes", () => {
    const allSources = [...sources, "sycl/hoohash/host-math.c"];
    const selected = allSources.filter(file => sourceCompiler(upstream, file, "oneapi", vendor) !== upstream).sort();
    assert.deepEqual(selected, [...mixedSources].sort());
    for (const file of allSources) {assert.equal(sourceCompiler(upstream, file, "oneapi", undefined), upstream);}
  });
  test("the mixed frontend does not change selected source objects or device-link inputs", () => {
    const objects = sources.map((_, index) => path.resolve(`/build/out/${index}.o`)), output = path.resolve("/build/out/device.o");
    const link = deviceLinkArgs(gyp, undefined, objects, output, "oneapi");
    assert.deepEqual(link.slice(-objects.length - 2), [...objects, "-o", output]);
    assert.ok(!link.includes(vendor));
    const code = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    assert.match(code, /run\(compiler, \[\.\.\.systemFlags, \.\.\.deviceLinkArgs\(/);
  });
  test("other compiler profiles remain upstream and reject the oneAPI override", () => {
    for (const profile of ["dpcpp", "dpcpp-opencl"]) {
      for (const file of mixedSources) {assert.equal(sourceCompiler(upstream, file, profile, vendor), upstream);}
      const env = {MOM_AOT_DPCPP_PROFILE: profile, MOM_AOT_ONEAPI_CXX: vendor};
      assert.throws(() => oneapiCompilerPath(env, profile), /only valid for the oneapi profile/);
      assert.throws(() => buildInputs(env, [path.resolve("/build/sdk"), path.resolve("/build/out")]), /only valid for the oneapi profile/);
    }
  });
  test("invalid vendor admission fails before tool launch or output creation", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-admission-")), output = path.join(root, "output");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    const result = spawnSync(process.execPath, [path.join(__dirname, "../scripts/build-native-gpu.js"), path.join(root, "sdk"), output], {
      encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_BUILD_ONLY: "1", MOM_AOT_DPCPP_PROFILE: "oneapi",
        MOM_AOT_ONEAPI_CXX: path.join(root, "missing", "icpx"), PATH: "/unavailable"},
    });
    assert.equal(result.status, 1); assert.match(result.stderr, /oneAPI compiler must be a regular executable/);
    assert.doesNotMatch(result.stderr, /Cannot query selected DPC\+\+ compiler/); assert.equal(fs.existsSync(output), false);
  });
  test("the admitted Intel frontend version and executable participate in provenance hashes", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-receipt-")), compiler = path.join(root, "icpx");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(compiler, "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf 'Intel vendor icpx test 2026.0\\n'; else exit 7; fi\n", {mode: 0o755});
    const env = {MOM_AOT_DPCPP_PROFILE: "oneapi", MOM_AOT_ONEAPI_CXX: compiler};
    assert.equal(oneapiCompilerPath(env, "oneapi"), compiler);
    const admitted = admitOneapiCompiler(env, "oneapi");
    assert.ok(admitted); assert.equal(admitted.compiler, compiler);
    assert.equal(admitted.versionSha256, createHash("sha256").update(admitted.version).digest("hex"));
    const sdkBefore = buildFileInputs([compiler]); assert.equal(sdkBefore[compiler], fileHash(compiler));
    fs.appendFileSync(compiler, "# changed input\n"); assert.notEqual(fileHash(compiler), sdkBefore[compiler]);
  });
  test("a changed reported compiler version fails final verification", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-version-")), compiler = path.join(root, "icpx"), version = path.join(root, "version");
    t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(version, "Intel oneAPI DPC++/C++ Compiler 2026.2.0\\n");
    fs.writeFileSync(compiler, `#!/bin/sh\nif [ "$1" = "--version" ]; then cat '${version}'; else exit 7; fi\n`, {mode: 0o755});
    const admitted = admitOneapiCompiler({MOM_AOT_ONEAPI_CXX: compiler}, "oneapi");
    assert.doesNotThrow(() => verifyOneapiCompiler(admitted));
    fs.writeFileSync(version, "Intel oneAPI DPC++/C++ Compiler 2026.3.0\\n");
    assert.throws(() => verifyOneapiCompiler(admitted), /version changed during compilation/);
  });
  test("path admission rejects relative and control-character paths", () => {
    for (const value of ["relative", path.resolve("/build/bad") + "\n", path.resolve("/build/bad") + "\0"]) {
      assert.throws(() => oneapiCompilerPath({MOM_AOT_DPCPP_PROFILE: "oneapi", MOM_AOT_ONEAPI_CXX: value}, "oneapi"), /absolute executable/);
    }
  });
  test("non-Intel output, non-executable and directory paths fail closed", {skip: process.platform !== "linux"}, t => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-oneapi-invalid-")), file = path.join(root, "icpx"), nonIntel = path.join(root, "not-intel");
    const directory = path.join(root, "directory"); t.after(() => fs.rmSync(root, {recursive: true, force: true}));
    fs.writeFileSync(file, "not executable\n", {mode: 0o644}); fs.mkdirSync(directory);
    fs.writeFileSync(nonIntel, "#!/bin/sh\nprintf 'clang version 99\\n'\n", {mode: 0o755});
    assert.throws(() => admitOneapiCompiler({MOM_AOT_ONEAPI_CXX: file}, "oneapi"), /regular executable/);
    assert.throws(() => admitOneapiCompiler({MOM_AOT_ONEAPI_CXX: directory}, "oneapi"), /regular executable/);
    assert.throws(() => admitOneapiCompiler({MOM_AOT_ONEAPI_CXX: nonIntel}, "oneapi"), /did not identify as Intel/);
  });
});

test.describe("Native Intel maintained fast-path graph", () => {
  const intel = dpcppGpuFlags(gyp, "oneapi");
  const env = {MOM_AOT_DPCPP_PROFILE: "oneapi"};
  const args = [path.resolve("/build/sdk"), path.resolve("/build/intel")];
  for (const flag of ["-DMOM_SYCL_INTEL_LAYOUT", "-DPEARLHASH_ESIMD", "-DWALAHASH_ESIMD", "-DOCTOPUS_ESIMD", "-DMOM_C29_INTEL",
    "-DMOM_BEAMHASH3_INTEL_PAIR32", "-DMOM_ZHASH_INTEL_LATE_BUCKETS"]) {
    test(`Intel retains the maintained ${flag} fast path`, () => {assert.ok(intel.includes(flag));});
  }
  test("Intel never inherits NVIDIA/HIP or generic compatibility image definitions", () => {
    assert.ok(intel.includes("-fsycl"));
    assert.doesNotMatch(intel.join(" "), /HAS_CUDA|HAS_HIP|PORTABLE_OPENCL|nvidia_gpu/);
  });
  test("Intel preserves size_t IDs for Pearl matrices larger than INT_MAX", () => {
    assert.ok(intel.includes("-fno-sycl-id-queries-fit-in-int"));
    assert.equal(32768 * 65536, 2 ** 31); // An admitted custom Pearl shape, not a launch reduced to fit int.
  });
  test("Windows Intel Engineering builds retain the same full-width ID contract", () => {
    assert.match(gyp, /"\/fsycl",\s*"\/clang:-fno-sycl-id-queries-fit-in-int"/);
    assert.match(gyp, /"\/DMOM_SYCL_INTEL_LAYOUT"/);
  });
  test("device linking retains explicit large-range metadata from the Intel compilation policy", () => {
    const range = "-fno-sycl-id-queries-fit-in-int";
    // Supply the option independently, so a missing compile flag cannot hide a second link-filter defect.
    const explicit = gyp.replaceAll(range, "").replace("-DPEARLHASH_ESIMD", range + " -DPEARLHASH_ESIMD");
    const linked = deviceLinkArgs(explicit, undefined, [path.resolve("/build/1.o")], path.resolve("/build/device.o"), "oneapi");
    assert.equal(linked.filter(flag => flag === range).length, 1);
  });
  test("Intel's range and arena policy does not change CUDA or portable OpenCL compilation", () => {
    for (const profile of ["dpcpp", "dpcpp-opencl"]) {
      assert.ok(!dpcppGpuFlags(gyp, profile).includes("-fno-sycl-id-queries-fit-in-int"));
      assert.ok(!dpcppGpuFlags(gyp, profile).includes("-DMOM_SYCL_INTEL_LAYOUT"));
    }
  });
  test("Intel's flags do not depend on an unrelated NVIDIA architecture default", () => {
    assert.deepEqual(dpcppGpuFlags(gyp.replace('"mom_cuda_arch%"', '"unrelated"'), "oneapi"), intel);
  });
  test("Intel cannot fall back to a combined or NVIDIA profile when its own condition disappears", () => {
    assert.throws(() => dpcppGpuFlags(gyp.replace('"mom_sycl_impl==\'dpcpp\'", {', '"unknown", {'), "oneapi"), /compiler profile/);
  });
  test("ambiguous Intel profiles fail closed instead of picking the first", () => {
    assert.throws(() => dpcppGpuFlags(gyp + '\n"mom_sycl_impl==\'dpcpp\'", {"cflags+":["-fsycl"]}', "oneapi"), /compiler profile/);
  });
  test("a missing Intel field cannot borrow a later condition's valid flags", () => {
    const missing = JSON.stringify({targets: [{target_name: "sycl", sources, conditions: [
      ["mom_sycl_impl=='dpcpp'", {unrelated: ["-fsycl"]}], ["other", {"cflags+": ["-fsycl"]}],
    ]}]});
    assert.throws(() => dpcppGpuFlags(missing, "oneapi"), /compiler profile/);
  });
  test("unknown compiler keys cannot select arbitrary gyp conditions", () => {
    for (const key of ["opencl", "dpcpp-combined", "../oneapi", "oneapi'"]) {
      assert.throws(() => dpcppGpuFlags(gyp, key), /compiler profile/);
      assert.throws(() => buildInputs({...env, MOM_AOT_DPCPP_PROFILE: key}, args), /explicit/);
    }
  });
  test("Intel needs only its pinned SDK and a new output root, never a CUDA toolkit", () => {
    assert.deepEqual(buildInputs(env, args), {sdk: args[0], output: args[1], cuda: undefined, nvrtc: undefined, headers: undefined});
    assert.deepEqual(buildInputs({...env, MOM_AOT_NVRTC_SDK: "unrelated", MOM_AOT_NVRTC_HEADERS: "unrelated"}, args), buildInputs(env, args));
  });
  test("both Intel positional paths remain explicit, absolute and free of controls", () => {
    for (let index = 0; index < args.length; ++index) {
      for (const value of ["relative", args[index] + "\n"]) {
        const invalid = [...args];invalid[index] = value;
        assert.throws(() => buildInputs(env, invalid), /absolute/);
      }
    }
    assert.throws(() => buildInputs(env, [...args, path.resolve("/extra")]), /absolute/);
  });
  test("Intel's device link keeps SPIR-V, PIC, per-kernel split and no application SYCL library", () => {
    const linked = deviceLinkArgs(gyp, undefined, [path.resolve("/build/1.o")], path.resolve("/build/device.o"), "oneapi");
    for (const flag of ["-fsycl-targets=spir64", "-fPIC", "-fsycl-device-code-split=per_kernel", "-nolibsycl"]) {assert.ok(linked.includes(flag));}
    assert.doesNotMatch(linked.join(" "), /cuda|nvidia|embed-ir|(?:^| )-D/);
    assert.throws(() => deviceLinkArgs(gyp, path.resolve("/unrelated/cuda"), [path.resolve("/build/1.o")], path.resolve("/out.o"), "oneapi"), /unique absolute/);
  });
});

test.describe("Native DPC++ device-image link policy", () => {
  /** @param {unknown} [values] */
  function profile(values = ["-fsycl-device-code-split=per_kernel"]) {
    return JSON.stringify({conditions: [["mom_sycl_impl=='dpcpp' or mom_sycl_impl=='dpcpp-combined'", {"ldflags+": values}]]});
  }
  test("the real release worker's final device split is retained", () => {
    assert.deepEqual(dpcppDeviceLinkFlags(gyp), ["-fsycl-device-code-split=per_kernel"]);
  });
  test("compile flags alone cannot silently become a complete device-link recipe", () => {
    assert.throws(() => dpcppDeviceLinkFlags(fixture()), /device-link profile/);
  });
  test("the maintained link condition is required", () => {assert.throws(() => dpcppDeviceLinkFlags(""), /device-link profile/);});
  test("duplicate link profiles are ambiguous, not a fallback choice", () => {
    assert.throws(() => dpcppDeviceLinkFlags(profile() + profile()), /device-link profile/);
  });
  test("unknown device grouping requires explicit requalification", () => {
    for (const mode of ["auto", "per_source", "off"]) {
      assert.throws(() => dpcppDeviceLinkFlags(profile(["-fsycl-device-code-split=" + mode])), /device-link profile/);
    }
  });
  test("an absent split cannot merge all algorithms into one code-generation unit", () => {
    assert.throws(() => dpcppDeviceLinkFlags(profile([])), /device-link profile/);
  });
  test("link options must remain a typed array", () => {
    for (const values of [null, 1, "-fsycl-device-code-split=per_kernel", [false]]) {
      assert.throws(() => dpcppDeviceLinkFlags(profile(values)), /device-link profile/);
    }
  });
  test("unrelated earlier link conditions cannot override the selected worker", () => {
    assert.deepEqual(dpcppDeviceLinkFlags('"other", {"ldflags+":["-DWRONG"]}\n' + profile()), dpcppDeviceLinkFlags(profile()));
  });
  test("extra options never implicitly change kernel optimization or dependencies", () => {
    assert.throws(() => dpcppDeviceLinkFlags(profile(["-fsycl-device-code-split=per_kernel", "-ffast-math"])), /device-link profile/);
  });
  test("compile-only IR embedding never reaches the device wrapper", () => {
    const args = deviceLinkArgs(gyp, path.resolve("/cuda"), [path.resolve("/build/1.o")], path.resolve("/build/device.o"));
    assert.ok(args.includes("-nolibsycl")); assert.ok(args.includes("-fsycl-link"));
    assert.ok(args.includes("-fPIC")); // The generated wrapper must link into a hardened PIE too.
    assert.ok(args.includes("-fsycl-device-code-split=per_kernel")); assert.ok(args.includes("-ffp-contract=off"));
    assert.ok(args.includes("-O3")); assert.doesNotMatch(args.join(" "), /embed-ir|(?:^| )-D/);
  });
  test("device objects cannot be duplicated or reinterpreted as relative compiler options", () => {
    const object = path.resolve("/build/1.o");
    assert.throws(() => deviceLinkArgs(gyp, path.resolve("/cuda"), [object, object], path.resolve("/out.o")), /unique absolute/);
    assert.throws(() => deviceLinkArgs(gyp, path.resolve("/cuda"), ["-plugin=evil"], path.resolve("/out.o")), /unique absolute/);
  });
  test("importing the build tool cannot query any device or SDK", () => {
    const result = spawnSync(process.execPath, ["-e", `require(${JSON.stringify(path.join(__dirname, "../scripts/build-native-gpu.js"))})`], {
      encoding: "utf8", timeout: 5000, env: {...process.env, MOM_AOT_BUILD_ONLY: "0", PATH: "/unavailable"},
    });
    assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, ""); assert.equal(result.stderr, "");
  });
  test("per-source progress evidence reuses the frozen SDK hashes instead of rehashing every input", () => {
    const code = fs.readFileSync(path.join(__dirname, "../scripts/build-native-gpu.js"), "utf8");
    const expression = code.match(/sdkFiles: (Object\.fromEntries\(explicitSdkFiles\.map\([^\n]+\)\)),\n/);
    assert.ok(expression?.[1]);
    const sdkBefore = {[path.resolve("/sdk/header.h")]: "a".repeat(64), [path.resolve("/sdk/ptxas")]: "b".repeat(64)};
    const evidence = runInNewContext(expression[1], {sdkBefore, explicitSdkFiles: Object.keys(sdkBefore),
      hash: () => {throw new Error("redundant SDK read");}});
    assert.deepEqual(Object.entries(evidence), Object.entries(sdkBefore));
    assert.match(code, /completeBuildInputs\([^\n]+startedAt, sdkBefore\)/, "final post-link revalidation must remain");
  });
});
