"use strict";

const fs = require("node:fs"), os = require("node:os"), path = require("node:path"), {spawnSync} = require("node:child_process");
const test = require("node:test"), assert = require("node:assert/strict");
const {fileHash} = require("../scripts/build-source-inputs");
const {isolateCoffObjects, rewriteCoffDirectives, coffSectionPayloads} = require("../scripts/native-compiler-isolation");

/** @param {string} file @returns {boolean} */
function regularFile(file) {return Boolean(file && fs.statSync(file, {throwIfNoEntry: false})?.isFile());}

/** @param {string} name @param {string | undefined} bin @returns {string | undefined} */
function discover(name, bin) {
  const suffix = process.platform === "win32" ? ".exe" : "", candidates = [];
  if (bin) {candidates.push(path.join(path.resolve(bin), name + suffix), path.join(path.resolve(bin), name));}
  const lookup = spawnSync(process.platform === "win32" ? "where.exe" : "which", [name + suffix, name], {encoding: "utf8", timeout: 5000});
  if (lookup.status === 0 && typeof lookup.stdout === "string") {candidates.push(...lookup.stdout.split(/\r?\n/).map(item => item.trim()).filter(Boolean));}
  return candidates.find(file => path.isAbsolute(file) && regularFile(file));
}

const llvmBin = process.env["MOM_TEST_LLVM_BIN"], toolchain = (() => {
  const clang = discover("clang-cl", llvmBin) || discover("clang", llvmBin), llvmObjcopy = discover("llvm-objcopy", llvmBin),
    llvmReadobj = discover("llvm-readobj", llvmBin), llvmNm = discover("llvm-nm", llvmBin), lldLink = discover("lld-link", llvmBin);
  return clang && llvmObjcopy && llvmReadobj && llvmNm && lldLink ? {clang, llvmObjcopy, llvmReadobj, llvmNm, lldLink} : undefined;
})();
const toolchainSkip = toolchain ? false : "headerless COFF LLVM toolchain unavailable; set MOM_TEST_LLVM_BIN";

/** @param {string} command @param {string[]} args @param {string} cwd @returns {string} */
function run(command, args, cwd) {
  const result = spawnSync(command, args, {cwd, encoding: "utf8", timeout: 120000, maxBuffer: 64 << 20});
  assert.equal(result.status, 0, result.error?.message || result.stderr);
  return result.stdout || "";
}

/** @param {import("node:test").TestContext} t @returns {{directory: string, objects: string[], entry: string, receipt: ReturnType<typeof isolateCoffObjects>, proof: string, tools: NonNullable<typeof toolchain>}} */
function fixture(t, directives = false) {
  assert.ok(toolchain);
  const tools = /** @type {NonNullable<typeof toolchain>} */ (toolchain);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-isolation-"));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  /** @param {number} value @param {string} factory */
  const source = (value, factory) => [
    ...(directives ? ['#pragma comment(linker, "/INCLUDE:sharedHelper")'] : []),
    "using Size = decltype(sizeof(0));",
    "struct ImageCandidates { int selected; };",
    `extern "C" __declspec(selectany) int sharedData = ${value};`,
    "extern \"C\" void mom_cuda_images_select_embedded(const char*, Size, int, ImageCandidates*) noexcept;",
    'extern "C" ' + (directives ? "__declspec(dllexport) " : "") + "__declspec(noinline) inline int sharedHelper() { volatile int local[4] = {sharedData, 1, 2, 3}; return local[0]; }",
    // Exercise only the external C import here; avoid a string COMDAT changing collision counts.
    `extern "C" __declspec(noinline) int ${factory}() { ImageCandidates candidates{}; mom_cuda_images_select_embedded(nullptr, 0, 80, &candidates); volatile int value = sharedHelper() + candidates.selected; __try { return value; } __except(1) { return -1; } }`
  ].join("\n") + "\n";
  const sourceOne = path.join(directory, "profile-one.cpp"), sourceTwo = path.join(directory, "profile-two.cpp"), sources = [sourceOne, sourceTwo];
  fs.writeFileSync(sourceOne, source(7, "factory1"), {flag: "wx"});
  fs.writeFileSync(sourceTwo, source(11, "factory2"), {flag: "wx"});
  const objects = sources.map((input, index) => {
    const output = path.join(directory, `profile-${index + 1}.obj`);
    run(tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", input, "-o", output], directory);
    return output;
  });
  const entrySource = path.join(directory, "entry.cpp"), entry = path.join(directory, "entry.obj");
  fs.writeFileSync(entrySource, [
    "using Size = decltype(sizeof(0));", "struct ImageCandidates { int selected; };",
    "extern \"C\" int factory1(); extern \"C\" int factory2();",
    "extern \"C\" void mom_cuda_images_select_embedded(const char*, Size, int, ImageCandidates* output) noexcept { if (output) output->selected = 1; }",
    "extern \"C\" int mainCRTStartup() { return factory1() + factory2(); }", "",
  ].join("\n"), {flag: "wx"});
  run(tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", entrySource, "-o", entry], directory);
  const isolated = path.join(directory, "isolated"), receipt = isolateCoffObjects({tools: {llvmObjcopy: tools.llvmObjcopy,
    llvmReadobj: tools.llvmReadobj, llvmNm: tools.llvmNm}, objects, output: isolated, prefix: "mom_coff_", entryPoints: ["factory1", "factory2"]});
  const proof = path.join(directory, "proof.exe");
  run(tools.lldLink, ["/entry:mainCRTStartup", "/subsystem:console", "/nodefaultlib", "/out:" + proof,
    ...receipt.outputs.map(output => output.path), entry], directory);
  return {directory, objects, entry, receipt, proof, tools};
}

test.describe("Maintained Windows COFF graph isolation", {skip: toolchainSkip}, () => {
  test("captures each original LLVM inspection once without repeating the bounded diagnostic", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t), calls = path.join(selected.directory, "readobj-calls.jsonl"), wrapper = path.join(selected.directory, "counting-readobj");
    fs.writeFileSync(wrapper, "#!" + process.execPath + "\nconst fs=require('node:fs'),cp=require('node:child_process'),args=process.argv.slice(2);\n" +
      "fs.appendFileSync(" + JSON.stringify(calls) + ",JSON.stringify(args)+'\\n');\n" +
      "const r=cp.spawnSync(" + JSON.stringify(selected.tools.llvmReadobj) + ",args,{stdio:'inherit'});process.exit(r.status===null?1:r.status);\n", {flag: "wx", mode: 0o755});
    fs.chmodSync(wrapper, 0o755);
    const receipt = isolateCoffObjects({tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: wrapper, llvmNm: selected.tools.llvmNm},
      objects: selected.objects, output: path.join(selected.directory, "single-inspection"), prefix: "mom_single_", entryPoints: ["factory1", "factory2"]});
    const invocations = fs.readFileSync(calls, "utf8").trim().split("\n").map(line => JSON.parse(line));
    for (const object of selected.objects) {assert.equal(invocations.filter(args => args.at(-1) === object).length, 1);}
    for (const [index, input] of receipt.inputs.entries()) {
      const captured = fs.readFileSync(path.join(receipt.output, "input-" + index + "-readobj.log"), "utf8");
      assert.match(captured, /Format: COFF-x86-64/); assert.equal(input.sha256, fileHash(input.path));
    }
  });

  test("keeps compact payload and metadata digests while preserving full raw LLVM evidence", t => {
    const selected = fixture(t), receipt = selected.receipt;
    for (const [index, output] of receipt.outputs.entries()) {
      assert.equal(Object.hasOwn(output, "metadata"), false);
      assert.equal(Object.hasOwn(output, "payloads"), false);
      assert.match(output.metadataSha256, /^[0-9a-f]{64}$/);
      assert.equal(output.payloadSha256, require("node:crypto").createHash("sha256").update(JSON.stringify(coffSectionPayloads(output.path))).digest("hex"));
      assert.equal(output.nonDirectivePayloadSha256, receipt.inputs[index]?.nonDirectivePayloadSha256);
      assert.equal(output.nonDirectiveMetadataSha256, receipt.inputs[index]?.nonDirectiveMetadataSha256);
      assert.match(fs.readFileSync(path.join(receipt.output, "output-" + index + "-readobj.log"), "utf8"), /Symbols \[/);
    }
  });

  test("rewrites compiler-generated linker roots and exports with their private definitions", t => {
    const selected = fixture(t, true);
    assert.equal(selected.receipt.checks.nonDirectivePayloadsPreserved, true);
    assert.equal(selected.receipt.checks.sectionPayloadsPreserved, false);
    for (const output of selected.receipt.outputs) {
      assert.ok(output.directives.rewritten >= 2);
      const text = run(selected.tools.llvmReadobj, ["--coff-directives", output.path], selected.directory);
      assert.match(text, /\/include:"?mom_coff_sharedHelper/i);
      assert.match(text, /\/export:"?mom_coff_sharedHelper/i);
      assert.doesNotMatch(text, /\/(?:include|export):"?sharedHelper(?:"|\s|$)/i);
    }
  });

  test("maps quoted aliases and alternates without changing CRT or unrelated linker policy", () => {
    const names = new Map([["?helper@@", "mom_private_?helper@@"], ["fallback", "mom_private_fallback"]]);
    const source = '/DEFAULTLIB:"C:\\SDK with space\\libcmt.lib" /FAILIFMISMATCH:"RuntimeLibrary=MT_StaticRelease" /EXPORT:"alias=?helper@@,DATA" /ALTERNATENAME:external=fallback /INCLUDE:"?helper@@"\0';
    const result = rewriteCoffDirectives(source, names);
    assert.equal(result.rewritten, 3);
    assert.equal(result.text, source.replace("alias=?helper@@", "alias=mom_private_?helper@@").replace("external=fallback", "external=mom_private_fallback").replace('/INCLUDE:"?helper@@"', '/INCLUDE:"mom_private_?helper@@"'));
    assert.throws(() => rewriteCoffDirectives("/ALTERNATENAME:missing_target", names), /Unsupported COFF symbol directive/);
  });

  test("preserves external fallback alias chains and strong CRT overrides across isolated profiles", t => {
    assert.ok(toolchain);
    const tools = toolchain, directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-alias-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const outputs = [], receipts = [];
    for (const index of [1, 2]) {
      const source = path.join(directory, `profile-${index}.cpp`), object = path.join(directory, `profile-${index}.obj`);
      fs.writeFileSync(source, [
        '#pragma comment(linker, "/alternatename:featureAlias=feature")',
        '#pragma comment(linker, "/alternatename:feature=weakFeature")',
        'extern "C" __declspec(selectany) int weakFeature = 0;',
        `extern "C" __declspec(selectany) int privateData = ${index};`,
        'extern "C" int featureAlias;',
        `extern "C" int factory${index}() { return featureAlias + privateData; }`,
      ].join("\n") + "\n", {flag: "wx"});
      run(tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", source, "-o", object], directory);
      const receipt = isolateCoffObjects({tools: {llvmObjcopy: tools.llvmObjcopy, llvmReadobj: tools.llvmReadobj, llvmNm: tools.llvmNm},
        objects: [object], output: path.join(directory, "isolated-" + index), prefix: `mom_profile${index}_`, entryPoints: ["factory" + index]});
      receipts.push(receipt); outputs.push(...receipt.outputs.map(row => row.path));
    }
    const entrySource = path.join(directory, "entry.cpp"), entry = path.join(directory, "entry.obj");
    fs.writeFileSync(entrySource, [
      '#pragma comment(linker, "/alternatename:feature=weakFeature")',
      'extern "C" __declspec(selectany) int weakFeature = 0;',
      'extern "C" int feature = 3;',
      'extern "C" int factory1(); extern "C" int factory2();',
      'extern "C" int mainCRTStartup() { return factory1() + factory2(); }',
    ].join("\n") + "\n", {flag: "wx"});
    run(tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", entrySource, "-o", entry], directory);
    run(tools.lldLink, ["/entry:mainCRTStartup", "/subsystem:console", "/nodefaultlib", "/out:" + path.join(directory, "proof.exe"), ...outputs, entry], directory);
    for (const receipt of receipts) {
      assert.deepEqual(receipt.map.entries.map(row => row.from), ["privateData"]);
      assert.deepEqual(receipt.sharedFallbackSymbols, ["feature", "featureAlias", "weakFeature"]);
      assert.equal(receipt.checks.nonDirectivePayloadsPreserved, true);
    }
  });

  test("rejects conflicting fallback bindings before creating output", t => {
    assert.ok(toolchain);
    const tools = toolchain, directory = fs.mkdtempSync(path.join(os.tmpdir(), "mom-coff-alias-conflict-"));
    t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
    const source = path.join(directory, "conflict.cpp"), object = path.join(directory, "conflict.obj"), output = path.join(directory, "isolated");
    fs.writeFileSync(source, '#pragma comment(linker, "/alternatename:feature=first")\n#pragma comment(linker, "/alternatename:feature=second")\nextern "C" int factory() { return 1; }\n', {flag: "wx"});
    run(tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", source, "-o", object], directory);
    assert.throws(() => isolateCoffObjects({tools: {llvmObjcopy: tools.llvmObjcopy, llvmReadobj: tools.llvmReadobj, llvmNm: tools.llvmNm},
      objects: [object], output, prefix: "mom_conflict_", entryPoints: ["factory"]}), /Conflicting COFF fallback aliases/);
    assert.equal(fs.existsSync(output), false);
  });

  test("renames complete duplicate definitions while preserving COMDAT and associative xdata", t => {
    const selected = fixture(t), receipt = selected.receipt;
    assert.deepEqual(receipt.entryPoints, ["factory1", "factory2"]);
    assert.equal(receipt.map.entries.length, 2);
    assert.deepEqual(receipt.map.entries.map(entry => entry.from).sort(), ["sharedData", "sharedHelper"]);
    assert.equal(receipt.checks.oldCollisionDefinitions, 0);
    assert.equal(receipt.checks.oldCollisionReferences, 0);
    assert.equal(receipt.checks.sectionMetadataPreserved, true);
    assert.equal(receipt.checks.sectionPayloadsPreserved, true);
    assert.equal(receipt.checks.comdatSelectionPreserved, true);
    assert.equal(receipt.checks.associativeXdataPreserved, true);
    assert.equal(receipt.checks.externalUndefinedImportsPreserved, true);
    assert.deepEqual(receipt.inputs.map(input => input.sha256), selected.objects.map(fileHash));
    for (const [index, output] of receipt.outputs.entries()) {
      assert.equal(output.sha256, fileHash(output.path));
      const definitions = run(selected.tools.llvmNm, ["-P", "-g", "--defined-only", output.path], selected.directory);
      assert.match(definitions, /^mom_coff_shared(?:Data|Helper)\s/m);
      assert.ok(definitions.split(/\r?\n/).some(line => line.startsWith("factory" + (index + 1) + " ")));
      assert.doesNotMatch(definitions, /^shared(?:Data|Helper)\s/m);
      assert.match(run(selected.tools.llvmNm, ["-P", "--undefined-only", output.path], selected.directory), /^mom_cuda_images_select_embedded\s/m);
      const command = JSON.parse(fs.readFileSync(path.join(receipt.output, `isolate-${index}.command.json`), "utf8"));
      assert.equal(command.args[0], "--redefine-syms=" + receipt.map.path);
    }
    assert.match(run(selected.tools.llvmNm, ["-P", "-g", "--defined-only", selected.entry], selected.directory),
      /^mom_cuda_images_select_embedded\s/m, "the one application registry object supplies the preserved C import");
    const pe = run(selected.tools.llvmReadobj, ["--file-headers", "--coff-imports", selected.proof], selected.directory);
    assert.match(pe, /^Format:\s+COFF-x86-64\s*$/m);
    assert.doesNotMatch(pe, /^Import \{/m);
  });

  test("keeps independently isolated profiles distinct instead of collapsing selectany COMDATs", t => {
    const selected = fixture(t), oneObject = selected.objects[0], twoObject = selected.objects[1],
      tools = {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm};
    assert.ok(oneObject && twoObject);
    for (const object of selected.objects) {
      const definitions = run(selected.tools.llvmNm, ["-P", "-g", "--defined-only", object], selected.directory);
      assert.match(definitions, /^sharedData /m);
      assert.match(definitions, /^sharedHelper /m);
    }

    const unisolated = path.join(selected.directory, "unisolated.exe"), unisolatedMap = path.join(selected.directory, "unisolated.map");
    run(selected.tools.lldLink, ["/entry:mainCRTStartup", "/subsystem:console", "/nodefaultlib", "/out:" + unisolated,
      "/map:" + unisolatedMap, ...selected.objects, selected.entry], selected.directory);
    const collapsedLines = fs.readFileSync(unisolatedMap, "utf8").split(/\r?\n/).filter(line => /\s(sharedData|sharedHelper)\s/.test(line)),
      collapsedOwners = new Set(collapsedLines.map(line => [path.basename(oneObject), path.basename(twoObject)].find(owner => line.trim().endsWith(owner))));
    assert.equal(collapsedLines.length, 2);
    assert.ok(collapsedOwners.size >= 1 && !collapsedOwners.has(undefined), "each surviving COMDAT must belong to one of the two inputs");

    const profileOne = isolateCoffObjects({tools, objects: [oneObject], output: path.join(selected.directory, "profile-one-isolated"),
      prefix: "mom_profile1_", entryPoints: ["factory1"]});
    const profileTwo = isolateCoffObjects({tools, objects: [twoObject], output: path.join(selected.directory, "profile-two-isolated"),
      prefix: "mom_profile2_", entryPoints: ["factory2"]});
    assert.deepEqual(profileOne.map.entries.map(entry => entry.from).sort(), ["sharedData", "sharedHelper"]);
    assert.deepEqual(profileTwo.map.entries.map(entry => entry.from).sort(), ["sharedData", "sharedHelper"]);
    const firstOutput = profileOne.outputs[0], secondOutput = profileTwo.outputs[0];
    assert.ok(firstOutput && secondOutput);

    const profileOneRelocations = run(selected.tools.llvmReadobj, ["--file-headers", "--sections", "--section-relocations", "--symbols", firstOutput.path], selected.directory),
      profileTwoRelocations = run(selected.tools.llvmReadobj, ["--file-headers", "--sections", "--section-relocations", "--symbols", secondOutput.path], selected.directory);
    assert.match(profileOneRelocations, /IMAGE_REL_AMD64_REL32 mom_profile1_sharedHelper/);
    assert.match(profileOneRelocations, /IMAGE_REL_AMD64_REL32 mom_profile1_sharedData/);
    assert.doesNotMatch(profileOneRelocations, /mom_profile2_shared(?:Data|Helper)/);
    assert.match(profileTwoRelocations, /IMAGE_REL_AMD64_REL32 mom_profile2_sharedHelper/);
    assert.match(profileTwoRelocations, /IMAGE_REL_AMD64_REL32 mom_profile2_sharedData/);
    assert.doesNotMatch(profileTwoRelocations, /mom_profile1_shared(?:Data|Helper)/);

    const isolated = path.join(selected.directory, "profiles-isolated.exe"), isolatedMap = path.join(selected.directory, "profiles-isolated.map");
    run(selected.tools.lldLink, ["/entry:mainCRTStartup", "/subsystem:console", "/nodefaultlib", "/out:" + isolated,
      "/map:" + isolatedMap, firstOutput.path, secondOutput.path, selected.entry], selected.directory);
    const isolatedMapText = fs.readFileSync(isolatedMap, "utf8");
    assert.match(isolatedMapText, /mom_profile1_sharedData/);
    assert.match(isolatedMapText, /mom_profile2_sharedData/);
    assert.match(isolatedMapText, /mom_profile1_sharedHelper/);
    assert.match(isolatedMapText, /mom_profile2_sharedHelper/);
  });

  test("rejects malformed paths, options, and unknown COFF before creating output", t => {
    const selected = fixture(t), base = {tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm},
      objects: selected.objects, prefix: "mom_bad_", entryPoints: ["factory1", "factory2"]};
    const invalidOutput = path.join(selected.directory, "invalid");
    assert.throws(() => isolateCoffObjects({...base, output: invalidOutput, prefix: "bad prefix"}), /private symbol prefix/);
    assert.equal(fs.existsSync(invalidOutput), false);
    assert.throws(() => isolateCoffObjects(/** @type {Parameters<typeof isolateCoffObjects>[0]} */ ({...base, output: invalidOutput, extra: true})), /unsupported fields/);
    assert.equal(fs.existsSync(invalidOutput), false);
    const first = selected.objects[0] || "", second = selected.objects[1] || "";
    assert.throws(() => isolateCoffObjects({...base, output: invalidOutput, objects: [first, first]}), /distinct/);
    assert.equal(fs.existsSync(invalidOutput), false);
    const unknown = path.join(selected.directory, "unknown.obj"); fs.writeFileSync(unknown, "not a COFF object\n", {flag: "wx"});
    assert.throws(() => isolateCoffObjects({...base, output: invalidOutput, objects: [unknown, second]}), /COFF input 0 readobj failed|unknown COFF|Unresolved COFF/);
    assert.equal(fs.existsSync(invalidOutput), false);
    assert.throws(() => isolateCoffObjects({...base, output: invalidOutput, tools: {...base.tools, llvmObjcopy: "/bin/sh"}}), /not an LLVM tool|version failed/);
    assert.equal(fs.existsSync(invalidOutput), false);
  });

  test("publishes literal input, tool, map, and output provenance", t => {
    const selected = fixture(t), result = /** @type {ReturnType<typeof isolateCoffObjects>} */ (JSON.parse(fs.readFileSync(path.join(selected.receipt.output, "result.json"), "utf8")));
    assert.equal(result.kind, "windows-coff-graph-isolation");
    assert.equal(result.tools.llvmObjcopy.path, selected.tools.llvmObjcopy);
    assert.equal(result.tools.llvmReadobj.path, selected.tools.llvmReadobj);
    assert.equal(result.tools.llvmNm.path, selected.tools.llvmNm);
    assert.equal(result.map.path, selected.receipt.map.path);
    assert.equal(result.map.sha256, fileHash(result.map.path));
    assert.deepEqual(result.inputs.map(input => input.path), selected.objects);
    assert.deepEqual(result.outputs.map(output => output.path), selected.receipt.outputs.map(output => output.path));
    for (const [index, output] of result.outputs.entries()) {
      const input = result.inputs[index]; assert.ok(input);
      assert.equal(output.payloadSha256, input.payloadSha256);
    }
  });

  test("preserves zero-initialized data when LLVM materializes its raw storage", t => {
    const selected = fixture(t), source = path.join(selected.directory, "zero.cpp"), object = path.join(selected.directory, "zero.obj");
    fs.writeFileSync(source, "extern \"C\" __declspec(selectany) int zeroData; extern \"C\" int bssFactory() { return zeroData; }\n", {flag: "wx"});
    run(selected.tools.clang, ["--target=x86_64-pc-windows-msvc", "-GS-", "-c", source, "-o", object], selected.directory);
    const receipt = isolateCoffObjects({tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm},
      objects: [object], output: path.join(selected.directory, "zero-isolated"), prefix: "mom_zero_", entryPoints: ["bssFactory"]});
    assert.equal(receipt.checks.sectionPayloadsPreserved, true);
    const firstOutput = receipt.outputs[0]; assert.ok(firstOutput);
    assert.ok(coffSectionPayloads(firstOutput.path).some(section => section.bytes === 4 && (section.characteristics & 0x80)));
  });

  test("rejects altered host code even when definitions and section metadata still match", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t), changedTool = path.join(selected.directory, "changing-objcopy");
    const source = "#!" + process.execPath + "\n" +
      'const fs=require("fs"),cp=require("child_process"),args=process.argv.slice(2);' +
      "const r=cp.spawnSync(" + JSON.stringify(selected.tools.llvmObjcopy) + ',args,{stdio:"inherit"});' +
      'if(r.status!==0||args[0]==="--version")process.exit(r.status===null?1:r.status);' +
      "const file=args.at(-1),b=fs.readFileSync(file);" +
      "for(let i=0;i<b.readUInt16LE(2);i++){const s=20+i*40,n=b.readUInt32LE(s+16),p=b.readUInt32LE(s+20);" +
      "if(n&&p){b[p]^=1;fs.writeFileSync(file,b);break;}}\n";
    fs.writeFileSync(changedTool, source, {flag: "wx", mode: 0o755});
    assert.throws(() => isolateCoffObjects({tools: {llvmObjcopy: changedTool, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm},
      objects: selected.objects, output: path.join(selected.directory, "changed-code"), prefix: "mom_changed_", entryPoints: ["factory1", "factory2"]}), /changed section payload bytes/);
  });

  test("rejects a COFF tool that mutates itself during version admission before output creation", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t), changingTool = path.join(selected.directory, "self-mutating-objcopy"), output = path.join(selected.directory, "self-mutating-output");
    const source = "#!" + process.execPath + "\n" +
      'const fs=require("fs"),args=process.argv.slice(2);' +
      'if(args[0]==="--version"){fs.appendFileSync(__filename,"\\n");process.stdout.write("LLVM self-mutating objcopy\\n");}' +
      "\n";
    fs.writeFileSync(changingTool, source, {flag: "wx", mode: 0o755}); fs.chmodSync(changingTool, 0o755);
    assert.throws(() => isolateCoffObjects({tools: {llvmObjcopy: changingTool, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm},
      objects: selected.objects, output, prefix: "mom_tool_mutation_", entryPoints: ["factory1", "factory2"]}), /COFF tool changed during isolation/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects synthetic symbol metadata over the aggregate cap before output creation", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t), noisyNm = path.join(selected.directory, "noisy-nm"), output = path.join(selected.directory, "too-many-symbols");
    const source = "#!" + process.execPath + "\n" +
      "const args=process.argv.slice(2);" +
      "if(args[0]===\"--version\"){process.stdout.write(\"LLVM synthetic nm\\n\");process.exit(0);}" +
      "if(!args.includes(\"--defined-only\"))process.exit(0);" +
      "let text=\"\";for(let index=0;index<200001;index++)text+=\"s\"+index.toString(36)+\" T 0 0\\n\";process.stdout.write(text);" +
      "\n";
    fs.writeFileSync(noisyNm, source, {flag: "wx", mode: 0o755}); fs.chmodSync(noisyNm, 0o755);
    assert.throws(() => isolateCoffObjects({tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: selected.tools.llvmReadobj, llvmNm: noisyNm},
      objects: selected.objects, output, prefix: "mom_symbol_cap_", entryPoints: ["factory1", "factory2"]}), /COFF aggregate metadata budget exceeded/);
    assert.equal(fs.existsSync(output), false);
  });

  test("rejects an objcopy output symlink", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t), symlinkTool = path.join(selected.directory, "symlink-objcopy"), output = path.join(selected.directory, "symlink-output");
    const source = "#!" + process.execPath + "\n" +
      "const fs=require(\"fs\"),args=process.argv.slice(2);" +
      "if(args[0]===\"--version\"){process.stdout.write(\"LLVM symlink objcopy\\n\");process.exit(0);}" +
      "fs.symlinkSync(args.at(-2),args.at(-1));\n";
    fs.writeFileSync(symlinkTool, source, {flag: "wx", mode: 0o755}); fs.chmodSync(symlinkTool, 0o755);
    assert.throws(() => isolateCoffObjects({tools: {llvmObjcopy: symlinkTool, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm},
      objects: selected.objects, output, prefix: "mom_symlink_", entryPoints: ["factory1", "factory2"]}), /COFF isolated output must not be a symlink/);
    assert.equal(fs.lstatSync(path.join(output, "object-0000.obj")).isSymbolicLink(), true);
  });

  test("rejects incomplete LLVM inspection before output creation and retains a failed-tool log", {skip: process.platform !== "linux"}, t => {
    const selected = fixture(t);
    /** @param {string} name @param {string} body @returns {string} */
    const makeTool = (name, body) => {
      const file = path.join(selected.directory, name); fs.writeFileSync(file, "#!/bin/sh\n" + body + "\n", {mode: 0o755, flag: "wx"}); fs.chmodSync(file, 0o755); return file;
    };
    const incompleteReadobj = makeTool("fake-readobj", 'if [ "$1" = "--version" ]; then echo "fake LLVM readobj"; else printf "Format: COFF-x86-64\\nArch: x86_64\\nAddressSize: 64bit\\n"; fi'), malformedOutput = path.join(selected.directory, "malformed");
    const base = {tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: incompleteReadobj, llvmNm: selected.tools.llvmNm}, objects: selected.objects,
      output: malformedOutput, prefix: "mom_malformed_", entryPoints: ["factory1", "factory2"]};
    assert.throws(() => isolateCoffObjects(base), /Unresolved COFF section format/);
    assert.equal(fs.existsSync(malformedOutput), false);
    const failedObjcopy = makeTool("fake-objcopy", 'if [ "$1" = "--version" ]; then echo "fake LLVM objcopy"; else echo "intentional objcopy failure" >&2; exit 17; fi'), failedOutput = path.join(selected.directory, "failed");
    const validBase = {...base, tools: {llvmObjcopy: selected.tools.llvmObjcopy, llvmReadobj: selected.tools.llvmReadobj, llvmNm: selected.tools.llvmNm}};
    assert.throws(() => isolateCoffObjects({...validBase, tools: {...validBase.tools, llvmObjcopy: failedObjcopy}, output: failedOutput}), /COFF isolate-0 failed/);
    assert.match(fs.readFileSync(path.join(failedOutput, "isolate-0.log"), "utf8"), /intentional objcopy failure/);
    assert.equal(JSON.parse(fs.readFileSync(path.join(failedOutput, "isolate-0.command.json"), "utf8")).tool, failedObjcopy);
  });
});
