"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {windowsHostSdkCatalog} = require("../scripts/fetch-native-sdk-sources");
const windowsIntelRuntimeCatalog = require("../runtime/windows-intel-runtime.json");
const {safeRelative, parseIdt, buildMsiFilePaths, parse7zListing, mapSelectedMember, normalizeMsiSource,
  validateMsiMedia, walkTree, addSelectedFile, headerOverlayEntries, prepareWindowsHostSdk,
  defaultWindowsNativeLibraryPolicy} = require("../scripts/prepare-native-windows-sdk");

const lock = windowsHostSdkCatalog();

/** @param {string} name @param {string[]} columns @param {string[][]} rows */
function table(name, columns, rows) {
  return [columns.join("\t"), columns.map(() => "s255").join("\t"), `${name}\t${columns[0]}`, ...rows.map(row => row.join("\t")), ""].join("\n");
}

test.describe("Windows host SDK composition", () => {
  test("default MT policy preserves qualified link order and isolates caller mutations", () => {
    const expected = {
      msvcLibraries: ["libcmt.lib", "libcpmt.lib", "oldnames.lib", "delayimp.lib", "libircmt.lib", "libmmt.lib", "svml_dispmt.lib",
        "libdecimal.lib", "libirc.lib"],
      ucrtLibraries: ["libvcruntime.lib", "libucrt.lib"],
      importLibraries: ["AdvAPI32.Lib", "Crypt32.Lib", "DbgHelp.Lib", "DnsAPI.Lib", "Gdi32.Lib", "iphlpapi.lib",
        "kernel32.Lib", "ncrypt.lib", "ntdll.lib", "Ole32.Lib", "powrprof.lib", "psapi.lib", "RpcRT4.Lib", "Secur32.Lib",
        "SetupAPI.Lib", "shell32.lib", "ShLwApi.Lib", "User32.Lib", "UserEnv.Lib", "Uuid.Lib", "Version.Lib", "winhttp.lib",
        "WinMM.Lib", "WS2_32.Lib", "WSock32.Lib", "OleAut32.Lib", "cfgmgr32.lib", "runtimeobject.lib"]
    };
    const first = defaultWindowsNativeLibraryPolicy();
    assert.deepEqual(first, expected);
    const runtimeNames = windowsIntelRuntimeCatalog.libraries.map(row => row.name).sort();
    assert.deepEqual(first.msvcLibraries.slice(4).sort(), runtimeNames);
    first.msvcLibraries.push("unexpected.lib"); first.ucrtLibraries.pop(); first.importLibraries[0] = "mutated.lib";
    assert.deepEqual(defaultWindowsNativeLibraryPolicy(), expected);
  });

  test("safe Windows SDK paths reject traversal, ADS, device names, and ambiguous components", () => {
    assert.equal(safeRelative("Program Files/Windows Kits/10/Include/sdk.h"), "Program Files/Windows Kits/10/Include/sdk.h");
    for (const value of ["", "/rooted", "../escape", "a/../../escape", "C:/sdk.h", "a:stream", "CON", "dir/NUL.txt", "trailing.", "space ", "a//b", "bad\nname"]) {
      assert.throws(() => safeRelative(value));
    }
  });

  test("standard MSI Directory/Component/File tables preserve the x64 payload of an x86-labelled package", () => {
    const dirs = table("Directory", ["Directory", "Directory_Parent", "DefaultDir"], [
      ["TARGETDIR", "", "SourceDir"], ["ProgramFilesFolder", "TARGETDIR", "."],
      ["WindowsKitsParent", "ProgramFilesFolder", "WIN~1|Windows Kits"], ["Win10", "WindowsKitsParent", "10:unused-source"],
      ["Lib", "Win10", "Lib"], ["Version", "Lib", "10.0.26100.0"], ["Um", "Version", "um"], ["X64", "Um", "x64"]
    ]);
    const components = table("Component", ["Component", "ComponentId", "Directory_", "Attributes", "Condition", "KeyPath"],
      [["C1", "{00000000-0000-0000-0000-000000000001}", "X64", "0", "", "F1"]]);
    const files = table("File", ["File", "Component_", "FileName", "FileSize", "Version", "Language", "Attributes", "Sequence"],
      [["F1", "C1", "AdvAPI32.Lib", "42", "", "", "0", "1"]]);
    const projection = buildMsiFilePaths(dirs, components, files);
    assert.equal(projection.expandedBytes, 42);
    assert.equal(projection.rows[0]?.relative, "SourceDir/Windows Kits/10/Lib/10.0.26100.0/um/x64/AdvAPI32.Lib");

    // This package is named x86 by its installer, but the fixed lock intentionally selects its x64 subtree.
    const source = normalizeMsiSource(projection.rows[0].relative);
    const selection = lock.selections.find(row => row.package === "windows-sdk-store-libs-x86");
    assert.ok(selection);
    assert.equal(source, "Program Files/Windows Kits/10/Lib/10.0.26100.0/um/x64/AdvAPI32.Lib");
    assert.equal(mapSelectedMember(selection, source), "sdk/lib/um/AdvAPI32.Lib");
  });

  test("MSI table parser rejects malformed rows, duplicate paths, and directory cycles", () => {
    assert.throws(() => parseIdt("File\tName\n", "File"));
    const dirs = table("Directory", ["Directory", "Directory_Parent", "DefaultDir"], [
      ["TARGETDIR", "", "SourceDir"], ["A", "B", "a"], ["B", "A", "b"]
    ]);
    const components = table("Component", ["Component", "Directory_"], [["C1", "TARGETDIR"]]);
    const files = table("File", ["File", "Component_", "FileName", "FileSize", "Sequence"], [["F1", "C1", "x", "1", "1"]]);
    assert.throws(() => buildMsiFilePaths(dirs, components, files), /cyclic/);
    const duplicateFiles = table("File", ["File", "Component_", "FileName", "FileSize", "Sequence"], [
      ["F1", "C1", "same.h", "1", "1"], ["F2", "C1", "SAME.H", "1", "2"]
    ]);
    const goodDirs = table("Directory", ["Directory", "Directory_Parent", "DefaultDir"], [["TARGETDIR", "", "SourceDir"]]);
    assert.throws(() => buildMsiFilePaths(goodDirs, components, duplicateFiles), /Duplicate/);
  });

  test("Media table rows must bind each external cabinet to the fixed MSI relation", () => {
    const msiKey = "windows-sdk-onecore-headers-x86";
    const relations = lock.msiCabinets[msiKey];
    assert.ok(relations?.length);
    const mediaRows = relations.map((relation, index) => {
      const cabinet = lock.cabinets[relation.cabinet];
      assert.ok(cabinet);
      return [relation.diskId, String(index + 1), "", cabinet.vendorFileName, "", ""];
    });
    const media = table("Media", ["DiskId", "LastSequence", "DiskPrompt", "Cabinet", "VolumeLabel", "Source"], mediaRows);
    const files = [{file: "F1", relative: "SourceDir/Windows Kits/a.h", bytes: 1, sequence: 1}];
    assert.equal(validateMsiMedia(media, msiKey, lock, files).length, relations.length);
    const firstRelation = relations[0];
    assert.ok(firstRelation);
    const firstCabinet = lock.cabinets[firstRelation.cabinet];
    assert.ok(firstCabinet);
    const firstName = firstCabinet.vendorFileName.replace(/\\/g, "/").split("/").at(-1);
    assert.ok(firstName);
    assert.throws(() => validateMsiMedia(media.replace(firstName, "unlocked.cab"), msiKey, lock, files), /lock|cabinet/i);
  });

  test("bounded 7-Zip listing parser accepts ZIP and signed-tail CAB shapes but rejects unsafe members", () => {
    const zip = [
      "Path = /tmp/package.vsix", "Type = zip", "Physical Size = 102", "Tail Size = 0", "",
      "Path = Contents/VC/include/a.h", "Folder = -", "Size = 2", "Packed Size = 2", ""
    ].join("\n");
    assert.equal(parse7zListing(zip, "zip").expandedBytes, 2);
    const cab = [
      "Path = /tmp/selected.cab", "Type = cab", "Physical Size = 93008", "Tail Size = 10672", "",
      "Path = Installers/member.bin", "Size = 103680", "Packed Size = 1", "Attributes = A", ""
    ].join("\n");
    assert.equal(parse7zListing(cab, "cab").expandedBytes, 103680);
    assert.throws(() => parse7zListing(zip.replace("Contents/VC/include/a.h", "../escape.h"), "zip"), /Unsafe/);
    assert.throws(() => parse7zListing(zip + "\nPath = contents/vc/INCLUDE/A.H\nFolder = -\nSize = 2\nPacked Size = 2\n", "zip"), /Duplicate/);
    assert.throws(() => parse7zListing(zip.replace("Folder = -", "Folder = -\nSymbolic Link = "), "zip"), /link/i);
  });

  test("fixed lock maps one-file CRT support selectors to the expected output roots", () => {
    const oldnames = lock.selections.find(row => row.destinationPrefix === "msvc/lib/oldnames.lib");
    const comsuppw = lock.selections.find(row => row.destinationPrefix === "msvc/lib/comsuppw.lib");
    const delayimpRows = lock.selections.filter(row => row.destinationPrefix === "msvc/lib/delayimp.lib");
    assert.ok(oldnames && comsuppw);
    assert.equal(delayimpRows.length, 1);
    const delayimp = delayimpRows[0]; assert.ok(delayimp);
    assert.deepEqual(delayimp, {package: "vc-crt-x64-store",
      sourcePrefix: "Contents/VC/Tools/MSVC/14.44.35207/lib/x64/delayimp.lib",
      destinationPrefix: "msvc/lib/delayimp.lib", kind: "file"});
    assert.equal(mapSelectedMember(oldnames, oldnames.sourcePrefix), "msvc/lib/oldnames.lib");
    assert.equal(mapSelectedMember(comsuppw, comsuppw.sourcePrefix), "msvc/lib/comsuppw.lib");
    assert.equal(mapSelectedMember(comsuppw, "Contents/VC/Tools/MSVC/14.44.35207/lib/x64/other.lib"), null);
    assert.equal(mapSelectedMember(delayimp, delayimp.sourcePrefix), "msvc/lib/delayimp.lib");
    assert.equal(mapSelectedMember(delayimp, "Contents/VC/Tools/MSVC/14.44.35207/lib/x64/store/delayimp.lib"), null);
  });

  test("selected file composition retains exact duplicates but rejects case and parent-path collisions", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-win-sdk-test-"));
    try {
      const out = path.join(root, "sdk"); fs.mkdirSync(out);
      const input = path.join(root, "input.lib"); fs.writeFileSync(input, "one");
      /** @type {Map<string,{path:string,sha256:string,bytes:number,origins:object[]}>} */ const claims = new Map();
      /** @type {Map<string,string>} */ const directories = new Map();
      addSelectedFile(input, out, "lib/one.lib", claims, directories, {package: "a"});
      addSelectedFile(input, out, "lib/one.lib", claims, directories, {package: "b"});
      assert.equal(claims.size, 1);
      assert.equal([...claims.values()][0]?.origins.length, 2);
      assert.throws(() => addSelectedFile(input, out, "LIB/another.lib", claims, directories, {package: "c"}), /case|collision/i);
      const other = path.join(root, "other.lib"); fs.writeFileSync(other, "two");
      const sameMtime = new Date("2020-01-02T03:04:05.000Z");
      fs.utimesSync(input, sameMtime, sameMtime); fs.utimesSync(other, sameMtime, sameMtime);
      assert.equal(fs.statSync(input).size, fs.statSync(other).size);
      assert.equal(fs.statSync(input).mtimeMs, fs.statSync(other).mtimeMs);
      assert.throws(() => addSelectedFile(other, out, "lib/one.lib", claims, directories, {package: "d"}), /Conflicting/);
      assert.throws(() => addSelectedFile(input, out, "lib", claims, directories, {package: "e"}), /collides/);
      /** @type {Map<string,{path:string,sha256:string,bytes:number,origins:object[]}>} */ const fileClaims = new Map();
      /** @type {Map<string,string>} */ const fileDirectories = new Map();
      addSelectedFile(input, out, "prefix", fileClaims, fileDirectories, {package: "a"});
      assert.throws(() => addSelectedFile(input, out, "prefix/child.lib", fileClaims, fileDirectories, {package: "b"}), /collision/i);
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("selected header output is stable across source modes and mtimes", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-win-sdk-metadata-"));
    try {
      const sourceA = path.join(root, "a.h"), sourceB = path.join(root, "b.h"), payload = "same header bytes\n";
      fs.writeFileSync(sourceA, payload); fs.writeFileSync(sourceB, payload);
      fs.chmodSync(sourceA, 0o444); fs.chmodSync(sourceB, 0o644);
      fs.utimesSync(sourceA, new Date("2020-01-02T03:04:05.000Z"), new Date("2020-01-02T03:04:05.000Z"));
      fs.utimesSync(sourceB, new Date("2024-05-06T07:08:09.000Z"), new Date("2024-05-06T07:08:09.000Z"));
      const before = [sourceA, sourceB].map(file => {
        const stat = fs.statSync(file);
        return {bytes: fs.readFileSync(file), mode: stat.mode & 0o777, mtimeMs: stat.mtimeMs};
      });
      assert.notEqual(before[0]?.mtimeMs, before[1]?.mtimeMs);
      assert.notEqual(before[0]?.mode, before[1]?.mode);

      const outputs = ["out-a", "out-b"].map(name => path.join(root, name));
      for (const output of outputs) {fs.mkdirSync(output);}
      for (const [index, source] of [sourceA, sourceB].entries()) {
        const output = outputs[index]; assert.ok(output);
        const claims = /** @type {Map<string,{path:string,sha256:string,bytes:number,origins:object[]}>} */ (new Map());
        const directories = /** @type {Map<string,string>} */ (new Map());
        addSelectedFile(source, output, "sdk/include/example.h", claims, directories, {package: `fixture-${index}`});
      }
      const targets = outputs.map(output => path.join(output, "sdk", "include", "example.h"));
      const outputBytes = targets.map(file => fs.readFileSync(file));
      assert.deepEqual(outputBytes, [before[0]?.bytes, before[1]?.bytes]);
      for (const file of targets) {
        const stat = fs.statSync(file);
        // Windows exposes writability rather than POSIX owner/group mode bits.
        if (process.platform === "win32") {assert.notEqual(stat.mode & 0o200, 0);} else {assert.equal(stat.mode & 0o777, 0o644);}
        assert.equal(stat.mtimeMs, 0);
      }
      const after = [sourceA, sourceB].map(file => {
        const stat = fs.statSync(file);
        return {bytes: fs.readFileSync(file), mode: stat.mode & 0o777, mtimeMs: stat.mtimeMs};
      });
      assert.deepEqual(after, before);
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("extracted tree refuses links and special files", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-win-sdk-tree-"));
    try {
      const nested = path.join(root, "include"); fs.mkdirSync(nested);
      fs.writeFileSync(path.join(nested, "ok.h"), "ok");
      assert.equal(walkTree(root).files.length, 1);
      try {fs.symlinkSync("ok.h", path.join(root, "link.h"));} catch (error) {if (process.platform !== "win32") {throw error;} return;}
      assert.throws(() => walkTree(root), /link|special/i);
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });

  test("preparation fails closed before touching paths without build-only Linux authorization", () => {
    assert.throws(() => prepareWindowsHostSdk("/does/not/exist", "/tmp/unused-native-sdk", {}), /MOM_AOT_BUILD_ONLY/);
  });

  test("header overlays use the existing physical spelling of same-byte case aliases", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "native-win-sdk-overlay-"));
    try {
      const out = path.join(root, "sdk"); fs.mkdirSync(out);
      const input = path.join(root, "input.h"); fs.writeFileSync(input, "header");
      /** @type {Map<string,{path:string,sha256:string,bytes:number,origins:object[]}>} */ const claims = new Map();
      /** @type {Map<string,string>} */ const directories = new Map();
      addSelectedFile(input, out, "msvc/include/Existing.h", claims, directories, {package: "first"});
      addSelectedFile(input, out, "msvc/include/EXISTING.H", claims, directories, {package: "duplicate"});
      addSelectedFile(input, out, "msvc/lib/support.lib", claims, directories, {package: "library"});
      const overlay = headerOverlayEntries([...claims.values()], out);
      assert.equal(overlay.length, 1);
      assert.equal(overlay[0]?.name, "/mom-sdk/msvc/include/Existing.h");
      assert.equal(fs.readFileSync(overlay[0]["external-contents"], "utf8"), "header");
    } finally {fs.rmSync(root, {recursive: true, force: true});}
  });
});
