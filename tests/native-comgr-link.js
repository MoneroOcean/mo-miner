"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const {nativeComgrLinkInputs} = require("../scripts/probe-native-controller");
const archive = "/component/build/libamd_comgr.a", receipt = "/component/result.json", sha = "a".repeat(64), receiptSha = "b".repeat(64);
function fixture() {
  const comgr = {archive, receipt, sha256: sha, receiptSha256: receiptSha};
  const graph = {comgr, sdkInputs: {[archive]: sha, [receipt]: receiptSha}}, sdk = {libraries: [archive], links: /** @type {string[]} */ ([])};
  const component = {archive, receipt, header: "/component/build/include/amd_comgr.h", llvmVersion: "21.1.8",
    roots: {llvm: "/llvm", lld: "/lld", device: "/device"}, files: {[archive]: sha, [receipt]: receiptSha}};
  const read = () => component;
  const admit = () => nativeComgrLinkInputs(graph, sdk, read);
  return {comgr, graph, sdk, component, read, admit};
}

test.describe("Native final-link COMGR producer handoff", () => {
  test("the source-receipted consumed archive binds the final library", () => {const f = fixture(); assert.deepEqual(f.admit(), f.comgr);});
  test("an archive in explicit link tokens is equally valid", () => {const f = fixture(); f.sdk.libraries = []; f.sdk.links = [archive]; assert.deepEqual(f.admit(), f.comgr);});
  test("literal linker archive groups preserve the same unique input", () => {const f = fixture(); f.sdk.libraries = []; f.sdk.links = ["-Wl,--whole-archive," + archive + ",--no-whole-archive"]; assert.deepEqual(f.admit(), f.comgr);});
  test("a bare legacy graph cannot bypass producer admission", () => {assert.throws(() => nativeComgrLinkInputs({}, {}, () => {throw Error("must not read");}), /complete COMGR/);});
  for (const key of ["receipt", "archive", "sha256", "receiptSha256"]) {
    test("an incomplete claim fails before filesystem access: " + key, () => {
      const f = fixture(); Reflect.deleteProperty(f.comgr, key);
      assert.throws(() => nativeComgrLinkInputs(f.graph, f.sdk, () => {throw Error("must not read");}), /complete COMGR/);
    });
  }
  test("a profile cannot use one producer and name another final prefix", () => {const f = fixture(); f.sdk.libraries = ["/old/sdk/libamd_comgr.a"]; assert.throws(f.admit, /source-receipted/);});
  test("unresolved short-library flags are not an exact final archive", () => {const f = fixture(); f.sdk.libraries = []; f.sdk.links = ["-lamd_comgr"]; assert.throws(f.admit, /source-receipted/);});
  test("duplicate archive paths are not one selected compiler component", () => {const f = fixture(); f.sdk.links = [archive]; assert.throws(f.admit, /exactly/);});
  test("a different archive cannot hide inside a literal linker group", () => {const f = fixture(); f.sdk.links = ["-Wl,--whole-archive,/old/libamd_comgr.a,--no-whole-archive"]; assert.throws(f.admit, /exactly/);});
  test("changed producer archive bytes invalidate a graph's old claim", () => {const f = fixture(); f.component.files[archive] = "c".repeat(64); assert.throws(f.admit, /provenance/);});
  test("changed producer metadata cannot be restamped at final link", () => {const f = fixture(); f.component.files[receipt] = "c".repeat(64); assert.throws(f.admit, /provenance/);});
  test("an unconsumed archive is not made valid by a correct filename", () => {const f = fixture(); Reflect.deleteProperty(f.graph.sdkInputs, archive); assert.throws(f.admit, /consumed SDK/);});
  test("unconsumed producer metadata cannot be claimed after compiling the graph", () => {const f = fixture(); Reflect.deleteProperty(f.graph.sdkInputs, receipt); assert.throws(f.admit, /consumed SDK/);});
  test("producer admission failure propagates, without a legacy fallback", () => {const f = fixture(); assert.throws(() => nativeComgrLinkInputs(f.graph, f.sdk, () => {throw Error("stale producer");}), /stale producer/);});
  test("malformed final-link data is not interpreted as compiler arguments", () => {const f = fixture(); assert.throws(() => nativeComgrLinkInputs(f.graph, {libraries: null, links: []}, f.read), /complete COMGR/);});
});
