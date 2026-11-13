"use strict";

const fs = require("node:fs"), path = require("node:path");
const {fileHash} = require("./build-source-inputs");
const {isolateCompilerObject} = require("./native-compiler-isolation");

// Compiler-specific algorithm graphs share the maintained public ABI, not their SYCL types or
// helper/COMDAT symbols. Generate its typed table from lib.h so there is no second API registry.
/** Parse only the maintained declaration subset; unexpected syntax must be reviewed, not guessed.
 * @param {string} source */
function gpuDeclarations(source) {
  const declarations = [...source.matchAll(/^MOM_SYCL_API ([\s\S]*?);/gm)].map((match) => {
    const declaration = (match[1] || "").trim();
    const parsed = declaration.match(/^([\w:<>,*& \t\r\n]+?)\b([A-Za-z_]\w*)\(\s*([\s\S]*?)\s*\)\s*(noexcept)?$/);
    if (!parsed || /[(){};"'\\]/.test(parsed[3] || "")) {throw new Error("Unsupported maintained GPU declaration");}
    // Parameter defaults belong only in lib.h. Removing only the supported literal avoids
    // accidentally reproducing a side effect or an ABI-affecting expression in generated code.
    const parameters = (parsed[3] || "").replace(/[=]\s*nullptr\b/g, "").trim();
    if (parameters.includes("=")) {throw new Error("Unsupported maintained GPU default argument");}
    const args = [];
    let depth = 0, start = 0;
    for (let index = 0; index <= parameters.length; ++index) {
      const char = parameters[index];
      if (char === "<") {++depth;} else if (char === ">") {--depth;}
      if (depth < 0) {throw new Error("Unbalanced maintained GPU parameter type");}
      if ((char === "," && depth === 0) || index === parameters.length) {
        const parameter = parameters.slice(start, index).trim();
        if (parameter) {
          const name = parameter.match(/^[\w:<>,*&\s]+[\s*&]([A-Za-z_]\w*)(?:\[\d+\])?$/)?.[1];
          if (!name) {throw new Error("Unnamed or unsupported maintained GPU parameter");}
          args.push(name);
        } else if (parameters) {throw new Error("Empty maintained GPU parameter");}
        start = index + 1;
      }
    }
    if (depth || new Set(args).size !== args.length) {throw new Error("Unbalanced or duplicate maintained GPU parameters");}
    return {name: parsed[2] || "", declaration: `${parsed[1]}${parsed[2]}(${parameters})${parsed[4] ? " noexcept" : ""}`, args};
  });
  const count = (source.match(/^MOM_SYCL_API /gm) || []).length;
  if (!count || declarations.length !== count || count > 64 || new Set(declarations.map((row) => row.name)).size !== count) {
    throw new Error("Incomplete or duplicate maintained GPU interface");
  }
  return declarations;
}

/** The C factory exposes only one immutable table; kernels and their owning runtime stay private.
 * All participating host graphs must share the C++ standard-library ABI, as checked by the
 * real link/exception tests. No SYCL object or compiler-specific type crosses this boundary.
 * @param {string} source @param {string} factory */
function gpuTable(source, factory) {
  if (!/^mom_native_gpu_[a-z][a-z0-9_]{0,31}$/.test(factory)) {throw new Error("Invalid private GPU factory name");}
  const entries = gpuDeclarations(source);
  const header = ["// Generated from the maintained sycl/lib.h; no duplicate algorithm interface.", "#pragma once", '#include "sycl/lib.h"',
    "namespace mom::runtime {", "struct GpuTable {", ...entries.map(({name}) => `  decltype(&::${name}) ${name};`), "};", "}",
    `extern "C" const mom::runtime::GpuTable* ${factory}();`, ""].join("\n");
  const definition = ['#include "gpu-table.h"', `extern "C" const mom::runtime::GpuTable* ${factory}() {`,
    `  static const mom::runtime::GpuTable table{${entries.map(({name}) => `&::${name}`).join(", ")}};`, "  return &table;", "}", ""].join("\n");
  return {entries, header, definition};
}

/** Generate only forwards to maintained implementations. The worker fixes its compiler once;
 * no per-algorithm switch, copied kernel body, dynamic library lookup or hot-loop policy logic.
 * @param {string} source */
function gpuForwards(source) {
  return ['#include "gpu-table.h"', '#include "runtime/gpu-dispatch.h"',
    ...gpuDeclarations(source).map(({name, declaration, args}) => `${declaration} {\n` +
      (name === "sycl_cleanup" ? "  mom::runtime::cleanupGpu();\n" : `  return mom::runtime::selectedGpu().${name}(${args.join(", ")});\n`) + "}"), ""].join("\n");
}

/** Build-only common boundary for original host graphs (plus DPC++'s registered images).
 * Both compilers use identical table generation and COMDAT isolation; only their input objects
 * differ. Runtime/driver/compiler-gate objects stay outside this graph, shared once per process.
 * @param {string} compiler @param {string} output @param {string[]} objects @param {string} factoryName
 * @param {(command: string, args: string[], label: string) => import('node:child_process').SpawnSyncReturns<string>} run */
function isolateGpuObjects(compiler, output, objects, factoryName, run) {
  const repo = path.resolve(__dirname, ".."), api = path.join(repo, "sycl/lib.h");
  const bridge = gpuTable(fs.readFileSync(api, "utf8"), factoryName);
  fs.writeFileSync(path.join(output, "gpu-table.h"), bridge.header, {flag: "wx"});
  const factory = path.join(output, "gpu-factory.cpp"), factoryObject = path.join(output, "gpu-factory.o");
  fs.writeFileSync(factory, bridge.definition, {flag: "wx"});
  if (run(compiler, ["-std=c++20", "-O2", "-fPIC", "-I", repo, "-I", output, "-c", factory, "-o", factoryObject], "factory").status !== 0) {
    throw new Error("Maintained GPU factory did not compile");
  }
  const raw = path.join(output, "gpu-graph.o");
  if (run("ld", ["-r", ...objects, factoryObject, "-o", raw], "partial-link").status !== 0) {throw new Error("Cannot retain complete GPU graph");}
  const prefix = factoryName.replace("mom_native_gpu_", "mom_") + "_gpu_";
  const receipt = isolateCompilerObject(raw, path.join(output, "isolated"), {prefix, entryPoints: [factoryName]});
  return {...receipt, factory: factoryName, apiSha256: fileHash(api)};
}

module.exports = {gpuDeclarations, gpuTable, gpuForwards, isolateGpuObjects};
