"use strict";

const {createHash} = require("node:crypto");
const path = require("node:path");
const {fileHash, sourceInputs} = require("./build-source-inputs");
const digest = /^[a-f0-9]{64}$/;

/** The final admitted native closure owns cache identity, not transport paths or receipt JSON.
 * Preserve link order; exclude the controller object containing the generated tag to avoid a cycle.
 * @param {"linux-x64" | "windows-x64"} target
 * @param {Array<{role: string, sha256: string}>} artifacts
 * @param {Record<string, string>} recipes @param {string[]} [linkPlan] */
function nativeJitCacheTag(target, artifacts, recipes, linkPlan = []) {
  if (!["linux-x64", "windows-x64"].includes(target) || !Array.isArray(artifacts) ||
      artifacts.length < 1 || artifacts.length > 32768 || !recipes || typeof recipes !== "object" || Array.isArray(recipes)) {
    throw new Error("Invalid native JIT cache closure");
  }
  const roles = new Set();
  if (!Array.isArray(linkPlan) || linkPlan.length > 32768 || linkPlan.some(token =>
    typeof token !== "string" || !token.length || token.length > 4096 ||
    Array.from(token).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127))) {
    throw new Error("Invalid native JIT link plan");
  }
  for (const row of artifacts) {
    if (!row || Array.isArray(row) || Object.keys(row).sort().join(",") !== "role,sha256" ||
        typeof row.role !== "string" || !/^[a-z][a-z0-9-]*\/[0-9]+$/.test(row.role) ||
        roles.has(row.role) || typeof row.sha256 !== "string" || !digest.test(row.sha256)) {throw new Error("Invalid native JIT cache artifact");}
    roles.add(row.role);
  }
  // Recipes hash all of runtime/, so the bound must leave room for that tree to grow.
  const entries = Object.entries(recipes).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0);
  if (!entries.length || entries.length > 4096 || entries.some(([name, hash]) =>
    name.length > 256 || !/^(runtime|scripts)\/[A-Za-z0-9_./-]+$/.test(name) ||
    name.split("/").some(part => !part || part === "." || part === "..") || typeof hash !== "string" || !digest.test(hash))) {
    throw new Error("Invalid native JIT cache recipe");
  }
  return createHash("sha256").update(JSON.stringify({domain: "mom-native-jit-closure-v1", target, artifacts, linkPlan, recipes: entries})).digest("hex");
}

/** Emit a tiny generated input for runtime/main.cpp; no runtime hash or crypto dependency.
 * @param {string} tag */
function nativeJitCacheHeader(tag) {
  if (typeof tag !== "string" || !digest.test(tag)) {throw new Error("Invalid native JIT cache tag");}
  return "#pragma once\ninline constexpr unsigned char momNativeJitCacheTag[32] = {" +
    (tag.match(/../g) || []).map(byte => "0x" + byte).join(",") + "};\n";
}

/** Only call after admission; hash the exact linked inputs, not mutable receipt paths.
 * @param {string} repo @param {"linux-x64" | "windows-x64"} target
 * @param {Record<string, string[]>} groups @param {string[]} [linkPlan] */
function admittedJitCacheTag(repo, target, groups, linkPlan = []) {
  const artifacts = Object.entries(groups).flatMap(([role, files]) => files.map((file, index) => {
    if (!path.isAbsolute(file)) {throw new Error("Native JIT cache inputs must be admitted absolute files");}
    return {role: role + "/" + index, sha256: fileHash(file)};
  }));
  // Bind fixed cross-group ordering and linker policy too, not only descriptor archive lists.
  const recipes = sourceInputs(repo, ["runtime"], ["scripts/native-jit-cache-tag.js",
    ...(target === "windows-x64" ? ["scripts/build-native-windows-controller.js", "scripts/native-windows-application.js"] : ["scripts/build-native-application.js"])]);
  return nativeJitCacheTag(target, artifacts, {...recipes.files, ...recipes.graph}, linkPlan);
}

module.exports = {nativeJitCacheTag, nativeJitCacheHeader, admittedJitCacheTag};
