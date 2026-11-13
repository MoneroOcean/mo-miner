"use strict";

/** Build-only immutable resource map, portable C++ rather than startup extraction or incbin.
 * Exact symbolic names are intentional: runtime callers cannot turn them into filesystem paths.
 * @param {Array<{name: string, data: Buffer}>} resources */
function nativeResources(resources) {
  const names = new Set();
  let total = 0;
  for (const {name, data} of resources) {
    if (typeof name !== "string" || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(name) || names.has(name)) {throw new Error("Invalid or duplicate native resource name");}
    if (!Buffer.isBuffer(data) || !data.length || data.length > 64 * 1024 * 1024 - total) {throw new Error("Native resource data exceeds its bounds");}
    total += data.length; names.add(name);
  }
  const lines = ["// Generated immutable build resources. No runtime file extraction.", "#include <string_view>",
    "static std::string_view embedded_resource(std::string_view name) {"];
  for (const [index, {name, data}] of resources.entries()) {
    lines.push(`  static constexpr unsigned char resource_${index}[] = {`);
    for (let offset = 0; offset < data.length; offset += 32) {
      lines.push("    " + [...data.subarray(offset, offset + 32)].map((byte) => "0x" + byte.toString(16).padStart(2, "0")).join(",") + ",");
    }
    lines.push("  };", `  if (name == ${JSON.stringify(name)}) return {reinterpret_cast<const char *>(resource_${index}), sizeof(resource_${index})};`);
  }
  lines.push("  return {};", "}");
  return lines.join("\n") + "\n";
}

module.exports = nativeResources;

// CMake and opt-in probes share the exact generator; Node is a build-only prerequisite.
if (require.main === module) {
  try {
    const fs = require("node:fs"), [output, ...pairs] = process.argv.slice(2);
    if (!output || !pairs.length || pairs.length % 2) {throw new Error("Usage: native-resources.js <output> <name> <file> [<name> <file> ...]");}
    const inputs = [];
    for (let index = 0; index < pairs.length; index += 2) {
      const name = pairs[index], file = pairs[index + 1];
      if (!name || !file) {throw new Error("Incomplete resource pair");}
      inputs.push({name, data: fs.readFileSync(file)});
    }
    fs.writeFileSync(output, nativeResources(inputs));
  } catch (error) {console.error(error instanceof Error ? error.message : error); process.exitCode = 1;}
}
