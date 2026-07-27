#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const SPIRV_MAGIC = 0x07230203;
const OP_EXTENSION = 10;
const OP_CAPABILITY = 17;
const OP_DECORATE = 71;
const DECORATION_LINKAGE_ATTRIBUTES = 41;
const CAPABILITY_GENERIC_POINTER = 38;
const LINKAGE_IMPORT = 1;

function stringOperand(words, start, end) {
  const bytes = [];
  let index = start;
  for (; index < end; ++index) {
    const word = words[index];
    for (let shift = 0; shift < 32; shift += 8) {
      const byte = (word >>> shift) & 0xff;
      if (byte === 0) {return {text: Buffer.from(bytes).toString("utf8"), next: index + 1};}
      bytes.push(byte);
    }
  }
  throw new Error("unterminated SPIR-V string operand");
}

function validateModule(filePath) {
  const data = fs.readFileSync(filePath);
  if (data.length < 20 || data.length % 4 !== 0) {
    return {errors: ["not a complete SPIR-V word stream"], warnings: []};
  }
  const words = new Uint32Array(data.buffer, data.byteOffset, data.length / 4);
  if (words[0] !== SPIRV_MAGIC) {return {errors: ["invalid SPIR-V magic"], warnings: []};}

  const errors = [];
  const warnings = [];
  for (let cursor = 5; cursor < words.length;) {
    const instruction = words[cursor];
    const wordCount = instruction >>> 16;
    const opcode = instruction & 0xffff;
    if (wordCount === 0 || cursor + wordCount > words.length) {
      errors.push(`malformed instruction at word ${cursor}`);
      break;
    }
    const end = cursor + wordCount;
    if (opcode === OP_EXTENSION) {
      const extension = stringOperand(words, cursor + 1, end).text;
      if (/^SPV_(?:INTEL|NV|AMD)_/.test(extension)) {
        errors.push(`vendor SPIR-V extension ${extension}`);
      }
    } else if (opcode === OP_CAPABILITY && words[cursor + 1] === CAPABILITY_GENERIC_POINTER) {
      warnings.push("GenericPointer capability (not accepted by every ICD; runtime compatibility test required)");
    } else if (opcode === OP_DECORATE && words[cursor + 2] === DECORATION_LINKAGE_ATTRIBUTES) {
      const linkage = stringOperand(words, cursor + 3, end);
      if (linkage.next < end && words[linkage.next] === LINKAGE_IMPORT && /^llvm\./.test(linkage.text)) {
        warnings.push(`LLVM intrinsic import ${linkage.text} (covered by runtime compatibility tests)`);
      }
    }
    cursor = end;
  }
  return {errors, warnings};
}

function validateDirectory(directory) {
  const files = fs.readdirSync(directory)
    .filter((name) => name.endsWith(".spv"))
    .sort((a, b) => a.localeCompare(b, undefined, {numeric: true}));
  if (!files.length) {return {files, errors: ["no dumped .spv images found"], warnings: []};}

  const errors = [];
  const warningSet = new Set();
  for (const file of files) {
    const result = validateModule(path.join(directory, file));
    errors.push(...result.errors.map((message) => `${file}: ${message}`));
    for (const warning of result.warnings) {warningSet.add(warning);}
  }
  return {files, errors, warnings: [...warningSet]};
}

function main() {
  const directory = process.argv[2];
  if (!directory) {
    console.error("Usage: validate-portable-opencl.js DUMP_DIRECTORY");
    process.exit(2);
  }
  const report = validateDirectory(directory);
  for (const warning of report.warnings) {console.warn(`Portable OpenCL warning: ${warning}`);}
  if (report.errors.length) {
    for (const error of report.errors) {console.error(`Portable OpenCL error: ${error}`);}
    process.exit(1);
  }
  console.log(`Validated ${report.files.length} portable OpenCL SPIR-V images`);
}

if (require.main === module) {main();}

module.exports = {validateDirectory, validateModule};
