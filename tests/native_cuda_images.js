"use strict";

const test = require("node:test"), assert = require("node:assert/strict");
const fs = require("node:fs"), os = require("node:os"), path = require("node:path");
const {spawnSync} = require("node:child_process");
const {fileHash} = require("../scripts/build-source-inputs");
const {cudaHeaderResources} = require("../scripts/native-jit-headers");
const {planCudaImages, generateCudaImageBundle, cudaImageInputs, cudaImageCompileCommand} = require("../scripts/native-cuda-images");

const repo = path.resolve(__dirname, "..");
/** @typedef {{major: number, minor: number, supportedArchs: number[], inputs?: Record<string, string>}} CompilerFixture */
const queries = [
  {major: 12, minor: 6, supportedArchs: [75, 80, 86, 87, 89, 90]},
  {major: 13, minor: 3, supportedArchs: [80, 86, 87, 89, 90, 100, 103, 110, 120, 121]},
];
/** @type {Record<string, string[]>} */
const exportsByProfile = {
  "cn-expand": ["cn_gpu_expand_cuda"], "pearl-rank128": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"],
  "pearl-rank256": ["pearlhash_cuda_gemm", "pearlhash_cuda_hash"], "pearl-compact": ["compact_gemm", "selector_prepare"],
};

/** @param {import("node:test").TestContext} t */
function directory(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cuda-images-"));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  return root;
}

/** @param {string} file @param {string | Buffer} data */
function write(file, data) {fs.mkdirSync(path.dirname(file), {recursive: true}); fs.writeFileSync(file, data, {flag: "wx"});}

/** @param {string} root */
function headerPack(root) {
  const cutlass = path.join(root, "sdk", "cutlass"), cuda = path.join(root, "sdk", "cuda");
  write(path.join(cutlass, "cute/tensor.hpp"), "// CUTE fixture\n");
  write(path.join(cuda, "cuda_runtime.h"), "// CUDA runtime fixture\n");
  write(path.join(cuda, "cccl/cuda/std/cstdint"), "// CCCL fixture\n");
  const file = path.join(root, "nvrtc-headers.inc"), {source, ...manifest} = cudaHeaderResources([cutlass, cuda]);
  write(file, source); write(file + ".json", JSON.stringify(manifest, null, 2) + "\n");
  return {file, roots: manifest.roots};
}

/** @param {string} root @param {{file: string}} headers @returns {CompilerFixture[]} */
function compilerQueries(root, headers) {
  return queries.map((query) => {
    /** @type {Record<string, string>} */
    const inputs = {};
    for (const name of ["worker", "nvrtc-api", "nvrtc-library", "nvrtc-builtins", "ptx-compiler", "compiler-receipt"]) {
      const file = path.join(root, `nvrtc-${query.major}.${query.minor}`, name + ".bin");
      write(file, `${query.major}.${query.minor}:${name}\n`); inputs[file] = fileHash(file);
    }
    inputs[headers.file] = fileHash(headers.file);
    inputs[headers.file + ".json"] = fileHash(headers.file + ".json");
    return {...query, inputs};
  });
}

/** @param {string} target @param {string} profile @param {string[]} includeRoots */
function options(target, profile, includeRoots) {
  const values = ["--std=c++17"];
  if (profile === "pearl-rank128" || profile === "pearl-compact") {values.push("-DPEARLHASH_RANK=128");}
  if (profile === "pearl-rank256") {values.push("-DPEARLHASH_RANK=256");}
  values.push(`--gpu-architecture=${target}`);
  for (const root of includeRoots) {values.push(`--include-path=${root}`);}
  values.push("--device-as-default-execution-space");
  if (profile !== "cn-expand") {values.push("--use_fast_math");}
  if (profile === "pearl-compact") {values.push("-DDIRECT_RANK_STRIDE=1");}
  return values;
}

/** @param {string} profile */
function source(profile) {
  return profile === "cn-expand" ? "sycl/cn_gpu/cuda_expand_kernel.inc" :
    profile === "pearl-compact" ? "sycl/pearlhash/cuda_compact_kernel.inc" : "sycl/pearlhash/cuda_kernel.inc";
}

/** @param {string} profile */
function program(profile) {return profile === "cn-expand" ? "cn-gpu-expand.cu" : "pearlhash-cute.cu";}

/** Construct the bounded ELF64/EM_CUDA sections and global function symbols used by admission fixtures.
 * @param {string[]} names @param {boolean} [nobits] @returns {Buffer} */
function cubin(names, nobits = false) {
  const strings = [Buffer.from([0])], nameOffsets = new Map();
  let stringBytes = 1;
  for (const name of names) {
    nameOffsets.set(name, stringBytes);
    const encoded = Buffer.from(name + "\0"); strings.push(encoded); stringBytes += encoded.length;
  }
  const stringTable = Buffer.concat(strings), symbols = Buffer.alloc((names.length + 1) * 24);
  for (const [index, name] of names.entries()) {
    const offset = (index + 1) * 24;
    symbols.writeUInt32LE(nameOffsets.get(name) || 0, offset);
    symbols[offset + 4] = 0x12; // STB_GLOBAL | STT_FUNC
    symbols.writeUInt16LE(1, offset + 6);
  }
  const sectionOffset = 64, sectionCount = nobits ? 5 : 4, textOffset = sectionOffset + sectionCount * 64;
  const stringsOffset = textOffset + 1, symbolsOffset = stringsOffset + stringTable.length;
  const data = Buffer.alloc(symbolsOffset + symbols.length);
  data.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1], 0);
  data.writeUInt16LE(2, 16); data.writeUInt16LE(190, 18); data.writeUInt32LE(1, 20);
  data.writeBigUInt64LE(BigInt(sectionOffset), 40); data.writeUInt16LE(64, 52);
  data.writeUInt16LE(64, 58); data.writeUInt16LE(sectionCount, 60); data.writeUInt16LE(0, 62);
  /** @param {number} index @param {number} type @param {number} offset @param {number} size
   * @param {number} [link] @param {number} [entrySize] */
  const section = (index, type, offset, size, link = 0, entrySize = 0) => {
    const at = sectionOffset + index * 64;
    data.writeUInt32LE(type, at + 4); data.writeBigUInt64LE(BigInt(offset), at + 24);
    data.writeBigUInt64LE(BigInt(size), at + 32); data.writeUInt32LE(link, at + 40);
    data.writeBigUInt64LE(BigInt(entrySize), at + 56);
  };
  section(1, 1, textOffset, 1); section(2, 3, stringsOffset, stringTable.length);
  section(3, 2, symbolsOffset, symbols.length, 2, 24);
  if (nobits) {section(4, 8, data.length + 4096, 1 << 20);}
  data[textOffset] = 0; stringTable.copy(data, stringsOffset); symbols.copy(data, symbolsOffset);
  return data;
}

/** @param {string} target @param {string[]} names @param {"8.5" | "9.3"} version @returns {Buffer} */
function ptx(target, names, version) {
  const lines = ["// NVRTC fixture", `.version ${version}`, `.target sm_${target.match(/\d+/)?.[0]}`, ".address_size 64",
    ...names.map((name) => `.visible .entry ${name}() { ret; }`)];
  return Buffer.from(lines.join("\n") + "\n\0", "ascii");
}

/** @param {ReturnType<typeof planCudaImages>} plan @param {string[]} headerRoots */
function artifactsFor(plan, headerRoots) {
  return plan.map((job) => {
    const names = exportsByProfile[job.profile] || [], roots = job.profile === "cn-expand" ? [] : headerRoots;
    const ptxVersion = /** @type {"8.5" | "9.3" | null} */ (job.format === "ptx" ? job.compiler.major === 12 ? "8.5" : "9.3" : null);
    const data = job.format === "ptx" ? ptx(job.target, names, ptxVersion || "8.5") : cubin(names);
    return {job, data, metadata: {status: "ok", nvrtc: {major: job.compiler.major, minor: job.compiler.minor},
      supportedArchs: job.compiler.supportedArchs, profile: job.profile, source: source(job.profile), program: program(job.profile),
      target: job.target, format: job.format, ptxVersion, includeRoots: roots, options: options(job.target, job.profile, roots),
      exports: names, outputBytes: data.length}};
  });
}

test.describe("Fixed-source CUDA image planner", () => {
  test("uses 12.6 exact legacy images, current modern images, and PTX for every queried SM", () => {
    const plan = planCudaImages(queries);
    for (const target of ["sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]) {
      const exact = plan.filter((job) => job.format === "cubin" && job.target === target && job.profile !== "pearl-compact");
      assert.equal(exact.length, 3); assert.ok(exact.every((job) => job.compiler.major === 12 && job.compiler.minor === 6));
    }
    for (const target of ["sm_100", "sm_103", "sm_110", "sm_120", "sm_121"]) {
      const exact = plan.filter((job) => job.format === "cubin" && job.target === target && job.profile !== "pearl-compact");
      assert.equal(exact.length, 3); assert.ok(exact.every((job) => job.compiler.major === 13 && job.compiler.minor === 3));
    }
    for (const arch of [80, 86, 87, 89, 90, 100, 103, 110, 120, 121]) {
      const primary = plan.filter((job) => job.format === "ptx" && job.target === `compute_${arch}` && job.role === "primary");
      assert.equal(primary.length, 3);
    }
    const compatibility = plan.filter((job) => job.format === "ptx" && job.role === "compatibility-candidate");
    assert.equal(compatibility.length, 3); assert.ok(compatibility.every((job) => job.target === "compute_80" && job.compiler.major === 12));
    const compact = plan.find((job) => job.profile === "pearl-compact");
    assert.equal(compact?.target, "sm_120a"); assert.equal(compact?.compiler.minor, 3);
    assert.ok(!plan.some((job) => job.target === "sm_75" || job.target === "compute_75"));
  });

  test("diagnostic planning uses only queried architectures and release admission rejects modern-only compilers", (t) => {
    const modern = [{major: 13, minor: 3, supportedArchs: [80, 120, 121]}];
    const plan = planCudaImages(modern);
    assert.equal(plan.filter((job) => job.format === "cubin").length, 10);
    assert.equal(plan.filter((job) => job.format === "ptx" && job.role === "primary").length, 9);
    assert.ok(plan.some((job) => job.target === "sm_121" && job.compiler.major === 13));
    assert.ok(!plan.some((job) => job.target === "sm_122" || job.target === "compute_122"));
    assert.equal(plan.find((job) => job.profile === "pearl-compact")?.target, "sm_120a");

    const root = directory(t), headers = headerPack(root), compiled = compilerQueries(root, headers);
    const modernOnly = compiled.filter((query) => query.major === 13);
    const modernBundle = artifactsFor(planCudaImages(modernOnly), headers.roots);
    assert.throws(() => generateCudaImageBundle({repo, compilers: modernOnly, artifacts: modernBundle,
      headerPacks: {"13.3": headers.file}}), /actual NVRTC 12\.6/);
  });

  test("requires real compute_80 support, gates compact on 13.3+, and rejects malformed queries", () => {
    assert.throws(() => planCudaImages([{major: 13, minor: 3, supportedArchs: [120]}]), /compute_80/);
    const plan = planCudaImages([{major: 13, minor: 2, supportedArchs: [80, 120]}]);
    assert.ok(!plan.some((job) => job.profile === "pearl-compact"));
    assert.throws(() => planCudaImages([{major: 13, minor: 3, supportedArchs: [80, 80]}]), /Invalid/);
    assert.throws(() => planCudaImages([queries[0], queries[0]]), /Duplicate/);
    assert.throws(() => planCudaImages(Array(17).fill(queries[0])), /bounded/);
  });
});

test.describe("Generated CUDA image registry and admission", () => {
  test("verifies true PTX/ELF exports, compiler provenance, source binding, and resource roles", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const plan = planCudaImages(compilers), artifacts = artifactsFor(plan, headers.roots);
    const result = generateCudaImageBundle({repo, compilers, artifacts, headerPacks: {"12.6": headers.file, "13.3": headers.file}});
    assert.match(result.source, /static std::string_view embedded_resource/);
    assert.match(result.source, /static constexpr mom::cuda_images::CudaImageRow kCudaImageRows\[\]/);
    assert.match(result.source, /Maintained source inputs SHA-256/);
    assert.match(result.source, /\{"pearl-rank128", 80, false, true,/);
    assert.match(result.source, /\{"pearl-rank128", 80, false, false,/);
    assert.equal(result.manifest.format, "mom-cuda-images-v1");
    assert.deepEqual(Object.keys(result.manifest.headerPacks).sort(), ["12.6", "13.3"]);
    const baselineHeaders = result.manifest.headerPacks["12.6"];
    assert.ok(baselineHeaders);
    assert.equal(baselineHeaders.manifest["sourceSha256"], fileHash(headers.file));
    assert.ok(result.manifest.resourceBytes < (64 << 20));
    const legacyPtx = result.manifest.images.find((image) => image.profile === "pearl-rank128" &&
      image.target === "compute_80" && image.compiler.major === 12);
    const currentPtx = result.manifest.images.find((image) => image.profile === "pearl-rank128" &&
      image.target === "compute_80" && image.role === "primary");
    assert.equal(legacyPtx?.ptxVersion, "8.5"); assert.equal(legacyPtx?.compatibility, true); assert.equal(legacyPtx?.embedded, true);
    assert.equal(currentPtx?.ptxVersion, "9.3"); assert.equal(currentPtx?.compiler.major, 13);
    for (const arch of [80, 86, 87, 89, 90, 100, 103, 110, 120, 121]) {
      assert.ok(result.manifest.images.some((image) => image.profile === "cn-expand" && image.target === `compute_${arch}` && image.role === "primary"));
    }
    assert.ok(result.manifest.images.some((image) => image.profile === "pearl-compact" && image.target === "sm_120a" && image.compiler.major === 13));
    assert.ok(result.manifest.compilers.every((compiler) => Object.keys(compiler.inputs).length >= 8));
    assert.ok(result.manifest.images.every((image) => image.profile === "cn-expand" ||
      (Array.isArray(image.options) && image.options.includes(`--include-path=${headers.roots[0]}`))));
    assert.equal(result.manifest.sourceSha256, require("node:crypto").createHash("sha256").update(result.source).digest("hex"));
  });

  test("CUDA string resources round-trip every byte and embedded/trailing NUL through C++", {
    skip: process.platform === "win32" ? "requires a host C++ compiler" : false,
  }, (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const plan = planCudaImages(compilers), artifacts = artifactsFor(plan, headers.roots);
    const payload = Buffer.from([...Array.from({length: 256}, (_, byte) => byte), 0x41, 0, 0x42, 0]);
    const artifact = artifacts.find(({job}) => job.profile === "cn-expand" && job.target === "sm_80" && job.format === "cubin");
    assert.ok(artifact);
    const prefixBytes = artifact.data.length;
    artifact.data = Buffer.concat([artifact.data, payload]);
    artifact.metadata.outputBytes = artifact.data.length;
    const bundle = generateCudaImageBundle({repo, compilers, artifacts, headerPacks: {"12.6": headers.file, "13.3": headers.file}});
    // Numeric byte initializers exceed the bounded compiler's memory on the real image set.
    assert.match(bundle.source, /alignas\(16\) static constexpr char resource_\d+\[\] =\n {4}"/);
    assert.doesNotMatch(bundle.source, /static (?:const|constexpr) unsigned char/);
    const record = bundle.manifest.images.find((image) => image.profile === "cn-expand" && image.target === "sm_80" && image.format === "cubin");
    assert.ok(record && record.resource);
    const registry = [
      '#include "cuda-images.h"', bundle.source,
      'extern "C" const char* fixture_resource(const char* name, std::size_t* size) {',
      "  const auto bytes = embedded_resource(name); *size = bytes.size();",
      "  (void)kCudaImageRows; return bytes.data();",
      "}",
    ].join("\n");
    const program = [
      "#include <cstddef>",
      'extern "C" const char* fixture_resource(const char*, std::size_t*);',
      "int main() {",
      "  std::size_t size = 0;",
      `  const auto bytes = fixture_resource(${JSON.stringify(record.resource)}, &size);`,
      `  if (!bytes || size != ${prefixBytes + payload.length}) return 1;`,
      `  for (std::size_t i = 0; i < 256; ++i) if (static_cast<std::size_t>(static_cast<unsigned char>(bytes[${prefixBytes} + i])) != i) return 2;`,
      `  if (bytes[${prefixBytes + 256}] != 65 || bytes[${prefixBytes + 257}] != 0 || bytes[${prefixBytes + 258}] != 66 || bytes[${prefixBytes + 259}] != 0) return 3;`,
      '  fixture_resource("unknown-resource", &size); if (size != 0) return 4;',
      "  return 0;",
      "}",
    ].join("\n");
    const sourceFile = path.join(root, "cuda-resource-roundtrip.cpp"), registryFile = path.join(root, "cuda-resource-registry.cpp"), executable = path.join(root, "cuda-resource-roundtrip");
    write(sourceFile, program); write(registryFile, registry);
    const compile = spawnSync(process.env["CXX"] || "c++", ["-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
      "-I", path.join(repo, "sycl"), registryFile, sourceFile, "-o", executable], {encoding: "utf8", timeout: 120000});
    assert.equal(compile.status, 0, compile.stderr || compile.error?.message);
    const run = spawnSync(executable, [], {encoding: "utf8", timeout: 10000});
    assert.equal(run.status, 0, run.stderr || run.error?.message);
  });

  test("rejects option drift, incomplete outputs, missing packs, and changed compiler inputs", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const plan = planCudaImages(compilers), inputs = {repo, compilers, headerPacks: {"12.6": headers.file, "13.3": headers.file}};
    const artifacts = artifactsFor(plan, headers.roots);
    const first = artifacts[0];
    assert.ok(first);
    first.metadata.options.push("--use_fast_math");
    assert.throws(() => generateCudaImageBundle({...inputs, artifacts}), /options/);
    assert.throws(() => generateCudaImageBundle({...inputs, artifacts: artifactsFor(plan, headers.roots).slice(1)}), /cover/);
    assert.throws(() => generateCudaImageBundle({...inputs, artifacts: artifactsFor(plan, headers.roots),
      headerPacks: {"12.6": path.join(root, "absent.inc"), "13.3": headers.file}}), /ENOENT/);
    const changed = compilers[0]?.inputs;
    assert.ok(changed);
    const firstFile = Object.keys(changed || {})[0];
    assert.ok(firstFile);
    fs.appendFileSync(firstFile, "changed\n");
    assert.throws(() => generateCudaImageBundle({...inputs, artifacts: artifactsFor(plan, headers.roots)}), /changed during CUDA image generation/);
  });

  test("rejects PTX with a false target and non-CUDA ELF CUBINs", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const plan = planCudaImages(compilers), input = {repo, compilers, headerPacks: {"12.6": headers.file, "13.3": headers.file}};
    const badPtx = artifactsFor(plan, headers.roots), ptxArtifact = badPtx.find((artifact) => artifact.job.format === "ptx");
    assert.ok(ptxArtifact);
    const ptxTarget = ptxArtifact.data.indexOf(Buffer.from("sm_80"));
    assert.notEqual(ptxTarget, -1);
    ptxArtifact.data[ptxTarget + 4] = "1".charCodeAt(0);
    assert.throws(() => generateCudaImageBundle({...input, artifacts: badPtx}), /PTX target/);

    const badCubin = artifactsFor(plan, headers.roots), cubinArtifact = badCubin.find((artifact) => artifact.job.format === "cubin");
    assert.ok(cubinArtifact); cubinArtifact.data.writeUInt16LE(0, 18);
    assert.throws(() => generateCudaImageBundle({...input, artifacts: badCubin}), /ELF64 little-endian EM_CUDA/);

    const newerIsa = artifactsFor(plan, headers.roots), baseline = newerIsa.find((artifact) =>
      artifact.job.format === "ptx" && artifact.job.compiler.major === 12);
    assert.ok(baseline);
    baseline.data = ptx(baseline.job.target, baseline.metadata.exports, "9.3");
    baseline.metadata.ptxVersion = "9.3";
    baseline.metadata.outputBytes = baseline.data.length;
    assert.throws(() => generateCudaImageBundle({...input, artifacts: newerIsa}), /12\.6 PTX.*8\.5/);
  });

  test("CUBIN NOBITS storage needs no file bytes, while file-backed sections stay bounded", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const input = {repo, compilers, headerPacks: {"12.6": headers.file, "13.3": headers.file}};
    const artifacts = artifactsFor(planCudaImages(compilers), headers.roots);
    const image = artifacts.find(({job}) => job.format === "cubin");
    assert.ok(image);
    image.data = cubin(image.metadata.exports, true);
    image.metadata.outputBytes = image.data.length;
    assert.doesNotThrow(() => generateCudaImageBundle({...input, artifacts}));
    // The same out-of-file extent is invalid if it claims to occupy file storage.
    image.data.writeUInt32LE(1, 64 + 4 * 64 + 4);
    assert.throws(() => generateCudaImageBundle({...input, artifacts}), /section data exceeds its bounds/);
  });

  test("admits a sidecar cross-host without opening producer compiler or SDK paths", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const bundle = generateCudaImageBundle({repo, compilers, artifacts: artifactsFor(planCudaImages(compilers), headers.roots),
      headerPacks: {"12.6": headers.file, "13.3": headers.file}});
    const inc = path.join(root, "cuda-images.inc");
    write(inc, bundle.source); write(inc + ".json", JSON.stringify(bundle.manifest, null, 2) + "\n");
    for (const compiler of compilers) {
      for (const file of Object.keys(compiler.inputs || {})) {if (file !== headers.file && file !== headers.file + ".json") {fs.unlinkSync(file);}}
    }
    fs.unlinkSync(headers.file); fs.unlinkSync(headers.file + ".json");
    const admitted = cudaImageInputs(inc);
    assert.deepEqual(Object.keys(admitted.files).sort(), [inc, inc + ".json"].sort());
    assert.equal(admitted.manifest["sourceSha256"], fileHash(inc));
    fs.appendFileSync(inc, "// changed\n");
    assert.throws(() => cudaImageInputs(inc), /Incomplete or changed/);
  });

  test("consumer admission preserves every exact image, PTX target, and compatibility role", (t) => {
    const root = directory(t), headers = headerPack(root), compilers = compilerQueries(root, headers);
    const bundle = generateCudaImageBundle({repo, compilers, artifacts: artifactsFor(planCudaImages(compilers), headers.roots),
      headerPacks: {"12.6": headers.file, "13.3": headers.file}});
    for (const target of ["compute_100", "sm_80"]) {
      const changed = structuredClone(bundle.manifest), image = changed.images.find((entry) =>
        entry.profile === "pearl-rank128" && entry.target === target);
      assert.ok(image);
      changed.resourceBytes -= image.bytes;
      image.embedded = false; image.resource = null;
      const inc = path.join(root, `${target}.inc`);
      write(inc, bundle.source); write(inc + ".json", JSON.stringify(changed));
      assert.throws(() => cudaImageInputs(inc), /sidecar entry/);
    }
    const changed = structuredClone(bundle.manifest), image = changed.images.find((entry) =>
      entry.profile === "pearl-rank128" && entry.target === "compute_90" && entry.role === "primary");
    assert.ok(image && image.resource);
    image.compatibility = true; image.resource = image.resource.replace("-primary-", "-compatibility-");
    const inc = path.join(root, "wrong-role.inc");
    write(inc, bundle.source); write(inc + ".json", JSON.stringify(changed));
    assert.throws(() => cudaImageInputs(inc), /sidecar entry/);
  });

  test("refuses malformed sidecars, stale maintained sources, and symlinked outputs", (t) => {
    const root = directory(t), inc = path.join(root, "cuda-images.inc");
    write(inc, "generated\n"); write(inc + ".json", JSON.stringify({format: "wrong"}));
    assert.throws(() => cudaImageInputs(inc), /Incomplete or changed/);
    const link = path.join(root, "link.inc"); fs.symlinkSync(inc, link);
    assert.throws(() => cudaImageInputs(link), /regular files/);
    assert.throws(() => cudaImageInputs("relative.inc"), /absolute/);

    const headers = headerPack(root), compilers = compilerQueries(root, headers);
    const bundle = generateCudaImageBundle({repo, compilers, artifacts: artifactsFor(planCudaImages(compilers), headers.roots),
      headerPacks: {"12.6": headers.file, "13.3": headers.file}});
    const stale = path.join(root, "stale.inc");
    write(stale, bundle.source);
    bundle.manifest.sources["sycl/cn_gpu/cuda_expand_kernel.inc"] = "0".repeat(64);
    write(stale + ".json", JSON.stringify(bundle.manifest));
    assert.throws(() => cudaImageInputs(stale), /Incomplete or changed/);
  });
});

test.describe("CUDA registry compile cache contract", () => {
  test("uses normal depfiles and explicitly tracks the generated image include", () => {
    const root = path.join(path.sep, "tmp", "fixture");
    const linux = cudaImageCompileCommand({repo, output: root, compiler: "/opt/dpcpp/bin/clang++", flags: ["-std=c++20", "-O2"]});
    assert.equal(linux.source, path.join(repo, "sycl/cuda-images.cpp"));
    assert.equal(linux.object, path.join(root, "cuda-images.o"));
    assert.ok(linux.args.includes("-MD")); assert.ok(linux.args.includes("-MF"));
    assert.ok(!linux.args.includes("-MV"));
    assert.deepEqual(linux.generatedInputs, [path.join(root, "cuda-images.inc")]);
    const windows = cudaImageCompileCommand({repo, output: root, compiler: "clang-cl", flags: ["-std=c++20"], windows: true});
    assert.equal(windows.object, path.join(root, "cuda-images.obj"));
    assert.ok(windows.args.includes("-MV")); assert.equal(windows.dependencyFormat, "nmake");
  });

  test("worker source uses only public NVRTC APIs, filesystem roots, and maintained kernel literals", () => {
    const worker = fs.readFileSync(path.join(repo, "scripts/native-cuda-images.cpp"), "utf8");
    for (const include of ["../sycl/cn_gpu/cuda_expand_kernel.inc", "../sycl/pearlhash/cuda_kernel.inc", "../sycl/pearlhash/cuda_compact_kernel.inc"]) {
      assert.ok(worker.includes(include));
    }
    for (const api of ["nvrtcVersion", "nvrtcGetNumSupportedArchs", "nvrtcGetSupportedArchs", "nvrtcCreateProgram",
      "nvrtcCompileProgram", "nvrtcGetPTX", "nvrtcGetCUBIN"]) {assert.ok(worker.includes(api), api);}
    assert.doesNotMatch(worker, /\b(?:cuInit|cuModuleLoad|cudaGetDevice|cudaSetDevice)\s*\(/);
    assert.match(worker, /--include-path=/); assert.match(worker, /--include <absolute-directory>/);
    assert.match(worker, /--use_fast_math/); assert.match(worker, /-DDIRECT_RANK_STRIDE=1/);
  });
});
