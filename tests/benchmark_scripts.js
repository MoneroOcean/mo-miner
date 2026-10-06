"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
/** @param {string} file */
function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

test("standalone installers authenticate downloaded helper scripts", () => {
  const scripts = path.join(__dirname, "..", "scripts");
  const shellHash = sha256(path.join(scripts, "install-cutlass.sh"));
  const shellInstaller = fs.readFileSync(path.join(scripts, "install.sh"), "utf8");
  assert.match(shellInstaller, new RegExp(`MOM_CUTLASS_HELPER_SHA256=${shellHash}`));
  assert.match(shellInstaller, /sha256sum -c -/);
  assert.match(shellInstaller, /trap "rm -f -- '\$cutlass_helper'" EXIT/);
  assert.match(shellInstaller, /mo-miner\/v0\.9\.0\/scripts\/install-cutlass\.sh/);
  assert.doesNotMatch(shellInstaller, /mo-miner\/master\/scripts\/install-cutlass\.sh/);

  const powershellHash = sha256(path.join(scripts, "install-cutlass.ps1"));
  const powershellInstaller = fs.readFileSync(path.join(scripts, "install.ps1"), "utf8");
  assert.match(powershellInstaller,
    new RegExp(`cutlassHelperSha256 = "${powershellHash}"`, "i"));
  assert.match(powershellInstaller, /Get-FileHash -Algorithm SHA256 \$helper/);
  assert.match(powershellInstaller, /mo-miner\/v0\.9\.0\/scripts\/install-cutlass\.ps1/);
  assert.doesNotMatch(powershellInstaller, /mo-miner\/master\/scripts\/install-cutlass\.ps1/);
  assert.match(powershellInstaller, /function Assert-Authenticode/);
  assert.match(powershellInstaller,
    /Invoke-WebRequest[^\n]*\$fallbackVsBuildToolsUrl[\s\S]*?Assert-Authenticode \$vsInstaller/);
  assert.match(powershellInstaller,
    /Invoke-WebRequest[^\n]*\$url[\s\S]*?Assert-Authenticode \$cudaInstaller/);
  assert.match(powershellInstaller,
    /Invoke-WebRequest[^\n]*\$openClCpuUrl[\s\S]*?Assert-Authenticode \$installer/);
  assert.match(powershellInstaller,
    /Get-FileHash -Algorithm SHA256 \$sevenZipMsi[\s\S]*?Install-MsiPackage \$sevenZipMsi/);
  assert.match(powershellInstaller,
    /if \(\$DryRun\)[\s\S]*?exit 0\s*}[\s\S]*?Install-OpenClCpuRuntime/);

  const oneApiInstaller = fs.readFileSync(path.join(__dirname, "..", ".github", "workflows",
    "scripts", "install-oneapi.bat"), "utf8");
  assert.match(oneApiInstaller, /\[Guid\]::NewGuid\(\)\.ToString\(\)/);
  assert.match(oneApiInstaller, /%TEMP%\\mom-oneapi-%RUN_ID%/);
  assert.doesNotMatch(oneApiInstaller, /oneapi_webimage_extracted/);
});

test("host installer preserves an explicit GPU vendor selection across sudo", {
  skip: process.platform === "win32",
}, () => {
  const installer = path.join(__dirname, "..", "scripts", "install.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-sudo-test-"));
  const bin = path.join(temp, "bin");
  const argsPath = path.join(temp, "sudo-args");
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, "id"), [
    "#!/bin/sh",
    "if [ \"$1\" = -u ]; then printf '%s\\n' 1000; else exit 1; fi",
  ].join("\n"));
  fs.writeFileSync(path.join(bin, "sudo"), [
    "#!/bin/sh",
    "printf '%s\\n' \"$@\" > \"$MOM_SUDO_ARGS\"",
    "exit 77",
  ].join("\n"));
  fs.chmodSync(path.join(bin, "id"), 0o755);
  fs.chmodSync(path.join(bin, "sudo"), 0o755);
  /** @param {Record<string, string>} configured */
  const run = (configured) => {
    fs.rmSync(argsPath, {force: true});
    /** @type {TestEnvironment} */
    const env = {...process.env, MOM_SUDO_ARGS: argsPath,
      PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`};
    for (const name of ["MOM_INSTALL_GPU_VENDORS", "ROCM_PATH", "HIP_PATH"]) {
      delete env[name];
    }
    Object.assign(env, configured);
    const result = spawnSync("bash", [installer, "--fixture-argument"], {
      encoding: "utf8", env,
    });
    assert.equal(result.status, 77, `${result.stdout}\n${result.stderr}`);
    return fs.readFileSync(argsPath, "utf8").trim().split(/\r?\n/);
  };
  try {
    assert.deepEqual(run({MOM_INSTALL_GPU_VENDORS: "nvidia"}), [
      "--preserve-env=MOM_INSTALL_GPU_VENDORS", "--", installer, "--fixture-argument",
    ]);
    assert.deepEqual(run({ROCM_PATH: "/fixture/rocm"}), [
      "--preserve-env=ROCM_PATH", "--", installer, "--fixture-argument",
    ]);
    assert.deepEqual(run({HIP_PATH: "/fixture/hip"}), [
      "--preserve-env=HIP_PATH", "--", installer, "--fixture-argument",
    ]);
    assert.deepEqual(run({
      MOM_INSTALL_GPU_VENDORS: "nvidia", ROCM_PATH: "/fixture/rocm", HIP_PATH: "/fixture/hip",
    }), [
      "--preserve-env=MOM_INSTALL_GPU_VENDORS,ROCM_PATH,HIP_PATH", "--", installer,
      "--fixture-argument",
    ]);
    assert.deepEqual(run({}), ["--", installer, "--fixture-argument"]);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("host installer adds the sudo user only to missing GPU access groups", {
  skip: process.platform === "win32",
}, () => {
  const installer = path.join(__dirname, "..", "scripts", "install.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-groups-test-"));
  const bin = path.join(temp, "bin");
  const usermodLog = path.join(temp, "usermod.log");
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, "id"), [
    "#!/bin/sh",
    "if [ \"$1\" = -u ] && [ \"$#\" = 1 ]; then printf '%s\\n' 0; exit 0; fi",
    "if [ \"$1\" = -u ] && [ \"$3\" = fixture-user ]; then printf '%s\\n' 1000; exit 0; fi",
    "if [ \"$1\" = -nG ] && [ \"$3\" = fixture-user ]; then printf '%s\\n' \"$MOM_FIXTURE_GROUPS\"; exit 0; fi",
    "exit 1",
  ].join("\n"), {mode: 0o755});
  fs.writeFileSync(path.join(bin, "getent"), [
    "#!/bin/sh",
    "[ \"$1\" = group ] && { [ \"$2\" = render ] || [ \"$2\" = video ]; }",
  ].join("\n"), {mode: 0o755});
  fs.writeFileSync(path.join(bin, "usermod"), [
    "#!/bin/sh",
    "printf '%s\\n' \"$*\" > \"$MOM_USERMOD_LOG\"",
  ].join("\n"), {mode: 0o755});
  for (const command of ["apt-get", "ldconfig"]) {
    fs.writeFileSync(path.join(bin, command), "#!/bin/sh\nexit 0\n", {mode: 0o755});
  }
  /** @type {TestEnvironment} */
  const env = {
    ...process.env,
    MOM_INSTALL_GPU_VENDORS: "opencl",
    MOM_FIXTURE_GROUPS: "fixture-user video",
    MOM_USERMOD_LOG: usermodLog,
    SUDO_UID: "1000",
    SUDO_USER: "fixture-user",
    PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
  };
  try {
    const result = spawnSync("bash", [installer], {encoding: "utf8", env});
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    assert.equal(fs.readFileSync(usermodLog, "utf8").trim(), "-aG render -- fixture-user");
    assert.match(result.stdout, /Added fixture-user to the GPU access group\(s\): render/);
    assert.match(result.stdout, /Sign out and back in/);

    fs.rmSync(usermodLog);
    const complete = spawnSync("bash", [installer], {encoding: "utf8", env: {
      ...env, MOM_FIXTURE_GROUPS: "fixture-user render video",
    }});
    assert.equal(complete.status, 0, `${complete.stdout}\n${complete.stderr}`);
    assert.equal(fs.existsSync(usermodLog), false);
    assert.doesNotMatch(complete.stdout, /Added fixture-user/);

    const directRootEnv = {...env};
    delete directRootEnv["SUDO_UID"];
    delete directRootEnv["SUDO_USER"];
    const directRoot = spawnSync("bash", [installer], {encoding: "utf8", env: directRootEnv});
    assert.equal(directRoot.status, 0, `${directRoot.stdout}\n${directRoot.stderr}`);
    assert.equal(fs.existsSync(usermodLog), false);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("host installer does not add distro ROCm beside a coherent vendor core", {
  skip: process.platform === "win32",
}, () => {
  const sourceInstaller = path.join(__dirname, "..", "scripts", "install.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-rocm-test-"));
  const installer = path.join(temp, "install.sh");
  const bin = path.join(temp, "bin");
  const aptLog = path.join(temp, "apt.log");
  const rocm = path.join(temp, "rocm", "core-7.14");
  const source = fs.readFileSync(sourceInstaller, "utf8");
  const defaultRoot = "  roots+=(/opt/rocm)";
  assert.equal(source.split(defaultRoot).length, 2);
  // Build containers can own a real /opt/rocm. Keep both fixture branches independent of it.
  fs.writeFileSync(installer, source.replace(defaultRoot,
    `  roots+=(${JSON.stringify(path.join(temp, "system-rocm"))})`));
  fs.mkdirSync(bin, {recursive: true});
  fs.mkdirSync(path.join(rocm, "bin"), {recursive: true});
  fs.mkdirSync(path.join(rocm, "lib"));
  fs.writeFileSync(path.join(rocm, "bin", "hipconfig"), "#!/bin/sh\n", {mode: 0o755});
  fs.symlinkSync(path.join(rocm, "bin", "hipconfig"), path.join(bin, "hipconfig"));
  for (const library of [
    "libamdhip64.so.7", "libamd_comgr.so.3", "libhsa-runtime64.so.1",
    "libhiprtc.so.7", "libhiprtc-builtins.so.7",
  ]) {
    fs.writeFileSync(path.join(rocm, "lib", library), "test");
  }
  fs.writeFileSync(path.join(bin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n", {mode: 0o755});
  fs.writeFileSync(path.join(bin, "apt-get"), [
    "#!/bin/sh",
    "printf '%s\\n' \"$*\" >> \"$MOM_APT_LOG\"",
  ].join("\n"), {mode: 0o755});
  for (const command of ["dpkg-query", "ldconfig"]) {
    fs.writeFileSync(path.join(bin, command), "#!/bin/sh\nexit 0\n", {mode: 0o755});
  }
  /** @type {NodeJS.ProcessEnv} */
  const env = {
    ...process.env,
    MOM_APT_LOG: aptLog,
    MOM_INSTALL_GPU_VENDORS: "amd",
    PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
  };
  delete env["ROCM_PATH"];
  delete env["HIP_PATH"];
  try {
    const preserved = spawnSync("bash", [installer], {encoding: "utf8", env});
    assert.equal(preserved.status, 0, `${preserved.stdout}\n${preserved.stderr}`);
    assert.match(preserved.stdout, /keeping the coherent ROCm core .*core-7\.14\/lib/);
    const preservedApt = fs.readFileSync(aptLog, "utf8");
    assert.match(preservedApt, /install -y --no-install-recommends libomp5/);
    assert.doesNotMatch(preservedApt, /libamdhip64/);

    fs.rmSync(path.join(rocm, "lib", "libhiprtc-builtins.so.7"));
    fs.rmSync(aptLog);
    const installed = spawnSync("bash", [installer], {encoding: "utf8", env});
    assert.equal(installed.status, 0, `${installed.stdout}\n${installed.stderr}`);
    assert.match(fs.readFileSync(aptLog, "utf8"),
      /libamdhip64-7 libamdhip64-dev libhiprtc7 libhiprtc-builtins7/);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("host installer preserves complete CUDA roots and rejects unsafe roots before download", {
  skip: process.platform === "win32",
}, () => {
  const sourceInstaller = path.join(__dirname, "..", "scripts", "install.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-install-cuda-test-"));
  const installer = path.join(temp, "install.sh");
  const bin = path.join(temp, "bin");
  const aptLog = path.join(temp, "apt.log");
  const cudaRoot = path.join(temp, "usr-local-cuda");
  const cudaTarget = path.join(temp, "cuda-target");
  const payloadRoot = path.join(temp, "nvidia-cuda-ubuntu");
  const ldConfig = path.join(temp, "mom-cuda.conf");
  const source = fs.readFileSync(sourceInstaller, "utf8");
  /** @type {Array<[string, string]>} */
  const rewrites = [
    ["/usr/local/cuda", cudaRoot],
    ["/opt/nvidia-cuda-ubuntu", payloadRoot],
    ["/etc/ld.so.conf.d/mom-cuda.conf", ldConfig],
  ];
  let isolated = source;
  for (const [realPath, fixturePath] of rewrites) {
    assert.ok(source.includes(realPath), `source path missing: ${realPath}`);
    isolated = isolated.replaceAll(realPath, fixturePath);
    assert.equal(isolated.includes(realPath), false, `real path leaked: ${realPath}`);
  }
  fs.writeFileSync(installer, isolated, {mode: 0o755});
  fs.writeFileSync(path.join(temp, "install-cutlass.sh"),
    "install_cutlass_headers() { :; }\n");
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n", {mode: 0o755});
  fs.writeFileSync(path.join(bin, "ldconfig"), [
    "#!/bin/sh",
    "if [ \"$1\" = -p ]; then printf '%s\\n' libcuda.so.1; fi",
  ].join("\n"), {mode: 0o755});
  fs.writeFileSync(path.join(bin, "apt-cache"), [
    "#!/bin/sh",
    "printf '%s\\n' \"apt-cache $*\" >> \"$MOM_APT_LOG\"",
    "exit 0",
  ].join("\n"), {mode: 0o755});
  fs.writeFileSync(path.join(bin, "apt-get"), [
    "#!/bin/sh",
    "printf '%s\\n' \"$*\" >> \"$MOM_APT_LOG\"",
    "for arg in \"$@\"; do",
    "  if [ \"$arg\" = download ]; then exit 23; fi",
    "done",
    "exit 0",
  ].join("\n"), {mode: 0o755});
  /** @type {TestEnvironment} */
  const env = {
    ...process.env,
    MOM_APT_LOG: aptLog,
    MOM_INSTALL_GPU_VENDORS: "nvidia",
    PATH: `${bin}${path.delimiter}/usr/bin${path.delimiter}/bin`,
  };
  const run = () => spawnSync("bash", [installer], {encoding: "utf8", env});
  const resetLog = () => fs.rmSync(aptLog, {force: true});
  const assertNoToolkitDownload = () => {
    const log = fs.readFileSync(aptLog, "utf8");
    assert.doesNotMatch(log, /\bdownload\b/);
    assert.doesNotMatch(log, /apt-cache/);
  };
  const ptxas = path.join(cudaTarget, "bin", "ptxas");
  const include = path.join(cudaTarget, "include");
  /** @type {[string, string, string, string]} */
  const headers = [
    path.join(include, "cuda.h"),
    path.join(include, "cuda_runtime.h"),
    path.join(include, "nvrtc.h"),
    path.join(include, "cuda", "std", "cstdint"),
  ];
  const libdevice = path.join(cudaTarget, "nvvm", "libdevice", "libdevice.10.bc");
  /** @type {[string, string]} */
  const libraries = [
    path.join(cudaTarget, "lib64", "libnvrtc.so.12"),
    path.join(cudaTarget, "lib64", "libnvrtc-builtins.so.12"),
  ];
  try {
    fs.mkdirSync(path.dirname(ptxas), {recursive: true});
    fs.mkdirSync(include, {recursive: true});
    fs.mkdirSync(path.dirname(headers[3]), {recursive: true});
    fs.mkdirSync(path.dirname(libdevice), {recursive: true});
    fs.mkdirSync(path.dirname(libraries[0]), {recursive: true});
    fs.writeFileSync(ptxas, "ptxas fixture");
    fs.chmodSync(ptxas, 0o755);
    for (const header of headers) {
      fs.writeFileSync(header, "CUDA fixture");
    }
    fs.writeFileSync(libdevice, "libdevice fixture");
    for (const library of libraries) {
      fs.writeFileSync(library, "NVRTC fixture");
    }
    fs.symlinkSync(cudaTarget, cudaRoot, "dir");
    const linkTarget = fs.readlinkSync(cudaRoot);
    /** @type {Map<string, Buffer>} */
    const sdkBytes = new Map();
    for (const file of [ptxas, ...headers, libdevice, ...libraries]) {
      sdkBytes.set(file, fs.readFileSync(file));
    }

    const complete = run();
    assert.equal(complete.status, 0, `${complete.stdout}\n${complete.stderr}`);
    assert.match(complete.stdout, /Keeping the complete CUDA source-JIT toolkit/);
    assert.doesNotMatch(complete.stdout, /Installing the NVIDIA source-JIT toolchain/);
    assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
    assert.equal(fs.readlinkSync(cudaRoot), linkTarget);
    for (const [file, bytes] of sdkBytes) {
      assert.deepEqual(fs.readFileSync(file), bytes);
    }
    assert.equal(fs.existsSync(payloadRoot), false);
    assert.equal(fs.existsSync(ldConfig), false);
    assertNoToolkitDownload();

    fs.rmSync(headers[2]);
    resetLog();
    const missing = run();
    assert.notEqual(missing.status, 0, `${missing.stdout}\n${missing.stderr}`);
    assert.match(missing.stderr, /repair the existing CUDA installation/);
    assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
    assert.equal(fs.readlinkSync(cudaRoot), linkTarget);
    assert.equal(fs.existsSync(payloadRoot), false);
    assertNoToolkitDownload();

    fs.writeFileSync(headers[2], "CUDA fixture");
    fs.rmSync(cudaRoot);
    const danglingTarget = path.join(temp, "missing-cuda-target");
    fs.symlinkSync(danglingTarget, cudaRoot, "dir");
    resetLog();
    const dangling = run();
    assert.notEqual(dangling.status, 0, `${dangling.stdout}\n${dangling.stderr}`);
    assert.match(dangling.stderr, /repair the existing CUDA installation/);
    assert.equal(fs.lstatSync(cudaRoot).isSymbolicLink(), true);
    assert.equal(fs.readlinkSync(cudaRoot), danglingTarget);
    assert.equal(fs.existsSync(payloadRoot), false);
    assertNoToolkitDownload();

    fs.rmSync(cudaRoot);
    resetLog();
    const absent = run();
    assert.notEqual(absent.status, 0, `${absent.stdout}\n${absent.stderr}`);
    assert.match(absent.stdout, /Installing the NVIDIA source-JIT toolchain/);
    assert.match(fs.readFileSync(aptLog, "utf8"), /\bdownload\b/);
    assert.equal(fs.existsSync(cudaRoot), false);
    assert.equal(fs.existsSync(payloadRoot), false);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("CUTLASS shell installer rejects files and symlinks before download", {
  skip: process.platform === "win32",
}, () => {
  const script = path.join(__dirname, "..", "scripts", "install-cutlass.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-cutlass-safety-test-"));
  const run = (/** @type {string} */ destination, /** @type {string} */ curlCalled) => spawnSync("bash", [
    "-c",
    'source "$1"; curl() { : > "$MOM_CURL_CALLED"; return 97; }; install_cutlass_headers "$2"',
    "mom-cutlass-safety", script, destination,
  ], {encoding: "utf8", env: {...process.env, MOM_CURL_CALLED: curlCalled}});
  try {
    const curlCalled = path.join(temp, "curl-called");
    const file = path.join(temp, "destination-file");
    fs.writeFileSync(file, "sentinel");
    const fileResult = run(file, curlCalled);
    assert.notEqual(fileResult.status, 0);
    assert.match(fileResult.stderr, /not a directory/);
    assert.equal(fs.readFileSync(file, "utf8"), "sentinel");
    assert.equal(fs.existsSync(curlCalled), false);

    const target = path.join(temp, "target");
    const link = path.join(temp, "destination-link");
    fs.mkdirSync(target);
    fs.writeFileSync(path.join(target, "sentinel"), "sentinel");
    fs.symlinkSync(target, link, "dir");
    const linkResult = run(link, curlCalled);
    assert.notEqual(linkResult.status, 0);
    assert.match(linkResult.stderr, /destination symlink/);
    assert.equal(fs.readFileSync(path.join(target, "sentinel"), "utf8"), "sentinel");
    assert.equal(fs.existsSync(curlCalled), false);

    const destination = path.join(temp, "destination");
    const marker = path.join(destination, ".mom-version");
    const sentinel = path.join(temp, "marker-sentinel");
    fs.mkdirSync(destination);
    fs.writeFileSync(sentinel, "keep");
    fs.symlinkSync(sentinel, marker, "file");
    const markerResult = run(destination, curlCalled);
    assert.notEqual(markerResult.status, 0);
    assert.match(markerResult.stderr, /version marker symlink/);
    assert.equal(fs.existsSync(curlCalled), false);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});

test("development installer authenticates artifacts at their acquisition boundary", () => {
  const installer = fs.readFileSync(path.join(__dirname, "..", "scripts", "install-dev.ps1"),
    "utf8");
  const helpers = fs.readFileSync(path.join(__dirname, "..", "scripts",
    "windows-install-helpers.ps1"), "utf8");
  const adaptiveCpp = fs.readFileSync(path.join(__dirname, "..", "scripts",
    "build-windows-adaptivecpp-base.ps1"), "utf8");
  const installMsi = installer.match(/function Install-Msi\([\s\S]*?\n}/)?.[0] ?? "";
  assert.doesNotMatch(installMsi, /Assert-Authenticode/);
  assert.match(installer,
    /Download 'https:\/\/www\.7-zip\.org\/a\/7z2409-x64\.msi' \$sevenZip\s+Assert-Sha256 \$sevenZip \$sevenZipSha256\s+Install-Msi \$sevenZip/);
  assert.match(installer,
    /Download \$hipSdkUrl \$sdkExe\s+Assert-Sha256 \$sdkExe \$hipSdkSha256\s+\$sevenZip/);
  assert.match(adaptiveCpp,
    /Download \$bootstrapUrl \$bootstrapInstaller }\s+Assert-Sha256 \$bootstrapInstaller \$bootstrapSha256\s+Remove-Item \$bootstrapDir/);
  assert.match(installer,
    /Download 'https:\/\/aka\.ms\/vs\/17\/release\/vs_BuildTools\.exe' \$installer\s+Assert-Authenticode \$installer/);
  assert.match(installer,
    /Download \$openClCpuUrl \$installer\s+Assert-Authenticode \$installer/);
  assert.match(installer,
    /Download \$cudaUrl \$installer\s+Assert-Authenticode \$installer/);
  assert.match(helpers,
    /'--retry','5','--retry-all-errors','--retry-delay','5',[\s\S]*?'--continue-at','-','-o',\$OutFile,\$Url/);
  assert.match(installer,
    /if \(-not \$KeepWorkspace\) \{\s+#[^\n]+\n\s+#[^\n]+\n\s+Set-Location -LiteralPath \$repo\s+Assert-NoReparseAncestor \$Workspace/);
});

test("development installers clean only dedicated workspace names", () => {
  const scripts = path.join(__dirname, "..", "scripts");
  const linux = path.join(scripts, "install-dev.sh");
  for (const workspace of ["/", "/tmp/compiler-workspace"]) {
    const result = spawnSync("bash", [linux, "--component", "base", "--validate-only",
      "--workspace", workspace], {encoding: "utf8"});
    assert.equal(result.status, 2, result.stderr);
    assert.match(result.stderr, /dedicated mom-dev-/);
  }
  const validationParent = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dev-validation-test-"));
  const validationWorkspace = path.join(validationParent, "mom-dev-validation");
  const validationBin = path.join(validationParent, "bin");
  fs.mkdirSync(validationBin);
  fs.writeFileSync(path.join(validationBin, "id"), "#!/bin/sh\nprintf '%s\\n' 0\n");
  fs.chmodSync(path.join(validationBin, "id"), 0o755);
  const validationEnv = {
    ...process.env,
    PATH: `${validationBin}${path.delimiter}${process.env["PATH"] ?? ""}`,
  };
  try {
    const validation = spawnSync("bash", [linux, "--component", "base", "--validate-only",
      "--workspace", validationWorkspace], {encoding: "utf8", env: validationEnv});
    assert.equal(validation.error, undefined, validation.error?.message);
    assert.equal(validation.status, 0, validation.stderr);
    assert.equal(fs.existsSync(validationWorkspace), false);

    const wrongMarkerWorkspace = path.join(validationParent, "mom-dev-wrong-marker");
    const wrongMarker = path.join(wrongMarkerWorkspace, ".mom-dev-workspace");
    const sentinel = path.join(wrongMarkerWorkspace, "sentinel");
    fs.mkdirSync(wrongMarkerWorkspace);
    fs.writeFileSync(wrongMarker, "wrong marker");
    fs.writeFileSync(sentinel, "keep");
    const wrongMarkerResult = spawnSync("bash", [linux, "--component", "base",
      "--workspace", wrongMarkerWorkspace], {encoding: "utf8", env: validationEnv});
    assert.equal(wrongMarkerResult.status, 2, wrongMarkerResult.stderr);
    assert.match(wrongMarkerResult.stderr, /\.mom-dev-workspace marker/);
    assert.equal(fs.readFileSync(sentinel, "utf8"), "keep");
  } finally {
    fs.rmSync(validationParent, {recursive: true, force: true});
  }
  const windows = fs.readFileSync(path.join(scripts, "install-dev.ps1"), "utf8");
  assert.match(windows, /\$workspaceLeaf -notmatch '\^mom-dev-/);
  assert.match(windows, /\$workspaceMarker = Join-Path \$Workspace '\.mom-dev-workspace'/);
  assert.match(windows, /\[IO\.FileAttributes\]::ReparsePoint/);
  assert.match(windows, /Test-MomMarker \$workspaceMarker 'mom development workspace'/);
  assert.match(windows, /Remove-Item -LiteralPath \$Workspace -Recurse/);
});

test("development installer rejects invalid configuration before privilege escalation", {
  skip: process.platform === "win32",
}, () => {
  const linux = path.join(__dirname, "..", "scripts", "install-dev.sh");
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "mom-dev-config-test-"));
  const bin = path.join(temp, "bin");
  const activity = path.join(temp, "activity");
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, "id"), [
    "#!/bin/sh",
    "printf '%s\\n' id >> \"$MOM_INSTALLER_ACTIVITY\"",
    "printf '%s\\n' 1000",
  ].join("\n"));
  fs.writeFileSync(path.join(bin, "sudo"), [
    "#!/bin/sh",
    "printf '%s\\n' sudo >> \"$MOM_INSTALLER_ACTIVITY\"",
    "exit 99",
  ].join("\n"));
  fs.chmodSync(path.join(bin, "id"), 0o755);
  fs.chmodSync(path.join(bin, "sudo"), 0o755);

  const valid = {
    MOM_NODE_VERSION: "24.15.0",
    MOM_NODE_SHA256: "0".repeat(64),
    MOM_DPCPP_RELEASE: "nightly-2026-07-11",
    MOM_DPCPP_ASSET: "sycl_linux.tar.gz",
    MOM_DPCPP_SHA256: "0".repeat(64),
    MOM_ADAPTIVECPP_COMMIT: "0".repeat(40),
    MOM_CUDA_VERSION: "12-6",
    MOM_ROCM_VERSION: "7.1.1",
    MOM_LLVM_VERSION: "21",
    MOM_BUILD_JOBS: "1",
  };
  /** @type {[string, string][]} */
  const invalid = [
    ["MOM_NODE_VERSION", "bad"],
    ["MOM_NODE_SHA256", "0".repeat(63)],
    ["MOM_DPCPP_RELEASE", ""],
    ["MOM_DPCPP_ASSET", "asset/name"],
    ["MOM_DPCPP_SHA256", "0".repeat(63)],
    ["MOM_ADAPTIVECPP_COMMIT", "0".repeat(39)],
    ["MOM_CUDA_VERSION", "12.6"],
    ["MOM_ROCM_VERSION", "7.1/evil"],
    ["MOM_LLVM_VERSION", "0"],
    ["MOM_BUILD_JOBS", "0"],
  ];
  try {
    for (const [name, value] of invalid) {
      fs.rmSync(activity, {force: true});
      const result = spawnSync("bash", [linux, "--component", "base", "--validate-only"], {
        encoding: "utf8",
        env: {
          ...process.env,
          ...valid,
          [name]: value,
          MOM_INSTALLER_ACTIVITY: activity,
          PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`,
        },
      });
      assert.equal(result.status, 2, `${name}: ${result.stderr}`);
      assert.equal(result.stderr.trim().split(/\r?\n/).length, 1, result.stderr);
      assert.match(result.stderr, new RegExp(name));
      assert.equal(fs.existsSync(activity), false, `${name} triggered activity`);
    }
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
});
