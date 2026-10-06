"use strict";

const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {createHash} = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const repo = path.resolve(__dirname, "..");
const PACKAGE_SCRIPT = path.join(repo, ".github", "workflows", "scripts",
  "package-linux-combined.sh");
const LINUX_RELEASE_TEST_SCRIPT = path.join(repo, ".github", "workflows", "scripts",
  "test-release-linux.sh");
const WINDOWS_RELEASE_TEST_SCRIPT = path.join(repo, ".github", "workflows", "scripts",
  "test-release-windows.ps1");
const RELEASE_WORKFLOW = path.join(repo, ".github", "workflows", "build-release-artifacts.yml");
const PUBLISH_WORKFLOW = path.join(repo, ".github", "workflows", "release.yml");
const RELEASE_ROOT = `mom-v${require(path.join(repo, "package.json")).version}`;
const KAWPOW_DEVICE_MEMBER = "libs/dpcpp/kawpow_device.inc";
const KAWPOW_KECCAK_MEMBER = "libs/dpcpp/kawpow_keccak.inc";
const KAWPOW_SIDECAR_MEMBERS = [KAWPOW_DEVICE_MEMBER, KAWPOW_KECCAK_MEMBER];

function linuxReleaseLauncher() {
  const source = fs.readFileSync(PACKAGE_SCRIPT, "utf8");
  const match = source.match(/cat >"\$package_dir\/mom" <<'EOF'\n([\s\S]*?)\nEOF\nchmod \+x "\$package_dir\/mom"/);
  assert.ok(match?.[1], "Linux release launcher template is missing");
  return match[1];
}

function linuxQemuWrapper() {
  const source = fs.readFileSync(LINUX_RELEASE_TEST_SCRIPT, "utf8");
  const match = source.match(/cat >"\$package_dir\/mom-bin" <<'EOF'\n([\s\S]*?)\nEOF\n\x20{2}chmod 0755 "\$package_dir\/mom-bin"/);
  assert.ok(match?.[1], "Linux QEMU wrapper template is missing");
  return match[1];
}

function linuxQemuPreload() {
  const source = fs.readFileSync(LINUX_RELEASE_TEST_SCRIPT, "utf8");
  const match = source.match(/cat >"\$package_dir\/mom-qemu-preload\.cjs" <<'EOF'\n([\s\S]*?)\nEOF\n\x20{2}chmod 0644 "\$package_dir\/mom-qemu-preload\.cjs"/);
  assert.ok(match?.[1], "Linux QEMU preload template is missing");
  return match[1];
}

/** @param {string} command @returns {string} */
function makeDriver(command) {
  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    'libs_dir="$1"',
    command,
    "",
  ].join("\n");
}

/**
 * @param {string} command
 * @param {string[]} args
 * @param {Pick<import("node:child_process").SpawnSyncOptions, "cwd" | "env">} [options]
 */
function run(command, args, options = {}) {
  const result = spawnSync(command, args, {encoding: "utf8", timeout: 30000, ...options});
  assert.equal(result.error, undefined, result.error?.message);
  return result;
}

/** @param {string} file @param {string} contents */
function writeExecutable(file, contents) {
  fs.writeFileSync(file, contents);
  fs.chmodSync(file, 0o755);
}

/** @param {string} source @returns {string} */
function extractCleanupFunction(source) {
  const start = source.indexOf("cleanup_containers() {");
  const end = source.indexOf("\n}\n\ncompilers=", start);
  assert.notEqual(start, -1, "archive cleanup function is missing");
  assert.notEqual(end, -1, "archive cleanup function boundary is missing");
  return source.slice(start, end + 2);
}

/** @param {string} source @returns {string} */
function extractPublication(source) {
  const guard = '[ "$failed" -eq 0 ] || exit 1';
  const guardIndex = source.indexOf(guard);
  assert.notEqual(guardIndex, -1, "archive validation guard is missing");
  const start = source.indexOf("\n", guardIndex);
  assert.notEqual(start, -1, "archive publication boundary is missing");
  return source.slice(start + 1).trim();
}

/** @param {string} directory @returns {string[]} */
function archiveStageEntries(directory) {
  return fs.readdirSync(directory).filter(name => name.startsWith(".mom-archive."));
}

/** @param {string} archive @param {string} workDirectory @param {NodeJS.ProcessEnv} [extraEnv] */
function validateLinuxArchive(archive, workDirectory, extraEnv = {}) {
  return run("bash", [LINUX_RELEASE_TEST_SCRIPT, archive], {
    cwd: repo,
    env: {...process.env,
      MOM_RELEASE_ARCHIVE_VALIDATION_ONLY: "1",
      MOM_RELEASE_TEST_DIR: workDirectory,
      ...extraEnv,
    },
  });
}

/** @param {string} archive @param {string} baseDirectory @param {string[]} entries
 * @param {string} [transform] */
function createTarArchive(archive, baseDirectory, entries, transform) {
  const args = ["-C", baseDirectory, "-czf", archive];
  if (transform) {
    args.push("--transform", transform);
  }
  args.push(...entries);
  const result = run("tar", args);
  assert.equal(result.status, 0, result.stdout + result.stderr);
}

/** @param {string} archive @param {string} root @param {string} linkTarget */
function createHardlinkArchive(archive, root, linkTarget) {
  const script = [
    "import sys, tarfile",
    "archive, root, link_target = sys.argv[1:]",
    "with tarfile.open(archive, 'w:gz') as payload:",
    "    directory = tarfile.TarInfo(root + '/')",
    "    directory.type = tarfile.DIRTYPE",
    "    payload.addfile(directory)",
    "    link = tarfile.TarInfo(root + '/link')",
    "    link.type = tarfile.LNKTYPE",
    "    link.linkname = link_target",
    "    payload.addfile(link)",
  ].join("\n");
  const result = run("python3", ["-c", script, archive, root, linkTarget]);
  assert.equal(result.status, 0, result.stdout + result.stderr);
}

test("release archive validation enforces the versioned canonical root", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-validation-"));
  try {
    const packageDirectory = path.join(fixture, RELEASE_ROOT);
    fs.mkdirSync(packageDirectory);
    fs.writeFileSync(path.join(packageDirectory, "payload.txt"), "fixture payload\n");
    for (const member of KAWPOW_SIDECAR_MEMBERS) {
      const file = path.join(packageDirectory, member);
      fs.mkdirSync(path.dirname(file), {recursive: true});
      fs.writeFileSync(file, "fixture " + path.basename(member) + "\n");
    }

    const validArchive = path.join(fixture, "valid.tgz");
    createTarArchive(validArchive, fixture, [RELEASE_ROOT]);
    const valid = validateLinuxArchive(validArchive, path.join(fixture, "mom-release-valid"));
    assert.equal(valid.status, 0, valid.stdout + valid.stderr);

    /** @type {Array<[string, string[], string | undefined, RegExp]>} */
    const cases = [
      ["wrong root", ["mom-vwrong"], undefined, /expected package root/i],
      ["multiple roots", [RELEASE_ROOT, "second"], undefined, /expected package root/i],
      ["missing root entry", ["payload.txt"], `s,^payload[.]txt$,${RELEASE_ROOT}/payload.txt,`,
        /explicit package root directory/i],
      ["repeated separator", [RELEASE_ROOT, `${RELEASE_ROOT}/payload.txt`],
        `s,^${RELEASE_ROOT}/payload[.]txt$,${RELEASE_ROOT}//payload.txt,`, /noncanonical member/i],
      ["dot component", [RELEASE_ROOT, `${RELEASE_ROOT}/payload.txt`],
        `s,^${RELEASE_ROOT}/payload[.]txt$,${RELEASE_ROOT}/./payload.txt,`, /noncanonical member/i],
      ["backslash", [RELEASE_ROOT, `${RELEASE_ROOT}/payload.txt`],
        `s,^${RELEASE_ROOT}/payload[.]txt$,${RELEASE_ROOT}\\\\payload.txt,`, /noncanonical member/i],
      ["trailing separator collision", [RELEASE_ROOT, `${RELEASE_ROOT}/payload.txt`],
        `s,^${RELEASE_ROOT}/payload[.]txt$,${RELEASE_ROOT},`, /duplicate member/i],
    ];
    for (const [label, entries, transform, expected] of cases) {
      const archive = path.join(fixture, `${label.replaceAll(" ", "-")}.tgz`);
      if (label === "wrong root") {
        fs.mkdirSync(path.join(fixture, "mom-vwrong"));
      }
      if (label === "multiple roots") {
        fs.mkdirSync(path.join(fixture, "second"));
      }
      if (label === "missing root entry") {
        fs.writeFileSync(path.join(fixture, "payload.txt"), "fixture payload\n");
      }
      createTarArchive(archive, fixture, entries, transform);
      const result = validateLinuxArchive(archive, path.join(fixture, `mom-release-${label.replaceAll(" ", "-")}`));
      assert.notEqual(result.status, 0, `${label} unexpectedly accepted`);
      assert.match(result.stderr, expected, `${label}: ${result.stderr}`);
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux release archive validation requires nonempty KawPow sidecars after extraction", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-sidecars-"));
  try {
    const packageDirectory = path.join(fixture, RELEASE_ROOT);
    fs.mkdirSync(path.join(packageDirectory, "libs", "dpcpp"), {recursive: true});
    fs.writeFileSync(path.join(packageDirectory, KAWPOW_DEVICE_MEMBER),
      "fixture kawpow device\n");
    const missingArchive = path.join(fixture, "missing.tgz");
    createTarArchive(missingArchive, fixture, [RELEASE_ROOT]);
    const missing = validateLinuxArchive(missingArchive, path.join(fixture, "mom-release-missing"), {
      MOM_RELEASE_ARCHIVE_VALIDATION_ONLY: "0",
    });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /missing nonempty dpcpp\/kawpow_keccak\.inc/);

    fs.writeFileSync(path.join(packageDirectory, KAWPOW_KECCAK_MEMBER),
      "fixture kawpow keccak\n");
    fs.writeFileSync(path.join(packageDirectory, KAWPOW_DEVICE_MEMBER), "");
    const emptyArchive = path.join(fixture, "empty.tgz");
    createTarArchive(emptyArchive, fixture, [RELEASE_ROOT]);
    const empty = validateLinuxArchive(emptyArchive, path.join(fixture, "mom-release-empty"), {
      MOM_RELEASE_ARCHIVE_VALIDATION_ONLY: "0",
    });
    assert.notEqual(empty.status, 0);
    assert.match(empty.stderr, /missing nonempty dpcpp\/kawpow_device\.inc/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux release archive validation rejects absolute and escaping hardlinks", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-hardlink-"));
  try {
    /** @type {Array<[string, string]>} */
    const cases = [["absolute", "/outside"], ["escaping", "../outside"]];
    for (const [label, target] of cases) {
      const archive = path.join(fixture, `${label}.tgz`);
      createHardlinkArchive(archive, RELEASE_ROOT, target);
      const result = validateLinuxArchive(archive, path.join(fixture, `mom-release-${label}`));
      assert.notEqual(result.status, 0, `${label} hardlink unexpectedly accepted`);
      assert.match(result.stderr, /hardlink target|unsafe member/i);
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux release archive validation uses its private archive copy", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-private-copy-"));
  try {
    const packageDirectory = path.join(fixture, RELEASE_ROOT);
    fs.mkdirSync(packageDirectory);
    fs.writeFileSync(path.join(packageDirectory, "payload.txt"), "fixture payload\n");
    for (const member of KAWPOW_SIDECAR_MEMBERS) {
      const file = path.join(packageDirectory, member);
      fs.mkdirSync(path.dirname(file), {recursive: true});
      fs.writeFileSync(file, "fixture " + path.basename(member) + "\n");
    }
    const archive = path.join(fixture, "valid.tgz");
    createTarArchive(archive, fixture, [RELEASE_ROOT]);
    const replacement = path.join(fixture, "replacement.tgz");
    fs.writeFileSync(replacement, "not an archive\n");
    const fakeBin = fs.mkdtempSync(path.join(fixture, "fake-bin-"));
    const realTar = run("bash", ["-c", "command -v tar"]).stdout.trim();
    writeExecutable(path.join(fakeBin, "tar"), [
      "#!/usr/bin/env bash",
      "\"$MOM_REAL_TAR\" \"$@\"",
      "status=$?",
      "if [[ \"$1\" == \"-tzf\" ]]; then cp -- \"$MOM_REPLACEMENT\" \"$MOM_SOURCE\"; fi",
      "exit \"$status\"",
      "",
    ].join("\n"));
    const result = validateLinuxArchive(archive, path.join(fixture, "mom-release-private"), {
      PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] ?? ""}`,
      MOM_REAL_TAR: realTar,
      MOM_REPLACEMENT: replacement,
      MOM_SOURCE: archive,
    });
    assert.equal(result.status, 0, result.stdout + result.stderr);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Windows release archive validation uses exact version and case-insensitive canonical members", () => {
  const source = fs.readFileSync(WINDOWS_RELEASE_TEST_SCRIPT, "utf8");
  assert.match(source, /\$expectedRoot = "mom-v\$packageVersion"/);
  assert.match(source, /HashSet\[string\]\]::new\(\[StringComparer\]::OrdinalIgnoreCase\)/);
  assert.match(source, /\$normalizedName = \$rawName\.Replace/);
  assert.match(source, /\$normalizedName\.TrimEnd\('\/'\)/);
  assert.match(source, /\$member -cne \$expectedRoot/);
  assert.match(source, /\$hasRootEntry = \$false/);
  assert.match(source, /if \(\$member -ceq \$expectedRoot\)[\s\S]*?\$hasRootEntry = \$true/);
  assert.match(source, /if \(-not \$hasRootEntry\) \{\s*throw 'Release archive must contain an explicit package root directory\.'/);
  assert.match(source, /\$hasRootPayload = \$true/);
  assert.match(source, /if \(-not \$hasRootPayload\)/);
  assert.match(source, /explicit package root directory/);
});

test("Windows release validation authenticates the source and extracted tree", () => {
  const source = fs.readFileSync(WINDOWS_RELEASE_TEST_SCRIPT, "utf8");
  assert.match(source, /Assert-NoReparseAncestor \$Archive 'Release archive'/);
  assert.match(source, /Get-Item -LiteralPath \$Archive/);
  assert.match(source, /FileAttributes\]::ReparsePoint/);
  assert.match(source, /Copy-Item -LiteralPath \$archivePath -Destination \$privateArchive/);
  assert.match(source, /ZipFile\]::OpenRead\(\$privateArchive\)/);
  assert.match(source, /Expand-Archive -LiteralPath \$privateArchive/);
  assert.match(source, /EnumerateFileSystemEntries/);
  assert.doesNotMatch(source, /Expand-Archive -LiteralPath \$Archive/);
  assert.match(source, /foreach \(\$sidecar in @\('kawpow_device\.inc', 'kawpow_keccak\.inc'\)\)/);
  assert.match(source, /Test-Path -LiteralPath \$sidecarPath -PathType Leaf/);
  assert.match(source, /\$sidecarPath[\s\S]*?\.Length -le 0/);
});

test("Windows portable CPU gate mirrors only a missing OpenCL adapter binding", (t) => {
  const source = fs.readFileSync(WINDOWS_RELEASE_TEST_SCRIPT, "utf8");
  const binding = source.match(/if \(\$Suite -eq 'gpu-portable-cpu' -and -not \$env:UR_ADAPTERS_FORCE_LOAD\) \{\r?\n[\s\S]*?\r?\n\}(?=\r?\n\$sharedOneApiDir)/)?.[0];
  assert.ok(binding, "portable CPU adapter binding is missing");
  if (process.platform !== "win32") {
    t.skip("actual PowerShell environment behavior requires Windows");
    return;
  }
  const script = String.raw`
$ErrorActionPreference = 'Stop'
$workerDir = 'C:\package with spaces & (punctuation)\libs\dpcpp-opencl'
$cases = @(
  @{suite = 'gpu-portable-cpu'; value = $null},
  @{suite = 'gpu-portable-cpu'; value = '"C:\caller-runtime\ur_adapter_opencl.dll"'},
  @{suite = 'gpu'; value = $null}
)
$results = @(foreach ($case in $cases) {
  $Suite = $case.suite
  if ($null -eq $case.value) { Remove-Item Env:UR_ADAPTERS_FORCE_LOAD -ErrorAction SilentlyContinue }
  else { $env:UR_ADAPTERS_FORCE_LOAD = $case.value }
` + binding + String.raw`
  [pscustomobject]@{suite = $Suite; value = [Environment]::GetEnvironmentVariable('UR_ADAPTERS_FORCE_LOAD', 'Process')}
})
ConvertTo-Json -InputObject $results -Compress
`;
  const result = run("powershell.exe", ["-NoProfile", "-EncodedCommand",
    Buffer.from(script, "utf16le").toString("base64")]);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout.trim()), [
    {suite: "gpu-portable-cpu", value: String.raw`"C:\package with spaces & (punctuation)\libs\dpcpp-opencl\ur_adapter_opencl.dll"`},
    {suite: "gpu-portable-cpu", value: String.raw`"C:\caller-runtime\ur_adapter_opencl.dll"`},
    {suite: "gpu", value: null},
  ]);
});

test("release publication binds the package tag and exact canonical archive names", () => {
  const source = fs.readFileSync(PUBLISH_WORKFLOW, "utf8");
  assert.match(source, /actions\/checkout@v7/);
  assert.match(source, /require\("\.\/package\.json"\)\.version/);
  assert.match(source, /expected_tag="v\$version"/);
  assert.match(source, /GITHUB_REF_NAME.*expected_tag/);
  assert.match(source, /linux_name="mom-\$\{expected_tag\}-lin\.tgz"/);
  assert.match(source, /windows_name="mom-\$\{expected_tag\}-win\.zip"/);
  assert.match(source, /-name "\$linux_name" -print0/);
  assert.match(source, /-name "\$windows_name" -print0/);
  assert.match(source, /Unexpected release archive name/);
  assert.ok(source.indexOf("checksums=()") > source.indexOf("Unexpected release archive name"));
});

test("Windows multicompiler docs use none for packaging and explicit vendor tests", () => {
  const docs = fs.readFileSync(path.join(repo, "scripts", "windows-multicompiler.md"), "utf8");
  const releaseExample = docs.indexOf("Build the archive once");
  const exampleStart = docs.indexOf("```bash", releaseExample);
  const exampleEnd = docs.indexOf("```", exampleStart + 7);
  assert.notEqual(exampleStart, -1);
  assert.notEqual(exampleEnd, -1);
  const example = docs.slice(exampleStart, exampleEnd);
  assert.match(example, /GPU_GROUP=none[\s\S]{0,180}build-windows-multicompiler\.ps1/);
  assert.match(example, /GPU_GROUP=none[\s\S]{0,300}npm install --ignore-scripts --no-audit --no-fund[\s\S]{0,180}package-windows\.ps1/);
  assert.match(example, /ln -f mom-v0\.9\.0-win\.zip build\/win-package\/mom-v0\.9\.0-win\.zip/);
  assert.doesNotMatch(example, /--release --download mom-v0\.9\.0-win\.zip/);
  for (const vendor of ["intel", "nvidia", "amd"]) {
    assert.match(example, new RegExp(`GPU_GROUP=${vendor}[^\\n]*--release[\\s\\S]{0,300}test-release-windows\\.ps1[\\s\\S]{0,120}-Suite gpu`));
  }
  assert.doesNotMatch(example, /test-windows-unified-release\.ps1/);
});

test("Linux release packager uses only tag refs as implicit versions", {
  skip: process.platform === "win32",
}, () => {
  const source = fs.readFileSync(PACKAGE_SCRIPT, "utf8");
  const boundary = source.indexOf("\nroot=");
  assert.notEqual(boundary, -1, "version selection boundary is missing");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-ref-version-"));
  try {
    fs.copyFileSync(path.join(repo, "package.json"), path.join(fixture, "package.json"));
    const script = source.slice(0, boundary) + '\nprintf "%s\\n" "$version"\n';
    const packageVersion = require(path.join(repo, "package.json")).version;
    const cases = [
      {type: "branch", ref: "123/merge", supplied: "", expected: packageVersion},
      {type: "branch", ref: "0.8-maintenance", supplied: "", expected: packageVersion},
      {type: "branch", ref: "master", supplied: "", expected: packageVersion},
      {type: "tag", ref: "v9.8.7", supplied: "", expected: "9.8.7"},
      {type: "", ref: "", supplied: "", expected: packageVersion},
      {type: "branch", ref: "123/merge", supplied: "7.6.5", expected: "7.6.5"},
      {type: "tag", ref: "v1/bad", supplied: "", expected: null},
      {type: "branch", ref: "master", supplied: "1/bad", expected: null},
    ];
    for (const entry of cases) {
      const result = run("bash", ["-c", script, "ref-version", entry.supplied], {
        cwd: fixture, env: {...process.env, GITHUB_REF_TYPE: entry.type, GITHUB_REF_NAME: entry.ref},
      });
      assert.equal(result.signal, null, result.stdout + result.stderr);
      if (entry.expected === null) {
        assert.equal(result.status, 2, result.stdout + result.stderr);
        assert.match(result.stderr, /Invalid release version/);
      } else {
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.equal(result.stdout.trim(), entry.expected);
      }
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("release workflow discovers exactly the package-versioned archives", () => {
  const source = fs.readFileSync(RELEASE_WORKFLOW, "utf8");
  assert.match(source, /archives=\(mom-v"\$version"-lin\.tgz\)/);
  assert.match(source, /\[\s*"\$\{#archives\[@\]\}"\s+-ne 1\s*\]/);
  assert.match(source, /\$expectedArchive = "mom-v\$version-win\.zip"/);
  assert.match(source, /\$archives\.Count -ne 1/);
  assert.doesNotMatch(source, /find\s+\.\s+-maxdepth 1\s+-name ['"]mom-\*-/);
  assert.doesNotMatch(source, /Select-Object\s+-First\s+1/);
});

/** @param {string} archive */
function assertArchivePayload(archive) {
  const extracted = run("tar", ["-xOf", archive, "mom-vfixture/payload.txt"]);
  assert.equal(extracted.status, 0, extracted.stderr);
  assert.equal(extracted.stdout, "fixture payload\n");
  for (const member of KAWPOW_SIDECAR_MEMBERS) {
    const sidecar = run("tar", ["-xOf", archive, `mom-vfixture/${member}`]);
    assert.equal(sidecar.status, 0, `${member}: ${sidecar.stderr}`);
    assert.notEqual(sidecar.stdout.length, 0, `${member} is empty`);
  }
}

/** @param {string} source @returns {string} */
function makePublicationHarness(source) {
  return [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    'archive="$1"',
    'root="mom-vfixture"',
    "archive_stage=",
    "node_container=",
    "container=",
    extractCleanupFunction(source),
    "trap cleanup_containers EXIT",
    "failed=0",
    extractPublication(source),
    "",
  ].join("\n");
}

/**
 * @param {string} source
 * @param {{existing: boolean, archiveSymlink?: boolean, tarFailure?: number, mvFailure?: number}} options
 * @param {(result: ReturnType<typeof run>, archive: string, archiveDirectory: string) => void} check
 */
function exercisePublication(source, options, check) {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-publication-"));
  try {
    const archiveDirectory = path.join(fixture, "published");
    const archive = path.join(archiveDirectory, "mom-vfixture.tgz");
    const releaseRoot = path.join(fixture, "release-combined", "mom-vfixture");
    const oldArchiveTarget = path.join(fixture, "old-target.tgz");
    const script = path.join(fixture, "publish.sh");
    const bin = path.join(fixture, "bin");
    fs.mkdirSync(archiveDirectory);
    fs.mkdirSync(releaseRoot, {recursive: true});
    fs.mkdirSync(bin);
    fs.writeFileSync(path.join(releaseRoot, "payload.txt"), "fixture payload\n");
    const sidecarDirectory = path.join(releaseRoot, "libs", "dpcpp");
    fs.mkdirSync(sidecarDirectory, {recursive: true});
    for (const member of KAWPOW_SIDECAR_MEMBERS) {
      fs.writeFileSync(path.join(releaseRoot, member), `fixture ${path.basename(member)}\n`);
    }
    if (options.archiveSymlink) {
      fs.writeFileSync(oldArchiveTarget, "old archive\n");
      fs.symlinkSync(path.relative(archiveDirectory, oldArchiveTarget), archive);
    } else if (options.existing) {
      fs.writeFileSync(archive, "old archive\n");
    }
    fs.writeFileSync(script, makePublicationHarness(source));
    fs.chmodSync(script, 0o755);

    if (options.tarFailure !== undefined) {
      writeExecutable(path.join(bin, "tar"), [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "archive=",
        "while [ \"$#\" -gt 0 ]; do",
        "  case \"$1\" in",
        "    -*f) archive=\"$2\"; shift 2 ;;",
        "    *) shift ;;",
        "  esac",
        "done",
        "[ -n \"$archive\" ]",
        "printf 'partial archive\\n' > \"$archive\"",
        `exit ${options.tarFailure}`,
        "",
      ].join("\n"));
    }
    if (options.mvFailure !== undefined) {
      writeExecutable(path.join(bin, "mv"), [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        `exit ${options.mvFailure}`,
        "",
      ].join("\n"));
    }

    const result = run("bash", [script, archive], {
      cwd: fixture,
      env: {...process.env, PATH: `${bin}${path.delimiter}${process.env["PATH"] ?? ""}`},
    });
    check(result, archive, archiveDirectory);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
}

test("Linux combined archive keeps the compatibility mom.node as a hardlink", {
  skip: process.platform === "win32",
}, () => {
  const source = fs.readFileSync(PACKAGE_SCRIPT, "utf8");
  const packagingLine = source.split("\n").find(line =>
    line.includes('"$libs_dir/oneapi/mom.node"') &&
    line.includes('"$libs_dir/mom.node"'));
  assert.ok(packagingLine, "compatibility mom.node packaging command is missing");

  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-archive-"));
  try {
    const libs = path.join(fixture, "libs");
    const oneapi = path.join(libs, "oneapi");
    const sourceNode = path.join(oneapi, "mom.node");
    const compatibilityNode = path.join(libs, "mom.node");
    const payload = Buffer.from("mom-node-fixture\0payload\n");
    fs.mkdirSync(oneapi, {recursive: true});
    fs.writeFileSync(sourceNode, payload);

    const driver = path.join(fixture, "hardlink.sh");
    fs.writeFileSync(driver, makeDriver(packagingLine.trim()));
    const linked = run("bash", [driver, libs]);
    assert.equal(linked.status, 0, linked.stdout + linked.stderr);
    assert.equal(fs.lstatSync(compatibilityNode).isSymbolicLink(), false);
    assert.equal(fs.statSync(sourceNode).ino, fs.statSync(compatibilityNode).ino);
    assert.ok(fs.statSync(sourceNode).nlink >= 2);
    assert.deepEqual(fs.readFileSync(compatibilityNode), payload);

    const archive = path.join(fixture, "fixture.tar.gz");
    const packed = run("tar", ["-C", fixture, "-czf", archive, "libs"]);
    assert.equal(packed.status, 0, packed.stdout + packed.stderr);
    const extracted = path.join(fixture, "extracted");
    fs.mkdirSync(extracted);
    const unpacked = run("tar", ["-C", extracted, "-xzf", archive]);
    assert.equal(unpacked.status, 0, unpacked.stdout + unpacked.stderr);
    const extractedSource = path.join(extracted, "libs", "oneapi", "mom.node");
    const extractedCompatibility = path.join(extracted, "libs", "mom.node");
    assert.equal(fs.statSync(extractedSource).ino, fs.statSync(extractedCompatibility).ino);
    assert.deepEqual(fs.readFileSync(extractedCompatibility), payload);

    const missing = path.join(fixture, "missing-libs");
    fs.mkdirSync(path.join(missing, "oneapi"), {recursive: true});
    const failed = run("bash", [driver, missing]);
    assert.notEqual(failed.status, 0, failed.stdout + failed.stderr);
    assert.equal(fs.existsSync(path.join(missing, "mom.node")), false);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux release launcher honors the documented OpenCL device type", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-launcher-"));
  try {
    const launcher = path.join(fixture, "mom");
    const executable = path.join(fixture, "mom-bin");
    fs.mkdirSync(path.join(fixture, "libs", "dpcpp-opencl"), {recursive: true});
    fs.writeFileSync(launcher, linuxReleaseLauncher());
    fs.writeFileSync(executable, [
      "#!/usr/bin/env sh",
      "printf '%s\\n' \"$ONEAPI_DEVICE_SELECTOR\" \"$MOM_NATIVE_PATH\" \"$MOM_COMMAND\"",
    ].join("\n"));
    fs.chmodSync(launcher, 0o755);
    fs.chmodSync(executable, 0o755);

    const cpu = run(launcher, ["algorithms"], {
      env: {
        PATH: process.env["PATH"] ?? "",
        MOM_GPU_BACKEND: "opencl",
        MOM_OPENCL_DEVICE_TYPE: "cpu",
      },
    });
    assert.equal(cpu.status, 0, cpu.stdout + cpu.stderr);
    assert.deepEqual(cpu.stdout.trim().split(/\r?\n/), [
      "opencl:cpu", path.join(fixture, "libs", "dpcpp-opencl", "mom.node"), "./mom",
    ]);

    const gpu = run(launcher, ["algorithms"], {
      env: {
        PATH: process.env["PATH"] ?? "",
        MOM_COMMAND: "custom-mom",
        MOM_GPU_BACKEND: "opencl",
      },
    });
    assert.equal(gpu.status, 0, gpu.stdout + gpu.stderr);
    assert.deepEqual(gpu.stdout.trim().split(/\r?\n/), [
      "opencl:gpu", path.join(fixture, "libs", "dpcpp-opencl", "mom.node"), "custom-mom",
    ]);

    const invalid = run(launcher, [], {
      env: {
        PATH: process.env["PATH"] ?? "",
        MOM_GPU_BACKEND: "opencl",
        MOM_OPENCL_DEVICE_TYPE: "accelerator",
      },
    });
    assert.equal(invalid.status, 2, invalid.stdout + invalid.stderr);
    assert.match(invalid.stderr, /MOM_OPENCL_DEVICE_TYPE must be gpu or cpu/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux QEMU release wrapper re-enters emulation for fork and subprocess workers", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom qemu-wrapper-"));
  const packageDir = path.join(fixture, "package");
  const fakeBin = path.join(fixture, "bin");
  const wrapper = path.join(packageDir, "mom-bin");
  const real = path.join(packageDir, "mom-bin.real");
  const preload = path.join(packageDir, "mom-qemu-preload.cjs");
  const qemuLog = path.join(fixture, "qemu.log");
  fs.mkdirSync(packageDir, {recursive: true});
  fs.mkdirSync(fakeBin);
  writeExecutable(wrapper, `${linuxQemuWrapper()}\n`);
  fs.writeFileSync(preload, `${linuxQemuPreload()}\n`);
  writeExecutable(real, [
    "#!/usr/bin/env node",
    "const cluster = require('node:cluster');",
    "const {spawn} = require('node:child_process');",
    "const readline = require('node:readline');",
    "const mode = process.argv[2];",
    "const childRole = process.env.MOM_FIXTURE_ROLE === 'child';",
    "const childId = process.env.MOM_FIXTURE_ID || process.argv[3] || '0';",
    "let exits = 0;",
    "let failed = 0;",
    "function runChild() {",
    "  process.stdout.write('READY:' + childId + '\\n');",
    "  const input = readline.createInterface({input: process.stdin});",
    "  input.on('line', (line) => {",
    "    if (line === 'go') { process.exit(childId === '1' ? 7 : 0); }",
    "  });",
    "}",
    "function attach(id, child, input, output) {",
    "  if (!input || !output) { process.exit(88); }",
    "  output.setEncoding('utf8');",
    "  let pending = '';",
    "  output.on('data', (chunk) => {",
    "    pending += chunk;",
    "    let end;",
    "    while ((end = pending.indexOf('\\n')) !== -1) {",
    "      const line = pending.slice(0, end);",
    "      pending = pending.slice(end + 1);",
    "      if (line === 'READY:' + id) {",
    "        process.stdout.write('STREAM_READY:' + id + '\\n');",
    "        input.write('go\\n');",
    "        process.stdout.write('SENT:' + id + '\\n');",
    "      }",
    "    }",
    "  });",
    "  child.on('exit', (code) => {",
    "    process.stdout.write('EXIT:' + id + ':' + code + '\\n');",
    "    exits += 1;",
    "    failed = Math.max(failed, code === null ? 99 : code);",
    "    if (exits === 2) { process.exit(failed); }",
    "  });",
    "}",
    "if (cluster.isWorker || childRole || mode === 'child') {",
    "  runChild();",
    "} else {",
    "  if (mode === 'fork') { cluster.setupPrimary({exec: process.argv[1], silent: true}); }",
    "  for (const id of [0, 1]) {",
    "    if (mode === 'fork') {",
    "      const worker = cluster.fork({MOM_FIXTURE_ROLE: 'child', MOM_FIXTURE_ID: String(id)});",
    "      attach(id, worker.process, worker.process.stdin, worker.process.stdout);",
    "    } else {",
    "      const child = spawn(process.execPath, [process.argv[1], 'child', String(id)], {",
    "        env: {...process.env, MOM_FIXTURE_ROLE: 'child', MOM_FIXTURE_ID: String(id)},",
    "        stdio: ['pipe', 'pipe', 'pipe', 'ipc'],",
    "      });",
    "      attach(id, child, child.stdin, child.stdout);",
    "    }",
    "  }",
    "}",
    "",
  ].join("\n"));
  writeExecutable(path.join(fakeBin, "qemu-x86_64"), [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    'printf "%s\\n" "$*" >> "$MOM_QEMU_LOG"',
    'exec "$@"',
    "",
  ].join("\n"));
  try {
    for (const mode of ["fork", "spawn"]) {
      fs.writeFileSync(qemuLog, "");
      /** @type {NodeJS.ProcessEnv} */
      const env = {
        ...process.env,
        MOM_QEMU_LOG: qemuLog,
        PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] || ""}`,
      };
      delete env["MOM_RELEASE_QEMU_CHILD"];
      const result = spawnSync(wrapper, [mode], {
        cwd: packageDir, env, encoding: "utf8", timeout: 30000,
      });
      const qemuTrace = fs.existsSync(qemuLog) ? fs.readFileSync(qemuLog, "utf8") : "<missing>";
      assert.equal(result.error, undefined,
        `${mode}: ${result.error?.message || "spawn failed"}\n${result.stdout}\n${result.stderr}\n${qemuTrace}`);
      assert.equal(result.status, 7, `${mode}: ${result.stdout}${result.stderr}`);
      assert.match(result.stdout, /STREAM_READY:0[\s\S]*SENT:0/);
      assert.match(result.stdout, /STREAM_READY:1[\s\S]*SENT:1/);
      const invocations = fs.readFileSync(qemuLog, "utf8").trim().split(/\r?\n/);
      assert.equal(invocations.length, 3, `${mode}: ${invocations.join(" | ")}`);
      assert.ok(invocations.every((line) => line.includes("mom-bin.real")), mode);
    }
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux combined archive preserves an existing output before publication", {
  skip: process.platform === "win32",
}, () => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom-release-packager-"));
  try {
    const script = path.join(fixture, ".github", "workflows", "scripts",
      "package-linux-combined.sh");
    const buildRoot = path.join(fixture, "build", "lin", "Release");
    const archive = path.join(fixture, "published.tgz");
    const fakeBin = path.join(fixture, "fake-bin");
    const dockerLog = path.join(fixture, "docker.log");
    const npxLog = path.join(fixture, "npx.log");
    const seaNode = path.join(fixture, "sea-node");
    const compilerKeys = ["oneapi", "dpcpp", "dpcpp-opencl", "acpp-cuda", "acpp-hip"];
    fs.mkdirSync(path.dirname(script), {recursive: true});
    fs.mkdirSync(fakeBin);
    fs.copyFileSync(PACKAGE_SCRIPT, script);
    fs.chmodSync(script, 0o755);
    for (const key of compilerKeys) {
      fs.mkdirSync(path.join(buildRoot, key), {recursive: true});
      const addon = path.join(buildRoot, key, "mom.node");
      const contents = `${key}\n`;
      fs.writeFileSync(addon, contents);
      fs.writeFileSync(`${addon}.build-profile`, [
        "schema=1", `worker=${key}`,
        `sha256=${createHash("sha256").update(contents).digest("hex")}`,
        "portable=1", "cpu=unset", "",
      ].join("\n"));
    }
    fs.writeFileSync(archive, "old archive\n");
    fs.writeFileSync(seaNode, "NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2\n");
    writeExecutable(path.join(fakeBin, "docker"), [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"',
      'if [ "$#" -ge 2 ] && [ "$1" = image ] && [ "$2" = inspect ]; then exit 0; fi',
      "exit 99",
      "",
    ].join("\n"));
    writeExecutable(path.join(fakeBin, "npx"), [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      'printf \'%s\\n\' "$*" >> "$NPX_LOG"',
      "exit 37",
      "",
    ].join("\n"));

    const result = run("bash", [script, "1.2.3", archive], {
      cwd: fixture,
      env: {
        ...process.env,
        DOCKER_LOG: dockerLog,
        MOM_MULTICOMPILER_IMAGE: "fake-image",
        NODE_BIN: seaNode,
        NPX_LOG: npxLog,
        PATH: `${fakeBin}${path.delimiter}${process.env["PATH"] ?? ""}`,
      },
    });
    assert.equal(result.status, 37, result.stdout + result.stderr);
    assert.deepEqual(fs.readFileSync(archive), Buffer.from("old archive\n"));
    assert.equal(fs.readFileSync(dockerLog, "utf8"), "image inspect fake-image\n");
    assert.match(fs.readFileSync(npxLog, "utf8"), /--no-install esbuild/);
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});

test("Linux combined archive publishes atomically and cleans failed stages", {
  skip: process.platform === "win32",
}, () => {
  const source = fs.readFileSync(PACKAGE_SCRIPT, "utf8");

  exercisePublication(source, {existing: true}, (result, archive, archiveDirectory) => {
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.notDeepEqual(fs.readFileSync(archive), Buffer.from("old archive\n"));
    assert.deepEqual(archiveStageEntries(archiveDirectory), []);
    const listed = run("tar", ["-tzf", archive]);
    assert.equal(listed.status, 0, listed.stderr);
    assert.match(listed.stdout, /mom-vfixture\/payload\.txt/);
    assertArchivePayload(archive);
  });

  exercisePublication(source, {existing: false}, (result, archive, archiveDirectory) => {
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.equal(fs.existsSync(archive), true);
    assert.deepEqual(archiveStageEntries(archiveDirectory), []);
    assertArchivePayload(archive);
  });

  exercisePublication(source, {existing: true, tarFailure: 23},
    (result, archive, archiveDirectory) => {
      assert.equal(result.status, 23, result.stdout + result.stderr);
      assert.deepEqual(fs.readFileSync(archive), Buffer.from("old archive\n"));
      assert.deepEqual(archiveStageEntries(archiveDirectory), []);
    });

  exercisePublication(source, {existing: false, tarFailure: 23},
    (result, archive, archiveDirectory) => {
      assert.equal(result.status, 23, result.stdout + result.stderr);
      assert.equal(fs.existsSync(archive), false);
      assert.deepEqual(archiveStageEntries(archiveDirectory), []);
    });

  exercisePublication(source, {existing: true, mvFailure: 24},
    (result, archive, archiveDirectory) => {
      assert.equal(result.status, 24, result.stdout + result.stderr);
      assert.deepEqual(fs.readFileSync(archive), Buffer.from("old archive\n"));
      assert.deepEqual(archiveStageEntries(archiveDirectory), []);
    });

  exercisePublication(source, {existing: true, archiveSymlink: true},
    (result, archive, archiveDirectory) => {
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.equal(fs.lstatSync(archive).isSymbolicLink(), false);
      assert.deepEqual(fs.readFileSync(path.join(path.dirname(archiveDirectory), "old-target.tgz")),
        Buffer.from("old archive\n"));
      assert.deepEqual(archiveStageEntries(archiveDirectory), []);
      assertArchivePayload(archive);
    });
});

test("Linux packaging replaces SDK RPATH with isolated relocatable runtime RUNPATH", {
  skip: process.platform === "win32",
}, () => {
  const source = fs.readFileSync(PACKAGE_SCRIPT, "utf8");
  const start = source.indexOf("normalize_runtime_elfs() {");
  const end = source.indexOf("\n}\n\ncopy_oneapi_opencl_runtime()", start);
  assert.notEqual(start, -1, "runtime normalization function is missing");
  assert.notEqual(end, -1, "runtime normalization function boundary is missing");
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "mom runtime-isolation-"));
  try {
    const libs = path.join(fixture, "package", "libs");
    const runtime = path.join(libs, "dpcpp");
    const nested = path.join(runtime, "hipSYCL", "ext", "llvm", "bin");
    const shared = path.join(libs, "dpcpp-opencl");
    const poison = path.join(fixture, "poison-sdk");
    for (const dir of [nested, shared, poison]) {
      fs.mkdirSync(dir, {recursive: true});
    }
    const dep = path.join(fixture, "dep.c");
    const parent = path.join(fixture, "parent.c");
    const main = path.join(fixture, "main.c");
    fs.writeFileSync(dep, "int dep(void) { return VALUE; }\n");
    fs.writeFileSync(parent, "extern int dep(void); int parent(void) { return dep() + EXTRA; }\n");
    fs.writeFileSync(main,
      '#include <stdio.h>\nextern int parent(void); int main(void) { printf("%d\\n", parent()); }\n');
    /** @param {string[]} args */
    const compile = (args) => {
      const result = run("cc", args);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.equal(result.signal, null);
    };
    /** @type {Array<[string, number, number]>} */
    const libraryConfigs = [[runtime, 7, 1], [poison, 70, 10]];
    for (const [dir, value, extra] of libraryConfigs) {
      compile(["-shared", "-fPIC", `-DVALUE=${value}`, dep, "-Wl,-soname,libfixture-dep.so",
        "-o", path.join(dir, "libfixture-dep.so")]);
      compile(["-shared", "-fPIC", `-DEXTRA=${extra}`, parent, `-L${dir}`, "-lfixture-dep",
        "-Wl,-soname,libfixture-parent.so", "-Wl,--disable-new-dtags",
        `-Wl,-rpath,${poison}:$ORIGIN`, "-o", path.join(dir, "libfixture-parent.so")]);
    }
    const tool = path.join(nested, "tool");
    const sharedWorker = path.join(shared, "mom.node");
    for (const file of [tool, sharedWorker]) {
      compile([main, `-L${runtime}`, "-lfixture-parent", "-Wl,--disable-new-dtags",
        `-Wl,-rpath,${poison}:$ORIGIN:$ORIGIN/llvm-to-backend::`, "-o", file]);
    }
    const object = path.join(runtime, "compiler-input.o");
    compile(["-c", "-DVALUE=7", dep, "-o", object]);
    const objectBefore = fs.readFileSync(object);
    const sdkBefore = fs.readFileSync(path.join(poison, "libfixture-parent.so"));
    const negative = run(tool, [], {env: {PATH: process.env["PATH"] ?? "", LD_LIBRARY_PATH: runtime}});
    assert.equal(negative.status, 0, negative.stdout + negative.stderr);
    assert.equal(negative.stdout, "80\n", "legacy SDK RPATH must demonstrate the poisoned host first");
    const driver = path.join(fixture, "normalize.sh");
    fs.writeFileSync(driver, makeDriver(source.slice(start, end + 2) +
      '\nnormalize_runtime_elfs "$libs_dir/dpcpp"\nnormalize_runtime_elfs "$libs_dir/dpcpp-opencl"'));
    const normalized = run("bash", [driver, libs]);
    assert.equal(normalized.status, 0, normalized.stdout + normalized.stderr);
    assert.equal(normalized.signal, null);
    assert.deepEqual(fs.readFileSync(object), objectBefore, "relocatable compiler inputs must not be rewritten");
    assert.deepEqual(fs.readFileSync(path.join(poison, "libfixture-parent.so")), sdkBefore,
      "host SDK must not be modified");
    const inspect = run("readelf", ["-d", tool]);
    assert.equal(inspect.status, 0, inspect.stdout + inspect.stderr);
    assert.match(inspect.stdout, /\(RUNPATH\)/);
    assert.doesNotMatch(inspect.stdout, /\(RPATH\)|poison-sdk|::/);
    assert.match(inspect.stdout, /\$ORIGIN\/llvm-to-backend/, "safe nested plugin search must survive");
    for (const file of [tool, sharedWorker]) {
      const isolated = run(file, [], {env: {PATH: process.env["PATH"] ?? ""}});
      assert.equal(isolated.status, 0, isolated.stdout + isolated.stderr);
      assert.equal(isolated.stdout, "8\n",
        "both nested/transitive and dpcpp-opencl shared runtime must resolve package libraries");
      const selected = run(file, [], {env: {PATH: process.env["PATH"] ?? "", LD_LIBRARY_PATH: `${runtime}:${poison}`}});
      assert.equal(selected.status, 0, selected.stdout + selected.stderr);
      assert.equal(selected.stdout, "8\n", "launcher-selected runtime must beat poisoned host library paths");
    }
    fs.renameSync(path.join(fixture, "package"), path.join(fixture, "relocated"));
    const relocated = run(path.join(fixture, "relocated", "libs", "dpcpp", "hipSYCL", "ext", "llvm", "bin", "tool"),
      [], {env: {PATH: process.env["PATH"] ?? ""}});
    assert.equal(relocated.status, 0, relocated.stdout + relocated.stderr);
    assert.equal(relocated.stdout, "8\n");
    for (const dependency of ["../outside/libfixture-dep.so", "/host-sdk/libfixture-dep.so"]) {
      const qualified = path.join(fixture, dependency.startsWith("/") ? "absolute-needed" : "relative-needed");
      const qualifiedRuntime = path.join(qualified, "dpcpp");
      fs.mkdirSync(qualifiedRuntime, {recursive: true});
      fs.mkdirSync(path.join(qualified, "dpcpp-opencl"));
      const file = path.join(qualifiedRuntime, "libfixture-parent.so");
      fs.copyFileSync(path.join(poison, "libfixture-parent.so"), file);
      const altered = run("patchelf", ["--replace-needed", "libfixture-dep.so", dependency, file]);
      assert.equal(altered.status, 0, altered.stdout + altered.stderr);
      const before = fs.readFileSync(file);
      const rejected = run("bash", [driver, qualified]);
      assert.notEqual(rejected.status, 0, "path-qualified dependencies bypass package search paths");
      assert.match(rejected.stderr, /path-qualified DT_NEEDED/);
      assert.deepEqual(fs.readFileSync(file), before, "rejection must precede any metadata rewrite");
    }
    const malformed = path.join(fixture, "malformed");
    fs.mkdirSync(path.join(malformed, "dpcpp"), {recursive: true});
    fs.mkdirSync(path.join(malformed, "dpcpp-opencl"));
    fs.writeFileSync(path.join(malformed, "dpcpp", "bad.so"), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 0]));
    const refused = run("bash", [driver, malformed]);
    assert.notEqual(refused.status, 0, "malformed ELF must fail normalization before publication");
  } finally {
    fs.rmSync(fixture, {recursive: true, force: true});
  }
});
