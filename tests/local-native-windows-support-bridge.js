"use strict";

const {describe, test} = require("node:test"), assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path");
const bridge = fs.readFileSync(path.resolve(__dirname, "../scripts/build-local-native-windows-support.ps1"), "utf8");

describe("Native Windows support PowerShell bridge", () => {
  test("composes public tool roles in-process and invokes the strict Node CLI", () => {
    assert.match(bridge, /\[string\]\$InputFile/);
    assert.match(bridge, /\[string\]\$OutputFile/);
    assert.match(bridge, /\. \(Join-Path \$PSScriptRoot 'native-windows-tools\.ps1'\)/);
    assert.match(bridge, /Get-MomNativeWindowsBuildTools -Role core/);
    assert.match(bridge, /Get-MomNativeWindowsBuildTools -Role io/);
    assert.match(bridge, /Get-MomNativeWindowsBuildTools -Role hermes/);
    assert.match(bridge, /openssl = \$ioTools\.openssl/);
    assert.match(bridge, /\$OutputEncoding = \$utf8/);
    assert.match(bridge, /\[Console\]::OutputEncoding = \$utf8/);
    assert.match(bridge, /\$toolsJson \| & \$nodePath \$commandScript --run \$InputFile \$OutputFile/);
    assert.match(bridge, /\$nodePath = Get-MomNativeBuildNode/);
    assert.ok(bridge.indexOf("Get-MomNativeBuildNode") < bridge.indexOf("Get-MomNativeWindowsBuildTools -Role core"));
    assert.doesNotMatch(bridge, /\b(?:Invoke-Expression|Start-Process|winget|choco|Invoke-WebRequest)\b/i);
  });
});
