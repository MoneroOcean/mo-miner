"use strict";

// Single-vendor GPU backend detection for launches without a release shell launcher (the packed
// standalone executables). The launchers' rule is kept: a mixed-vendor host stays unselected.

const fs = require("node:fs");
const path = require("node:path");

/** @typedef {{vendor: number, device: number, bus: number}} DisplayAdapter */

// PCI display adapters: Linux sysfs, or the Windows standalone executable's SetupAPI query. Other
// Windows launches (source packages) are selected by their mom.cmd launcher instead.
/** @param {string} [root] @returns {DisplayAdapter[]} */
function displayAdapters(root = "/sys/bus/pci/devices") {
  if (process.platform === "win32") {
    const values = globalThis.__momNativeCore?.displayAdapters?.();
    return Array.isArray(values) ? values : [];
  }
  let names;
  try {names = fs.readdirSync(root);} catch {return [];}
  /** @type {DisplayAdapter[]} */
  const adapters = [];
  for (const name of names) {
    /** @param {string} file */
    const read = (file) => {
      try {return fs.readFileSync(path.join(root, name, file), "utf8").trim();} catch {return "";}
    };
    if (!read("class").startsWith("0x03")) {continue;}
    const bus = /^[0-9a-f]{4,}:([0-9a-f]{2}):/i.exec(name);
    adapters.push({vendor: parseInt(read("vendor"), 16), device: parseInt(read("device"), 16),
      bus: bus ? parseInt(bus[1] ?? "", 16) : NaN});
  }
  return adapters;
}

// Display-only consoles never compete as a compute backend: QEMU Cirrus/std VGA/QXL/virtio-gpu,
// VMware SVGA and Hyper-V video by device, ASPEED and Matrox server BMCs by vendor.
const consoleDevices = new Set([0x101300b8, 0x12341111, 0x1b360100, 0x1af41050, 0x15ad0405, 0x14145353]);
const consoleVendors = new Set([0x1a03, 0x102b]);

// Intel counts only with a discrete card (off PCI bus 0, where Intel iGPUs sit): the intel backend
// excludes integrated GPUs, so an iGPU-only host stays unselected and keeps mining on its iGPU.
// An unreadable vendor counts as an unknown GPU, which can only make the host look mixed.
/** @param {DisplayAdapter[]} [adapters] @returns {string} */
function detectGpuBackend(adapters = displayAdapters()) {
  const vendors = new Set();
  let intelDiscrete = false;
  for (const {vendor, device, bus} of adapters) {
    if (consoleVendors.has(vendor) || consoleDevices.has(vendor * 0x10000 + device)) {continue;}
    vendors.add(vendor === 0x8086 ? "intel" : vendor === 0x10de ? "nvidia" : vendor === 0x1002 ? "amd" : "opencl");
    if (vendor === 0x8086 && Number.isInteger(bus) && bus > 0) {intelDiscrete = true;}
  }
  const [only] = vendors;
  return vendors.size === 1 && (only !== "intel" || intelDiscrete) ? only : "";
}

module.exports = {displayAdapters, detectGpuBackend};
