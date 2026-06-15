"use strict";

// Loaded via `node --require` so its side effects (patching console + an exit
// hook) install before any test runs. Buffers console output and, when the
// process exits non-zero with NODE_TEST_FLUSH_BUFFERED_OUTPUT=1, replays it so
// debug logs from failing tests are visible without spamming passing runs.

const METHODS = ["log", "info", "warn", "error"];

const originalConsole = {};
const buffered = [];
let flushed = false;

for (const method of METHODS) {
  originalConsole[method] = console[method].bind(console);
  console[method] = (...args) => buffered.push({ method, args });
}

function flushBufferedOutput() {
  if (flushed || buffered.length === 0) return;
  flushed = true;

  originalConsole.error("");
  originalConsole.error("Suppressed debug output:");
  for (const { method, args } of buffered) {
    originalConsole[method](...args);
  }
}

process.on("exit", (code) => {
  if (code !== 0 && process.env.NODE_TEST_FLUSH_BUFFERED_OUTPUT === "1") flushBufferedOutput();
});
