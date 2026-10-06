"use strict";

// Loaded via `node --require` so its side effects (patching console + an exit
// hook) install before any test runs. Buffers console output and, when the
// process exits non-zero with NODE_TEST_FLUSH_BUFFERED_OUTPUT=1, replays it so
// debug logs from failing tests are visible without spamming passing runs.

const {format} = require("node:util");
const fs = require("node:fs");

/** @type {ConsoleMethod[]} */
const METHODS = ["log", "info", "warn", "error"];
const MAX_BUFFERED_CHARACTERS = 256 * 1024;

/** @typedef {"log" | "info" | "warn" | "error"} ConsoleMethod */
/** @type {Array<{method: ConsoleMethod, text: string}>} */
const buffered = [];
let bufferedCharacters = 0;
let flushed = false;

/** @param {ConsoleMethod} method @param {unknown[]} args */
function bufferConsoleOutput(method, args) {
  const text = format(...args);
  const retained = text.length > MAX_BUFFERED_CHARACTERS
    ? text.slice(-MAX_BUFFERED_CHARACTERS) : text;
  buffered.push({method, text: retained});
  bufferedCharacters += retained.length + 1;
  while (bufferedCharacters > MAX_BUFFERED_CHARACTERS && buffered.length > 1) {
    const removed = buffered.shift();
    if (removed) {bufferedCharacters -= removed.text.length + 1;}
  }
}

for (const method of METHODS) {
  console[method] = (...args) => bufferConsoleOutput(method, args);
}

function flushBufferedOutput() {
  if (flushed || buffered.length === 0) {return;}
  flushed = true;

  // The process `exit` event cannot wait for asynchronous console streams. Synchronous writes make
  // the failure tail deterministic under a heavily parallel test run.
  fs.writeSync(2, "\nSuppressed debug output:\n");
  for (const {method, text} of buffered) {
    fs.writeSync(method === "log" || method === "info" ? 1 : 2, text + "\n");
  }
}

process.on("exit", (code) => {
  if (code !== 0 && process.env["NODE_TEST_FLUSH_BUFFERED_OUTPUT"] === "1") {flushBufferedOutput();}
});
