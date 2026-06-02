"use strict";

const { Transform } = require("node:stream");
const { spec } = require("node:test/reporters");

function formatDurationMs(durationMs, raw = String(durationMs)) {
  if (durationMs >= 60 * 1000) return `${(durationMs / (60 * 1000)).toFixed(2)} min`;
  if (durationMs >= 1000) return `${(durationMs / 1000).toFixed(2)} s`;
  return `${raw}ms`;
}

function rewriteReporterDurations(text) {
  return text
    .replace(/\(([0-9]+(?:\.[0-9]+)?)ms\)/g, (_match, raw) =>
      `(${formatDurationMs(Number.parseFloat(raw), raw)})`)
    .replace(/\bduration_ms ([0-9]+(?:\.[0-9]+)?)/g, (match, raw) => {
      const formatted = formatDurationMs(Number.parseFloat(raw), raw);
      return formatted.endsWith("ms") ? match : `duration ${formatted}`;
    });
}

class SpacedSpecReporter extends Transform {
  constructor() {
    super({ writableObjectMode: true });
    this.pendingText = "";
    this.lastPrintedNonEmptyLine = "";
    this.reporter = spec();
    this.reporter.on("data", (chunk) => {
      this.push(this.rewriteText(Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk)));
    });
    this.reporter.on("error", (error) => {
      this.destroy(error);
    });
  }

  rewriteText(text) {
    this.pendingText += text;
    let output = "";
    let newlineIndex = this.pendingText.indexOf("\n");
    while (newlineIndex !== -1) {
      const line = this.pendingText.slice(0, newlineIndex + 1);
      this.pendingText = this.pendingText.slice(newlineIndex + 1);
      output += this.rewriteLine(line);
      newlineIndex = this.pendingText.indexOf("\n");
    }
    return output;
  }

  rewriteLine(line) {
    let rewritten = rewriteReporterDurations(line);
    if (/^\s*▶ /.test(rewritten) && this.lastPrintedNonEmptyLine) rewritten = "\n" + rewritten;
    if (rewritten.trim()) this.lastPrintedNonEmptyLine = rewritten.trimEnd();
    return rewritten;
  }

  _transform(event, encoding, callback) {
    if (this.reporter.write(event, encoding)) return callback();
    this.reporter.once("drain", callback);
  }

  _flush(callback) {
    this.reporter.end();
    this.reporter.once("end", () => {
      if (this.pendingText) {
        let output = this.pendingText;
        if (/^\s*▶ /.test(output) && this.lastPrintedNonEmptyLine) output = "\n" + output;
        this.push(output);
        this.pendingText = "";
      }
      callback();
    });
  }
}

module.exports = SpacedSpecReporter;
module.exports.formatDurationMs = formatDurationMs;
module.exports.rewriteReporterDurations = rewriteReporterDurations;
