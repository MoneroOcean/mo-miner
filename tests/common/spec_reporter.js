"use strict";

const { Transform } = require("node:stream");
const { spec } = require("node:test/reporters");

// `raw` is the original numeric token from the reporter; sub-second durations
// echo it verbatim so we never reformat (and thus widen) the printed value.
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
    this.reporter.on("data", (chunk) =>
      this.push(this.rewriteText(Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk))));
    this.reporter.on("error", (error) => this.destroy(error));
  }

  // Rewrite complete lines as they arrive; buffer any trailing partial line.
  rewriteText(text) {
    this.pendingText += text;
    let output = "";
    for (let nl; (nl = this.pendingText.indexOf("\n")) !== -1; ) {
      output += this.rewriteLine(this.pendingText.slice(0, nl + 1));
      this.pendingText = this.pendingText.slice(nl + 1);
    }
    return output;
  }

  // Insert a blank line before each test-group header (▶) so groups stand apart,
  // but only once we've already printed something.
  spaceBeforeGroupHeader(text) {
    return /^\s*▶ /.test(text) && this.lastPrintedNonEmptyLine ? "\n" + text : text;
  }

  rewriteLine(line) {
    const rewritten = this.spaceBeforeGroupHeader(rewriteReporterDurations(line));
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
        this.push(this.spaceBeforeGroupHeader(this.pendingText));
        this.pendingText = "";
      }
      callback();
    });
  }
}

module.exports = SpacedSpecReporter;
module.exports.formatDurationMs = formatDurationMs;
module.exports.rewriteReporterDurations = rewriteReporterDurations;
