"use strict";

// Minimal reader for the YAML subset used by .github/: block mappings and sequences, plain and quoted
// scalars, single-line flow sequences, and literal/folded block scalars. Scalars stay strings. Anything
// outside this subset throws, so a workflow change that needs more YAML must extend this reader.

/** @typedef {string | YamlValue[] | {[key: string]: YamlValue}} YamlValue */
/** @typedef {{indent: number, text: string, number: number}} Line */

/** Remove a trailing comment that starts after whitespace outside quotes. @param {string} text */
function stripComment(text) {
  let quote = "";
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quote) {
      if (char === quote) {quote = "";}
    } else if (char === "\"" || char === "'") {
      if (index === 0 || /[\s:[,-]/.test(text[index - 1] || "")) {quote = char;}
    } else if (char === "#" && (index === 0 || /\s/.test(text[index - 1] || ""))) {
      return text.slice(0, index).trimEnd();
    }
  }
  return text.trimEnd();
}

/** Parse one inline scalar or single-line flow sequence. @param {string} text @param {number} number @returns {YamlValue} */
function scalar(text, number) {
  if (text.startsWith("\"")) {
    if (!text.endsWith("\"") || text.length < 2) {throw new Error("Unterminated double-quoted scalar at line " + number);}
    return JSON.parse(text);
  }
  if (text.startsWith("'")) {
    if (!text.endsWith("'") || text.length < 2) {throw new Error("Unterminated single-quoted scalar at line " + number);}
    return text.slice(1, -1).replaceAll("''", "'");
  }
  if (text.startsWith("[")) {
    if (!text.endsWith("]")) {throw new Error("Multi-line flow sequences are unsupported at line " + number);}
    const body = text.slice(1, -1).trim();
    return body ? body.split(",").map(item => scalar(item.trim(), number)) : [];
  }
  if (text.startsWith("{")) {
    if (text !== "{}") {throw new Error("Flow mappings are unsupported at line " + number);}
    return {};
  }
  return text;
}

/** Split "key: value" outside quotes and expressions. @param {string} text */
function splitKey(text) {
  const match = /^([A-Za-z0-9_.-]+|"[^"]*"|'[^']*'):(?:\s+(.*))?$/.exec(text);
  if (!match || !match[1]) {return null;}
  const key = match[1].replace(/^["']|["']$/g, "");
  return {key, rest: match[2] || ""};
}

class Reader {
  /** @param {string} source */
  constructor(source) {
    /** @type {Line[]} */
    this.lines = [];
    /** @type {string[]} */
    this.raw = source.replace(/\r\n/g, "\n").split("\n");
    this.position = 0;
  }

  /** Next significant line, skipping blanks and comments. */
  peek() {
    while (this.position < this.raw.length) {
      const raw = this.raw[this.position] || "";
      if (raw.includes("\t")) {throw new Error("Tabs are unsupported at line " + (this.position + 1));}
      const text = stripComment(raw);
      if (text.trim()) {return {indent: text.length - text.trimStart().length, text: text.trim(), number: this.position + 1};}
      this.position += 1;
    }
    return null;
  }

  /** Collect a literal or folded block scalar that belongs to a key at `parentIndent`.
   * @param {string} style @param {number} parentIndent */
  block(style, parentIndent) {
    /** @type {string[]} */
    const lines = [];
    let indent = -1;
    while (this.position < this.raw.length) {
      const raw = this.raw[this.position] || "";
      if (raw.trim()) {
        const current = raw.length - raw.trimStart().length;
        if (current <= parentIndent) {break;}
        if (indent < 0) {indent = current;}
        lines.push(raw.slice(indent));
      } else {
        lines.push("");
      }
      this.position += 1;
    }
    while (lines.length && !lines[lines.length - 1]) {lines.pop();}
    const text = style.startsWith(">") ? lines.join(" ").replace(/ +\n/g, "\n") : lines.join("\n");
    return style.endsWith("-") ? text : text + "\n";
  }

  /** Value after "key:" (or "- "): inline scalar, block scalar, or nested block.
   * @param {string} rest @param {number} indent @param {number} number @returns {YamlValue} */
  value(rest, indent, number) {
    if (/^[|>]-?$/.test(rest)) {return this.block(rest, indent);}
    if (rest) {return scalar(rest, number);}
    const next = this.peek();
    if (!next || next.indent < indent || next.indent === indent && !next.text.startsWith("- ") && next.text !== "-") {return "";}
    return this.node(next.indent);
  }

  /** Parse the mapping or sequence starting at `indent`. @param {number} indent @returns {YamlValue} */
  node(indent) {
    const first = this.peek();
    if (!first) {return "";}
    return first.text === "-" || first.text.startsWith("- ") ? this.sequence(indent) : this.mapping(indent);
  }

  /** @param {number} indent */
  sequence(indent) {
    /** @type {YamlValue[]} */
    const items = [];
    for (let line = this.peek(); line && line.indent === indent && (line.text === "-" || line.text.startsWith("- ")); line = this.peek()) {
      const rest = line.text.slice(1).trimStart();
      this.position += 1;
      const pair = splitKey(rest);
      if (pair && !rest.startsWith("\"") && !rest.startsWith("'")) {
        // "- key: value" opens a mapping whose further keys align with "key".
        this.raw.splice(this.position, 0, " ".repeat(indent + 2) + rest);
        items.push(this.mapping(indent + 2));
      } else {
        items.push(this.value(rest, indent + 1, line.number));
      }
    }
    return items;
  }

  /** @param {number} indent */
  mapping(indent) {
    /** @type {{[key: string]: YamlValue}} */
    const map = {};
    for (let line = this.peek(); line && line.indent === indent && !line.text.startsWith("- "); line = this.peek()) {
      const pair = splitKey(line.text);
      if (!pair) {throw new Error("Expected a mapping key at line " + line.number + ": " + line.text);}
      if (Object.hasOwn(map, pair.key)) {throw new Error("Duplicate key " + pair.key + " at line " + line.number);}
      this.position += 1;
      map[pair.key] = this.value(pair.rest, indent, line.number);
    }
    const next = this.peek();
    if (next && next.indent > indent) {throw new Error("Unexpected indentation at line " + next.number);}
    return map;
  }
}

/** Parse a workflow or action file. @param {string} source @returns {YamlValue} */
function parseWorkflowYaml(source) {
  const reader = new Reader(source), root = reader.node(0);
  if (reader.peek()) {throw new Error("Trailing content at line " + (reader.peek()?.number || 0));}
  return root;
}

module.exports = {parseWorkflowYaml};
