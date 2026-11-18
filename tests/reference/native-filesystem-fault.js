"use strict";

const fs = require("node:fs");
const file = process.env["MOM_AOT_FILE_FAULT_PATH"];
if (!file) {throw new Error("Missing private filesystem fault path");}
const data = "partial-é-💎-write";
let error;
try {fs.writeFileSync(file, data, {mode: 0o600});} catch (value) {
  error = value instanceof Error ? {code: Reflect.get(value, "code"), syscall: Reflect.get(value, "syscall")} : "unknown";
}
console.log(JSON.stringify({error, bytes: fs.readFileSync(file).toString("hex")}));
