"use strict";

process.env["MOM_CLUSTER_WORKER"] = "1";
process.env["thread_id"] = "0";
process.env["log_level"] = "0";

const {EventEmitter} = require("node:events");
const fs = require("node:fs");
const helper = require("../../helper");
const from = new EventEmitter();
const mode = process.env["MOM_HELPER_SHUTDOWN_MODE"] || "";
/** @type {{closeCalls: number, closeEvents: number, emittedTypes: string[]}} */
const observed = {closeCalls: 0, closeEvents: 0, emittedTypes: []};
let emergencyExits = 0;

process.on("exit", (code) => {
  fs.writeSync(1, "HELPER_SHUTDOWN_RESULT " + JSON.stringify({
    code,
    closeCalls: observed.closeCalls,
    closeEvents: observed.closeEvents,
    emittedTypes: observed.emittedTypes,
    emergencyExits,
  }) + "\n");
});

helper.exit_now = (code) => {
  ++emergencyExits;
  process.exit(code);
};
helper.create_core = () => ({
  from,
  emit_to(type) {
    observed.emittedTypes.push(type);
    if (type === "close") {
      observed.closeCalls++;
      if (observed.closeCalls > 1) {throw new Error("duplicate native close");}
      if (mode === "ipc-signal") {
        process.nextTick(() => process.emit("SIGTERM"));
      }
      setImmediate(() => setImmediate(() => {
        observed.closeEvents++;
        from.emit("close");
      }));
      return;
    }
    if (observed.closeCalls > 0) {
      throw new Error("command after native close: " + type);
    }
  },
});

if (!helper.cluster_process()) {throw new Error("shutdown fixture did not enter worker mode");}
if (mode === "second-signal") {
  process.nextTick(() => {
    process.emit("SIGTERM");
    process.emit("SIGTERM");
  });
}
