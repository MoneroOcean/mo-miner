"use strict";

process.env["MOM_CLUSTER_WORKER"] = "1";
process.env["thread_id"] = process.env["MOM_TEST_THREAD_ID"] ?? "0";
process.env["log_level"] = "0";

const {EventEmitter} = require("node:events");
const helper = require("../../helper");
const from = new EventEmitter();

helper.create_core = () => ({
  from,
  emit_to(type, data) {
    if (type === "bench") {
      process.stdout.write("HELPER_STDIN_BENCH " + JSON.stringify(data) + "\n");
    }
    if (type === "close") {setImmediate(() => from.emit("close"));}
  },
});
helper.exit_now = (code) => process.exit(code);
helper.cluster_process();
