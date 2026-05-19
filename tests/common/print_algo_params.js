"use strict";

const { resolveMinerCommand, spawnAndExit } = require("./miner_command");

const command = resolveMinerCommand(["mominer.js", "algo_params"]);
spawnAndExit(command[0], command.slice(1));
