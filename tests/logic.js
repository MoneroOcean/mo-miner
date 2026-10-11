"use strict";

const {describe} = require("./logic/support");

describe("JavaScript logic tests", () => {
  describe("Core configuration and runtime", () => {
    require("./logic/core");
  });
  describe("Mining orchestration", () => {
    require("./logic/mining");
  });
  describe("Pool transport and protocols", () => {
    require("./logic/pool");
  });
  describe("Native proxy submission", () => {
    require("./logic/proxy_submit");
  });
  describe("Native proxy pool negotiation", () => {
    require("./logic/proxy_pool");
  });
  describe("Proof-of-work protocol adapters", () => {
    require("./logic/pow_protocols");
  });
  describe("ZHash-family protocols", () => {
    require("./logic/zelhash");
  });
  describe("FishHash and Kaspa-family protocols", () => {
    require("./logic/fishhash");
  });
  describe("BeamHash III protocol", () => {
    require("./logic/beamhash");
  });
  describe("Cortex protocol", () => {
    require("./logic/cortex");
  });
  describe("Verthash protocol", () => {
    require("./logic/verthash");
  });
  describe("XELIS v3 protocol", () => {
    require("./logic/xelishashv3");
  });
  describe("NexaPoW protocol", () => {
    require("./logic/nexapow");
  });
  describe("IPC follow-on lifecycle", () => {
    require("./ipc_followon");
  });
});
