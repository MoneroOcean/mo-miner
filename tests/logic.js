"use strict";

const { describe } = require("./logic/support");

describe("JavaScript logic tests", () => {
  require("./logic/core");
  require("./logic/mining");
  require("./logic/pool");
  require("./logic/proxy_submit");
  require("./logic/proxy_pool");
  require("./logic/pow_protocols");
  require("./logic/zelhash");
  require("./logic/fishhash");
  require("./logic/beamhash");
  require("./logic/cortex");
  require("./logic/nexapow");
  require("./logic/verthash");
});
