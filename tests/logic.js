"use strict";

const { describe } = require("./logic/support");

describe("JavaScript logic tests", () => {
  require("./logic/core");
  require("./logic/mining");
  require("./logic/pool");
  require("./logic/pow_protocols");
  require("./logic/zelhash");
  require("./logic/fishhash");
  require("./logic/beamhash");
});
