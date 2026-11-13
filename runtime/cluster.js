"use strict";

// Native mode uses the existing explicit subprocess-worker route on both OSes. It does not
// emulate Node's cluster IPC: worker role/stdio framing already live in the maintained helper.
module.exports = {
  isPrimary: true,
  fork() {throw new Error("Native workers must use the subprocess route");},
};
