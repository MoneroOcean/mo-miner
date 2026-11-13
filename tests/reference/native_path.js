"use strict";

const path = require("node:path");

module.exports = function pathFixture() {
  return [path.posix, path.win32].map((api) => ({
    join: [[], [""], ["a", "."], ["a", "../"], ["//server//", "share/"],
      ["/root", "/child", "..", "file"], ["C:\\work", "..", "file"], ["é", "💎"]].map((parts) => api.join(...parts)),
    paths: ["", ".", "./", "/", "///", "..", "a/..", "a/../", "/a/../../b", "a\\b", "/é/💎.txt",
      "C:", "C:\\", "C:relative", "C:\\work\\..\\file.txt", "\\\\server\\share\\work\\",
      "\\\\?\\C:\\path"].map((value) => ({
      normalize: api.normalize(value), absolute: api.isAbsolute(value), dirname: api.dirname(value),
      basename: api.basename(value), extname: api.extname(value), parsed: api.parse(value),
      formatted: api.format(api.parse(value)), namespaced: api.toNamespacedPath(value),
    })),
    // Absolute inputs keep this transcript independent of the host/Windows drive working directory.
    resolve: api.resolve("C:\\base", "/root", "..", "é"),
    relative: [["/a/b", "/a/c"], ["C:\\a", "C:\\b"], ["\\\\s\\share\\a", "\\\\s\\share\\b"]]
      .map(([from, to]) => api.relative(from || "", to || "")),
    extension: api.format({dir: "/root", name: "file", ext: "txt"}),
    separators: [api.sep, api.delimiter],
  }));
};
