"use strict";

// Flat ESLint config (eslint v9+). Enforces the MoneroOcean JS conventions in ../style.md
// (2-space indent IN THIS REPO, double quotes, semicolons, const/let) plus a strict correctness
// baseline. camelcase is OFF: snake_case identifiers here mirror external contracts (stratum/RPC/
// wire/config keys), which style.md exempts and renaming would break (~687 sites).

const js = require("@eslint/js");
const stylistic = require("@stylistic/eslint-plugin");
const globals = require("globals");

module.exports = [
  {
    ignores: [
      "build/**",
      "node_modules/**",
      "release*/**",
      "**/*.bundle.cjs",
      "coverage/**",
    ],
  },
  js.configs.recommended,
  {
    files: ["**/*.js"],
    plugins: { "@stylistic": stylistic },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "commonjs",
      globals: { ...globals.node },
    },
    rules: {
      // ../style.md formatting (the code already follows these -> near-zero-churn guards).
      "@stylistic/indent": ["error", 2, { SwitchCase: 1 }],
      "@stylistic/quotes": ["error", "double", { avoidEscape: true }],
      "@stylistic/semi": ["error", "always"],
      "@stylistic/no-trailing-spaces": "error",
      "@stylistic/eol-last": ["error", "always"],
      // ../style.md declarations.
      "no-var": "error",
      "prefer-const": "error",
      "strict": ["error", "global"],
      "camelcase": "off",
      // strict correctness baseline. null:ignore keeps the intentional `== null`/`!= null` idiom
      // (matches null AND undefined) -- the only loose-equality the code relies on.
      "eqeqeq": ["error", "always", { null: "ignore" }],
      "no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "no-implicit-globals": "error",
      "no-throw-literal": "error",
    },
  },
  {
    // logic.js wraps the whole file in ONE top-level describe() and (deliberately) does not indent
    // its body, to avoid shifting ~1000 lines right by 2 columns for no readability gain. Exempt it
    // from indent rather than churn the diff (style.md: large mechanical reformatting is discouraged).
    files: ["tests/logic.js"],
    rules: { "@stylistic/indent": "off" },
  },
];
