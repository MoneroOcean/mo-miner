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
      "array-callback-return": "error",
      "accessor-pairs": "error",
      "block-scoped-var": "error",
      "curly": ["error", "all"],
      "default-case": "error",
      "default-case-last": "error",
      "dot-notation": "error",
      "grouped-accessor-pairs": "error",
      "no-alert": "error",
      "no-array-constructor": "error",
      "no-async-promise-executor": "error",
      "no-caller": "error",
      "no-constructor-return": "error",
      "no-div-regex": "error",
      "no-duplicate-imports": "error",
      "no-else-return": ["error", { allowElseIf: false }],
      "no-empty-function": "error",
      "no-eval": "error",
      "no-extend-native": "error",
      "no-extra-bind": "error",
      "no-floating-decimal": "error",
      "no-implied-eval": "error",
      "no-iterator": "error",
      "no-labels": "error",
      "no-lone-blocks": "error",
      "no-lonely-if": "error",
      "no-loop-func": "error",
      "no-multi-assign": "error",
      "no-new": "error",
      "no-new-func": "error",
      "no-new-wrappers": "error",
      "no-object-constructor": "error",
      "no-return-assign": "error",
      "no-proto": "error",
      "no-return-await": "error",
      "no-script-url": "error",
      "no-self-compare": "error",
      "no-sequences": "error",
      "no-template-curly-in-string": "error",
      "no-unneeded-ternary": "error",
      "no-unreachable-loop": "error",
      "no-useless-call": "error",
      "no-useless-computed-key": "error",
      "no-useless-concat": "error",
      "no-useless-rename": "error",
      "no-useless-return": "error",
      "prefer-object-has-own": "error",
      "prefer-promise-reject-errors": "error",
      "prefer-regex-literals": "error",
      "radix": "error",
      "symbol-description": "error",
      "unicode-bom": ["error", "never"],
      "yoda": "error",
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
