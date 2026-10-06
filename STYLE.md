# MoM source style

These rules apply to MoM-owned JavaScript, C++, SYCL, shell, and PowerShell. Keep `xmrig/**`
unchanged except for a narrowly required integration fix.

- Match the surrounding language, use two-space indentation and no tabs, and put opening braces on
  the same line. Do not combine sequential statements; a short guard clause may stay on one line.
- Prefer short functions, early returns, and direct ownership of state. Do not hide simple control
  flow behind wrappers or introduce an abstraction without more than one concrete use.
- Use descriptive internal names. Preserve snake_case names that are part of native, configuration,
  environment, pool, or wire contracts.
- JavaScript uses strict CommonJS, `node:` for built-ins, double quotes, semicolons, `const`/`let`,
  and no padding spaces inside object or destructuring braces.
- Keep external values unknown or optional only at genuine input or lookup boundaries; normalize or
  validate once, then keep downstream state and contracts definite. Do not add fake defaults or
  scattered fallback operators just to reduce literal `undefined` occurrences.
- C++ uses fixed-width integer types at binary boundaries, RAII for owned resources, and explicit
  validation before data enters a kernel or worker.
- Comments explain protocol rules, performance constraints, hardware/runtime limits, or surprising
  choices. Do not narrate obvious code, retain stale experiments, or compress logic to save lines.
- Keep tests behavior-focused and deterministic. A bug fix includes the smallest regression test
  that fails without it; avoid tests that pin source spelling or formatting.

Run `npm run lint` after JavaScript changes and the relevant host/build tests after native changes.
