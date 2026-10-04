// Keeps src/wasm/abi.js in step with wasm/src/abi.rs. The Rust file is the
// source of truth; every constant the bridge reads must match it.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import * as abi from "../../src/ir/abi.js";

const rustSource = readFileSync(
  new URL("../../wasm/src/abi.rs", import.meta.url),
  "utf8",
);

function evalRustInt(expr) {
  const cleaned = expr.replace(/_/g, "").trim();
  if (cleaned === "u32::MAX") return 0xffffffff;
  if (/^[0-9x*\s()a-f]+$/i.test(cleaned))
    return Function(`return ${cleaned}`)();
  throw new Error(`cannot evaluate Rust constant expression: ${expr}`);
}

const rust = Object.fromEntries(
  [...rustSource.matchAll(/pub const (\w+): \w+ = ([^;]+);/g)].map(
    ([, name, value]) => [name, evalRustInt(value)],
  ),
);

function rustGroup(prefix) {
  return Object.fromEntries(
    Object.entries(rust)
      .filter(([name]) => name.startsWith(prefix))
      .map(([name, value]) => [name.slice(prefix.length), value]),
  );
}

describe("ABI constants", () => {
  it("parses constants out of abi.rs", () => {
    expect(Object.keys(rust).length).toBeGreaterThan(20);
  });

  it("magic, version and NONE match", () => {
    expect(abi.ABI_MAGIC).toBe(rust.ABI_MAGIC);
    expect(abi.ABI_VERSION).toBe(rust.ABI_VERSION);
    expect(abi.NONE).toBe(rust.NONE);
  });

  it.each([
    ["HDR_", "HDR"],
    ["STRIDE_", "STRIDE"],
    ["STATUS_", "STATUS"],
    ["KIND_", "KIND"],
  ])("%s* matches abi.js %s", (prefix, jsName) => {
    expect({ ...abi[jsName] }).toEqual(rustGroup(prefix));
  });

  it("DIAG_* codes all have a message", () => {
    const codes = Object.values(rustGroup("DIAG_")).sort();
    expect(Object.keys(abi.DIAG_CODE).map(Number).sort()).toEqual(codes);
  });

  it("every status has text and every kind has a style", () => {
    for (const code of Object.values(abi.STATUS)) {
      expect(abi.STATUS_TEXT[code]).toBeTypeOf("string");
    }
    for (const kind of Object.values(abi.KIND)) {
      expect(abi.KIND_STYLE[kind]).toBeDefined();
    }
  });

  it("TOO_LARGE text names the real input capacity", () => {
    const mib = rust.INPUT_CAP / (1024 * 1024);
    expect(abi.STATUS_TEXT[abi.STATUS.TOO_LARGE]).toContain(`${mib} MiB`);
  });
});
