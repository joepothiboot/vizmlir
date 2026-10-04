// @vitest-environment jsdom

import { describe, expect, it } from "vitest";
import { bindHighlighting, highlightMlir } from "../../src/app/mlir-highlight.js";

function tokens(source) {
  const layer = document.createElement("code");
  layer.innerHTML = highlightMlir(source);
  return [...layer.querySelectorAll("span")].map((span) => [
    span.className.replace("syntax-", ""),
    span.textContent,
  ]);
}

describe("highlightMlir", () => {
  it.each([
    ["// comment {", "comment"],
    ['"str \\" still"', "string"],
    ["'c'", "string"],
    ["%arg0", "ssa"],
    ["%a.b$1", "ssa"],
    ["@main", "symbol"],
    ["module", "keyword"],
    ["func", "keyword"],
    ["dense", "keyword"],
    ["tensor", "type"],
    ["i1", "type"],
    ["i64", "type"],
    ["f16", "type"],
    ["bf16", "type"],
    ["index", "type"],
    ["42", "number"],
    ["1.5e-3", "number"],
    ["arith.addi", "operation"],
    ["foo", "identifier"],
  ])("%s is %s", (source, kind) => {
    expect(tokens(source)).toEqual([[kind, source]]);
  });

  // Known bug: `func` precedes `func\.func` in tokenPattern, so the longer
  // alternative never matches and `func.func` is split into `func` `.` `func`.
  it.fails("func.func is one keyword token", () => {
    expect(tokens("func.func")).toEqual([["keyword", "func.func"]]);
  });

  it("tokenizes a full line", () => {
    expect(tokens("%0 = arith.addi %a, %b : i32 // sum")).toEqual([
      ["ssa", "%0"],
      ["operation", "arith.addi"],
      ["ssa", "%a"],
      ["ssa", "%b"],
      ["type", "i32"],
      ["comment", "// sum"],
    ]);
  });

  it("escapes HTML inside and outside tokens", () => {
    const source = `<img src=x onerror="alert(1)"> & 'x' <script>alert(1)</script>`;
    const html = highlightMlir(source);
    expect(html).not.toMatch(/<(img|script)/);
    const layer = document.createElement("code");
    layer.innerHTML = html;
    expect(layer.querySelector("img, script")).toBeNull();
    expect(layer.textContent).toBe(`${source}\n`);
  });

  it("round-trips any text and adds a trailing newline", () => {
    const source =
      'module {\n  %x = "t.op"() {a = "<&>"} : () -> tensor<2xf32>\n}\n\t// ü';
    const layer = document.createElement("code");
    layer.innerHTML = highlightMlir(source);
    expect(layer.textContent).toBe(`${source}\n`);
  });
});

describe("bindHighlighting", () => {
  function setup(value) {
    const textarea = document.createElement("textarea");
    const pre = document.createElement("pre");
    const layer = pre.appendChild(document.createElement("code"));
    textarea.value = value;
    bindHighlighting(textarea, layer);
    return { textarea, layer, pre };
  }

  it("renders immediately and again on input", () => {
    const { textarea, layer } = setup("%a");
    expect(layer.querySelector(".syntax-ssa").textContent).toBe("%a");
    textarea.value = "@b";
    textarea.dispatchEvent(new Event("input"));
    expect(layer.querySelector(".syntax-symbol").textContent).toBe("@b");
  });

  it("follows the textarea scroll position", () => {
    const { textarea, pre } = setup("x\n".repeat(200));
    Object.defineProperty(textarea, "scrollTop", {
      value: 120,
      configurable: true,
    });
    Object.defineProperty(textarea, "scrollLeft", {
      value: 30,
      configurable: true,
    });
    textarea.dispatchEvent(new Event("scroll"));
    expect(pre.scrollTop).toBe(120);
    expect(pre.scrollLeft).toBe(30);
  });

  it("toggles the focused class", () => {
    const { textarea, layer } = setup("");
    textarea.dispatchEvent(new Event("focus"));
    expect(layer.classList.contains("focused")).toBe(true);
    textarea.dispatchEvent(new Event("blur"));
    expect(layer.classList.contains("focused")).toBe(false);
  });
});
