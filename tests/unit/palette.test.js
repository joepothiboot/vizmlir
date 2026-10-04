import { describe, expect, it } from "vitest";
import { filterItems, fuzzyScore } from "../../src/app/palette.js";

const item = (text, hint = "") => ({ group: "op", text, hint, run() {} });

describe("fuzzyScore", () => {
  it("prefers direct substring matches", () => {
    expect(fuzzyScore("fill", "linalg.fill")).toBe(7);
    expect(fuzzyScore("lfl", "linalg.fill")).toBeGreaterThanOrEqual(100);
  });

  it("rejects queries that are not a subsequence", () => {
    expect(fuzzyScore("zzz", "linalg.fill")).toBeNull();
  });

  it("matches everything for an empty query", () => {
    expect(fuzzyScore("", "anything")).toBe(0);
  });
});

describe("filterItems", () => {
  it("ranks better matches first and keeps order for ties", () => {
    const items = [
      item("arith.constant"),
      item("linalg.matmul"),
      item("@matmul"),
    ];
    expect(filterItems(items, "matmul").map((i) => i.text)).toEqual([
      "@matmul",
      "linalg.matmul",
    ]);
    expect(filterItems(items, "").map((i) => i.text)).toEqual([
      "arith.constant",
      "linalg.matmul",
      "@matmul",
    ]);
  });

  it("searches hints too, below direct matches", () => {
    const items = [
      item("linalg.fill", "#4 · in func.func @matmul"),
      item("func.func @matmul", "#2"),
    ];
    expect(filterItems(items, "@matmul").map((i) => i.text)).toEqual([
      "func.func @matmul",
      "linalg.fill",
    ]);
  });
});
