// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createScrubber } from "../../src/app/scrubber.js";

const page = new DOMParser().parseFromString(
  readFileSync(new FileURL("../../index.html", import.meta.url), "utf8"),
  "text/html",
);

describe("scrubber op marks", () => {
  let root;
  let scrubber;
  const classes = () =>
    [...root.querySelectorAll(".scrub-ticks span")].map((t) =>
      ["op", "life"].find((c) => t.classList.contains(c)) ?? "",
    );

  beforeEach(() => {
    document.body.replaceChildren();
    root = document.importNode(page.getElementById("pass-scrubber"), true);
    root.hidden = false;
    document.body.append(root);
    scrubber = createScrubber(root, { onSelect: vi.fn(), schedule: (run) => run() });
    scrubber.setPasses([{ failed: false }, { failed: false }, { failed: false }, { failed: false }]);
  });

  it("marks changes and the passes an op merely exists in differently", () => {
    scrubber.setOpMarks([
      { pass: 1, kind: "op" },
      { pass: 2, kind: "life" },
    ]);
    expect(classes()).toEqual(["", "op", "life", ""]);
  });

  it("replaces earlier marks, and clears", () => {
    scrubber.setOpMarks([{ pass: 0, kind: "op" }]);
    scrubber.setOpMarks([{ pass: 3, kind: "life" }]);
    expect(classes()).toEqual(["", "", "", "life"]);
    scrubber.setOpMarks([]);
    expect(classes()).toEqual(["", "", "", ""]);
  });

  it("keeps them when the selected pass changes, and alongside a breakpoint mark", () => {
    scrubber.setOpMarks([{ pass: 1, kind: "op" }]);
    scrubber.setMarks([1]);
    scrubber.setValue(1);
    scrubber.setValue(3);
    const tick = root.querySelectorAll(".scrub-ticks span")[1];
    expect(tick.classList.contains("op")).toBe(true);
    expect(tick.classList.contains("bp")).toBe(true);
  });

  it("survive the ticks being rebuilt for the same trace", () => {
    scrubber.setOpMarks([{ pass: 2, kind: "op" }]);
    scrubber.setPasses([{ failed: false }, { failed: false }, { failed: false }]);
    expect(classes()).toEqual(["", "", "op"]);
  });
});

describe("scrubber breakpoint marks", () => {
  let root;
  let scrubber;
  const marked = () =>
    [...root.querySelectorAll(".scrub-ticks span")].map((t) => t.classList.contains("bp"));

  beforeEach(() => {
    document.body.replaceChildren();
    root = document.importNode(page.getElementById("pass-scrubber"), true);
    root.hidden = false;
    document.body.append(root);
    scrubber = createScrubber(root, { onSelect: vi.fn(), schedule: (run) => run() });
    scrubber.setPasses([{ failed: false }, { failed: true }, { failed: false }, { failed: false }]);
  });

  it("marks the given passes and no others", () => {
    scrubber.setMarks([1, 3]);
    expect(marked()).toEqual([false, true, false, true]);
  });

  it("replaces earlier marks", () => {
    scrubber.setMarks([0]);
    scrubber.setMarks([2]);
    expect(marked()).toEqual([false, false, true, false]);
  });

  it("keeps the marks when the selected pass changes", () => {
    scrubber.setMarks([2]);
    scrubber.setValue(2);
    scrubber.setValue(0);
    expect(marked()).toEqual([false, false, true, false]);
  });

  it("keeps the marks on a failed pass's tick", () => {
    scrubber.setMarks([1]);
    const tick = root.querySelectorAll(".scrub-ticks span")[1];
    expect(tick.classList.contains("failed")).toBe(true);
    expect(tick.classList.contains("bp")).toBe(true);
  });

  it("clears them", () => {
    scrubber.setMarks([1, 2]);
    scrubber.setMarks([]);
    expect(marked().some(Boolean)).toBe(false);
  });
});
