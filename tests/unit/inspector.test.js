// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createInspector } from "../../src/app/inspector.js";
import { createScrubber } from "../../src/app/scrubber.js";

// The real markup from index.html, so the ids and roles under test are the
// ones the app ships.
const page = new DOMParser().parseFromString(
  readFileSync(new FileURL("../../index.html", import.meta.url), "utf8"),
  "text/html",
);
const take = (id) => document.importNode(page.getElementById(id), true);

const key = (target, k) =>
  target.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true }));
const panel = (name) => document.querySelector(`[data-panel="${name}"]`);

describe("inspector", () => {
  let root;
  let toggle;
  let editor;
  let onShow;
  let inspector;

  beforeEach(() => {
    document.body.replaceChildren();
    editor = document.createElement("textarea");
    toggle = take("inspector-toggle");
    root = take("inspector");
    document.body.append(editor, toggle, root);
    onShow = vi.fn();
    inspector = createInspector(root, {
      toggle,
      closeButton: root.querySelector("#inspector-close"),
      onShow,
    });
  });

  it("starts closed, with the tabs from the page", () => {
    expect(root.hidden).toBe(true);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    const tabs = [...root.querySelectorAll('[role="tab"]')].map((t) => t.textContent.trim());
    expect(tabs).toEqual(["Changes", "Line", "Timing", "Buffers", "Op count", "Symbols", "History"]);
  });

  it("opens on the tab for a selection, and draws it", () => {
    editor.focus();
    inspector.show("line", { from: editor });
    expect(root.hidden).toBe(false);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(inspector.active).toBe("line");
    expect(panel("line").hidden).toBe(false);
    expect(panel("changes").hidden).toBe(true);
    expect(onShow).toHaveBeenLastCalledWith("line");
    // Opening does not take focus from what was selected.
    expect(document.activeElement).toBe(editor);
  });

  it("closes on Escape and returns focus to the selected element", () => {
    inspector.show("line", { from: editor });
    const tab = root.querySelector("#tab-line");
    tab.focus();
    key(tab, "Escape");
    expect(root.hidden).toBe(true);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(editor);
  });

  it("leaves Escape in a field to the page, which closes it after", () => {
    inspector.show("opcount");
    const filter = root.querySelector("#opcount-filter");
    filter.focus();
    key(filter, "Escape");
    expect(root.hidden).toBe(false);
    expect(inspector.handleEscape()).toBe(true);
    expect(root.hidden).toBe(true);
    expect(inspector.handleEscape()).toBe(false);
  });

  it("falls back to the toggle when the selected element is gone", () => {
    const row = document.createElement("button");
    document.body.append(row);
    inspector.show("line", { from: row });
    row.remove();
    inspector.close();
    expect(document.activeElement).toBe(toggle);
  });

  it("switches tabs by click and arrow keys", () => {
    inspector.open();
    root.querySelector("#tab-timing").click();
    expect(inspector.active).toBe("timing");
    expect(panel("timing").hidden).toBe(false);
    expect(panel("changes").hidden).toBe(true);
    expect(onShow).toHaveBeenLastCalledWith("timing");
    const timing = root.querySelector("#tab-timing");
    expect(timing.getAttribute("aria-selected")).toBe("true");
    expect(timing.tabIndex).toBe(0);

    key(timing, "ArrowRight");
    expect(inspector.active).toBe("buffers");
    expect(document.activeElement).toBe(root.querySelector("#tab-buffers"));
    key(document.activeElement, "End");
    expect(inspector.active).toBe("ophistory");
    key(document.activeElement, "ArrowRight");
    expect(inspector.active).toBe("changes");
  });

  it("toggles a tab: shows it, then closes when it is already shown", () => {
    inspector.toggleTab("buffers");
    expect(root.hidden).toBe(false);
    expect(inspector.active).toBe("buffers");
    inspector.toggleTab("symbols");
    expect(inspector.active).toBe("symbols");
    expect(root.hidden).toBe(false);
    inspector.toggleTab("symbols");
    expect(root.hidden).toBe(true);
  });

  it("reopens from the toggle button on the tab it had", () => {
    inspector.show("symbols");
    root.querySelector("#inspector-close").click();
    expect(root.hidden).toBe(true);
    toggle.click();
    expect(root.hidden).toBe(false);
    expect(inspector.active).toBe("symbols");
    expect(document.activeElement).toBe(root.querySelector("#tab-symbols"));
    toggle.click();
    expect(root.hidden).toBe(true);
  });

  it("reports open and tab changes, for remembering them", () => {
    const onChange = vi.fn();
    document.body.replaceChildren(take("inspector"));
    const other = createInspector(document.getElementById("inspector"), {
      onChange,
      tab: "opcount",
    });
    expect(other.active).toBe("opcount");
    other.open();
    expect(onChange).toHaveBeenLastCalledWith({ open: true, tab: "opcount" });
    other.show("line");
    expect(onChange).toHaveBeenLastCalledWith({ open: true, tab: "line" });
    other.close();
    expect(onChange).toHaveBeenLastCalledWith({ open: false, tab: "line" });
  });
});

describe("pass scrubber", () => {
  let root;
  let range;
  let onSelect;
  let scrubber;
  const names = ["canonicalize", "cse", "gpu-kernel-outlining", "convert-gpu-to-nvvm"];

  beforeEach(() => {
    document.body.replaceChildren(take("pass-scrubber"));
    root = document.getElementById("pass-scrubber");
    range = document.getElementById("pass-range");
    onSelect = vi.fn();
    scrubber = createScrubber(root, {
      onSelect,
      schedule: (run) => run(),
      label: (i) => `Pass ${i + 1} of 4: ${names[i]}`,
      preview: (i) => ({ title: `${i + 1}. ${names[i]}`, lines: [`IR ${i + 1} KB`, "changed ops: 2"] }),
    });
    scrubber.setPasses(names.map((_, i) => ({ failed: i === 3, share: i / 3 })));
    scrubber.setValue(0);
  });

  it("is a 1 … N range with a readable value", () => {
    expect(range.min).toBe("1");
    expect(range.max).toBe("4");
    expect(range.value).toBe("1");
    expect(range.getAttribute("aria-valuetext")).toBe("Pass 1 of 4: canonicalize");
    expect(root.querySelectorAll(".scrub-ticks span")).toHaveLength(4);
    expect(root.querySelector(".scrub-ticks .failed")).toBe(root.querySelectorAll(".scrub-ticks span")[3]);
    expect(root.querySelector(".scrub-prev").disabled).toBe(true);
  });

  it("changes the active pass when the slider moves", () => {
    range.value = "3";
    range.dispatchEvent(new Event("input", { bubbles: true }));
    expect(onSelect).toHaveBeenLastCalledWith(2);
    expect(range.getAttribute("aria-valuetext")).toBe("Pass 3 of 4: gpu-kernel-outlining");
    expect(root.querySelector(".scrub-label").textContent).toBe("Pass 3 of 4: gpu-kernel-outlining");
  });

  it("steps with the next and previous buttons, within 1 … N", () => {
    root.querySelector(".scrub-next").click();
    expect(onSelect).toHaveBeenLastCalledWith(1);
    scrubber.step(5);
    expect(onSelect).toHaveBeenLastCalledWith(3);
    expect(root.querySelector(".scrub-next").disabled).toBe(true);
    onSelect.mockClear();
    scrubber.step(1);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("coalesces a drag into one selection per frame", () => {
    const frames = [];
    document.body.replaceChildren(take("pass-scrubber"));
    const select = vi.fn();
    const other = createScrubber(document.getElementById("pass-scrubber"), {
      onSelect: select,
      schedule: (run) => frames.push(run),
    });
    other.setPasses(names.map(() => ({ failed: false, share: null })));
    const slider = document.getElementById("pass-range");
    for (const value of ["2", "3", "4"]) {
      slider.value = value;
      slider.dispatchEvent(new Event("input"));
    }
    expect(select).not.toHaveBeenCalled();
    frames.forEach((run) => run());
    expect(select).toHaveBeenCalledTimes(1);
    expect(select).toHaveBeenCalledWith(3);
  });

  it("previews a pass on focus and hover without selecting it", () => {
    const tip = document.getElementById("pass-preview");
    range.dispatchEvent(new FocusEvent("focus"));
    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toContain("1. canonicalize");
    expect(tip.textContent).toContain("changed ops: 2");

    range.getBoundingClientRect = () => ({ left: 0, width: 300, top: 0, height: 20 });
    range.dispatchEvent(new MouseEvent("pointermove", { clientX: 200, bubbles: true }));
    expect(tip.textContent).toContain("3. gpu-kernel-outlining");
    expect(onSelect).not.toHaveBeenCalled();

    range.dispatchEvent(new FocusEvent("blur"));
    expect(tip.hidden).toBe(true);
  });

  it("shows a selected pass without reporting it back", () => {
    scrubber.setValue(2);
    expect(range.value).toBe("3");
    expect(root.querySelector(".scrub-ticks .current")).toBe(root.querySelectorAll(".scrub-ticks span")[2]);
    expect(onSelect).not.toHaveBeenCalled();
  });
});
