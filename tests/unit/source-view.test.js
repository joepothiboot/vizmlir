// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { URL as FileURL } from "node:url";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createSourceView } from "../../src/app/source-view.js";

// The real markup from index.html, so the classes under test are the ones the
// app ships.
const page = new DOMParser().parseFromString(
  readFileSync(new FileURL("../../index.html", import.meta.url), "utf8"),
  "text/html",
);

const SRC = "fn a():\n    pass\n\nfn b():\n    pass\n";

describe("source view", () => {
  let root;
  let view;
  let onPick;
  let onAdd;
  const rows = () => [...root.querySelectorAll(".src-line")];

  beforeEach(() => {
    document.body.replaceChildren();
    root = document.importNode(page.getElementById("source-panel"), true);
    document.body.append(root);
    onPick = vi.fn();
    onAdd = vi.fn();
    view = createSourceView(root, { onPick, onAdd });
  });

  it("lists the lines of the first file, numbered from 1", () => {
    view.setSources({ "k.mojo": SRC });
    expect(rows().map((r) => r.querySelector(".src-t").textContent)).toEqual([
      "fn a():",
      "    pass",
      "",
      "fn b():",
      "    pass",
    ]);
    expect(rows()[4].querySelector(".src-n").textContent).toBe("5");
  });

  it("shows the file picker only when there are several files", () => {
    view.setSources({ "k.mojo": SRC });
    expect(root.querySelector(".src-file").hidden).toBe(true);
    view.setSources({ "k.mojo": SRC, "m.mojo": "x\n" });
    expect(root.querySelector(".src-file").hidden).toBe(false);
    expect(root.querySelectorAll(".src-file option")).toHaveLength(2);
  });

  it("asks for the files the IR names when none are loaded", () => {
    view.setSources({}, ["saxpy.mojo"]);
    const empty = root.querySelector(".src-empty");
    expect(empty.hidden).toBe(false);
    expect(empty.textContent).toContain("saxpy.mojo");
    expect(rows()).toHaveLength(0);
  });

  it("names the files still missing next to loaded ones", () => {
    view.setSources({ "k.mojo": SRC }, ["other.mojo"]);
    expect(root.querySelector(".src-empty").textContent).toContain("other.mojo");
  });

  it("shows how many ops each line produced", () => {
    view.setSources({ "k.mojo": SRC });
    view.setCounts(new Map([[1, 2], [4, 1]]));
    expect(rows()[0].querySelector(".src-ops").textContent).toBe("2");
    expect(rows()[0].classList.contains("has-ops")).toBe(true);
    expect(rows()[1].querySelector(".src-ops").textContent).toBe("");
    expect(rows()[3].title).toBe("1 op from this line");
  });

  it("marks lines, switches to their file, and clears them", () => {
    view.setSources({ "k.mojo": SRC, "m.mojo": "x\ny\n" });
    view.mark("m.mojo", [2]);
    expect(view.file).toBe("m.mojo");
    expect(rows().map((r) => r.classList.contains("hit"))).toEqual([false, true]);
    view.mark(null, []);
    expect(rows().some((r) => r.classList.contains("hit"))).toBe(false);
    expect(view.file).toBe("m.mojo");
  });

  it("reports a click on a line with its file", () => {
    view.setSources({ "k.mojo": SRC });
    rows()[3].querySelector(".src-t").click();
    expect(onPick).toHaveBeenCalledWith("k.mojo", 4);
  });

  it("hands over files that are dropped on it", () => {
    view.setSources({}, []);
    const file = new File(["x"], "k.mojo");
    const drop = new Event("drop", { bubbles: true, cancelable: true });
    drop.dataTransfer = { files: [file], types: ["Files"] };
    root.dispatchEvent(drop);
    expect(onAdd).toHaveBeenCalledWith([file]);
    expect(drop.defaultPrevented).toBe(true);
  });
});
