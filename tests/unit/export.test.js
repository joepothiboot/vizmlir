import { describe, expect, it } from "vitest";
import {
  diffRecords,
  diffToJSON,
  diffToMarkdown,
  diffToPatch,
  slug,
} from "../../src/session/export.js";
import { sessionFromFile, sessionToFile } from "../../src/session/storage.js";

const rows = [
  { type: "added", after: { label: "memref.alloc", parent: 1 } },
  {
    type: "changed",
    before: { label: "linalg.fill", parent: 1 },
    after: { label: "linalg.fill_relu", parent: 1 },
  },
  { type: "removed", before: { label: "tensor.empty", parent: -1 } },
];
const parentOf = (row, parent) => (parent === 1 ? "func.func @matmul" : "");

describe("diff export", () => {
  it("builds plain records", () => {
    expect(diffRecords(rows, parentOf)).toEqual([
      { type: "added", op: "memref.alloc", parent: "func.func @matmul" },
      {
        type: "modified",
        op: "linalg.fill_relu",
        parent: "func.func @matmul",
        was: "linalg.fill",
      },
      { type: "removed", op: "tensor.empty", parent: null },
    ]);
  });

  it("writes a Markdown table and escapes pipes", () => {
    const md = diffToMarkdown("t", [
      ...diffRecords(rows, parentOf),
      { type: "added", op: "a|b", parent: null },
    ]);
    expect(md).toContain("| + | `memref.alloc` | in `func.func @matmul` |");
    expect(md).toContain("| ~ | `linalg.fill_relu` | was `linalg.fill` |");
    expect(md).toContain("| − | `tensor.empty` | top level |");
    expect(md).toContain("`a\\|b`");
  });

  it("writes a patch snippet grouped by parent", () => {
    expect(diffToPatch("t", diffRecords(rows, parentOf))).toBe(
      [
        "--- baseline",
        "+++ current  (t)",
        "@@ func.func @matmul @@",
        "+ memref.alloc",
        "- linalg.fill",
        "+ linalg.fill_relu",
        "@@ top level @@",
        "- tensor.empty",
        "",
      ].join("\n"),
    );
  });

  it("says when there is nothing to report", () => {
    expect(diffToMarkdown("t", [])).toContain("No structural changes.");
    expect(JSON.parse(diffToJSON("t", []))).toEqual({
      title: "t",
      changes: [],
    });
  });

  it("makes safe file names", () => {
    expect(slug("trace.txt · pass 3/4")).toBe("trace-txt-pass-3-4");
    expect(slug("…")).toBe("vizmlir");
  });
});

describe("session files", () => {
  it("round-trips", () => {
    const state = { current: "module {}", trace: null };
    expect(sessionFromFile(sessionToFile("demo", state))).toEqual({
      name: "demo",
      state,
    });
  });

  it("rejects other JSON and non-JSON", () => {
    expect(sessionFromFile('{"title":"x"}')).toBeNull();
    expect(sessionFromFile("not json")).toBeNull();
  });
});
