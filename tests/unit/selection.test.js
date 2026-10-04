import { describe, expect, it, vi } from "vitest";
import { createSelection } from "../../src/app/selection.js";

describe("createSelection", () => {
  it("starts with nothing selected", () => {
    expect(createSelection().state).toEqual({ pass: -1, node: -1, line: -1 });
  });

  it("merges a pick into the state and tells listeners which keys it set", () => {
    const selection = createSelection();
    const listener = vi.fn();
    selection.subscribe(listener);
    selection.select({ line: 4 });
    selection.select({ pass: 2 });
    expect(selection.state).toEqual({ pass: 2, node: -1, line: 4 });
    expect(listener).toHaveBeenNthCalledWith(1, { pass: -1, node: -1, line: 4 }, ["line"]);
    expect(listener).toHaveBeenNthCalledWith(2, { pass: 2, node: -1, line: 4 }, ["pass"]);
  });

  it("notifies when the same value is picked again", () => {
    const selection = createSelection();
    const listener = vi.fn();
    selection.subscribe(listener);
    selection.select({ line: 4 });
    selection.select({ line: 4 });
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("ignores unknown keys", () => {
    const selection = createSelection();
    const listener = vi.fn();
    selection.subscribe(listener);
    selection.select({ colour: 1 });
    expect(listener).not.toHaveBeenCalled();
    expect(selection.state).toEqual({ pass: -1, node: -1, line: -1 });
  });

  it("clears a key without notifying", () => {
    const selection = createSelection();
    selection.select({ line: 4, node: 1 });
    const listener = vi.fn();
    selection.subscribe(listener);
    selection.clear("line");
    expect(selection.state).toEqual({ pass: -1, node: 1, line: -1 });
    expect(listener).not.toHaveBeenCalled();
  });

  it("stops notifying after unsubscribe", () => {
    const selection = createSelection();
    const listener = vi.fn();
    const off = selection.subscribe(listener);
    off();
    selection.select({ pass: 1 });
    expect(listener).not.toHaveBeenCalled();
  });
});
