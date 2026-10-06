// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createDescentView } from "../../src/render/descent.js";

function stubContext() {
  const calls = [];

  return new Proxy(
    { calls },
    {
      get(target, name) {
        if (name === "calls") return calls;
        if (name === "measureText") return () => ({ width: 10 });

        return (...args) => calls.push([name, ...args]);
      },
      set: () => true,
    },
  );
}

const shape = (extra = 0) => ({
  parent: [-1, 0, 1, 1, ...Array(extra).fill(1)],
  kind: [0, 1, 2, 2, ...Array(extra).fill(2)],
  line: [0, 1, 2, 3, ...Array(extra).fill(0)],
  edges: [
    [0, 1],
    [1, 2],
    [2, 3],
  ],
});

describe("descent view", () => {
  let root;
  let ctx;
  let handlers;
  let view;

  const data = () => ({
    shapes: [shape(), shape(), shape(1)],
    titles: ["After a", "After b", "After c"],
    names: ["After a", "After b", "After c"],
    labelOf: (pass, node) => `op${node}@${pass}`,
  });

  beforeEach(() => {
    document.body.replaceChildren();
    root = document.createElement("div");
    document.body.append(root);
    ctx = stubContext();
    HTMLCanvasElement.prototype.getContext = () => ctx;

    handlers = {
      onPickNode: vi.fn(),
      onPickPass: vi.fn(),
      onStep: vi.fn(),
      onGpu: vi.fn(),
    };

    view = createDescentView(root, handlers);
  });

  it("asks for a trace until it has data", () => {
    expect(root.querySelector(".descent-empty").hidden).toBe(false);
    expect(root.querySelector("canvas").hidden).toBe(true);
    view.setData(data());
    expect(root.querySelector(".descent-empty").hidden).toBe(true);
    expect(root.querySelector("canvas").hidden).toBe(false);
  });

  it("draws every layer and says which pass is lit", () => {
    view.setData(data());
    view.setState({ pass: 1 });
    view.draw();

    expect(
      ctx.calls
        .filter(([n]) => n === "fillText")
        .some((c) => c[1].startsWith("#2")),
    ).toBe(true);

    expect(root.querySelector(".descent-status").textContent).toContain(
      "After b",
    );

    expect(root.querySelector(".descent-status").textContent).toContain(
      "4 ops",
    );

    expect(root.querySelector("canvas").getAttribute("aria-label")).toContain(
      "pass 2 of 3",
    );
  });

  it("reports a click on a node of the lit layer", () => {
    view.setData(data());
    view.setState({ pass: 1 });
    view.draw();

    const at = view.nodeScreen(1, 2);
    expect(at).not.toBeNull();

    expect(view.hitTest(at.x, at.y)).toEqual({
      kind: "node",
      pass: 1,
      node: 2,
    });
  });

  it("reports a click on a layer away from its nodes as that pass", () => {
    view.setData(data());
    view.setState({ pass: 0 });
    view.draw();

    let found = null;

    for (let x = 0; x < 640 && !found; x += 8) {
      for (let y = 0; y < 420 && !found; y += 8) {
        const hit = view.hitTest(x, y);
        if (hit?.kind === "pass" && hit.pass === 2) found = hit;
      }
    }

    expect(found).toEqual({ kind: "pass", pass: 2 });
  });

  it("passes clicks and keys on to the app", () => {
    view.setData(data());
    view.setState({ pass: 1 });
    view.draw();

    const canvas = root.querySelector("canvas");
    canvas.getBoundingClientRect = () => ({ left: 0, top: 0 });

    const at = view.nodeScreen(1, 3);

    canvas.dispatchEvent(
      new MouseEvent("pointerdown", {
        clientX: at.x,
        clientY: at.y,
        bubbles: true,
      }),
    );

    canvas.dispatchEvent(
      new MouseEvent("pointerup", {
        clientX: at.x,
        clientY: at.y,
        bubbles: true,
      }),
    );

    expect(handlers.onPickNode).toHaveBeenCalledWith(1, 3);

    canvas.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }),
    );

    canvas.dispatchEvent(
      new KeyboardEvent("keydown", { key: "End", bubbles: true }),
    );

    canvas.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true }),
    );

    expect(handlers.onStep.mock.calls.map((c) => c[0])).toEqual([
      1,
      Infinity,
      -1,
    ]);
  });

  it("turns on drag instead of picking", () => {
    view.setData(data());
    view.draw();

    const canvas = root.querySelector("canvas");
    const yaw = view.view.yaw;

    canvas.dispatchEvent(
      new MouseEvent("pointerdown", {
        clientX: 100,
        clientY: 100,
        bubbles: true,
      }),
    );

    canvas.dispatchEvent(
      new MouseEvent("pointermove", {
        clientX: 160,
        clientY: 100,
        bubbles: true,
      }),
    );

    canvas.dispatchEvent(
      new MouseEvent("pointerup", {
        clientX: 160,
        clientY: 100,
        bubbles: true,
      }),
    );

    expect(view.view.yaw).not.toBe(yaw);
    expect(handlers.onPickNode).not.toHaveBeenCalled();
    expect(handlers.onPickPass).not.toHaveBeenCalled();
  });

  it("zooms with the wheel within limits, and resets", () => {
    view.setData(data());

    const canvas = root.querySelector("canvas");

    const wheel = (deltaY) =>
      canvas.dispatchEvent(
        new WheelEvent("wheel", { deltaY, cancelable: true, bubbles: true }),
      );

    wheel(-400);
    expect(view.view.zoom).toBeGreaterThan(1);
    for (let i = 0; i < 40; i++) wheel(-1000);
    expect(view.view.zoom).toBe(4);
    for (let i = 0; i < 80; i++) wheel(1000);
    expect(view.view.zoom).toBe(0.4);

    [...root.querySelectorAll("button")]
      .find((b) => b.textContent === "Reset view")
      .click();

    expect(view.view.zoom).toBe(1);
  });

  it("draws the op's path, in dashes where ops merged", () => {
    view.setData(data());

    view.setState({
      pass: 1,
      node: 2,
      lineage: [
        { pass: 0, node: 2, change: "created", from: [] },
        {
          pass: 1,
          node: 2,
          change: "fused",
          from: [
            { pass: 0, node: 2 },
            { pass: 0, node: 3 },
          ],
        },
      ],
    });

    ctx.calls.length = 0;
    view.draw();

    expect(
      ctx.calls.some(([n, dash]) => n === "setLineDash" && dash.length === 2),
    ).toBe(true);
  });

  it("shows the launch button only when asked", () => {
    view.setData(data());

    const gpu = [...root.querySelectorAll("button")].find((b) =>
      b.textContent.includes("launch"),
    );

    expect(gpu.hidden).toBe(true);
    view.setState({ gpu: true });
    expect(gpu.hidden).toBe(false);
    gpu.click();
    expect(handlers.onGpu).toHaveBeenCalled();
  });

  it("copes with a pass whose IR did not parse and with a huge layer", () => {
    view.setData({ ...data(), shapes: [shape(), null, shape(600)] });
    view.setState({ pass: 2 });
    expect(() => view.draw()).not.toThrow();

    expect(root.querySelector(".descent-status").textContent).toContain(
      "not drawn",
    );

    view.setState({ pass: 1 });

    expect(root.querySelector(".descent-status").textContent).toContain(
      "did not parse",
    );
  });
});

describe("descent view animation", () => {
  let queue;
  let clock;
  let ctx;
  let root;
  let view;

  const shape = () => ({
    parent: [-1, 0, 1, 1],
    kind: [0, 1, 2, 2],
    line: [0, 1, 2, 3],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
    ],
  });

  const data = () => ({
    shapes: [shape(), shape(), shape(), shape()],
    titles: ["a", "b", "c", "d"],
    names: ["a", "b", "c", "d"],
    labelOf: () => "op",
  });

  const lineage = [
    { pass: 0, node: 2, change: "created", from: [] },
    { pass: 1, node: 2, change: "kept", from: [{ pass: 0, node: 2 }] },
    { pass: 2, node: 2, change: "kept", from: [{ pass: 1, node: 2 }] },
  ];

  function settle(max = 400) {
    let frames = 0;

    while (queue.length && frames < max) {
      clock += 16;

      const run = queue.splice(0);
      for (const cb of run) cb(clock);
      frames++;
    }

    return frames;
  }

  beforeEach(() => {
    queue = [];
    clock = 1000;
    vi.stubGlobal("requestAnimationFrame", (cb) => queue.push(cb));
    vi.spyOn(performance, "now").mockImplementation(() => clock);
    vi.stubGlobal("matchMedia", () => ({ matches: false }));
    ctx = stubContext();
    HTMLCanvasElement.prototype.getContext = () => ctx;
    document.body.replaceChildren();
    root = document.createElement("div");
    document.body.append(root);

    view = createDescentView(root, {
      onPickNode() {},
      onPickPass() {},
      onStep() {},
    });

    view.setData(data());
    view.setState({ pass: 0 });
    settle();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("starts at the lit layer and stays put when nothing changes", () => {
    expect(view.focus).toBe(0);
    expect(queue).toHaveLength(0);
  });

  it("glides to the next layer over several frames, then stops drawing", () => {
    view.setState({ pass: 2 });

    const target = view.focus;
    const seen = [];
    let frames = 0;

    while (queue.length && frames < 400) {
      clock += 16;
      queue.splice(0).forEach((cb) => cb(clock));
      seen.push(view.focus);
      frames++;
    }

    expect(frames).toBeGreaterThan(5);
    expect(frames).toBeLessThan(120);

    for (let i = 1; i < seen.length; i++) {
      expect(seen[i]).toBeLessThanOrEqual(seen[i - 1]);
    }

    expect(seen.at(-1)).toBeLessThan(target);
    expect(seen.at(-1)).toBe(-2 * 6);
    expect(queue).toHaveLength(0);
  });

  it("sends a pulse down the op's path, and only while it travels", () => {
    view.setState({ pass: 2, node: 2, lineage });

    const arcs = [];

    while (queue.length) {
      ctx.calls.length = 0;
      clock += 16;
      queue.splice(0).forEach((cb) => cb(clock));
      arcs.push(ctx.calls.some(([n]) => n === "arc"));
    }

    expect(arcs.some(Boolean)).toBe(true);
    expect(arcs.at(-1)).toBe(false);
  });

  it("sends no pulse when the op's life does not cover both passes", () => {
    view.setState({ pass: 3, node: 2, lineage });

    const arcs = [];

    while (queue.length) {
      ctx.calls.length = 0;
      clock += 16;
      queue.splice(0).forEach((cb) => cb(clock));
      arcs.push(ctx.calls.some(([n]) => n === "arc"));
    }

    expect(arcs.some(Boolean)).toBe(false);
  });

  it("jumps with no pulse when the person prefers reduced motion", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: true }));
    view.setState({ pass: 2, node: 2, lineage });

    let arcs = false;

    const frames = (() => {
      let n = 0;

      while (queue.length && n < 50) {
        ctx.calls.length = 0;
        clock += 16;
        queue.splice(0).forEach((cb) => cb(clock));
        arcs ||= ctx.calls.some(([name]) => name === "arc");
        n++;
      }

      return n;
    })();

    expect(frames).toBe(1);
    expect(view.focus).toBe(-12);
    expect(arcs).toBe(false);
  });
});
