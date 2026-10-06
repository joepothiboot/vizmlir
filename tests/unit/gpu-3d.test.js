import { describe, expect, it } from "vitest";
import { camera, elementTiles, layoutLaunch } from "../../src/render/gpu-3d.js";

describe("elementTiles", () => {
  const memref = { dims: [1024, 1024], element: "f32" };

  const tiles = (offsetOf, space = "global") => {
    const lanes = Array.from({ length: 32 }, (_, lane) => ({
      lane,
      offset: offsetOf(lane),
      group: lane,
    }));

    return elementTiles(
      { access: { buffer: "%out" }, memref, space },
      { lanes, elementBytes: 4 },
      new Map(lanes.map((l) => [l.group, l.lane])),
    );
  };

  it("lays a coalesced warp along one row", () => {
    const t = tiles((lane) => lane);
    expect([t.rows, t.cols]).toEqual([1, 32]);
    expect(t.of.get(5)).toEqual([0, 5]);
    expect(t.label).toBe("%out · rows 0–0 × columns 0–31");
  });

  it("lays a strided warp down one column", () => {
    const t = tiles((lane) => lane * 1024);
    expect([t.rows, t.cols]).toEqual([32, 32]);
    expect(t.of.get(5)).toEqual([5, 0]);
    expect(t.cells.get(5 * 32).lane).toBe(5);
    expect([t.shade(0, 7), t.shade(0, 8)]).toEqual([false, true]);
  });
});

describe("layoutLaunch", () => {
  it("places blocks by id and threads by id, with warps of 32", () => {
    const layout = layoutLaunch([2, 3, 1], [32, 2, 1]);
    expect(layout.blocks).toHaveLength(6);
    expect(layout.blocks.at(-1).id).toEqual([1, 2, 0]);
    expect(layout.blocks[1].at[0]).toBeGreaterThan(layout.blocks[0].at[0]);
    expect(layout.blocks[2].at[2]).toBeLessThan(layout.blocks[0].at[2]);
    expect(layout.threads).toHaveLength(64);

    expect(layout.threads[33]).toMatchObject({
      id: [1, 1, 0],
      warp: 1,
      lane: 1,
    });

    expect(layout.threads[0].at[0]).toBeGreaterThan(layout.blocks[1].at[0] + 1);
    expect(layout.clipped).toEqual({ grid: false, threads: false });
  });

  it("clips large launches and wraps wide 1-D blocks into warp rows", () => {
    const layout = layoutLaunch([1024, 1, 1], [256, 1, 1]);
    expect(layout.blocks).toHaveLength(16);
    expect(layout.clipped.grid).toBe(true);
    expect(layout.shownGrid).toEqual([16, 1, 1]);

    const [a, b] = [layout.threads[0], layout.threads[32]];
    expect(b.id).toEqual([32, 0, 0]);
    expect(b.at[0]).toBe(a.at[0]);
    expect(b.at[2]).toBeLessThan(a.at[2]);
  });

  it("draws runtime sizes as 1", () => {
    const layout = layoutLaunch([null, 4, 1], null);
    expect(layout.grid).toEqual([1, 4, 1]);
    expect(layout.threads).toHaveLength(1);
  });
});

describe("camera", () => {
  it("projects the center to the middle and keeps far floor points higher", () => {
    const cam = camera({
      yaw: 0,
      pitch: 0.8,
      center: [0, 0, 0],
      scale: 10,
      cx: 100,
      cy: 50,
    });

    expect(cam.screen(cam.view([0, 0, 0]))).toEqual([100, 50]);

    const near = cam.screen(cam.view([0, 0, -5]));
    const far = cam.screen(cam.view([0, 0, 5]));
    expect(far[1]).toBeLessThan(near[1]);
  });
});
