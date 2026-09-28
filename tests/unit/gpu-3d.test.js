import { describe, expect, it } from "vitest";
import { camera, layoutLaunch } from "../../src/gpu-3d.js";

describe("layoutLaunch", () => {
  it("places blocks by id and threads by id, with warps of 32", () => {
    const layout = layoutLaunch([2, 3, 1], [32, 2, 1]);
    expect(layout.blocks).toHaveLength(6);
    expect(layout.blocks.at(-1).id).toEqual([1, 2, 0]);
    // Block y runs away from the viewer (-z), block x to the right.
    expect(layout.blocks[1].at[0]).toBeGreaterThan(layout.blocks[0].at[0]);
    expect(layout.blocks[2].at[2]).toBeLessThan(layout.blocks[0].at[2]);
    expect(layout.threads).toHaveLength(64);
    expect(layout.threads[33]).toMatchObject({ id: [1, 1, 0], warp: 1, lane: 1 });
    // The opened block sits to the right of the grid.
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
    const cam = camera({ yaw: 0, pitch: 0.8, center: [0, 0, 0], scale: 10, cx: 100, cy: 50 });
    expect(cam.screen(cam.view([0, 0, 0]))).toEqual([100, 50]);
    const near = cam.screen(cam.view([0, 0, -5]));
    const far = cam.screen(cam.view([0, 0, 5]));
    expect(far[1]).toBeLessThan(near[1]);
  });
});
