import { kernelFlow } from "../gpu/index.js";
import { NODE, TILE, bounds, layoutLaunch } from "./gpu-3d-geometry.js";

export function buildSceneModel(launch, kernel) {
  const layout = layoutLaunch(launch.grid, launch.block);
  const floor = [...layout.blocks, ...layout.threads];
  const { min, max } = bounds(floor);
  const threadBox = layout.threads.length ? bounds(layout.threads) : null;

  const tileArea =
    threadBox && kernel?.accesses?.length
      ? { x: threadBox.max[0] + 2, far: threadBox.max[2] }
      : null;

  const tileBounds = tileArea
    ? [
        { at: [tileArea.x, 0, tileArea.far - TILE.rows * TILE.pitch], size: 0 },
        { at: [tileArea.x + TILE.cols * TILE.pitch, 0, tileArea.far], size: 0 },
      ]
    : [];

  const plateBox = bounds([...floor, ...tileBounds]);

  const threadCube = new Map(
    layout.threads.map((cube) => [cube.id.join(","), cube]),
  );

  const flow = kernelFlow(kernel);
  const ranks = Math.max(1, ...flow.nodes.map((n) => n.rank + 1));
  const stacked = new Map();
  const boardFar = min[2] - NODE.gap;

  const nodes = [...flow.nodes]
    .sort((a, b) => a.rank - b.rank || a.line - b.line)
    .map((node) => {
      const level = stacked.get(node.rank) ?? 0;
      stacked.set(node.rank, level + 1);

      const x =
        min[0] +
        (node.rank / Math.max(1, ranks - 1)) * (max[0] - min[0] - NODE.size);

      return {
        kind: "node",
        node,
        at: [x, 0, boardFar - NODE.size - level * NODE.step],
        size: NODE.size,
      };
    });

  const byId = new Map(nodes.map((cube) => [cube.node.id, cube]));
  const boardNear = Math.min(boardFar, ...nodes.map((c) => c.at[2]));

  const spaceOf = new Map(
    (kernel?.buffers ?? []).map((b) => [b.name, b.space]),
  );

  const gridBox = bounds(layout.blocks);
  const buffers = kernel?.buffers ?? [];

  const bytesIn = (space) =>
    buffers
      .filter((b) => b.space === space)
      .reduce((sum, b) => sum + (b.bytes ?? 0), 0);

  const memory = {
    global: bytesIn("global"),
    shared: bytesIn("shared"),
    private: bytesIn("private"),
  };

  return {
    layout,
    floor,
    min,
    max,
    threadBox,
    tileArea,
    tileBounds,
    plateBox,
    threadCube,
    flow,
    nodes,
    byId,
    boardNear,
    boardFar,
    spaceOf,
    gridBox,
    buffers,
    memory,
  };
}
