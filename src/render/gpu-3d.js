import { flowSlice } from "../gpu/index.js";
import { THREAD_MAX, WARP } from "../constants.js";
import { el } from "../dom.js";
import { drawScene } from "./gpu-3d-draw.js";
import {
  CORNERS,
  HOME,
  bounds,
  camera,
  dimText,
  elementTiles,
  idText,
  inside,
} from "./gpu-3d-geometry.js";
import { buildSceneModel } from "./gpu-3d-model.js";

export {
  camera,
  elementTiles,
  layoutLaunch,
  matrixWidth,
} from "./gpu-3d-geometry.js";

const scenes = new Set();

if (
  typeof MutationObserver !== "undefined" &&
  typeof document !== "undefined"
) {
  new MutationObserver(() => {
    for (const scene of scenes) {
      if (scene.canvas.isConnected) scene.schedule();
      else if (scene.shown) scenes.delete(scene);
    }
  }).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
}

export function gpuScene(launch, kernel, { onLine } = {}) {
  const model = buildSceneModel(launch, kernel);
  const { layout, floor, tileBounds, flow, nodes } = model;

  let showIR = nodes.length > 0;
  let slice = null;
  let focusLine = null;
  let all;
  let center;
  let box;

  const rebuild = () => {
    all = showIR ? [...floor, ...nodes] : floor;
    box = bounds([...all, ...tileBounds]);
    center = box.min.map((v, i) => (v + box.max[i]) / 2);
  };

  rebuild();

  const view = { ...HOME, zoom: 1, pan: [0, 0] };
  let fit = 1;
  let selected = layout.blocks[0];
  let hovered = null;
  let access = null;
  let size = [0, 0];
  let focusSpace = null;
  let inset = 0;
  let topSpace = 0;

  const figure = el("figure", "gpu-3d");
  const stage = el("div", "gpu-3d-stage");
  const canvas = el("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");

  const tip = el("div", "gpu-3d-tip");
  tip.hidden = true;

  const tools = el("div", "gpu-3d-tools");

  const tool = (text, title, action) => {
    const button = el("button", "", text);
    button.type = "button";
    button.title = title;

    button.addEventListener("click", () => {
      action();
      schedule();
    });

    tools.append(button);
  };

  tool("−", "Zoom out (−)", () => (view.zoom /= 1.25));
  tool("+", "Zoom in (+)", () => (view.zoom *= 1.25));

  tool("Top", "Look straight down (t)", () =>
    Object.assign(view, { yaw: 0, pitch: Math.PI / 2 - 0.01 }),
  );

  tool("Reset", "Reset the view (0)", () =>
    Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] }),
  );

  const elements = el("div", "gpu-3d-elements");
  elements.hidden = true;

  let showTop = false;

  tool(
    "Elements",
    "Show the elements the picked warp touches, seen from above (e)",
    () => toggleElements(),
  );

  const elementsButton = tools.lastChild;
  elementsButton.setAttribute("aria-pressed", "false");
  elementsButton.disabled = true;

  function toggleElements(on = !showTop) {
    showTop = on && !elementsButton.disabled;
    elementsButton.setAttribute("aria-pressed", String(showTop));
    placeElements();
  }

  function placeElements() {
    elements.hidden = !showTop;
    elements.style.top = `${topSpace + 4}px`;
    elements.style.right = `${inset ? inset : 8}px`;
  }

  let irButton = null;

  if (nodes.length) {
    tool("IR", "Show or hide the kernel IR wall (i)", () => toggleIR());
    irButton = tools.lastChild;
    irButton.setAttribute("aria-pressed", "true");
  }

  function toggleIR() {
    showIR = !showIR;
    irButton.setAttribute("aria-pressed", String(showIR));
    rebuild();
    resize();
  }

  const hint = el("div", "gpu-3d-hint");
  stage.append(canvas, elements, tip, hint);
  figure.append(tools, stage);

  function describe() {
    const perBlock = layout.block[0] * layout.block[1] * layout.block[2];
    const warps = Math.ceil(perBlock / WARP);
    const notes = [];

    if (layout.clipped.grid) {
      notes.push(
        `showing ${dimText(layout.shownGrid)} of the ${dimText(layout.grid)} blocks`,
      );
    }

    if (layout.clipped.threads) {
      notes.push(`showing the first ${THREAD_MAX} of ${perBlock} threads`);
    }

    if ([launch.grid, launch.block].some((d) => !d || d.includes(null))) {
      notes.push("sizes known only at runtime are drawn as 1");
    }

    hint.textContent =
      (notes.length ? `${notes.join(" · ")}. ` : "") +
      "Drag to turn · shift-drag to pan · ⌘/Ctrl + scroll to zoom · click a block to open it" +
      (nodes.length ? " · click an IR node or a line of code to trace it" : "");

    canvas.setAttribute(
      "aria-label",
      `3D view. Left: the grid of ${dimText(layout.grid)} blocks, which run independently. ` +
        `Right: block ${idText(selected.id)} opened into ${perBlock} threads in ${warps} warp${warps > 1 ? "s" : ""} of ${WARP}, ` +
        "each row of 32 running in lockstep. Floor plates: global memory under everything, shared memory under the block.",
    );
  }

  function refit() {
    const cam = camera({ ...HOME, center, scale: 1, cx: 0, cy: 0 });
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];

    for (const cube of [
      { at: box.min, size: 0 },
      { at: box.max, size: 0 },
      ...all.filter((_, i) => i % 7 === 0),
    ]) {
      for (const corner of CORNERS) {
        const [x, y] = cam.screen(
          cam.view(cube.at.map((v, i) => v + corner[i] * cube.size)),
        );

        [x0, y0, x1, y1] = [
          Math.min(x0, x),
          Math.min(y0, y),
          Math.max(x1, x),
          Math.max(y1, y),
        ];
      }
    }

    fit = Math.min(
      (size[0] - inset - 40) / (x1 - x0),
      (size[1] - topSpace - 110) / (y1 - y0),
    );
  }

  let faces = [];
  let frame = 0;

  function schedule() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  }

  function draw() {
    const drawn = drawScene(canvas, model, {
      access,
      all,
      center,
      fit,
      focusLine,
      focusSpace,
      hovered,
      inset,
      selected,
      showIR,
      size,
      slice,
      topSpace,
      view,
    });

    if (drawn) faces = drawn;
  }

  function hit(x, y) {
    for (let i = faces.length - 1; i >= 0; i--) {
      if (inside([x, y], faces[i].poly)) return faces[i].cube;
    }

    return null;
  }

  function tipText(cube) {
    if (cube.kind === "node") {
      return `line ${cube.node.line} · ${cube.node.text}\nclick to find it in the code`;
    }

    if (cube.kind === "block") {
      return `block ${idText(cube.id)}${cube === selected ? " · opened" : " · click to open"}`;
    }

    let text = `thread ${idText(cube.id)} · warp ${cube.warp}, lane ${cube.lane}`;
    const lane = access?.lanes.get(cube.id.join(","));

    if (lane) {
      text += `\n${access.buffer}[${lane.index.join(", ")}] · ${access.unit} ${lane.group}`;
    }

    return text;
  }

  let drag = null;

  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, moved: false, pan: e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener("pointermove", (e) => {
    const box = canvas.getBoundingClientRect();

    if (drag) {
      const [dx, dy] = [e.clientX - drag.x, e.clientY - drag.y];
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;

      if (drag.moved) {
        if (drag.pan) {
          view.pan[0] += dx;
          view.pan[1] += dy;
        } else {
          view.yaw -= dx * 0.008;

          view.pitch = Math.min(
            Math.PI / 2 - 0.01,
            Math.max(0.08, view.pitch + dy * 0.006),
          );
        }

        drag.x = e.clientX;
        drag.y = e.clientY;
        tip.hidden = true;
        schedule();

        return;
      }
    }

    const cube = hit(e.clientX - box.left, e.clientY - box.top);

    if (cube !== hovered) {
      hovered = cube;

      canvas.style.cursor =
        cube?.kind === "block" || cube?.kind === "node" ? "pointer" : "grab";

      schedule();
    }

    tip.hidden = !cube;

    if (cube) {
      tip.textContent = tipText(cube);
      tip.style.left = `${e.clientX - box.left + 12}px`;
      tip.style.top = `${e.clientY - box.top + 12}px`;
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    if (drag && !drag.moved) {
      const box = canvas.getBoundingClientRect();
      const cube = hit(e.clientX - box.left, e.clientY - box.top);

      if (cube?.kind === "node") {
        showLine(cube.node.line);
        onLine?.(cube.node.line);
      } else if (cube?.kind === "block" && cube !== selected) {
        selected = cube;
        describe();
        tip.textContent = tipText(cube);
        schedule();
      }
    }

    drag = null;
  });

  canvas.addEventListener("pointerleave", () => {
    tip.hidden = true;

    if (hovered) {
      hovered = null;
      schedule();
    }
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();

      view.zoom = Math.min(
        8,
        Math.max(0.3, view.zoom * Math.exp(-e.deltaY * 0.004)),
      );

      schedule();
    },
    { passive: false },
  );

  canvas.addEventListener("dblclick", () => {
    Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] });
    schedule();
  });

  const KEYS = {
    ArrowLeft: () => (view.yaw += 0.12),
    ArrowRight: () => (view.yaw -= 0.12),
    ArrowUp: () =>
      (view.pitch = Math.min(Math.PI / 2 - 0.01, view.pitch + 0.1)),
    ArrowDown: () => (view.pitch = Math.max(0.08, view.pitch - 0.1)),
    "+": () => (view.zoom *= 1.25),
    "=": () => (view.zoom *= 1.25),
    "-": () => (view.zoom /= 1.25),
    t: () => Object.assign(view, { yaw: 0, pitch: Math.PI / 2 - 0.01 }),
    i: () => nodes.length && toggleIR(),
    e: () => toggleElements(),
    0: () => Object.assign(view, { ...HOME, zoom: 1, pan: [0, 0] }),
  };

  canvas.addEventListener("keydown", (e) => {
    const action = KEYS[e.key];
    if (!action || e.metaKey || e.ctrlKey) return;
    e.preventDefault();
    e.stopPropagation();
    action();
    schedule();
  });

  const resize = () => {
    const width = stage.clientWidth;
    if (!width) return;

    const height = Math.round(
      Math.min(640, Math.max(380, width * (showIR ? 0.7 : 0.58))),
    );

    const panel = stage.querySelector(".gpu-3d-panel");
    inset = panel?.open && width >= 640 ? panel.offsetWidth + 16 : 0;
    topSpace = (stage.querySelector(".gpu-3d-facts")?.offsetHeight ?? 0) + 8;
    placeElements();

    const ratio = window.devicePixelRatio || 1;
    size = [width, height];
    scene.shown = true;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    refit();
    draw();
  };

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(resize).observe(stage);
  }

  const scene = { canvas, schedule, shown: false };

  for (const old of scenes) {
    if (old.shown && !old.canvas.isConnected) scenes.delete(old);
  }

  scenes.add(scene);
  describe();

  function showLine(line) {
    const lit = line ? flowSlice(flow, line) : new Set();
    slice = lit.size ? lit : null;
    focusLine = slice ? line : null;
    schedule();

    return lit.size;
  }

  return {
    figure,
    showLine,
    flow,
    lit: (id) => !!slice?.has(id),
    showSpace(space) {
      focusSpace = space;
      schedule();
    },
    overlay(facts, panelBody, summary) {
      facts.classList.add("gpu-3d-facts");
      stage.append(facts);
      if (!panelBody) return;

      const panel = el("details", "gpu-3d-panel");
      panel.open = true;

      const head = el("summary", "", summary);
      panel.append(head, panelBody);

      requestAnimationFrame(() => {
        if (stage.clientWidth && stage.clientWidth < 760) panel.open = false;
      });

      panel.addEventListener("toggle", resize);
      stage.append(panel);
    },
    showElements(node) {
      elements.replaceChildren(...(node ? [node] : []));
      elementsButton.disabled = !node;

      elementsButton.title = node
        ? "Show the elements the picked warp touches, seen from above (e)"
        : "No elements to show: pick an access that could be analyzed";

      toggleElements(showTop);
    },
    showAccess(judged, { formula = false } = {}) {
      const result = judged?.result;

      if (!result?.analyzed) {
        access = null;
      } else {
        const groups = [...new Set(result.lanes.map((l) => l.group))];
        const order = new Map(groups.map((g, i) => [g, i]));
        const unit = judged.space === "shared" ? "bank" : "sector";

        access = {
          buffer: judged.access.buffer,
          unit,
          proven: result.proof?.status === "proven",
          good: ["coalesced", "broadcast", "conflict-free"].includes(
            result.verdict,
          ),
          formula: formula ? (result.proof?.formula ?? null) : null,
          text: `${judged.access.kind === "load" ? "load" : "store"} ${judged.access.buffer} · ${groups.length} ${unit}${groups.length > 1 ? "s" : ""}`,
          lanes: new Map(
            result.lanes.map((l) => [
              [l.tx, l.ty, l.tz].join(","),
              { ...l, order: order.get(l.group) },
            ]),
          ),
          tiles: elementTiles(judged, result, order),
        };
      }

      schedule();
    },
  };
}
