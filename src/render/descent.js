// The Descent view: the IR of every pass as a layer in a 3D stack, the first
// pass at the top and the last at the bottom. The pass on show is the lit
// layer; stepping slides the camera down to the next one and sends a pulse
// along the selected op's path, so you watch the op go down through the
// compiler. The op's source line lights its nodes and their edges.
//
// Plain canvas 2D with the small perspective camera from gpu-3d.js: no WebGL
// and no library. Drag to turn, scroll to zoom, click a node to follow it or
// a layer to go to that pass, arrow keys to step.

import {
  approach,
  layoutStack,
  lineageSegments,
  neighbours,
  nodePoint,
  pointAlong,
} from "../trace/index.js";
import { camera } from "./gpu-3d.js";

const HOME = { yaw: -0.55, pitch: 0.5, zoom: 1 };
const PULSE_MS = 900;
// Layers this far from the lit one are drawn as plates only, in tall stacks.
const NEAR = 3;
const MANY_LAYERS = 12;
const NODE = 0.5;
const FALLBACK = { width: 640, height: 420 };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function inside([px, py], poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

const reducedMotion = () =>
  typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * @param {HTMLElement} root
 * @param {{
 *   onPickNode: (pass: number, node: number) => void,
 *   onPickPass: (pass: number) => void,
 *   onStep: (delta: number) => void,
 *   onGpu?: () => void,
 * }} options  `onStep` gets +1 / -1, or ±Infinity for the last / first pass.
 */
export function createDescentView(root, { onPickNode, onPickPass, onStep, onGpu }) {
  const bar = el("div", "descent-bar");
  const status = el("span", "descent-status");
  const reset = el("button", "", "Reset view");
  const bottom = el("button", "", "Bottom ⤓");
  const gpu = el("button", "", "Open launch view");
  for (const b of [reset, bottom, gpu]) b.type = "button";
  gpu.hidden = true;
  bar.append(status, reset, bottom, gpu);
  const canvas = el("canvas", "descent-canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  const empty = el("p", "descent-empty", "Open a pass trace to see its passes as layers.");
  root.replaceChildren(bar, canvas, empty);

  let data = null; // { shapes, titles, labelOf, stack }
  let state = { pass: 0, node: -1, lineNodes: [], lineage: [], gpu: false };
  let view = { ...HOME };
  let focusY = 0;
  let pulse = null;
  let move = null; // a pass change waiting for the op's path to arrive
  let raf = 0;
  let last = 0;
  let hits = []; // nodes on screen: { pass, node, x, y, z, r }
  let plates = []; // layers on screen: { pass, poly, z }
  let size = FALLBACK;

  // ---- Data and state -----------------------------------------------------

  function setData(next) {
    data = next
      ? { ...next, stack: layoutStack(next.shapes) }
      : null;
    empty.hidden = !!data;
    canvas.hidden = !data;
    bar.hidden = !data;
    if (data) focusY = data.stack.heightOf(Math.min(state.pass, data.shapes.length - 1));
    refresh();
  }

  function setState(next) {
    const before = state.pass;
    state = { ...state, ...next };
    if (next.pass !== undefined && next.pass !== before) move = { from: before, to: next.pass, at: performance.now() };
    if (move && performance.now() - move.at > 1500) move = null;
    if (move && !reducedMotion()) startPulse();
    gpu.hidden = !(state.gpu && onGpu);
    refresh();
  }

  // The op's path between the pass it was on and the one just loaded, when
  // its life covers both.
  function startPulse() {
    if (!data || !state.lineage.length) return;
    const [a, b] = move.from < move.to ? [move.from, move.to] : [move.to, move.from];
    const points = [];
    const on = new Map(state.lineage.filter((e) => e.node >= 0).map((e) => [e.pass, e.node]));
    if (!on.has(a) || !on.has(b)) return;
    for (let p = a; p <= b; p++) {
      if (!on.has(p)) return;
      const at = nodePoint(data.stack, p, on.get(p));
      if (!at) return;
      points.push(at);
    }
    if (move.from > move.to) points.reverse();
    pulse = { points, start: performance.now() };
    move = null;
  }

  function refresh() {
    if (!data) return;
    const n = data.shapes.length;
    const pass = Math.max(0, Math.min(state.pass, n - 1));
    const layer = data.stack.layers[pass];
    status.textContent =
      `${data.titles[pass] ?? `#${pass + 1}`}` +
      (layer ? ` · ${layer.shown} ops` : " · IR did not parse") +
      (layer?.hidden ? ` (+${layer.hidden} not drawn)` : "");
    canvas.setAttribute(
      "aria-label",
      `Descent: ${n} passes as layers, pass ${pass + 1} of ${n} lit. Arrow keys step through the passes.`,
    );
    schedule();
  }

  // ---- Drawing --------------------------------------------------------------

  function schedule() {
    if (raf) return;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    const dt = Math.min(64, now - last);
    last = now;
    let busy = false;
    if (data) {
      const target = data.stack.heightOf(Math.max(0, Math.min(state.pass, data.shapes.length - 1)));
      focusY = reducedMotion() ? target : approach(focusY, target, dt);
      if (focusY !== target) busy = true;
    }
    if (pulse) {
      if (now - pulse.start < PULSE_MS) busy = true;
      else pulse = null;
    }
    draw();
    if (busy) raf = requestAnimationFrame(frame);
  }

  function fit() {
    const dpr = (typeof devicePixelRatio === "number" && devicePixelRatio) || 1;
    const width = canvas.clientWidth || FALLBACK.width;
    const height = canvas.clientHeight || FALLBACK.height;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    size = { width, height, dpr };
    return size;
  }

  function colors() {
    const style = getComputedStyle(document.documentElement);
    const token = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
    return {
      bg: token("--bg", "#111"),
      raised: token("--raised", "#222"),
      line: token("--line", "#444"),
      fg: token("--fg", "#eee"),
      dim: token("--dim", "#999"),
      accent: token("--accent", "#6aa7ff"),
      good: token("--good", "#7bd88f"),
      warn: token("--warn", "#f2c94c"),
      err: token("--err", "#ff6b5e"),
      font: token("--ui", "sans-serif"),
    };
  }

  function draw() {
    if (!data || root.hidden) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height, dpr } = fit();
    const c = colors();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const { stack, shapes } = data;
    const n = shapes.length;
    const reach = Math.max(stack.plate.width, stack.plate.depth * 1.3, 24);
    const scale = (Math.min(width, height * 1.4) / (reach * 1.25)) * view.zoom;
    const cam = camera({ yaw: view.yaw, pitch: view.pitch, center: [0, focusY, 0], scale, cx: width / 2, cy: height / 2 });
    const project = (p) => {
      const v = cam.view(p);
      const [x, y] = cam.screen(v);
      return [x, y, v[2]];
    };

    hits = [];
    plates = [];
    const lit = Math.max(0, Math.min(state.pass, n - 1));
    const near = (p) => n <= MANY_LAYERS || Math.abs(p - lit) <= NEAR;
    const hw = stack.plate.width / 2;
    const hd = stack.plate.depth / 2;
    const order = [...Array(n).keys()]
      .map((p) => ({ p, z: project([0, stack.heightOf(p), 0])[2] }))
      .sort((a, b) => b.z - a.z);

    const half = NODE * Math.max(0.35, stack.shrink);
    const ink = new Set([state.node]);
    const lineSet = new Set(state.lineNodes);
    const around = state.node >= 0 ? neighbours(shapes[lit], state.node) : { inputs: [], outputs: [] };
    for (const l of state.lineNodes) {
      const { inputs, outputs } = neighbours(shapes[lit], l);
      around.inputs.push(...inputs);
      around.outputs.push(...outputs);
    }

    for (const { p } of order) {
      const active = p === lit;
      const y = stack.heightOf(p);
      const alpha = active ? 1 : Math.max(0.16, 0.5 - 0.1 * Math.abs(p - lit));
      const poly = [[-hw, y, -hd], [hw, y, -hd], [hw, y, hd], [-hw, y, hd]].map(project);
      plates.push({ pass: p, poly: poly.map(([x, yy]) => [x, yy]), z: project([0, y, 0])[2] });

      ctx.globalAlpha = alpha * (active ? 0.9 : 0.55);
      ctx.fillStyle = c.raised;
      ctx.strokeStyle = active ? c.accent : c.line;
      ctx.lineWidth = active ? 1.6 : 1;
      ctx.beginPath();
      poly.forEach(([x, yy], i) => (i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Layer label on the plate's near-left corner.
      ctx.globalAlpha = active ? 1 : Math.max(0.35, alpha + 0.2);
      ctx.fillStyle = active ? c.fg : c.dim;
      ctx.font = `${active ? 600 : 400} 11px ${c.font}`;
      ctx.textAlign = "left";
      ctx.fillText(`#${p + 1} ${data.names?.[p] ?? ""}`.trim(), poly[3][0] + 4, poly[3][1] + 12);

      const layer = stack.layers[p];
      if (!layer || !near(p)) continue;

      // Edges of the lit layer: all of them when there are few, else only the
      // ones at the picked op and the lit line.
      if (active) {
        const few = layer.shown <= 150;
        ctx.lineWidth = 1;
        for (const [a, b] of shapes[p].edges) {
          const pa = layer.pos[a];
          const pb = layer.pos[b];
          if (!pa || !pb) continue;
          const touched = ink.has(a) || ink.has(b) || lineSet.has(a) || lineSet.has(b);
          if (!few && !touched) continue;
          const [x1, y1] = project([pa[0], y, pa[1]]);
          const [x2, y2] = project([pb[0], y, pb[1]]);
          ctx.globalAlpha = touched ? 0.95 : 0.22;
          ctx.strokeStyle = touched ? (ink.has(a) || lineSet.has(a) ? c.accent : c.good) : c.dim;
          ctx.lineWidth = touched ? 1.8 : 1;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }

      layer.pos.forEach((pos, node) => {
        if (!pos) return;
        const corners = [
          [pos[0] - half, y, pos[1] - half],
          [pos[0] + half, y, pos[1] - half],
          [pos[0] + half, y, pos[1] + half],
          [pos[0] - half, y, pos[1] + half],
        ].map(project);
        const [cx, cy, cz] = project([pos[0], y, pos[1]]);
        const kind = shapes[p].kind[node];
        const picked = active && node === state.node;
        const onLine = active && lineSet.has(node);
        const isIn = active && around.inputs.includes(node);
        const isOut = active && around.outputs.includes(node);
        ctx.globalAlpha = active ? 1 : alpha + 0.15;
        ctx.fillStyle = picked ? c.accent : onLine ? c.warn : isIn ? c.good : isOut ? c.accent : kind === 4 ? c.err : kind === 1 || kind === 0 ? c.fg : c.dim;
        ctx.beginPath();
        corners.forEach(([x, yy], i) => (i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy)));
        ctx.closePath();
        ctx.fill();
        if (picked || onLine) {
          ctx.strokeStyle = c.fg;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        const r = Math.max(3, Math.hypot(corners[0][0] - corners[2][0], corners[0][1] - corners[2][1]) / 2);
        hits.push({ pass: p, node, x: cx, y: cy, z: cz, r });
      });

      if (active && state.node >= 0 && layer.pos[state.node]) {
        const label = data.labelOf(p, state.node);
        if (label) {
          const [lx, ly] = project([layer.pos[state.node][0], y, layer.pos[state.node][1]]);
          ctx.globalAlpha = 1;
          ctx.fillStyle = c.fg;
          ctx.font = `600 11px ${c.font}`;
          ctx.fillText(label, lx + 8, ly - 8);
        }
      }
    }

    // The op's path down through the layers.
    ctx.setLineDash([]);
    for (const seg of lineageSegments(state.lineage)) {
      const a = nodePoint(stack, seg.from.pass, seg.from.node);
      const b = nodePoint(stack, seg.to.pass, seg.to.node);
      if (!a || !b) continue;
      const [x1, y1] = project(a);
      const [x2, y2] = project(b);
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = seg.kind === "fused" || seg.kind === "inlined" ? c.warn : c.accent;
      ctx.lineWidth = 2;
      ctx.setLineDash(seg.kind === "fused" || seg.kind === "inlined" ? [5, 4] : []);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    if (pulse) {
      const t = Math.min(1, (performance.now() - pulse.start) / PULSE_MS);
      const at = pointAlong(pulse.points, 1 - (1 - t) ** 2);
      if (at) {
        const [x, y] = project(at);
        ctx.globalAlpha = 1;
        ctx.fillStyle = c.fg;
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  // ---- Picking ----------------------------------------------------------------

  function hitTest(x, y) {
    let best = null;
    for (const h of hits) {
      const d = Math.hypot(h.x - x, h.y - y);
      if (d <= h.r + 3 && (!best || h.z < best.z - 1e-6 || (Math.abs(h.z - best.z) < 1e-6 && d < best.d)))
        best = { ...h, d };
    }
    if (best) return { kind: "node", pass: best.pass, node: best.node };
    let plate = null;
    for (const p of plates)
      if (inside([x, y], p.poly) && (!plate || p.z < plate.z)) plate = p;
    return plate ? { kind: "pass", pass: plate.pass } : null;
  }

  // Where node `node` of pass `pass` is on screen (as last drawn).
  function nodeScreen(pass, node) {
    const h = hits.find((e) => e.pass === pass && e.node === node);
    return h ? { x: h.x, y: h.y } : null;
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture?.(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, moved: false };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    view.yaw += dx * 0.008;
    view.pitch = Math.max(0.12, Math.min(1.4, view.pitch + dy * 0.006));
    drag.x = e.clientX;
    drag.y = e.clientY;
    schedule();
  });
  canvas.addEventListener("pointerup", (e) => {
    const was = drag;
    drag = null;
    if (!was || was.moved) return;
    const box = canvas.getBoundingClientRect();
    const hit = hitTest(e.clientX - box.left, e.clientY - box.top);
    if (hit?.kind === "node") onPickNode(hit.pass, hit.node);
    else if (hit?.kind === "pass") onPickPass(hit.pass);
  });
  canvas.addEventListener("pointercancel", () => (drag = null));
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      view.zoom = Math.max(0.4, Math.min(4, view.zoom * Math.exp(-e.deltaY * 0.0015)));
      schedule();
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (e) => {
    const step = { ArrowDown: 1, PageDown: 1, ArrowUp: -1, PageUp: -1, End: Infinity, Home: -Infinity }[e.key];
    if (step === undefined) return;
    e.preventDefault();
    onStep(step);
  });
  reset.addEventListener("click", () => {
    view = { ...HOME };
    schedule();
  });
  bottom.addEventListener("click", () => onStep(Infinity));
  gpu.addEventListener("click", () => onGpu?.());

  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => schedule()).observe(canvas);

  setData(null);
  return {
    setData,
    setState,
    draw,
    hitTest,
    nodeScreen,
    get view() {
      return view;
    },
    // Height the camera is looking at, which glides to the lit layer.
    get focus() {
      return focusY;
    },
  };
}
