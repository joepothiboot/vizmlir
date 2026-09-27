import { KIND_STYLE, STRIDE } from "../wasm/abi.js";

const MONO = `"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace`;

const BADGE = { added: "+", changed: "~" };

function readTheme() {
  const css = getComputedStyle(document.documentElement);
  const v = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  return {
    background: v("--bg", "#0f0f0d"),
    gridDot: v("--grid-dot", "#26251f"),
    edge: v("--edge", "rgba(142,138,128,0.45)"),
    dim: v("--dim", "#8e8a80"),
    selected: v("--accent", "#ff7a1a"),
    added: v("--good", "#9be564"),
    changed: v("--warn", "#f2c94c"),
    kinds: Object.fromEntries(
      Object.entries(KIND_STYLE).map(([kind, style]) => [
        kind,
        {
          fill: v(`--node-${kind}-fill`, style.fill),
          stroke: v(`--node-${kind}-stroke`, style.stroke),
          text: v(`--node-${kind}-text`, style.text),
        },
      ]),
    ),
  };
}

export class CanvasRenderer {
  constructor(canvas, { onSelect = null, onViewChange = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
    this.theme = readTheme();
    this.onSelect = onSelect;
    this.onViewChange = onViewChange;

    this.camera = { x: 0, y: 0, scale: 1 };
    this.snapshot = null;
    this.marks = new Map();
    this.selected = -1;
    this.dpr = 1;
    this._raf = 0;

    this.#bindInput();
    this.resize();
    document.fonts?.ready.then(() => this.requestDraw());
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    this.width = rect.width;
    this.height = rect.height;
    this.requestDraw();
  }

  setSnapshot(snapshot, { fit = true } = {}) {
    this.snapshot = snapshot;
    this.marks = new Map();
    this.selected = -1;
    this.onSelect?.(-1, snapshot);
    if (fit) this.fit();
    this.requestDraw();
  }

  /** Re-read colours after the page theme changes. */
  refreshTheme() {
    this.theme = readTheme();
    this.requestDraw();
  }

  /** Map of node index -> "added" | "changed", drawn as coloured outlines. */
  setMarks(marks) {
    this.marks = marks;
    this.requestDraw();
  }

  select(index, { center = false } = {}) {
    const s = this.snapshot;
    this.selected = s && index >= 0 && index < s.nodeCount ? index : -1;
    if (center && this.selected >= 0) {
      const o = this.selected * STRIDE.NODE_XYWH;
      const { scale } = this.camera;
      this.camera.x = this.width / 2 - (s.xywh[o] + s.xywh[o + 2] / 2) * scale;
      this.camera.y =
        this.height / 2 - (s.xywh[o + 1] + s.xywh[o + 3] / 2) * scale;
    }
    this.onSelect?.(this.selected, s);
    this.requestDraw();
  }

  fit(padding = 48) {
    const s = this.snapshot;
    if (!s || s.nodeCount === 0) return;
    const [minX, minY, maxX, maxY] = s.bounds;
    const w = Math.max(1, maxX - minX);
    const h = Math.max(1, maxY - minY);
    const scale = Math.min(
      (this.width - padding * 2) / w,
      (this.height - padding * 2) / h,
      2,
    );
    this.camera.scale = Math.max(scale, 0.05);
    this.camera.x = this.width / 2 - ((minX + maxX) / 2) * this.camera.scale;
    this.camera.y = this.height / 2 - ((minY + maxY) / 2) * this.camera.scale;
  }

  /** Zoom by `factor` around a point in canvas pixels (default: centre). */
  zoomBy(factor, mx = this.width / 2, my = this.height / 2) {
    const next = Math.min(4, Math.max(0.05, this.camera.scale * factor));
    const k = next / this.camera.scale;
    this.camera.x = mx - (mx - this.camera.x) * k;
    this.camera.y = my - (my - this.camera.y) * k;
    this.camera.scale = next;
    this.requestDraw();
  }

  /** Render the whole graph (not just the view) to a PNG blob. */
  exportPNG({ scale = 2, padding = 32 } = {}) {
    const s = this.snapshot;
    if (!s || s.nodeCount === 0) return Promise.resolve(null);
    const [minX, minY, maxX, maxY] = s.bounds;
    const width = maxX - minX + padding * 2;
    const height = maxY - minY + padding * 2;
    // Browsers cap canvas size; shrink very large graphs instead of failing.
    const pixelScale = Math.min(scale, 8000 / width, 8000 / height);
    const out = document.createElement("canvas");
    out.width = Math.max(1, Math.round(width * pixelScale));
    out.height = Math.max(1, Math.round(height * pixelScale));

    const saved = {
      ctx: this.ctx,
      width: this.width,
      height: this.height,
      dpr: this.dpr,
      camera: this.camera,
      selected: this.selected,
    };
    Object.assign(this, {
      ctx: out.getContext("2d"),
      width,
      height,
      dpr: pixelScale,
      camera: { x: padding - minX, y: padding - minY, scale: 1 },
      selected: -1,
      exporting: true,
    });
    try {
      this.draw();
    } finally {
      Object.assign(this, saved, { exporting: false });
      this.requestDraw();
    }
    return new Promise((resolve) => out.toBlob(resolve, "image/png"));
  }

  /** The whole graph as a standalone SVG document string. */
  exportSVG({ padding = 32 } = {}) {
    const s = this.snapshot;
    if (!s || s.nodeCount === 0) return null;
    const { xywh, edges } = s;
    const t = this.theme;
    const [minX, minY, maxX, maxY] = s.bounds;
    const w = maxX - minX + padding * 2;
    const h = maxY - minY + padding * 2;
    const esc = (text) =>
      text.replace(
        /[&<>"]/g,
        (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
      );
    const out = [
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${minX - padding} ${minY - padding} ${w} ${h}" width="${w}" height="${h}" font-family='${MONO}' font-size="12">`,
      `<rect x="${minX - padding}" y="${minY - padding}" width="${w}" height="${h}" fill="${t.background}"/>`,
      `<g fill="none" stroke="${t.edge}" stroke-width="1">`,
    ];
    for (let i = 0; i < s.edgeCount; i++) {
      const a = edges[i * STRIDE.EDGE] * STRIDE.NODE_XYWH;
      const b = edges[i * STRIDE.EDGE + 1] * STRIDE.NODE_XYWH;
      const x1 = xywh[a] + xywh[a + 2];
      const y1 = xywh[a + 1] + xywh[a + 3] / 2;
      const x2 = xywh[b];
      const y2 = xywh[b + 1] + xywh[b + 3] / 2;
      const mid = (x1 + x2) / 2;
      out.push(
        `<path d="M${x1} ${y1}C${mid} ${y1} ${mid} ${y2} ${x2} ${y2}"/>`,
      );
    }
    out.push("</g>");
    for (let i = 0; i < s.nodeCount; i++) {
      const o = i * STRIDE.NODE_XYWH;
      const [x, y, nw, nh] = [xywh[o], xywh[o + 1], xywh[o + 2], xywh[o + 3]];
      const style = t.kinds[s.kindOf(i)] ?? t.kinds[2];
      const mark = this.marks.get(i);
      const r = Math.min(6, nh / 2);
      out.push(
        `<rect x="${x}" y="${y}" width="${nw}" height="${nh}" rx="${r}" fill="${style.fill}" stroke="${mark ? t[mark] : style.stroke}" stroke-width="${mark ? 2 : 1.25}"/>`,
        // A nested <svg> clips long labels to the node, like the canvas does.
        `<svg x="${x + 6}" y="${y}" width="${Math.max(0, nw - 12)}" height="${nh}" overflow="hidden"><text x="4" y="${nh / 2}" dominant-baseline="central" fill="${style.text}">${esc(s.labelOf(i))}</text></svg>`,
      );
      if (mark)
        out.push(
          `<circle cx="${x + nw}" cy="${y}" r="7" fill="${t[mark]}"/>`,
          `<text x="${x + nw}" y="${y}" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="800" fill="${t.background}">${BADGE[mark]}</text>`,
        );
    }
    out.push("</svg>");
    return out.join("\n");
  }

  requestDraw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.draw();
    });
  }

  draw() {
    const { ctx, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.theme.background;
    ctx.fillRect(0, 0, this.width, this.height);
    if (!this.exporting) {
      this.#drawGrid();
      this.onViewChange?.(this.camera.scale);
    }

    const s = this.snapshot;
    if (!s || s.nodeCount === 0) {
      this.#placeholder();
      return;
    }

    const { x: cx, y: cy, scale } = this.camera;
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * cx, dpr * cy);

    const view = {
      l: -cx / scale,
      t: -cy / scale,
      r: (this.width - cx) / scale,
      b: (this.height - cy) / scale,
    };

    this.#drawEdges(s, view, scale);
    this.#drawNodes(s, view, scale);
  }

  #drawGrid() {
    const { ctx } = this;
    const step = 18;
    const ox = ((this.camera.x % step) + step) % step;
    const oy = ((this.camera.y % step) + step) % step;
    ctx.fillStyle = this.theme.gridDot;
    for (let x = ox; x < this.width; x += step) {
      for (let y = oy; y < this.height; y += step) ctx.fillRect(x, y, 1, 1);
    }
  }

  #drawEdges(s, view, scale) {
    const { ctx } = this;
    const { xywh, edges } = s;
    ctx.lineWidth = 1 / scale;
    ctx.strokeStyle = this.theme.edge;
    ctx.beginPath();

    for (let i = 0; i < s.edgeCount; i++) {
      const a = edges[i * STRIDE.EDGE] * STRIDE.NODE_XYWH;
      const b = edges[i * STRIDE.EDGE + 1] * STRIDE.NODE_XYWH;

      const x1 = xywh[a] + xywh[a + 2];
      const y1 = xywh[a + 1] + xywh[a + 3] / 2;
      const x2 = xywh[b];
      const y2 = xywh[b + 1] + xywh[b + 3] / 2;

      if (Math.max(x1, x2) < view.l || Math.min(x1, x2) > view.r) continue;
      if (Math.max(y1, y2) < view.t || Math.min(y1, y2) > view.b) continue;

      const mid = (x1 + x2) / 2;
      ctx.moveTo(x1, y1);
      ctx.bezierCurveTo(mid, y1, mid, y2, x2, y2);
    }
    ctx.stroke();
  }

  #drawNodes(s, view, scale) {
    const { ctx } = this;
    const { xywh } = s;
    const showText = scale > 0.42;
    ctx.textBaseline = "middle";
    ctx.font = `12px ${MONO}`;

    for (let i = 0; i < s.nodeCount; i++) {
      const o = i * STRIDE.NODE_XYWH;
      const x = xywh[o];
      const y = xywh[o + 1];
      const w = xywh[o + 2];
      const h = xywh[o + 3];

      if (x + w < view.l || x > view.r || y + h < view.t || y > view.b)
        continue;

      const style = this.theme.kinds[s.kindOf(i)] ?? this.theme.kinds[2];
      const mark = this.marks.get(i);
      ctx.fillStyle = style.fill;
      ctx.strokeStyle =
        i === this.selected
          ? this.theme.selected
          : mark
            ? this.theme[mark]
            : style.stroke;
      ctx.lineWidth = (i === this.selected || mark ? 2 : 1.25) / scale;

      const r = Math.min(6, h / 2);
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
      ctx.fill();
      ctx.stroke();

      if (mark && showText) this.#drawBadge(x + w, y, mark);

      if (showText) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x + 6, y, w - 12, h);
        ctx.clip();
        ctx.fillStyle = style.text;
        ctx.fillText(s.labelOf(i), x + 10, y + h / 2 + 0.5);
        ctx.restore();
      }
    }
  }

  #drawBadge(x, y, mark) {
    const { ctx } = this;
    ctx.save();
    ctx.fillStyle = this.theme[mark];
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = this.theme.background;
    ctx.font = `800 11px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(BADGE[mark], x, y + 0.5);
    ctx.restore();
  }

  #placeholder() {
    const { ctx } = this;
    ctx.fillStyle = this.theme.dim;
    ctx.font = `13px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("No module loaded", this.width / 2, this.height / 2);
    ctx.textAlign = "left";
  }

  hitTest(clientX, clientY) {
    const s = this.snapshot;
    if (!s) return -1;
    const rect = this.canvas.getBoundingClientRect();
    const wx = (clientX - rect.left - this.camera.x) / this.camera.scale;
    const wy = (clientY - rect.top - this.camera.y) / this.camera.scale;

    for (let i = s.nodeCount - 1; i >= 0; i--) {
      const o = i * STRIDE.NODE_XYWH;
      if (
        wx >= s.xywh[o] &&
        wx <= s.xywh[o] + s.xywh[o + 2] &&
        wy >= s.xywh[o + 1] &&
        wy <= s.xywh[o + 1] + s.xywh[o + 3]
      ) {
        return i;
      }
    }
    return -1;
  }

  #bindInput() {
    const c = this.canvas;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let moved = 0;

    c.addEventListener("pointerdown", (e) => {
      dragging = true;
      moved = 0;
      lastX = e.clientX;
      lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
    });

    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      lastX = e.clientX;
      lastY = e.clientY;
      this.camera.x += dx;
      this.camera.y += dy;
      this.requestDraw();
    });

    c.addEventListener("pointerup", (e) => {
      dragging = false;
      c.releasePointerCapture(e.pointerId);
      if (moved < 4) this.select(this.hitTest(e.clientX, e.clientY));
    });

    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = c.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        this.zoomBy(Math.exp(-e.deltaY * 0.0015), mx, my);
      },
      { passive: false },
    );

    // Pane layout changes (pass strip, resized editor) also resize the canvas.
    new ResizeObserver(() => this.resize()).observe(c);
  }
}
