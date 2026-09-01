// Canvas2D painter. Reads WASM-owned buffers directly — no intermediate objects.
import { KIND_STYLE, STRIDE } from '../wasm/abi.js';

export class CanvasRenderer {
  constructor(canvas, { background = '#0b1120', onSelect = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    this.background = background;
    this.onSelect = onSelect;

    this.camera = { x: 0, y: 0, scale: 1 };
    this.snapshot = null;
    this.selected = -1;
    this.dpr = 1;
    this._raf = 0;

    this.#bindInput();
    this.resize();
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
    this.selected = -1;
    if (fit) this.fit();
    this.requestDraw();
  }

  fit(padding = 48) {
    const s = this.snapshot;
    if (!s || s.nodeCount === 0) return;
    const [minX, minY, maxX, maxY] = s.bounds;
    const w = Math.max(1, maxX - minX);
    const h = Math.max(1, maxY - minY);
    const scale = Math.min((this.width - padding * 2) / w, (this.height - padding * 2) / h, 2);
    this.camera.scale = Math.max(scale, 0.05);
    this.camera.x = this.width / 2 - ((minX + maxX) / 2) * this.camera.scale;
    this.camera.y = this.height / 2 - ((minY + maxY) / 2) * this.camera.scale;
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
    ctx.fillStyle = this.background;
    ctx.fillRect(0, 0, this.width, this.height);

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

  #drawEdges(s, view, scale) {
    const { ctx } = this;
    const { xywh, edges } = s;
    ctx.lineWidth = 1 / scale;
    ctx.strokeStyle = 'rgba(148,163,184,0.38)';
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
    ctx.textBaseline = 'middle';
    ctx.font = `${13 / 1}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.lineWidth = 1.25 / scale;

    for (let i = 0; i < s.nodeCount; i++) {
      const o = i * STRIDE.NODE_XYWH;
      const x = xywh[o];
      const y = xywh[o + 1];
      const w = xywh[o + 2];
      const h = xywh[o + 3];

      if (x + w < view.l || x > view.r || y + h < view.t || y > view.b) continue;

      const style = KIND_STYLE[s.kindOf(i)] ?? KIND_STYLE[2];
      ctx.fillStyle = style.fill;
      ctx.strokeStyle = i === this.selected ? '#f8fafc' : style.stroke;

      const r = Math.min(6, h / 2);
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
      ctx.fill();
      ctx.stroke();

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

  #placeholder() {
    const { ctx } = this;
    ctx.fillStyle = '#475569';
    ctx.font = '14px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No module loaded', this.width / 2, this.height / 2);
    ctx.textAlign = 'left';
  }

  /** Screen -> world, then linear scan for a hit. */
  hitTest(clientX, clientY) {
    const s = this.snapshot;
    if (!s) return -1;
    const rect = this.canvas.getBoundingClientRect();
    const wx = (clientX - rect.left - this.camera.x) / this.camera.scale;
    const wy = (clientY - rect.top - this.camera.y) / this.camera.scale;

    for (let i = s.nodeCount - 1; i >= 0; i--) {
      const o = i * STRIDE.NODE_XYWH;
      if (wx >= s.xywh[o] && wx <= s.xywh[o] + s.xywh[o + 2] &&
          wy >= s.xywh[o + 1] && wy <= s.xywh[o + 1] + s.xywh[o + 3]) {
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

    c.addEventListener('pointerdown', (e) => {
      dragging = true;
      moved = 0;
      lastX = e.clientX;
      lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
    });

    c.addEventListener('pointermove', (e) => {
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

    c.addEventListener('pointerup', (e) => {
      dragging = false;
      c.releasePointerCapture(e.pointerId);
      if (moved < 4) {
        this.selected = this.hitTest(e.clientX, e.clientY);
        this.onSelect?.(this.selected, this.snapshot);
        this.requestDraw();
      }
    });

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const next = Math.min(4, Math.max(0.05, this.camera.scale * factor));
      const k = next / this.camera.scale;
      this.camera.x = mx - (mx - this.camera.x) * k;
      this.camera.y = my - (my - this.camera.y) * k;
      this.camera.scale = next;
      this.requestDraw();
    }, { passive: false });

    window.addEventListener('resize', () => this.resize());
  }
}