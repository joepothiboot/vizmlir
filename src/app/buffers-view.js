// Renders the Buffers dialog: a summary, one lifetime chart per function
// (baseline vs current), and, for traces, buffer totals at every pass.

import { formatBytes } from "../trace/index.js";

const GLYPH = { added: "+", removed: "−", changed: "~", same: "" };
const FREED = {
  dealloc: "freed by dealloc",
  returned: "returned",
  "last-use": "no dealloc; live to last use",
  scope: "stack; live to last use",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function signedBytes(delta) {
  if (!delta) return "±0";
  return `${delta > 0 ? "+" : "−"}${formatBytes(Math.abs(delta))}`;
}

// "32.0 KB → 48.0 KB (+16.0 KB)", or just the value without a baseline.
function compare(before, after, format = formatBytes) {
  if (before === null || before === undefined) return format(after);
  if (before === after) return format(after);
  const delta = after - before;
  const change =
    format === formatBytes ? signedBytes(delta) : delta > 0 ? `+${delta}` : `−${-delta}`;
  return `${format(before)} → ${format(after)} (${change})`;
}

export function renderSummary(dl, { before, after, globals }) {
  const count = (n) => String(n);
  const facts = [
    ["Buffers", compare(before?.buffers, after.buffers, count)],
    ["Allocated", compare(before?.allocated, after.allocated)],
    ["Peak live", compare(before?.peak, after.peak)],
  ];
  if (after.dynamic || before?.dynamic)
    facts.push([
      "Dynamic size",
      `${compare(before?.dynamic, after.dynamic, count)} (not in totals)`,
    ]);
  if (globals.length) {
    const bytes = globals.reduce((sum, g) => sum + (g.bytes ?? 0), 0);
    facts.push(["Globals", `${globals.length} · ${formatBytes(bytes)}`]);
  }
  dl.replaceChildren(
    ...facts.flatMap(([term, value]) => [el("dt", "", term), el("dd", "", value)]),
  );
}

// Step line of live bytes, scaled to the function's own length.
function curve(fn, scale, className) {
  if (!fn) return "";
  const w = 1000;
  const h = 40;
  const x = (p) => ((p / fn.length) * w).toFixed(1);
  const y = (bytes) => (h - (scale ? (bytes / scale) * (h - 2) : 0)).toFixed(1);
  let d = `M0 ${h}`;
  fn.live.forEach((bytes, p) => {
    d += ` L${x(p)} ${y(bytes)} L${x(p + 1)} ${y(bytes)}`;
  });
  d += ` L${w} ${h}`;
  return `<path class="${className}" d="${d}" vector-effect="non-scaling-stroke"/>`;
}

function bar(buffer, fn, className) {
  const span = el("span", `buf-bar ${className}`);
  span.style.left = `${(buffer.start / fn.length) * 100}%`;
  span.style.width = `${Math.max(((buffer.end - buffer.start + 1) / fn.length) * 100, 0.6)}%`;
  return span;
}

function describe(buffer, fn) {
  return (
    `${buffer.op} at line ${buffer.line + 1}` +
    ` · live ${buffer.start + 1}–${buffer.end + 1} of ${fn.length}` +
    ` · ${FREED[buffer.freed]}` +
    (buffer.aliases.length ? ` · aliases ${buffer.aliases.join(", ")}` : "")
  );
}

/**
 * One section per function that has buffers on either side. `onLine(line)`
 * is called with a 0-based current-IR line when a current buffer is clicked.
 */
export function renderFunctions(container, comparison, { hasBaseline, onLine }) {
  const sections = comparison
    .filter((entry) => entry.rows.length)
    .map((entry) => {
      const { before, after } = entry;
      const section = el("section", "buf-fn");
      const head = el("div", "buf-fn-head");
      head.append(el("code", "", entry.name));
      const status = !after ? " (removed)" : !before && hasBaseline ? " (added)" : "";
      head.append(
        el(
          "span",
          "dialog-note",
          `peak ${compare(hasBaseline ? before?.peak ?? 0 : null, after?.peak ?? 0)}` + status,
        ),
      );
      section.append(head);

      const scale = Math.max(before?.peak ?? 0, after?.peak ?? 0);
      const svg = el("div", "buf-curve");
      svg.innerHTML =
        `<svg viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true">` +
        (hasBaseline ? curve(before, scale, "before") : "") +
        curve(after, scale, "after") +
        `</svg>`;
      svg.title =
        "Live bytes through the function body" +
        (hasBaseline ? " (dashed: baseline)" : "");
      section.append(svg);

      const list = el("ul", "buf-rows");
      for (const row of entry.rows) {
        const buffer = row.after ?? row.before;
        const fn = row.after ? after : before;
        const li = el("li", row.status);
        const label = el(row.after ? "button" : "div", "buf-label");
        label.append(
          el("span", "glyph", GLYPH[row.status]),
          el("span", "name", buffer.name),
          el("span", "type", buffer.type),
          el(
            "span",
            "size",
            buffer.bytes === null ? "dynamic" : formatBytes(buffer.bytes),
          ),
        );
        label.title =
          describe(buffer, fn) +
          (row.status === "changed"
            ? `\nbaseline: ${describe(row.before, before)}`
            : "");
        if (row.after) {
          label.type = "button";
          label.addEventListener("click", () => onLine(row.after.line));
        }
        const track = el("div", "buf-track");
        if (row.status === "changed") track.append(bar(row.before, before, "ghost"));
        track.append(bar(buffer, fn, buffer.freed));
        li.append(label, track);
        list.append(li);
      }
      section.append(list);
      return section;
    });
  container.replaceChildren(...sections);
  return sections.length;
}

/**
 * Buffer totals at each trace dump. `rows` are
 * `{ header, title, totals }`; `onPass(i)` selects a dump.
 */
export function renderPasses(table, rows, { current, onlyChanged, onPass }) {
  const changed = (i) =>
    i > 0 &&
    ["buffers", "allocated", "peak"].some(
      (key) => rows[i].totals[key] !== rows[i - 1].totals[key],
    );
  const shown = rows
    .map((_, i) => i)
    .filter((i) => !onlyChanged || i === 0 || i === current || changed(i));

  const head = el("tr");
  for (const [text, className] of [
    ["#", ""],
    ["Pass", "pass"],
    ["Buffers", ""],
    ["Allocated", ""],
    ["Peak live", ""],
  ])
    head.append(el("th", className, text));
  const thead = el("thead");
  thead.append(head);

  const tbody = el("tbody");
  for (const i of shown) {
    const { header, title, totals } = rows[i];
    const previous = i > 0 ? rows[i - 1].totals : totals;
    const tr = el("tr", i === current ? "current" : "");
    tr.append(el("td", "", header));
    const pass = el("td", "pass");
    const link = el("button", "link-btn", title);
    link.type = "button";
    link.title = `${title}\nGo to this pass`;
    link.addEventListener("click", () => onPass(i));
    pass.append(link);
    tr.append(pass);
    for (const [key, format] of [
      ["buffers", String],
      ["allocated", formatBytes],
      ["peak", formatBytes],
    ]) {
      const delta = totals[key] - previous[key];
      const td = el(
        "td",
        delta > 0 ? "up" : delta < 0 ? "down" : "same",
        format(totals[key]),
      );
      if (delta)
        td.title = `${key === "buffers" ? (delta > 0 ? `+${delta}` : `−${-delta}`) : signedBytes(delta)} in this pass`;
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.replaceChildren(thead, tbody);
  return { shown: shown.length, changed: rows.filter((_, i) => changed(i)).length };
}
