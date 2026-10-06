import { diffRecords, diffToPatch } from "../session/index.js";
import { diffCopy, diffTitle, setStatus } from "./dom.js";
import { renderer } from "./graph.js";
import { state } from "./state.js";

const DIFF_GLYPH = { added: "+", removed: "−", changed: "~" };
const diffSummary = document.getElementById("diff-summary");
const diffList = document.getElementById("diff-list");
const inspectorBadge = document.getElementById("inspector-badge");
let diffPicked = new Set();

export function parentLabel(snapshot, index) {
  if (!snapshot || index < 0) return "";

  return snapshot.nodes
    ? (snapshot.nodes[index]?.label ?? "")
    : snapshot.labelOf(index);
}

function rowContext(row, parent) {
  if (row.type === "changed") return `was ${row.before.label}`;
  if (parent) return `in ${parent}`;

  return "top level";
}

export function renderDiff(rows, before, after) {
  diffTitle.textContent = state.traceDiffTitle || "diff baseline → current";

  const counts = rows.reduce(
    (result, row) => {
      result[row.type] += 1;

      return result;
    },
    { added: 0, removed: 0, changed: 0 },
  );

  const total = counts.added + counts.removed + counts.changed;

  if (total) {
    diffSummary.replaceChildren(
      ...["added", "removed", "changed"].flatMap((type) => {
        const span = document.createElement("span");
        span.className = type;
        span.textContent = `${DIFF_GLYPH[type]}${counts[type]}`;

        return [span, " "];
      }),
    );
  } else {
    diffSummary.textContent = "No structural changes";
  }

  inspectorBadge.replaceChildren(
    ...(total
      ? [...diffSummary.childNodes].map((node) => node.cloneNode(true))
      : []),
  );

  state.diffRows = rows;
  state.diffBefore = before;
  state.diffAfter = after;
  state.diffCursor = -1;
  diffPicked = new Set();

  diffList.replaceChildren(
    ...rows.map((row, index) => {
      const item = document.createElement("li");
      item.className = row.type;
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", "false");

      const glyph = document.createElement("span");
      glyph.className = "glyph";
      glyph.textContent = DIFF_GLYPH[row.type];

      const op = row.after ?? row.before;
      const body = document.createElement("span");
      body.className = "body";

      const label = document.createElement("span");
      label.className = "label";
      label.textContent = op.label;

      const sub = document.createElement("span");
      sub.className = "sub";

      const parent = parentLabel(row.after ? after : before, op.parent);

      sub.textContent = rowContext(row, parent);

      body.append(label, sub);
      item.append(glyph, body);
      item.setAttribute("aria-checked", "false");

      item.addEventListener("click", (e) => {
        if (e.shiftKey && state.diffCursor >= 0) {
          pickRange(state.diffCursor, index);
        } else if (e.metaKey || e.ctrlKey) {
          togglePick(index);
        }

        focusChange(index);
      });

      return item;
    }),
  );

  syncPicks();

  renderer.setMarks(
    new Map(
      rows.filter((row) => row.after).map((row) => [row.after.index, row.type]),
    ),
  );
}

export function togglePick(index) {
  if (!diffPicked.delete(index)) diffPicked.add(index);
  syncPicks();
}

function pickRange(from, to) {
  for (let i = Math.min(from, to); i <= Math.max(from, to); i++) {
    diffPicked.add(i);
  }

  syncPicks();
}

export function clearPicks() {
  if (!diffPicked.size) return false;
  diffPicked.clear();
  syncPicks();

  return true;
}

function syncPicks() {
  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-checked", String(diffPicked.has(i))),
  );

  diffCopy.textContent = diffPicked.size ? `copy ${diffPicked.size}` : "copy";
  diffCopy.disabled = !state.diffRows.length;
}

function rowsToCopy() {
  if (diffPicked.size) return [...diffPicked].sort((a, b) => a - b);
  if (state.diffCursor >= 0) return [state.diffCursor];

  return state.diffRows.map((_, i) => i);
}

export async function copyChanges() {
  const indices = rowsToCopy();

  if (!indices.length) return;

  const rows = indices.map((i) => state.diffRows[i]);

  const records = diffRecords(rows, (row, parent) =>
    parentLabel(row.after ? state.diffAfter : state.diffBefore, parent),
  );

  try {
    await navigator.clipboard.writeText(
      diffToPatch(diffTitle.textContent, records),
    );

    setStatus(
      `copied ${rows.length} change${rows.length === 1 ? "" : "s"} as a patch`,
    );
  } catch {
    setStatus("clipboard unavailable", { error: true });
  }
}

export function syncDiffSelection(nodeIndex) {
  state.diffCursor = state.diffRows.findIndex(
    (row) => row.after?.index === nodeIndex,
  );

  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-selected", String(i === state.diffCursor)),
  );

  diffList.children[state.diffCursor]?.scrollIntoView({ block: "nearest" });
}

export function focusChange(index) {
  if (!state.diffRows.length) return;
  state.diffCursor = (index + state.diffRows.length) % state.diffRows.length;

  [...diffList.children].forEach((item, i) =>
    item.setAttribute("aria-selected", String(i === state.diffCursor)),
  );

  diffList.children[state.diffCursor].scrollIntoView({ block: "nearest" });

  const row = state.diffRows[state.diffCursor];
  if (row.after) renderer.select(row.after.index, { center: true });
}
