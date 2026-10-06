import { explainLine } from "../anatomy.js";
import { localLineInfo, memorySpace, warpAccess } from "../gpu/index.js";
import { localExplain } from "../render/index.js";
import { parseMemref } from "../trace/index.js";
import {
  GPU_VIEW_OPTIONS,
  drawGpuView,
  focusGpu,
  placeAnswer,
  renderPath,
  setCanvasView,
} from "./canvas-views.js";
import { input, leBody } from "./dom.js";
import { inspect } from "./layout.js";
import { showSourceLine } from "./source-tabs.js";
import { selection, state } from "./state.js";

const leTitle = document.getElementById("le-title");
const leDocs = document.getElementById("le-docs");

const VERDICT_MEANING = {
  coalesced:
    "Neighboring threads use neighboring items, so the GPU fetches them in one trip.",
  strided:
    "Neighboring threads use items far apart, so the GPU fetches much more than it uses.",
  broadcast:
    "Every thread uses the same item, so it is fetched once and shared.",
  "conflict-free":
    "Every thread gets its own shared-memory counter (bank), so nobody waits.",
};

function caretLine() {
  return input.value.slice(0, input.selectionStart).split("\n").length - 1;
}

function partElement(tag, part, text) {
  const element = document.createElement(tag);
  element.className = `k-${part.kind}`;
  element.textContent = text;

  return element;
}

function markedLineElement(line, parts) {
  const pre = document.createElement("div");
  pre.className = "le-line";

  let cursor = 0;

  for (const part of parts) {
    const at = line.indexOf(part.text, cursor);
    if (at < 0) continue;
    pre.append(line.slice(cursor, at), partElement("mark", part, part.text));
    cursor = at + part.text.length;
  }

  pre.append(line.slice(cursor));

  return pre;
}

function laneLead(i, count) {
  if (i === 0) return "thread ";
  if (i === count - 1) return ", and thread ";

  return ", thread ";
}

function gpuExplain(index) {
  if (!state.gpuModel) return null;

  for (const launch of state.gpuModel.launches) {
    const kernel = state.gpuModel.kernels[launch.kernel];
    const access = kernel?.accesses?.find((a) => a.line === index + 1);
    if (!access) continue;

    const memref = parseMemref(access.type);
    const space = memorySpace(memref?.space ?? "");

    const result = warpAccess(access, memref, space, {
      defs: kernel.defs,
      args: kernel.args,
      block: launch.block,
      grid: launch.grid,
    });

    const box = document.createElement("div");
    box.className = "le-gpu";

    const head = document.createElement("p");

    if (!result.analyzed) {
      head.textContent = `On the GPU: not worked out, because ${result.reason}.`;
      box.append(head);
    } else {
      const strong = document.createElement("strong");
      strong.textContent = `On the GPU (${space} memory): ${result.verdict.replace("-", " ")}. `;

      head.append(
        strong,
        result.verdict === "bank-conflict"
          ? `Up to ${result.ways} threads queue at the same shared-memory counter (bank), so they are served one after another.`
          : (VERDICT_MEANING[result.verdict] ?? ""),
      );

      const example = document.createElement("p");
      const lanes = result.lanes.slice(0, 3);

      example.append(
        "For example, ",
        ...lanes.flatMap((lane, i) => {
          const code = document.createElement("code");
          code.textContent = `${access.buffer}[${lane.index.join(", ")}]`;

          return [
            laneLead(i, lanes.length),
            `(${lane.tx}, ${lane.ty}, ${lane.tz}) uses `,
            code,
          ];
        }),
        ".",
      );

      box.append(head, example);
    }

    const show = document.createElement("button");
    show.textContent = "Show in the GPU view";

    show.addEventListener("click", () => {
      setCanvasView("gpu");
      drawGpuView({ ...GPU_VIEW_OPTIONS, focusLine: index + 1 });
      focusGpu(index);
    });

    box.append(show);

    return box;
  }

  return null;
}

export function renderExplain(index) {
  state.explainedLine = index;

  const lines = input.value.split("\n");
  const explained = index >= 0 ? explainLine(lines, index) : null;
  leDocs.hidden = !explained?.op;

  if (!explained) {
    placeAnswer();
    leTitle.textContent = "What's this line?";

    const hint = document.createElement("p");
    hint.className = "le-hint";

    hint.textContent =
      "Click any line of code, or a box in the diagram, to see what each part of it means.";

    leBody.replaceChildren(hint);

    return;
  }

  leTitle.textContent =
    `Line ${index + 1}` + (explained.op ? ` · ${explained.op}` : "");

  if (explained.op) {
    leDocs.href = explained.docs;
    leDocs.title = `The ${explained.dialect} family in the MLIR documentation`;
  }

  const summary = document.createElement("p");
  summary.className = "le-summary";

  summary.textContent =
    explained.summary ??
    "There is no plain description of this operation yet, but its parts are labeled below.";

  const children = [summary];

  if (explained.parts.length) {
    children.push(markedLineElement(lines[index].trim(), explained.parts));

    const list = document.createElement("dl");
    list.className = "le-parts";

    for (const part of explained.parts) {
      const dt = partElement("dt", part, part.label);
      const dd = document.createElement("dd");
      const code = document.createElement("code");
      code.textContent = part.text;

      if (part.detail.startsWith(part.text)) {
        dd.append(code, part.detail.slice(part.text.length));
      } else {
        dd.append(code, " ", part.detail);
      }

      if (part.ref !== undefined) {
        const go = document.createElement("button");
        go.className = "link-btn";
        go.textContent = `go to line ${part.ref + 1}`;
        go.addEventListener("click", () => showSourceLine(part.ref));
        dd.append(" ", go);
      }

      list.append(dt, dd);
    }

    children.push(list);
  }

  const gpu = state.canvasView === "gpu" ? null : gpuExplain(index);
  if (gpu) children.push(gpu);

  const local = localExplain(localLineInfo(state.localModel, index));
  if (local) children.push(local);
  leBody.replaceChildren(...children);
  placeAnswer();
}

export function renderLinePanel() {
  renderPath();
}

export function setupLineExplain() {
  input.addEventListener("click", () => {
    selection.select({ line: caretLine() });
    inspect("line", input);
  });

  input.addEventListener("keyup", (e) => {
    if (!/^(Arrow|Page|Home|End)/.test(e.key)) return;
    selection.select({ line: caretLine() });
  });

  input.addEventListener("input", () => renderExplain(-1));
}
