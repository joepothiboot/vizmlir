import { WARP } from "../constants.js";
import { el } from "../dom.js";
import { gpuScene } from "./gpu-3d.js";
import { answerCard, cards, isCompilerDepth } from "./gpu-answer.js";
import { elementMap, renderLaneDetail, renderLayout } from "./gpu-layout.js";
import { memorySection, memorySummary } from "./gpu-memory.js";
import { GOOD, groupOrder, judge, reach, verdictChip } from "./gpu-verdict.js";

const dim = (dims) => (dims ? dims.map((d) => d ?? "?").join(" × ") : "?");

export const count = (dims) =>
  dims && dims.every((d) => d !== null)
    ? dims.reduce((a, b) => a * b, 1)
    : null;

function kernelOrigin(kernel, launch) {
  if (kernel?.triton) return " · Triton GPU IR";

  if (kernel?.inline) {
    return " · written inline, not outlined into a kernel yet";
  }

  if (launch.host) return ` launched from ${launch.host}`;

  return "";
}

function launchSection(launch, kernel, options) {
  const section = el("section", "gpu-launch");
  const head = el("h3");

  head.append(
    el("code", "", kernel?.path ?? kernel?.name ?? "kernel"),
    el("span", "gpu-dim", kernelOrigin(kernel, launch)),
  );

  section.append(head);

  const blocks = count(launch.grid);
  const perBlock = count(launch.block);
  const facts = el("dl", "gpu-facts");

  const rows = kernel?.triton
    ? [
        ["Programs", "set at launch (not in this IR)"],
        ["Program", `${kernel.triton.numWarps} warps = ${perBlock} threads`],
      ]
    : [
        ["Grid", `${dim(launch.grid)} = ${blocks ?? "?"} blocks`],
        ["Block", `${dim(launch.block)} = ${perBlock ?? "?"} threads`],
        [
          "Threads",
          launch.threads === null
            ? "some sizes are only known at runtime"
            : `${launch.threads.toLocaleString("en-US")} in ${(Math.ceil((perBlock ?? 0) / WARP) * (blocks ?? 0)).toLocaleString("en-US")} warps of ${WARP}`,
        ],
      ];

  for (const [term, value] of rows) {
    const pair = el("div");
    pair.append(el("dt", "", term), el("dd", "", value));
    facts.append(pair);
  }

  const scene = gpuScene(launch, kernel, { onLine: options.onLine });

  const answer = kernel?.accesses?.length
    ? answerCard(kernel, { ...options, launch })
    : null;

  if (answer && !options.onAnswer) section.append(answer);

  const layout = kernel?.triton && answer ? el("figure", "gpu-layout") : null;
  if (layout) section.append(layout);
  section.append(scene.figure);

  const onPick = (judged, { initial = false } = {}) => {
    answer.show(judged);

    options.onAnswer?.(answer, judged, {
      initial,
      launchIndex: options.launchIndex,
    });

    if (layout) renderLayout(layout, judged, kernel);
    scene.showAccess(judged, { formula: isCompilerDepth() });

    answer.onDepth = () =>
      scene.showAccess(judged, { formula: isCompilerDepth() });

    scene.showElements(
      judged.result.analyzed
        ? elementMap(
            judged.access,
            judged.result,
            judged.memref,
            groupOrder(judged.result),
          )
        : null,
    );
  };

  const accesses = kernel?.accesses?.length
    ? accessSection(kernel, launch, { ...options, onPick })
    : null;

  const memory = kernel
    ? memorySection(kernel, launch, {
        accesses,
        onLine: options.onLine,
        onSpace: scene.showSpace,
        heading: false,
      })
    : null;

  scene.overlay(facts, memory, memorySummary(kernel));
  if (accesses) section.append(accesses);

  shown.push({
    section,
    scene,
    accesses,
    kernel,
    name: kernel?.path ?? kernel?.name ?? "kernel",
  });

  return section;
}

let shown = [];
let focused = null;

export function focusGpuLine(line) {
  shown = shown.filter((entry) => entry.section.isConnected);
  focused = null;

  for (const entry of shown) {
    const lit = entry.scene.showLine(focused ? null : line);
    if (!lit || focused) continue;
    focused = { entry, line };
    entry.accesses?.pickLine(line);
  }

  if (focused) reveal(focused.entry.scene.figure);

  return !!focused;
}

function reveal(figure) {
  const view = figure.closest("#gpu-view");
  if (!view || view.hidden) return;

  const outer = view.getBoundingClientRect();
  const inner = figure.getBoundingClientRect();

  if (inner.top < outer.top || inner.bottom > outer.bottom) {
    view.scrollTo({
      top: view.scrollTop + inner.top - outer.top - 12,
      behavior: "smooth",
    });
  }

  figure.classList.remove("gpu-flash");
  void figure.offsetWidth;
  figure.classList.add("gpu-flash");
}

export function renderGpuPath(container, { onLine } = {}) {
  shown = shown.filter((entry) => entry.section.isConnected);

  const step = (node, entry) => {
    const button = el("button", "gpu-step");
    button.type = "button";

    button.append(
      el("span", "gpu-dim", `line ${node.line}`),
      el("code", "", node.text),
    );

    if (node.kind === "access") {
      const judged = entry.accesses?.judged.find(
        (j) => j.access.line === node.line,
      );

      if (judged) {
        button.append(el("span", "gpu-dim", ` ${judged.space} memory `));
      }

      if (judged && node.line !== focused?.line) {
        button.append(verdictChip(judged.result));
      }
    } else if (
      node.kind === "thread" ||
      node.kind === "block" ||
      node.kind === "size"
    ) {
      button.append(el("span", "gpu-dim", ` ${node.label}`));
    }

    if (focused && node.line === focused.line) button.classList.add("current");
    button.addEventListener("click", () => onLine?.(node.line));

    return button;
  };

  const children = [];

  if (!shown.length) {
    children.push(
      el("p", "gpu-note", "No kernels with loads or stores to trace here."),
    );
  } else if (!focused) {
    children.push(
      el("h3", "", "How the IR reaches the GPU"),
      el(
        "p",
        "gpu-note",
        "Click a line of kernel code, a row in Memory accesses, or a node on the IR wall in the 3D view to trace it from the thread and block ids, through the index math, to the memory it touches.",
      ),
    );
  } else {
    const { entry, line } = focused;
    const { flow } = entry.scene;

    const lit = new Set(
      flow.nodes.filter((n) => entry.scene.lit?.(n.id)).map((n) => n.id),
    );

    const nodes = flow.nodes
      .filter((n) => lit.has(n.id))
      .sort((a, b) => a.rank - b.rank || a.line - b.line);

    children.push(
      el("h3", "", `Line ${line} on its way to the GPU`),
      el("p", "gpu-note", entry.name),
    );

    for (const [title, kinds, note] of [
      [
        "Where the thread is",
        ["thread", "block", "size"],
        "The ids the launch gives each thread and block.",
      ],
      [
        "Index math",
        ["math", "const", "loop", "arg"],
        "Ops that turn those ids into an element index.",
      ],
      [
        "Memory",
        ["access"],
        "The element each thread reads or writes, and how the warp's accesses land.",
      ],
    ]) {
      const group = nodes.filter((n) => kinds.includes(n.kind));
      if (!group.length) continue;
      children.push(el("h4", "", title), el("p", "gpu-note", note));

      const list = el("ol", "gpu-steps");

      for (const node of group) {
        const item = el("li");
        item.append(step(node, entry));
        list.append(item);
      }

      children.push(list);
    }

    const clear = el("button", "link-btn", "show all loads and stores");

    clear.addEventListener("click", () => {
      focusGpuLine(null);
      renderGpuPath(container, { onLine });
    });

    children.push(clear);
  }

  container.replaceChildren(...children);
}

function checkedCell(result) {
  if (!result.analyzed) return el("span", "gpu-dim", "");

  if (result.proof?.status === "varies") {
    return el("span", "gpu-chip bad", "varies");
  }

  return el("span", "gpu-dim", reach(result.proof));
}

function accessSection(kernel, launch, options) {
  const section = el("div", "gpu-accesses");

  section.append(
    el("h4", "", "Memory accesses"),
    el(
      "p",
      "gpu-note",
      "The lanes shown are warp 0 of block (0, 0, 0) on the first loop iteration: which element each of its 32 threads touches. Global memory is judged by 32-byte sectors, shared memory by its 32 banks. When an index is a linear function of the thread ids, block ids and loop counters, the verdict is also checked for every warp and iteration.",
    ),
  );

  const judged = kernel.accesses.map((access, index) => ({
    access,
    index,
    ...judge(access, kernel, launch),
  }));

  const focused = judged.findIndex((j) => j.access.line === options.focusLine);

  let picked =
    focused >= 0
      ? focused
      : judged.findIndex(
          (j) => j.result.analyzed && !GOOD.has(j.result.verdict),
        );

  if (picked < 0) picked = 0;
  if (focused >= 0) section.dataset.focus = "true";

  const table = el("table", "gpu-access-table");
  const head = el("tr");

  for (const title of ["Line", "Access", "Space", "Verdict", "Checked"]) {
    head.append(el("th", "", title));
  }

  const thead = el("thead");
  thead.append(head);

  const tbody = el("tbody");
  const detail = el("div", "gpu-lane-detail");

  const pick = (i, { mark = false, initial = false } = {}) => {
    picked = i;

    [...tbody.children].forEach((row, k) =>
      row.setAttribute("aria-selected", String(k === i)),
    );

    renderLaneDetail(detail, judged[i]);
    options.onPick?.(judged[i], { initial });
    if (mark) options.onLine?.(judged[i].access.line);
  };

  judged.forEach((j, i) => {
    const row = el("tr");
    row.tabIndex = 0;

    const text = `${j.access.kind === "load" ? "load" : "store"} ${j.access.buffer}[${j.access.indices.join(", ")}]`;

    row.append(
      el("td", "gpu-dim", String(j.access.line)),
      el("td", "gpu-code", text + (j.access.inLoop ? "  ↻" : "")),
      el("td", "", j.space),
    );

    const verdict = el("td");
    verdict.append(verdictChip(j.result));

    const checked = checkedCell(j.result);

    const reachCell = el("td");
    reachCell.append(checked);
    row.append(verdict, reachCell);

    row.title =
      (j.access.inLoop && j.result.proof?.status !== "proven"
        ? "Inside a loop: first iteration shown\n"
        : "") + "Show its lanes and mark its line in the source";

    row.addEventListener("click", () => pick(i, { mark: true }));

    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick(i, { mark: true });
      }
    });

    tbody.append(row);
  });

  table.append(thead, tbody);
  section.append(table, detail);
  pick(picked, { initial: true });
  section.judged = judged;

  section.pick = (i) => {
    pick(i, { mark: true });
    tbody.children[i].scrollIntoView({ block: "nearest" });
  };

  section.pickLine = (line) => {
    const i = judged.findIndex((j) => j.access.line === line);
    if (i >= 0) pick(i);
  };

  section.highlight = (name) =>
    judged.forEach((j, i) =>
      tbody.children[i].classList.toggle(
        "gpu-linked",
        j.access.buffer === name,
      ),
    );

  return section;
}

export function renderGpuView(container, model, options = {}) {
  shown = [];
  focused = null;
  cards.clear();

  const children = [
    el(
      "p",
      "gpu-note gpu-intro",
      "Read from the IR, not measured: launch sizes come from constants and buffers from their memref types. Occupancy, caching and timing need a profiler.",
    ),
  ];

  if (!model) {
    children.push(
      el("p", "gpu-note", "No GPU kernels or launches in this IR."),
    );

    container.replaceChildren(...children);

    return;
  }

  const launched = new Set();

  model.launches.forEach((launch, launchIndex) => {
    launched.add(launch.kernel);

    children.push(
      launchSection(launch, model.kernels[launch.kernel], {
        ...options,
        launchIndex,
      }),
    );
  });

  model.kernels.forEach((kernel, i) => {
    if (launched.has(i)) return;

    const section = el("section", "gpu-launch");
    const head = el("h3");

    head.append(
      el("code", "", kernel.path ?? kernel.name),
      el(
        "span",
        "gpu-dim",
        " · not launched in this IR, so launch sizes are unknown",
      ),
    );

    section.append(
      head,
      memorySection(kernel, null, { onLine: options.onLine }),
    );

    children.push(section);
  });

  container.replaceChildren(...children);

  container
    .querySelector('[data-focus="true"]')
    ?.scrollIntoView({ block: "start" });
}
