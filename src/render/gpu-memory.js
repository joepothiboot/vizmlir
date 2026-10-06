import { el } from "../dom.js";
import { formatBytes, parseMemref } from "../trace/index.js";
import { verdictChip } from "./gpu-verdict.js";
import { count } from "./gpu-view.js";

const SPACE_TEXT = {
  global: ["Global", "device memory, visible to every thread in the launch"],
  shared: ["Shared", "one copy per block, visible to that block's threads"],
  private: ["Private", "one copy per thread (registers or local memory)"],
};

export function memorySummary(kernel) {
  const n = kernel?.buffers?.length ?? 0;

  return n ? `Memory · ${n} buffer${n > 1 ? "s" : ""}` : "Memory";
}

const SOURCE_TEXT = {
  argument: "Passed in by the host when the kernel launches.",
  workgroup:
    "Declared with workgroup(...) on the kernel: the GPU sets aside one copy per block.",
  private: "Declared with private(...) on the kernel: one copy per thread.",
  alloc: "Allocated inside the kernel.",
  alloca: "Allocated on each thread's stack inside the kernel.",
  captured: "A host value that the launch body uses directly.",
};

function copiesText(space, blocks, threads) {
  if (space === "shared" && blocks) {
    return ` per block, ${blocks.toLocaleString()} copies across the grid`;
  }

  if (space === "private" && threads) {
    return ` per thread, ${threads.toLocaleString()} copies`;
  }

  return "";
}

function bufferDetail(buffer, { blocks, threads, accesses, onLine, kernel }) {
  const detail = el("div", "gpu-buffer-detail");
  const memref = parseMemref(buffer.type);
  const facts = el("dl", "gpu-facts");

  const shape = memref?.dims?.length
    ? `${memref.dims.map((d) => d ?? "?").join(" × ")} of ${memref.element}`
    : (memref?.element ?? buffer.type);

  const size =
    buffer.bytes === null ? "only known at runtime" : formatBytes(buffer.bytes);

  const copies = copiesText(buffer.space, blocks, threads);

  const [spaceName, spaceBlurb] = SPACE_TEXT[buffer.space] ?? [
    buffer.space,
    "a custom memory space",
  ];

  for (const [term, value] of [
    ["Holds", shape],
    ["Size", size + copies],
    ["Lives in", `${spaceName} memory: ${spaceBlurb}`],
    ["From", SOURCE_TEXT[buffer.source] ?? buffer.source],
  ]) {
    facts.append(el("dt", "", term), el("dd", "", value));
  }

  detail.append(facts);

  const uses = (
    accesses?.judged ?? (kernel.accesses ?? []).map((access) => ({ access }))
  )
    .map((j, i) => ({ ...j, i }))
    .filter((j) => j.access.buffer === buffer.name);

  if (!uses.length) {
    detail.append(
      el(
        "p",
        "gpu-note",
        buffer.loads || buffer.stores
          ? "Its loads and stores could not be placed on a line here."
          : "Never read or written in this kernel.",
      ),
    );

    return detail;
  }

  const list = el("ul", "gpu-buffer-uses");

  for (const j of uses) {
    const button = el("button");
    button.type = "button";

    button.append(
      el("span", "gpu-dim", `line ${j.access.line}`),
      el(
        "span",
        "gpu-code",
        ` ${j.access.kind === "load" ? "load" : "store"} ${j.access.buffer}[${j.access.indices.join(", ")}]${j.access.inLoop ? "  ↻" : ""} `,
      ),
    );

    if (j.result) button.append(verdictChip(j.result));

    button.title = accesses
      ? "Show its lanes below and mark its line"
      : "Mark its line in the source";

    button.addEventListener("click", () =>
      accesses ? accesses.pick(j.i) : onLine?.(j.access.line),
    );

    const item = el("li");
    item.append(button);
    list.append(item);
  }

  detail.append(list);

  return detail;
}

export function memorySection(
  kernel,
  launch,
  { accesses = null, onLine, onSpace, heading = true } = {},
) {
  const section = el("div", "gpu-memory");
  if (heading) section.append(el("h4", "", "Memory"));

  if (kernel.lowered && !kernel.buffers.length && !kernel.ptx) {
    section.append(
      el(
        "p",
        "gpu-note",
        `This kernel is ${kernel.op} here, so its buffers are no longer memrefs. Step back to a pass where it is a gpu.func to see them.`,
      ),
    );

    return section;
  }

  if (!kernel.op && !kernel.buffers.length) {
    section.append(el("p", "gpu-note", "The kernel's body is not in this IR."));

    return section;
  }

  if (!kernel.buffers.length && kernel.ptx) {
    section.append(
      el(
        "p",
        "gpu-note",
        "Only the generated PTX is left here, where buffers are plain pointers. Step back to a pass where the kernel is a gpu.func to see them by memory space.",
      ),
    );

    if (kernel.ptx.sharedBytes) {
      section.append(
        el(
          "p",
          "gpu-total",
          `PTX declares ${formatBytes(kernel.ptx.sharedBytes)} of shared memory per block`,
        ),
      );
    }

    if (kernel.ptx.registers.length) section.append(ptxNote(kernel.ptx));

    return section;
  }

  const columns = el("div", "gpu-spaces");

  const spaces = new Map([
    ["global", []],
    ["shared", []],
    ["private", []],
  ]);

  for (const buffer of kernel.buffers) {
    if (!spaces.has(buffer.space)) spaces.set(buffer.space, []);
    spaces.get(buffer.space).push(buffer);
  }

  const blocks = count(launch?.grid);
  const threads = launch?.threads ?? null;

  for (const [space, buffers] of spaces) {
    if (!buffers.length && space === "private" && !kernel.ptx) continue;

    const column = el("div", `gpu-space ${space.replace(/\s+/g, "-")}`);
    const [title, blurb] = SPACE_TEXT[space] ?? [space, ""];
    const known = buffers.every((b) => b.bytes !== null);
    const bytes = buffers.reduce((sum, b) => sum + (b.bytes ?? 0), 0);
    column.append(el("h5", "", title), el("p", "gpu-note", blurb));

    const list = el("ul");

    for (const buffer of buffers) {
      const item = el("li");
      const toggle = el("button", "gpu-buffer");
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.title = "Show its size, origin, and every load and store";

      const access = [
        buffer.loads
          ? `${buffer.loads} load${buffer.loads > 1 ? "s" : ""}`
          : "",
        buffer.stores
          ? `${buffer.stores} store${buffer.stores > 1 ? "s" : ""}`
          : "",
      ].filter(Boolean);

      toggle.append(
        el("code", "", buffer.name),
        el("span", "gpu-type", ` ${buffer.type}`),
        el(
          "span",
          "gpu-dim",
          ` · ${buffer.bytes === null ? "dynamic size" : formatBytes(buffer.bytes)}` +
            ` · ${buffer.source}` +
            (access.length ? ` · ${access.join(", ")}` : " · not accessed"),
        ),
      );

      let detail = null;

      toggle.addEventListener("click", () => {
        detail ??= bufferDetail(buffer, {
          blocks,
          threads,
          accesses,
          onLine,
          kernel,
        });

        const open = toggle.getAttribute("aria-expanded") !== "true";
        toggle.setAttribute("aria-expanded", String(open));
        if (open) item.append(detail);
        else detail.remove();
      });

      const light = (on) => {
        accesses?.highlight(on ? buffer.name : null);
        onSpace?.(on ? buffer.space : null);
      };

      toggle.addEventListener("pointerenter", () => light(true));
      toggle.addEventListener("pointerleave", () => light(false));
      toggle.addEventListener("focus", () => light(true));
      toggle.addEventListener("blur", () => light(false));
      item.append(toggle);
      list.append(item);
    }

    if (!buffers.length) list.append(el("li", "gpu-dim", "none"));
    column.append(list);

    if (buffers.length && space === "shared") {
      column.append(
        el(
          "p",
          "gpu-total",
          `${known ? formatBytes(bytes) : "≥ " + formatBytes(bytes)} per block` +
            (blocks ? ` · ${blocks} copies across the grid` : ""),
        ),
      );
    } else if (buffers.length && space === "private") {
      column.append(
        el(
          "p",
          "gpu-total",
          `${formatBytes(bytes)} per thread` +
            (threads ? ` · ${threads.toLocaleString()} copies` : ""),
        ),
      );
    } else if (buffers.length) {
      column.append(
        el(
          "p",
          "gpu-total",
          `${known ? "" : "≥ "}${formatBytes(bytes)} in total`,
        ),
      );
    }

    if (space === "private" && kernel.ptx?.registers.length) {
      column.append(ptxNote(kernel.ptx));
    }

    columns.append(column);
  }

  section.append(columns);

  return section;
}

function ptxNote(ptx) {
  const text = ptx.registers
    .map((reg) => `${reg.count} × .${reg.type}`)
    .join(", ");

  return el(
    "p",
    "gpu-note",
    `PTX declares ${text} virtual registers per thread. ptxas maps them to physical registers later, so the real count can differ.`,
  );
}
