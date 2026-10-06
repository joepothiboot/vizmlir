import { el } from "../dom.js";
import { analyzeGpu } from "../gpu/index.js";
import { GOOD, VERDICT_TEXT, judge } from "./gpu-verdict.js";

const passModels = new Map();

function passModel(ir) {
  if (!passModels.has(ir)) {
    if (passModels.size > 200) passModels.clear();
    passModels.set(ir, analyzeGpu(ir));
  }

  return passModels.get(ir);
}

export function acrossPasses(passes, launchIndex, kernel, judged) {
  return passes.events.map((event) => {
    const model = event.ir ? passModel(event.ir) : null;
    const launch = model?.launches[launchIndex];
    const other = launch ? model.kernels[launch.kernel] : null;
    const access = other?.accesses?.[judged.index];

    if (
      !access ||
      other.accesses.length !== kernel.accesses.length ||
      access.kind !== judged.access.kind
    ) {
      return { event, result: null };
    }

    return { event, access, ...judge(access, other, launch) };
  });
}

function passState(result) {
  if (!result?.analyzed) return "none";

  if (GOOD.has(result.verdict) && result.proof?.status !== "varies") {
    return "good";
  }

  return "bad";
}

function passResultText(result) {
  if (result?.analyzed) return verdictText(result);
  if (result) return `not analyzed: ${result.reason}`;

  return "not found: the kernel was compiled to a binary, or its loads and stores changed";
}

export function passStrip(passes, history) {
  const wrap = el("div", "gpu-passes");
  const cells = el("div", "gpu-pass-cells");
  const width = Math.max(2, String(history.length).length);

  history.forEach(({ event, result }, i) => {
    const state = passState(result);

    const cell = el(
      "button",
      `gpu-pass ${state}`,
      String(i + 1).padStart(width, "0"),
    );

    cell.type = "button";
    if (i === passes.current) cell.setAttribute("aria-current", "step");

    cell.title =
      `${i + 1}. ${passes.describe(event)}\n` + passResultText(result);

    cell.addEventListener("click", () => passes.select(i));
    cells.append(cell);
  });

  wrap.append(el("span", "gpu-kicker", "Across passes"), cells);
  wrap.append(el("p", "gpu-note", passSummary(history)));

  return [wrap];
}

function verdictText(result) {
  let cost = "";

  if (result.sectors !== undefined) {
    cost = ` · ${result.sectors} sector${result.sectors === 1 ? "" : "s"}`;
  } else if (result.ways > 1) {
    cost = ` · ${result.ways}-way`;
  }

  const varies =
    result.proof?.status === "varies"
      ? " for warp 0, varying across warps"
      : "";

  return VERDICT_TEXT[result.verdict] + cost + varies;
}

const span = (first, last) =>
  first === last ? `${first + 1}` : `${first + 1}–${last + 1}`;

export function passSummary(history) {
  const seen = history
    .map((h, i) => ({
      i,
      text: h.result?.analyzed ? verdictText(h.result) : null,
    }))
    .filter((h) => h.text);

  if (!seen.length) return "Not readable in any pass.";

  const first = seen[0].i;
  const last = seen.at(-1).i;

  const lowered =
    last < history.length - 1
      ? ` From pass ${last + 2} on it can't be found: the kernel was compiled to a binary, or its loads and stores changed.`
      : "";

  const changes = seen.filter((h, k) => k && h.text !== seen[k - 1].text);

  if (!changes.length) {
    return `${seen[0].text[0].toUpperCase()}${seen[0].text.slice(1)} in passes ${span(first, last)}: no pass changes it, so a fix belongs in the source.${lowered}`;
  }

  const steps = changes.map((h) => {
    const before = seen[seen.indexOf(h) - 1];

    return `pass ${h.i + 1} turns ${before.text} into ${h.text}`;
  });

  return `${steps.join("; ")}.${lowered}`.replace(/^./, (c) => c.toUpperCase());
}
