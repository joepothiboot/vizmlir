import { STORAGE_KEYS } from "../constants.js";
import { el } from "../dom.js";
import { expectedCost } from "../gpu/index.js";
import { acrossPasses, passStrip } from "./gpu-passes.js";
import {
  compilerBody,
  judge,
  plainBody,
  reachChip,
  verdictChip,
} from "./gpu-verdict.js";

let depth = "plain";

try {
  if (localStorage.getItem(STORAGE_KEYS.depth) === "compiler") {
    depth = "compiler";
  }
} catch {}

export const cards = new Set();

export function isCompilerDepth() {
  return depth === "compiler";
}

export function answerCard(kernel, options) {
  const card = el("div", "gpu-answer");
  let judged = null;

  const render = () => {
    if (!judged) return;

    const { access, result, space } = judged;
    const head = el("div", "gpu-answer-top");

    head.append(
      el(
        "span",
        "gpu-kicker",
        `Line ${access.line} · ${access.kind} ${access.buffer}`,
      ),
      verdictChip(result),
    );

    if (result.analyzed) head.append(reachChip(result.proof));

    const toggle = el("div", "gpu-depth");
    toggle.setAttribute("role", "group");
    toggle.setAttribute("aria-label", "Explain in");

    for (const [value, label] of [
      ["plain", "Plain"],
      ["compiler", "Compiler"],
    ]) {
      const button = el("button", "", label);
      button.type = "button";
      button.setAttribute("aria-pressed", String(depth === value));

      button.addEventListener("click", () => {
        depth = value;

        try {
          localStorage.setItem(STORAGE_KEYS.depth, value);
        } catch {}

        for (const other of cards) {
          other.render();
          other.onDepth?.();
        }
      });

      toggle.append(button);
    }

    head.append(toggle);

    const body =
      depth === "compiler" && result.analyzed
        ? compilerBody(judged)
        : plainBody(result, space);

    const passes = options.passes?.();

    const history = passes
      ? passStrip(
          passes,
          acrossPasses(passes, options.launchIndex, kernel, judged),
        )
      : [];

    card.replaceChildren(
      head,
      ...body,
      ...measuredLine(kernel, options, judged),
      ...history,
    );
  };

  card.render = render;

  card.show = (next) => {
    judged = next;
    render();
  };

  cards.add(card);

  return card;
}

const MEASURED_TOLERANCE = 0.05;

const COUNTER = {
  global: {
    load: "globalLoad",
    store: "globalStore",
    unit: ["chunk of 32 bytes", "chunks of 32 bytes", "per warp request"],
  },
  shared: {
    load: "sharedLoad",
    store: "sharedStore",
    unit: ["pass", "passes", "per warp"],
  },
};

function measuredLine(kernel, options, { access, result, space }) {
  const measured = options.measured?.();
  const counter = COUNTER[space];
  if (!measured || !counter || !result.analyzed || kernel.triton) return [];

  const name = kernel.inline
    ? `${options.launch.host.slice(1)}_kernel`
    : kernel.name;

  const value = measured.counters.get(name)?.[counter[access.kind]];
  if (value === null || value === undefined) return [];

  const peers = kernel.accesses
    .filter((a) => a.kind === access.kind)
    .map((a) => judge(a, kernel, options.launch))
    .filter((j) => j.space === space);

  if (peers.some((j) => !j.result.analyzed)) return [];

  const predicted =
    peers.reduce((sum, j) => sum + expectedCost(j.result, space).mean, 0) /
    peers.length;

  const agrees = Math.abs(value - predicted) <= MEASURED_TOLERANCE * predicted;
  const n = (x) => String(+x.toFixed(3));

  let scope = `; predicted ${n(predicted)}`;

  if (peers.length > 1) {
    scope = `, averaged over this kernel's ${peers.length} ${access.kind}s; predicted ${n(predicted)}`;
  } else if (agrees && n(value) === n(predicted)) {
    scope = ", as predicted";
  }

  const line = el("p", `gpu-measured ${agrees ? "good" : "bad"}`);

  line.append(
    el("span", "gpu-measured-mark", agrees ? "✓ " : "✗ "),
    `Measured on a ${measured.device}: ${n(value)} ${counter.unit[value === 1 ? 0 : 1]} ${counter.unit[2]}${scope}.`,
  );

  return [line];
}
