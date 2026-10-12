import { compareTraces, parsePassTrace, rowDiff } from "../trace/index.js";

const inputA = document.getElementById("compare-a");
const inputB = document.getElementById("compare-b");
const verdictEl = document.getElementById("compare-verdict");
const rowsEl = document.getElementById("compare-rows");
const diffEl = document.getElementById("compare-diff");

const KIND_TEXT = {
  same: "same",
  ir: "IR differs",
  pass: "different pass",
  length: "missing in one log",
};

const model = { a: null, b: null, result: null, selected: -1 };

function verdictText() {
  const { a, b, result } = model;
  if (!a || !b) return "Choose both logs to compare them.";

  const passes = `${a.events.length} vs ${b.events.length} passes`;
  const end = result.finalSame ? "same final IR" : "different final IR";

  if (result.firstDivergence < 0) return `No divergence · ${passes} · ${end}.`;

  return (
    `Diverges at pass ${result.firstDivergence + 1} · ` +
    `${result.differing} of ${result.rows.length} passes differ · ${end}.`
  );
}

function renderRows() {
  const items = (model.result?.rows ?? []).map((row) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "link-btn";
    button.textContent = `${row.index + 1}. ${row.labelA || "—"} | ${row.labelB || "—"} · ${KIND_TEXT[row.kind]}`;
    button.dataset.kind = row.kind;
    button.setAttribute("aria-pressed", String(row.index === model.selected));
    button.addEventListener("click", () => select(row.index));
    item.append(button);

    return item;
  });

  rowsEl.replaceChildren(...items);
}

function renderDiff() {
  if (model.selected < 0) {
    diffEl.replaceChildren();

    return;
  }

  const marks = { same: " ", add: "+", del: "-" };

  diffEl.replaceChildren(
    ...rowDiff(model.a, model.b, model.selected).map(({ type, text }) => {
      const line = document.createElement("div");
      line.className = `compare-line ${type}`;
      line.textContent = `${marks[type]} ${text}`;

      return line;
    }),
  );
}

function select(index) {
  model.selected = index;
  renderRows();
  renderDiff();
}

export function renderComparePanel() {
  verdictEl.textContent = verdictText();
  renderRows();
  renderDiff();
}

function update() {
  const { a, b } = model;
  model.result = a && b ? compareTraces(a, b) : null;
  model.selected = model.result?.firstDivergence ?? -1;
  renderComparePanel();
}

async function load(input, key) {
  const file = input.files[0];
  model[key] = file ? parsePassTrace(await file.text()) : null;
  update();
}

export function setupComparePanel() {
  inputA.addEventListener("change", () => load(inputA, "a"));
  inputB.addEventListener("change", () => load(inputB, "b"));
}
