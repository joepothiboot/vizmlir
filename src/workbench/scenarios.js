import { STORAGE_KEYS } from "../constants.js";
import { SAMPLES, groupSamples, loadSampleState, scopeOf } from "../samples.js";
import { sessionFromFile } from "../session/index.js";
import { isPassTrace } from "../trace/index.js";
import { VIEWS, setCanvasView } from "./canvas-views.js";
import { setDebug } from "./debugger.js";
import { fileInput, input, setStatus, sourceName, statusEl } from "./dom.js";
import { run } from "./pipeline.js";
import { state } from "./state.js";
import { openSymbols } from "./symbol-history.js";
import { clearTrace, loadTrace } from "./trace-nav.js";
import { applyState, sessionsDialog } from "./workspace.js";

const samplesDialog = document.getElementById("samples");
const sampleList = document.getElementById("sample-list");

export async function fetchSampleText(path) {
  const response = await fetch(`${import.meta.env.BASE_URL}samples/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);

  return response.text();
}

export async function openSample(sample) {
  samplesDialog.close();

  try {
    applyState(await loadSampleState(sample, fetchSampleText));
    applyScenario(sample);
    statusEl.textContent = `loaded ${sample.title} · ${statusEl.textContent}`;
    if (sample.benchmarks) openSymbols();
  } catch (error) {
    setStatus(`could not load sample: ${error.message}`, { error: true });
  }
}

const scenarioButton = document.getElementById("samples-open");
const sampleButtons = new Map();
const scopeHeads = new Map();
let currentSample = null;
let scenario = null;

function setScenario(sample) {
  currentSample = sample;
  scenario = sample ? scopeOf(sample) : null;
  scenarioButton.textContent = `Scenario: ${scenario ? scenario.short : "your IR"} ▾`;

  scenarioButton.title = scenario
    ? `${scenario.blurb} Click to choose another scenario.`
    : "Choose a scenario: a sample and the views that suit it";

  for (const [id, button] of sampleButtons) {
    if (id === sample?.id) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }

  for (const [id, head] of scopeHeads) {
    head.classList.toggle("current", id === scenario?.id);
  }

  try {
    if (sample) localStorage.setItem(STORAGE_KEYS.sample, sample.id);
    else localStorage.removeItem(STORAGE_KEYS.sample);
  } catch {}
}

export function applyScenario(sample) {
  const scope = scopeOf(sample);
  setScenario(sample);
  setDebug(scope.debug);
  state.localWanted = scope.view === "local";

  setCanvasView(VIEWS[scope.view]?.available() ? scope.view : "graph", {
    remember: false,
  });
}

function sampleKind(sample) {
  if (sample.benchmarks) return "trace + mock benchmarks";
  if (sample.trace && sample.handwritten) return "pass trace (hand-written)";
  if (sample.trace) return "pass trace";

  return "before / after";
}

function sampleItem(sample) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = sample.title;

  const kind = document.createElement("span");
  kind.className = "kind";

  kind.textContent = sampleKind(sample);

  const blurb = document.createElement("span");
  blurb.className = "blurb";
  blurb.textContent = sample.blurb;
  button.append(title, kind, blurb);
  button.addEventListener("click", () => openSample(sample));
  sampleButtons.set(sample.id, button);
  item.append(button);

  return item;
}

export function openSamples() {
  samplesDialog.showModal();

  const current = sampleButtons.get(currentSample?.id);
  current?.focus();
  current?.scrollIntoView?.({ block: "nearest" });
}

export function setupScenarios() {
  fileInput.addEventListener("change", async () => {
    const [file] = fileInput.files;
    fileInput.value = "";
    if (!file) return;

    const text = await file.text();
    window.location.hash = "#/";

    const session = file.name.endsWith(".json") ? sessionFromFile(text) : null;

    if (session) {
      sessionsDialog.close();
      applyState(session.state);

      return;
    }

    setScenario(null);
    sourceName.textContent = file.name;
    sourceName.title = file.name;

    if (isPassTrace(text)) {
      state.benchmarks = { baseline: null, current: null };
      state.measured = null;
      loadTrace(text);

      return;
    }

    clearTrace();
    input.value = text;
    input.dispatchEvent(new Event("input"));
    run();
  });

  try {
    currentSample =
      SAMPLES.find((s) => s.id === localStorage.getItem(STORAGE_KEYS.sample)) ??
      null;
  } catch {}

  sampleList.replaceChildren(
    ...groupSamples().flatMap(({ scope, samples }) => {
      const head = document.createElement("li");
      head.className = "scope-head";

      const name = document.createElement("strong");
      name.textContent = scope.title;

      const note = document.createElement("span");
      note.textContent = scope.blurb;
      head.append(name, note);
      scopeHeads.set(scope.id, head);

      return [head, ...samples.map(sampleItem)];
    }),
  );

  setScenario(currentSample);

  samplesDialog.addEventListener("click", (e) => {
    if (e.target !== samplesDialog) return;

    const box = samplesDialog.getBoundingClientRect();

    if (
      e.clientX < box.left ||
      e.clientX > box.right ||
      e.clientY < box.top ||
      e.clientY > box.bottom
    ) {
      samplesDialog.close();
    }
  });

  document
    .getElementById("samples-open")
    .addEventListener("click", () => openSamples());
}
