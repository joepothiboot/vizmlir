import { ABI_VERSION } from "./ir/index.js";
import { DEFAULT_SAMPLE_ID, SAMPLES, loadSampleState } from "./samples.js";
import { kv } from "./session/index.js";
import { renderPath, setupCanvasViews } from "./workbench/canvas-views.js";
import { setDebug, setupDebugger } from "./workbench/debugger.js";
import { input } from "./workbench/dom.js";
import { restoreWatch, setupFiles, updateWatchUi } from "./workbench/files.js";
import { inspector, setupLayout } from "./workbench/layout.js";
import { renderExplain, setupLineExplain } from "./workbench/line-explain.js";
import {
  applyScenario,
  fetchSampleText,
  openSamples,
  setupScenarios,
} from "./workbench/scenarios.js";
import { setupSourceTabs, showTab } from "./workbench/source-tabs.js";
import { state } from "./workbench/state.js";
import { applyState, setupWorkspace } from "./workbench/workspace.js";
import { setupPipeline } from "./workbench/pipeline.js";
import { setupSourceFiles } from "./workbench/source-files.js";
import { setupCommands } from "./workbench/commands.js";
import { setupTimingPanel } from "./workbench/timing-panel.js";
import { setupOpCountPanel } from "./workbench/opcount-panel.js";
import { setupBuffersPanel } from "./workbench/buffers-panel.js";
import { setupSymbolHistory } from "./workbench/symbol-history.js";
import { setupOpHistory } from "./workbench/op-history.js";
import { setupSymbolView } from "./workbench/symbol-view.js";
import { STORAGE_KEYS } from "./constants.js";

setupPipeline();
setupSourceTabs();
setupDebugger();
setupSourceFiles();
setupLayout();
setupCommands();
setupScenarios();
setupWorkspace();
setupTimingPanel();
setupOpCountPanel();
setupBuffersPanel();
setupSymbolHistory();
setupCanvasViews();
setupLineExplain();
setupOpHistory();
setupSymbolView();
setupFiles();

document.getElementById("abi").textContent = `wasm abi v${ABI_VERSION}`;

showTab(input);

renderExplain(-1);

renderPath();

updateWatchUi();

const saved = await kv.get("autosave");
let firstVisit = false;

const hasContent = (workspace) =>
  !!(
    workspace?.trace ||
    workspace?.baseline?.trim() ||
    workspace?.current?.trim()
  );

if (hasContent(saved)) {
  applyState(saved);
} else {
  const first =
    SAMPLES.find((sample) => sample.id === DEFAULT_SAMPLE_ID) ?? SAMPLES[0];

  try {
    applyState(await loadSampleState(first, fetchSampleText));
    applyScenario(first);
    firstVisit = true;
  } catch {
    applyState(await loadSampleState(SAMPLES.find((sample) => sample.inline)));
  }
}

state.restored = true;

try {
  if (localStorage.getItem(STORAGE_KEYS.debug) === "1") {
    setDebug(true, { save: false });
  }
} catch {}

if (state.inspectorSaved.open) inspector.open();

restoreWatch();

if (firstVisit) {
  try {
    if (!localStorage.getItem(STORAGE_KEYS.welcomed)) {
      localStorage.setItem("1");
      openSamples({ welcome: true });
    }
  } catch {
    openSamples({ welcome: true });
  }
}
