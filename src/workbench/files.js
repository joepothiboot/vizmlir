import {
  FileWatcher,
  canWatchFiles,
  diffRecords,
  diffToJSON,
  diffToMarkdown,
  download,
  kv,
  slug,
} from "../session/index.js";
import { isPassTrace } from "../trace/index.js";
import { copyChanges, parentLabel } from "./diff-panel.js";
import {
  diffCopy,
  diffTitle,
  input,
  setStatus,
  sourceName,
  statusEl,
} from "./dom.js";
import { renderer } from "./graph.js";
import { run } from "./pipeline.js";
import { state } from "./state.js";
import { clearTrace, loadTrace } from "./trace-nav.js";

function exportStem() {
  const name = sourceName.textContent
    .replace(/ ●$/, "")
    .replace(/\.[^.]+$/, "");

  return slug(state.trace ? `${name}-pass-${state.traceIndex + 1}` : name);
}

export async function exportPNG() {
  const blob = await renderer.exportPNG();
  if (blob) download(`${exportStem()}.png`, blob);
}

export function exportSVG() {
  const svg = renderer.exportSVG();
  if (svg) download(`${exportStem()}.svg`, svg, "image/svg+xml");
}

export function exportDiff(format) {
  const title = `${sourceName.textContent.replace(/ ●$/, "")} · ${diffTitle.textContent}`;

  const records = diffRecords(state.diffRows, (row, parent) =>
    parentLabel(row.after ? state.diffAfter : state.diffBefore, parent),
  );

  if (format === "md") {
    download(
      `${exportStem()}-diff.md`,
      diffToMarkdown(title, records),
      "text/markdown",
    );
  } else {
    download(
      `${exportStem()}-diff.json`,
      diffToJSON(title, records),
      "application/json",
    );
  }
}

const watchButton = document.getElementById("watch");
const watchLive = document.getElementById("watch-live");
const watchResume = document.getElementById("watch-resume");

export const watcher = new FileWatcher((change, error) => {
  if (!change) {
    updateWatchUi();

    setStatus(`stopped watching: ${error?.message ?? "file unavailable"}`, {
      error: true,
    });

    return;
  }

  sourceName.textContent = change.name;
  sourceName.title = `${change.name} · watching`;

  if (isPassTrace(change.text)) {
    state.measured = null;
    loadTrace(change.text, { keepIndex: !!state.trace });
  } else {
    clearTrace();
    input.value = change.text;
    input.dispatchEvent(new Event("input"));
    clearTimeout(state.timer);
    run();
  }

  statusEl.textContent = `reloaded ${new Date().toLocaleTimeString()} · ${statusEl.textContent}`;
});

export function updateWatchUi() {
  watchButton.hidden = !canWatchFiles || watcher.watching;
  watchLive.hidden = !watcher.watching;

  document
    .getElementById("menu-handle")
    .classList.toggle("watching", watcher.watching);

  if (watcher.watching) {
    watchResume.hidden = true;
    watchLive.title = `Watching ${watcher.handle.name} · click to stop`;
  }
}

export async function pickWatch() {
  const handle = await watcher.pick();
  if (handle) await kv.set("watch-handle", handle);
  updateWatchUi();
}

export function stopWatching() {
  watcher.stop();
  kv.delete("watch-handle");
  updateWatchUi();
}

export async function restoreWatch() {
  if (!canWatchFiles) return;

  const handle = await kv.get("watch-handle");
  if (!handle) return;

  if (!(await FileWatcher.needsPermission(handle))) {
    await watcher.start(handle);
    updateWatchUi();

    return;
  }

  watchResume.textContent = `Resume watching ${handle.name}`;
  watchResume.hidden = false;
  watchButton.hidden = true;

  watchResume.onclick = async () => {
    if (await FileWatcher.requestPermission(handle)) {
      await watcher.start(handle);
    } else {
      watchResume.hidden = true;
    }

    updateWatchUi();
  };
}

export function setupFiles() {
  document.getElementById("export-png").addEventListener("click", exportPNG);

  document.getElementById("export-svg").addEventListener("click", exportSVG);

  diffCopy.addEventListener("click", copyChanges);

  document
    .getElementById("export-md")
    .addEventListener("click", () => exportDiff("md"));

  document
    .getElementById("export-json")
    .addEventListener("click", () => exportDiff("json"));

  watchButton.addEventListener("click", pickWatch);

  watchLive.addEventListener("click", stopWatching);
}
