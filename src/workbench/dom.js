import { state } from "./state.js";

export const canvas = document.getElementById("canvas");
export const input = document.getElementById("editor");
export const baseline = document.getElementById("baseline");
export const statusEl = document.getElementById("status-text");
const modeEl = document.getElementById("mode");
export const diffCopy = document.getElementById("diff-copy");
export const fileInput = document.getElementById("file");
export const baselineTitle = document.getElementById("baseline-title");
export const currentTitle = document.getElementById("current-title");
export const sourcePane = document.getElementById("source-pane");
export const splitToggle = document.getElementById("split-toggle");
export const sourceTitle = document.getElementById("source-title");
export const sourcePanel = document.getElementById("source-panel");
export const sourceName = document.getElementById("source-name");
export const diffTitle = document.getElementById("diff-title");
export const descentEl = document.getElementById("descent-view");
export const localViewEl = document.getElementById("local-view");
export const leBody = document.getElementById("le-body");

export function setStatus(text, { error = false } = {}) {
  statusEl.textContent = text;
  modeEl.textContent = state.trace ? "TRACE" : "IR";
  if (error) modeEl.textContent = "ERROR";
  modeEl.classList.toggle("bad", error);
}
