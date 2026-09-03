import { MlirEngine } from './wasm/bridge.js';
import { CanvasRenderer } from './render/canvas-renderer.js';
import { STATUS } from './wasm/abi.js';
import { copySnapshot, diffSnapshots } from './diff.js';

const SAMPLE = `module {
  func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>) -> tensor<128x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128x64xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x64xf32>) -> tensor<128x64xf32>
    %out = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                         outs(%filled : tensor<128x64xf32>) -> tensor<128x64xf32>
    func.return %out : tensor<128x64xf32>
  }
}`;

const canvas = document.getElementById('canvas');
const input = document.getElementById('editor');
const baseline = document.getElementById('baseline');
const statusEl = document.getElementById('status');
const detailEl = document.getElementById('detail');
const diffSummary = document.getElementById('diff-summary');
const diffList = document.getElementById('diff-list');

const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    detailEl.textContent =
      index < 0 ? '—' : `#${index} · ${snap.labelOf(index)} · parent ${snap.parentOf(index)}`;
  },
});

const engine = await MlirEngine.load('/mlir_core.wasm');

function parse(text) {
  const t0 = performance.now();
  const status = engine.parse(text);
  return { status, ms: performance.now() - t0, snapshot: status === STATUS.OK ? engine.snapshot() : null };
}

function renderDiff(rows) {
  const counts = rows.reduce((result, row) => {
    result[row.type] += 1;
    return result;
  }, { added: 0, removed: 0, changed: 0 });
  const total = counts.added + counts.removed + counts.changed;
  diffSummary.textContent = total
    ? `${total} change(s): +${counts.added}  -${counts.removed}  ~${counts.changed}`
    : 'No structural changes';
  diffList.replaceChildren(...rows.map((row) => {
    const item = document.createElement('li');
    item.className = row.type;
    if (row.type === 'changed') item.textContent = `~ ${row.before.label} -> ${row.after.label}`;
    if (row.type === 'added') item.textContent = `+ ${row.after.label}`;
    if (row.type === 'removed') item.textContent = `- ${row.before.label}`;
    return item;
  }));
}

function run() {
  const before = parse(baseline.value);
  if (before.status !== STATUS.OK) {
    statusEl.textContent = `baseline error: ${engine.statusText}`;
    return;
  }
  const beforeCopy = copySnapshot(before.snapshot);
  const after = parse(input.value);
  if (after.status !== STATUS.OK) {
    statusEl.textContent = `current error: ${engine.statusText}`;
    renderer.setSnapshot(null);
    return;
  }

  const snap = after.snapshot;
  renderer.setSnapshot(snap);
  const diags = snap.diagnostics();
  const rows = diffSnapshots(beforeCopy, snap);
  renderDiff(rows);
  statusEl.textContent =
    `${snap.nodeCount} nodes · ${snap.edgeCount} edges · ${after.ms.toFixed(2)} ms` +
    (diags.length ? ` · ${diags.length} warning(s): ${diags[0].message} "${diags[0].symbol}"` : '');
}

let timer = 0;
baseline.value = SAMPLE;
input.value = SAMPLE.replace('linalg.fill', 'linalg.fill_relu');
function scheduleRun() {
  clearTimeout(timer);
  timer = setTimeout(run, 140);
}
baseline.addEventListener('input', scheduleRun);
input.addEventListener('input', scheduleRun);
document.getElementById('reparse')?.addEventListener('click', run);

document.getElementById('fit')?.addEventListener('click', () => {
  renderer.fit();
  renderer.requestDraw();
});

run();