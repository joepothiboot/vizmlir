import { MlirEngine } from './wasm/bridge.js';
import { CanvasRenderer } from './render/canvas-renderer.js';
import { STATUS } from './wasm/abi.js';

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

const canvas = document.getElementById('viewport');
const input = document.getElementById('source');
const statusEl = document.getElementById('status');
const detailEl = document.getElementById('detail');

const renderer = new CanvasRenderer(canvas, {
  onSelect(index, snap) {
    detailEl.textContent =
      index < 0 ? '—' : `#${index} · ${snap.labelOf(index)} · parent ${snap.parentOf(index)}`;
  },
});

const engine = await MlirEngine.load('/mlir_core.wasm');

function run(text) {
  const t0 = performance.now();
  const status = engine.parse(text);
  const ms = performance.now() - t0;

  if (status !== STATUS.OK) {
    statusEl.textContent = `error: ${engine.statusText}`;
    renderer.setSnapshot(null);
    return;
  }

  const snap = engine.snapshot();
  renderer.setSnapshot(snap);

  const diags = snap.diagnostics();
  statusEl.textContent =
    `${snap.nodeCount} nodes · ${snap.edgeCount} edges · ${ms.toFixed(2)} ms` +
    (diags.length ? ` · ${diags.length} warning(s): ${diags[0].message} "${diags[0].symbol}"` : '');
}

let timer = 0;
input.value = SAMPLE;
input.addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(() => run(input.value), 140);
});

document.getElementById('fit')?.addEventListener('click', () => {
  renderer.fit();
  renderer.requestDraw();
});

run(SAMPLE);