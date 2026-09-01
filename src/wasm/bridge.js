// The JS/WASM safety boundary. Nothing above this file touches raw pointers.
import { MemoryViews } from './views.js';
import { ABI_MAGIC, ABI_VERSION, HDR, STRIDE, STATUS, STATUS_TEXT, DIAG_CODE, NONE } from './abi.js';

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8');

export class MlirEngine {
  #exports;
  #views;
  #headerPtr;
  #inputPtr;
  #inputCap;

  constructor(instance) {
    this.#exports = instance.exports;
    const { memory, mlir_header_ptr, mlir_input_ptr, mlir_input_cap, mlir_abi_version } = this.#exports;
    if (!memory) throw new Error('wasm module does not export `memory`');

    const version = mlir_abi_version();
    if (version !== ABI_VERSION) {
      throw new Error(`ABI mismatch: wasm=${version}, js=${ABI_VERSION}. Rebuild and sync abi.js.`);
    }

    this.#views = new MemoryViews(memory);
    this.#headerPtr = mlir_header_ptr() >>> 0;
    this.#inputPtr = mlir_input_ptr() >>> 0;
    this.#inputCap = mlir_input_cap() >>> 0;

    const magic = this.#header()[HDR.MAGIC];
    if (magic !== ABI_MAGIC) {
      throw new Error(`bad ABI magic 0x${magic.toString(16)} — memory layout is not what JS expects`);
    }
  }

  static async load(url = '/mlir_core.wasm') {
    let instance;
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      try {
        ({ instance } = await WebAssembly.instantiateStreaming(fetch(url), {}));
      } catch {
        instance = null;
      }
    }
    if (!instance) {
      const bytes = await (await fetch(url)).arrayBuffer();
      ({ instance } = await WebAssembly.instantiate(bytes, {}));
    }
    return new MlirEngine(instance);
  }

  #header() {
    return this.#views.u32At(this.#headerPtr, HDR.LEN);
  }

  get status() {
    return this.#header()[HDR.STATUS];
  }

  get statusText() {
    return STATUS_TEXT[this.status] ?? `unknown status ${this.status}`;
  }

  get inputCapacity() {
    return this.#inputCap;
  }

  reset() {
    this.#exports.mlir_reset();
  }

  /** Encodes `text`, hands it to Rust, and parses. Returns a numeric STATUS. */
  parse(text) {
    const bytes = encoder.encode(text);
    if (bytes.length > this.#inputCap) return STATUS.TOO_LARGE;
    this.#views.writeBytes(this.#inputPtr, bytes);
    return this.#exports.mlir_parse(bytes.length) >>> 0;
  }

  /** Re-runs layout without re-parsing. */
  layout({ nodeWidth = 180, nodeHeight = 30, colGap = 70, rowGap = 14 } = {}) {
    return this.#exports.mlir_layout(nodeWidth, nodeHeight, colGap, rowGap) >>> 0;
  }

  /**
   * Zero-copy read model. Views are only valid until the next parse()/layout()
   * call — never cache them across a mutation.
   */
  snapshot() {
    const h = this.#header();
    const nodeCount = h[HDR.NODE_COUNT];
    const edgeCount = h[HDR.EDGE_COUNT];
    const diagCount = h[HDR.DIAG_COUNT];
    const stringsLen = h[HDR.STRINGS_LEN];

    const strings = this.#views.u8At(h[HDR.PTR_STRINGS], stringsLen);
    const meta = this.#views.u32At(h[HDR.PTR_NODE_META], nodeCount * STRIDE.NODE_META);

    return {
      status: h[HDR.STATUS],
      nodeCount,
      edgeCount,
      diagCount,
      xywh: this.#views.f32At(h[HDR.PTR_NODE_XYWH], nodeCount * STRIDE.NODE_XYWH),
      meta,
      edges: this.#views.u32At(h[HDR.PTR_EDGES], edgeCount * STRIDE.EDGE),
      bounds: this.#views.f32At(h[HDR.PTR_BOUNDS], 4),
      kindOf: (i) => meta[i * STRIDE.NODE_META],
      parentOf: (i) => {
        const p = meta[i * STRIDE.NODE_META + 3];
        return p === NONE ? -1 : p;
      },
      labelOf: (i) => {
        const b = i * STRIDE.NODE_META;
        const off = meta[b + 1];
        const len = meta[b + 2];
        return len === 0 ? '' : decoder.decode(strings.subarray(off, off + len));
      },
      diagnostics: () => {
        const dv = this.#views.u32At(h[HDR.PTR_DIAG], diagCount * STRIDE.DIAG);
        const out = [];
        for (let i = 0; i < diagCount; i++) {
          const b = i * STRIDE.DIAG;
          const off = dv[b + 2];
          const len = dv[b + 3];
          out.push({
            code: dv[b],
            line: dv[b + 1],
            symbol: len ? decoder.decode(strings.subarray(off, off + len)) : '',
            message: DIAG_CODE[dv[b]] ?? 'unknown diagnostic',
          });
        }
        return out;
      },
    };
  }
}