export const ABI_MAGIC = 0x4d4c5231;
export const ABI_VERSION = 3;

export const HDR = Object.freeze({
  MAGIC: 0,
  VERSION: 1,
  STATUS: 2,
  NODE_COUNT: 3,
  EDGE_COUNT: 4,
  DIAG_COUNT: 5,
  PTR_NODE_XYWH: 6,
  PTR_NODE_META: 7,
  PTR_EDGES: 8,
  PTR_STRINGS: 9,
  STRINGS_LEN: 10,
  PTR_DIAG: 11,
  PTR_BOUNDS: 12,
  LEN: 16,
});

export const STRIDE = Object.freeze({
  NODE_XYWH: 4,
  NODE_META: 4,
  EDGE: 2,
  DIAG: 4,
});

export const STATUS = Object.freeze({
  OK: 0,
  EMPTY: 1,
  BAD_UTF8: 2,
  OOM: 3,
  TOO_LARGE: 4,
});

export const STATUS_TEXT = Object.freeze({
  0: 'ok',
  1: 'empty input',
  2: 'input was not valid UTF-8',
  3: 'arena exhausted — module too large',
  4: 'input exceeds the 4 MiB buffer',
});

export const KIND = Object.freeze({
  MODULE: 0,
  FUNC: 1,
  OP: 2,
  BLOCK: 3,
  TERMINATOR: 4,
});

export const KIND_STYLE = Object.freeze({
  0: { fill: '#1e293b', stroke: '#475569', text: '#e2e8f0' },
  1: { fill: '#1e3a5f', stroke: '#3b82f6', text: '#dbeafe' },
  2: { fill: '#27303f', stroke: '#64748b', text: '#e2e8f0' },
  3: { fill: '#3f2d1e', stroke: '#d97706', text: '#fed7aa' },
  4: { fill: '#3f1e2b', stroke: '#e11d48', text: '#fecdd3' },
});

export const DIAG_CODE = Object.freeze({
  1: 'undefined SSA value',
  2: 'unbalanced brace',
});

export const NONE = 0xffffffff;