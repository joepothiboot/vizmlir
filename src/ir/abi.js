export const ABI_MAGIC = 0x4d4c5231;
export const ABI_VERSION = 4;

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
  PTR_NODE_LOC: 13,
  LEN: 16,
});

export const STRIDE = Object.freeze({
  NODE_XYWH: 4,
  NODE_META: 4,
  EDGE: 2,
  DIAG: 4,
  NODE_LOC: 8,
});

export const STATUS = Object.freeze({
  OK: 0,
  EMPTY: 1,
  BAD_UTF8: 2,
  OOM: 3,
  TOO_LARGE: 4,
});

export const STATUS_TEXT = Object.freeze({
  0: "ok",
  1: "empty input",
  2: "input was not valid UTF-8",
  3: "arena exhausted — module too large",
  4: "input exceeds the 4 MiB buffer",
});

export const KIND = Object.freeze({
  MODULE: 0,
  FUNC: 1,
  OP: 2,
  BLOCK: 3,
  TERMINATOR: 4,
});

export const KIND_STYLE = Object.freeze({
  0: { fill: "#1c1c1c", stroke: "#3a3a3a", text: "#e8e8e8" },
  1: { fill: "#262626", stroke: "#888888", text: "#f5f5f5" },
  2: { fill: "#1f1f1f", stroke: "#555555", text: "#e8e8e8" },
  3: { fill: "#24200f", stroke: "#f2c94c", text: "#f5e6ad" },
  4: { fill: "#2a1614", stroke: "#ff6b5e", text: "#ffd3cc" },
});

export const DIAG_CODE = Object.freeze({
  1: "undefined SSA value",
  2: "unbalanced brace",
});

export const LOC_FLAG = Object.freeze({
  HAS: 1,
  CALLSITE: 2,
  FUSED: 4,
  UNKNOWN: 8,
});

export const NONE = 0xffffffff;
