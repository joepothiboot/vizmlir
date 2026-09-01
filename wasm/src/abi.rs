//! Single source of truth for the JS <-> WASM memory contract.
//! ANY change here MUST be mirrored in `src/wasm/abi.js`.

pub const ABI_MAGIC: u32 = 0x4D4C_5231; // "MLR1"
pub const ABI_VERSION: u32 = 3;

// ---- header slot indices (u32 words) ----
pub const HDR_MAGIC: usize = 0;
pub const HDR_VERSION: usize = 1;
pub const HDR_STATUS: usize = 2;
pub const HDR_NODE_COUNT: usize = 3;
pub const HDR_EDGE_COUNT: usize = 4;
pub const HDR_DIAG_COUNT: usize = 5;
pub const HDR_PTR_NODE_XYWH: usize = 6;
pub const HDR_PTR_NODE_META: usize = 7;
pub const HDR_PTR_EDGES: usize = 8;
pub const HDR_PTR_STRINGS: usize = 9;
pub const HDR_STRINGS_LEN: usize = 10;
pub const HDR_PTR_DIAG: usize = 11;
pub const HDR_PTR_BOUNDS: usize = 12;
pub const HDR_LEN: usize = 16;

// ---- strides (elements per record) ----
pub const STRIDE_NODE_XYWH: usize = 4; // f32: x, y, w, h
pub const STRIDE_NODE_META: usize = 4; // u32: kind, label_off, label_len, parent
pub const STRIDE_EDGE: usize = 2; // u32: src_node, dst_node
pub const STRIDE_DIAG: usize = 4; // u32: code, line, sym_off, sym_len

// ---- status codes ----
pub const STATUS_OK: u32 = 0;
pub const STATUS_EMPTY: u32 = 1;
pub const STATUS_BAD_UTF8: u32 = 2;
pub const STATUS_OOM: u32 = 3;
pub const STATUS_TOO_LARGE: u32 = 4;

// ---- node kinds ----
pub const KIND_MODULE: u32 = 0;
pub const KIND_FUNC: u32 = 1;
pub const KIND_OP: u32 = 2;
pub const KIND_BLOCK: u32 = 3;
pub const KIND_TERMINATOR: u32 = 4;

// ---- diagnostic codes ----
pub const DIAG_UNDEF_SSA: u32 = 1;
pub const DIAG_UNBALANCED: u32 = 2;

pub const NONE: u32 = u32::MAX;

// ---- capacities ----
pub const INPUT_CAP: usize = 4 * 1024 * 1024; // 4 MiB of source text
pub const ARENA_WORDS: usize = 2 * 1024 * 1024; // 16 MiB (u64 words)