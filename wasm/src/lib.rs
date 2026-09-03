pub mod abi;
pub mod arena;
pub mod intern;

#[path = "parser/lexer.rs"]
pub mod lexer;
#[path = "parser/ast.rs"]
pub mod ast;

use abi::*;
use arena::Arena;
use ast::Ast;
use intern::Interner;

struct Layout {
    off_xywh: usize,
    off_meta: usize,
    off_edges: usize,
    off_strings: usize,
    off_diag: usize,
    off_bounds: usize,
}

pub struct Engine {
    header: Box<[u32; HDR_LEN]>,
    input: Vec<u8>,
    arena: Arena,
    interner: Interner,
    ast: Ast,
    layout: Option<Layout>,
}

impl Engine {
    fn new() -> Self {
        let mut header = Box::new([0u32; HDR_LEN]);
        header[HDR_MAGIC] = ABI_MAGIC;
        header[HDR_VERSION] = ABI_VERSION;
        header[HDR_STATUS] = STATUS_EMPTY;
        Self {
            header,
            input: vec![0u8; INPUT_CAP],
            arena: Arena::with_words(ARENA_WORDS),
            interner: Interner::new(),
            ast: Ast::new(),
            layout: None,
        }
    }
}

static mut ENGINE: Option<Engine> = None;

fn engine() -> &'static mut Engine {
    unsafe {
        let slot = &mut *core::ptr::addr_of_mut!(ENGINE);
        if slot.is_none() {
            *slot = Some(Engine::new());
        }
        slot.as_mut().unwrap()
    }
}

#[no_mangle]
pub extern "C" fn mlir_abi_version() -> u32 { ABI_VERSION }

#[no_mangle]
pub extern "C" fn mlir_header_ptr() -> u32 {
    engine().header.as_ptr() as u32
}

#[no_mangle]
pub extern "C" fn mlir_input_ptr() -> u32 {
    engine().input.as_ptr() as u32
}

#[no_mangle]
pub extern "C" fn mlir_input_cap() -> u32 { INPUT_CAP as u32 }

#[no_mangle]
pub extern "C" fn mlir_reset() {
    let e = engine();
    e.arena.reset();
    e.interner.clear();
    e.ast.clear();
    e.layout = None;
    e.header[HDR_STATUS] = STATUS_EMPTY;
    e.header[HDR_NODE_COUNT] = 0;
    e.header[HDR_EDGE_COUNT] = 0;
    e.header[HDR_DIAG_COUNT] = 0;
}

#[no_mangle]
pub extern "C" fn mlir_parse(len: u32) -> u32 {
    let e = engine();
    let len = len as usize;

    if len > INPUT_CAP {
        e.header[HDR_STATUS] = STATUS_TOO_LARGE;
        return STATUS_TOO_LARGE;
    }
    if len == 0 {
        mlir_reset();
        return STATUS_EMPTY;
    }

    let src = match std::str::from_utf8(&e.input[..len]) {
        Ok(s) => s,
        Err(_) => {
            e.header[HDR_STATUS] = STATUS_BAD_UTF8;
            return STATUS_BAD_UTF8;
        }
    };

    let src: &str = unsafe { &*(src as *const str) };
    ast::parse(src, &mut e.interner, &mut e.ast);

    e.arena.reset();
    let n = e.ast.nodes.len();
    let m = e.ast.edges.len();
    let d = e.ast.diags.len();
    let pool_len = e.interner.pool().len();

    let alloc = |a: &mut Arena, bytes: usize, align: usize| a.alloc(bytes, align);
    let off_xywh = match alloc(&mut e.arena, n * STRIDE_NODE_XYWH * 4, 4) { Some(o) => o, None => return oom(e) };
    let off_meta = match alloc(&mut e.arena, n * STRIDE_NODE_META * 4, 4) { Some(o) => o, None => return oom(e) };
    let off_edges = match alloc(&mut e.arena, m * STRIDE_EDGE * 4, 4) { Some(o) => o, None => return oom(e) };
    let off_diag = match alloc(&mut e.arena, d * STRIDE_DIAG * 4, 4) { Some(o) => o, None => return oom(e) };
    let off_bounds = match alloc(&mut e.arena, 4 * 4, 4) { Some(o) => o, None => return oom(e) };
    let off_strings = match alloc(&mut e.arena, pool_len.max(1), 8) { Some(o) => o, None => return oom(e) };

    e.arena.bytes(off_strings, pool_len).copy_from_slice(e.interner.pool());

    {
        let spans: Vec<(u32, u32)> = e.ast.nodes.iter().map(|nd| e.interner.span(nd.label)).collect();
        let kinds: Vec<(u32, u32)> = e.ast.nodes.iter().map(|nd| (nd.kind, nd.parent)).collect();
        let meta = e.arena.u32s(off_meta, n * STRIDE_NODE_META);
        for i in 0..n {
            let b = i * STRIDE_NODE_META;
            meta[b] = kinds[i].0;
            meta[b + 1] = spans[i].0;
            meta[b + 2] = spans[i].1;
            meta[b + 3] = kinds[i].1;
        }
    }

    {
        let pairs: Vec<(u32, u32)> = e.ast.edges.clone();
        let ed = e.arena.u32s(off_edges, m * STRIDE_EDGE);
        for (i, (s, t)) in pairs.iter().enumerate() {
            ed[i * STRIDE_EDGE] = *s;
            ed[i * STRIDE_EDGE + 1] = *t;
        }
    }

    {
        let items: Vec<(u32, u32, u32, u32)> = e
            .ast
            .diags
            .iter()
            .map(|dg| {
                let (o, l) = e.interner.span(dg.sym);
                (dg.code, dg.line, o, l)
            })
            .collect();
        let dv = e.arena.u32s(off_diag, d * STRIDE_DIAG);
        for (i, it) in items.iter().enumerate() {
            let b = i * STRIDE_DIAG;
            dv[b] = it.0;
            dv[b + 1] = it.1;
            dv[b + 2] = it.2;
            dv[b + 3] = it.3;
        }
    }

    e.layout = Some(Layout { off_xywh, off_meta, off_edges, off_strings, off_diag, off_bounds });

    let h = &mut e.header;
    h[HDR_STATUS] = STATUS_OK;
    h[HDR_NODE_COUNT] = n as u32;
    h[HDR_EDGE_COUNT] = m as u32;
    h[HDR_DIAG_COUNT] = d as u32;
    h[HDR_PTR_NODE_XYWH] = e.arena.addr(off_xywh);
    h[HDR_PTR_NODE_META] = e.arena.addr(off_meta);
    h[HDR_PTR_EDGES] = e.arena.addr(off_edges);
    h[HDR_PTR_STRINGS] = e.arena.addr(off_strings);
    h[HDR_STRINGS_LEN] = pool_len as u32;
    h[HDR_PTR_DIAG] = e.arena.addr(off_diag);
    h[HDR_PTR_BOUNDS] = e.arena.addr(off_bounds);

    mlir_layout(180.0, 30.0, 70.0, 14.0);
    STATUS_OK
}

fn oom(e: &mut Engine) -> u32 {
    e.header[HDR_STATUS] = STATUS_OOM;
    STATUS_OOM
}

#[no_mangle]
pub extern "C" fn mlir_layout(node_w: f32, node_h: f32, col_gap: f32, row_gap: f32) -> u32 {
    let e = engine();
    let lay = match &e.layout {
        Some(l) => Layout {
            off_xywh: l.off_xywh,
            off_meta: l.off_meta,
            off_edges: l.off_edges,
            off_strings: l.off_strings,
            off_diag: l.off_diag,
            off_bounds: l.off_bounds,
        },
        None => return STATUS_EMPTY,
    };

    let n = e.ast.nodes.len();
    if n == 0 {
        return STATUS_EMPTY;
    }

    let mut rows: Vec<u32> = vec![0; 64];
    let mut boxes: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(n);
    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);

    for nd in &e.ast.nodes {
        let col = nd.depth as usize;
        if col >= rows.len() {
            rows.resize(col + 1, 0);
        }
        let row = rows[col];
        rows[col] += 1;

        let (_, label_len) = e.interner.span(nd.label);
        let w = node_w.max(label_len as f32 * 7.2 + 26.0);
        let x = col as f32 * (node_w + col_gap);
        let y = row as f32 * (node_h + row_gap);

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x + w);
        max_y = max_y.max(y + node_h);
        boxes.push((x, y, w, node_h));
    }

    {
        let xywh = e.arena.f32s(lay.off_xywh, n * STRIDE_NODE_XYWH);
        for (i, b) in boxes.iter().enumerate() {
            let o = i * STRIDE_NODE_XYWH;
            xywh[o] = b.0;
            xywh[o + 1] = b.1;
            xywh[o + 2] = b.2;
            xywh[o + 3] = b.3;
        }
    }
    {
        let bb = e.arena.f32s(lay.off_bounds, 4);
        bb[0] = min_x;
        bb[1] = min_y;
        bb[2] = max_x;
        bb[3] = max_y;
    }

    e.layout = Some(lay);
    STATUS_OK
}