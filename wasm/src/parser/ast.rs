use crate::abi::*;
use crate::intern::{Interner, SymId};
use crate::lexer::{Lexer, Tok, Token};
use crate::loc::{self, LocInfo};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Node {
    pub label: SymId,
    pub kind: u32,
    pub parent: u32,
    pub depth: u32,
    pub line: u32,
    // Byte span of the text inside the op's `loc(...)`; empty when it has none.
    pub loc: (u32, u32),
}

#[derive(Clone, Copy)]
pub struct Diag {
    pub code: u32,
    pub line: u32,
    pub sym: SymId,
}

pub struct Ast {
    pub nodes: Vec<Node>,
    pub edges: Vec<(u32, u32)>,
    pub diags: Vec<Diag>,
    // One per node: its resolved source location.
    pub locs: Vec<LocInfo>,
}

impl Ast {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), diags: Vec::new(), locs: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.diags.clear();
        self.locs.clear();
    }
}

impl Default for Ast {
    fn default() -> Self { Self::new() }
}

fn classify(name: &str) -> u32 {
    if name.ends_with(".func") || name == "func" || name.ends_with(".global") {
        KIND_FUNC
    } else if name.ends_with(".return")
        || name.ends_with(".br")
        || name.ends_with(".cond_br")
        || name.ends_with(".yield")
        || name.ends_with(".terminator")
    {
        KIND_TERMINATOR
    } else {
        KIND_OP
    }
}

// The span inside the last `loc( .. )` of a run of tokens.
fn loc_span(toks: &[Token], src: &str) -> Option<(u32, u32)> {
    let mut found = None;
    for (n, t) in toks.iter().enumerate() {
        if t.kind != Tok::Ident || &src[t.start as usize..t.end as usize] != "loc" {
            continue;
        }
        if toks.get(n + 1).map(|t| t.kind) != Some(Tok::LParen) {
            continue;
        }
        let mut depth = 0i32;
        for u in toks.iter().skip(n + 1) {
            match u.kind {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        found = Some((toks[n + 1].end, u.start));
                        break;
                    }
                }
                Tok::Eof => break,
                _ => {}
            }
        }
    }
    found
}

pub fn parse(src: &str, interner: &mut Interner, ast: &mut Ast) {
    ast.clear();
    interner.clear();

    let bytes = src.as_bytes();
    let toks = Lexer::new(bytes).tokenize();
    let text = |t: &Token| -> &str { &src[t.start as usize..t.end as usize] };

    let root_label = interner.intern("module");
    ast.nodes.push(Node { label: root_label, kind: KIND_MODULE, parent: NONE, depth: 0, line: 0, loc: (0, 0) });

    let mut stack: Vec<u32> = vec![0];
    let mut defs: HashMap<SymId, u32> = HashMap::with_capacity(1024);
    // `#loc3 = loc(...)`: alias name to the span inside its `loc(...)`.
    let mut alias_spans: Vec<(&str, (u32, u32))> = Vec::new();
    let mut i = 0usize;
    let mut label_buf = String::with_capacity(128);

    while i < toks.len() {
        match toks[i].kind {
            Tok::Eof => break,
            Tok::Newline => {
                i += 1;
                continue;
            }
            Tok::RBrace => {
                let closed = if stack.len() > 1 { stack.pop() } else { None };
                if closed.is_none() {
                    ast.diags.push(Diag { code: DIAG_UNBALANCED, line: toks[i].line, sym: root_label });
                }
                i += 1;
                
                // `} loc(...)` and the trailing types of a region op (`} -> tensor<..>`,
                // `} : (..) -> ..`) belong to the op that owns the region.
                if toks
                    .get(i)
                    .is_some_and(|t| matches!(t.kind, Tok::Arrow | Tok::Colon) || (t.kind == Tok::Ident && text(t) == "loc"))
                {
                    let from = i;
                    while i < toks.len() && !matches!(toks[i].kind, Tok::Newline | Tok::Eof) {
                        i += 1;
                    }
                    if let (Some(op), Some(span)) = (closed, loc_span(&toks[from..i], src)) {
                        ast.nodes[op as usize].loc = span;
                    }
                }
                continue;
            }
            _ => {}
        }

        let stmt_start = i;
        let mut depth = 0i32;
        let mut opens_region = false;
        while i < toks.len() {
            match toks[i].kind {
                Tok::Eof => break,
                Tok::LParen | Tok::LBrack | Tok::Lt => depth += 1,
                Tok::RParen | Tok::RBrack | Tok::Gt => depth -= 1,
                Tok::LBrace if depth == 0 => {
                    let nxt = toks.get(i + 1).map(|t| t.kind).unwrap_or(Tok::Eof);
                    if matches!(nxt, Tok::Newline | Tok::Eof) {
                        opens_region = true;
                        i += 1;
                        break;
                    }
                    depth += 1;
                }
                // A brace nested in a type or attribute (`#ttg.slice<{dim = 1}>`)
                // pairs with its `}` below.
                Tok::LBrace => depth += 1,
                Tok::RBrace if depth == 0 => break,
                Tok::RBrace => depth -= 1,
                Tok::Newline if depth <= 0 => {
                    i += 1;
                    break;
                }
                _ => {}
            }
            i += 1;
        }

        let raw_stmt: &[Token] = &toks[stmt_start..i.min(toks.len())];
        let stmt: Vec<Token> = raw_stmt
            .iter()
            .copied()
            .filter(|t| !matches!(t.kind, Tok::Newline | Tok::Eof | Tok::LBrace))
            .collect();
        if stmt.is_empty() {
            continue;
        }

        let mut eq_at: Option<usize> = None;
        let mut d = 0i32;
        for (n, t) in stmt.iter().enumerate() {
            match t.kind {
                Tok::LParen | Tok::LBrack | Tok::Lt => d += 1,
                Tok::RParen | Tok::RBrack | Tok::Gt => d -= 1,
                Tok::Equal if d == 0 => {
                    eq_at = Some(n);
                    break;
                }
                _ => {}
            }
        }

        // Only `%a, %b:2 =` names results (and `#alias =` defines one); an `=`
        // after anything else is inside the op, as in `scf.for %i = %c0 to ..`
        // or `op {key = value}`.
        let eq_at = eq_at.filter(|&n| {
            (n == 1 && matches!(stmt[0].kind, Tok::Attr | Tok::Type))
                || stmt[..n].iter().all(|t| matches!(t.kind, Tok::Ssa | Tok::Comma | Tok::Colon | Tok::Number))
        });
        let (lhs, rhs) = match eq_at {
            Some(n) => (&stmt[..n], &stmt[n + 1..]),
            None => (&stmt[..0], &stmt[..]),
        };
        if rhs.is_empty() {
            continue;
        }
        // `#loc = loc(...)`, `#map = affine_map<...>`, `!t = ...` alias definitions.
        if lhs.len() == 1 && matches!(lhs[0].kind, Tok::Attr | Tok::Type) {
            if lhs[0].kind == Tok::Attr && text(&rhs[0]) == "loc" {
                if let Some(span) = loc_span(rhs, src) {
                    alias_spans.push((text(&lhs[0]), span));
                }
            }
            continue;
        }

        let head = &rhs[0];
        let kind;
        label_buf.clear();

        if head.kind == Tok::Block {
            kind = KIND_BLOCK;
            label_buf.push_str(text(head));
        } else {
            let raw = text(head);
            let name = raw.trim_matches('"');
            kind = classify(name);
            label_buf.push_str(name);
            if let Some(sym) = rhs.iter().find(|t| t.kind == Tok::Symbol) {
                label_buf.push(' ');
                label_buf.push_str(text(sym));
            }
        }
        if label_buf.is_empty() {
            label_buf.push_str("<op>");
        }
        if kind == KIND_OP && !lhs.is_empty() {
            let n = lhs.iter().filter(|t| t.kind == Tok::Ssa).count();
            if n > 1 {
                label_buf.push_str(&format!(" ({}×)", n));
            }
        }

        let node_idx = ast.nodes.len() as u32;
        let parent = *stack.last().unwrap_or(&0);
        let label = interner.intern(&label_buf);
        ast.nodes.push(Node {
            label,
            kind,
            parent,
            depth: stack.len() as u32,
            line: head.line,
            loc: loc_span(rhs, src).unwrap_or((0, 0)),
        });

        // Values an op's header defines rather than uses: region arguments
        // bound with `=` (`scf.for %i = ..`, `iter_args(%acc = ..)`), function
        // arguments (`func.func @f(%arg0: T)`), the ids and buffers gpu.launch
        // names in `blocks(..)`, `threads(..)`, `clusters(..)`, `workgroup(..)`
        // and `private(..)`, and the induction variables of `scf.forall (..)`
        // and `scf.parallel (..)`.
        let head_name = text(head).trim_matches('"');
        let mut binds = vec![false; rhs.len()];
        let mut groups: Vec<bool> = Vec::new();
        for k in 0..rhs.len() {
            match rhs[k].kind {
                Tok::LParen => {
                    let named = k >= 1
                        && rhs[k - 1].kind == Tok::Ident
                        && matches!(text(&rhs[k - 1]), "blocks" | "threads" | "clusters" | "workgroup" | "private");
                    let induction = k == 1 && matches!(head_name, "scf.forall" | "scf.parallel");
                    groups.push(named || induction);
                }
                Tok::RParen => {
                    groups.pop();
                }
                Tok::Ssa => {
                    let next = rhs.get(k + 1).map(|t| t.kind);
                    binds[k] = next == Some(Tok::Equal)
                        || (kind == KIND_FUNC && next == Some(Tok::Colon))
                        || groups.last() == Some(&true);
                }
                _ => {}
            }
        }
        let operand_start = if kind == KIND_BLOCK { 0 } else { 1 };
        for (k, t) in rhs.iter().enumerate().skip(operand_start) {
            if t.kind != Tok::Ssa || binds[k] {
                continue;
            }
            let sym = interner.intern(text(t));
            match defs.get(&sym) {
                Some(&src_idx) => {
                    if src_idx != node_idx {
                        ast.edges.push((src_idx, node_idx));
                    }
                }
                None => {
                    if kind != KIND_BLOCK {
                        ast.diags.push(Diag { code: DIAG_UNDEF_SSA, line: t.line, sym });
                    }
                }
            }
        }

        let bound = (0..rhs.len()).filter(|&k| binds[k]).map(|k| &rhs[k]);
        for t in lhs.iter().filter(|t| t.kind == Tok::Ssa).chain(bound) {
            let sym = interner.intern(text(t));
            defs.insert(sym, node_idx);
        }
        if kind == KIND_BLOCK {
            for t in rhs.iter().filter(|t| t.kind == Tok::Ssa) {
                let sym = interner.intern(text(t));
                defs.insert(sym, node_idx);
            }
        }

        if parent != NONE && parent != node_idx {
            let has_pred = ast.edges.iter().rev().take(8).any(|&(_, dst)| dst == node_idx);
            if !has_pred {
                ast.edges.push((parent, node_idx));
            }
        }

        if opens_region {
            stack.push(node_idx);
        }
    }

    // Aliases are defined away from their uses, so locations resolve last.
    let aliases: HashMap<&str, &str> = alias_spans
        .iter()
        .map(|&(name, (a, b))| (name, &src[a as usize..b as usize]))
        .collect();
    for node in &ast.nodes {
        let (a, b) = node.loc;
        let resolved = if b > a { loc::resolve(&src[a as usize..b as usize], &aliases) } else { String::new() };
        let flags = loc::flags_of(&resolved);
        ast.locs.push(if flags == 0 {
            LocInfo::NONE
        } else {
            let (file, line, col) = match loc::first_file_line_col(&resolved) {
                Some((file, line, col)) => (interner.intern(file), line, col),
                None => (NONE, 0, 0),
            };
            LocInfo { flags, line, col, file, text: interner.intern(&resolved) }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undefined(src: &str) -> Vec<String> {
        let mut interner = Interner::new();
        let mut ast = Ast::new();
        parse(src, &mut interner, &mut ast);
        ast.diags
            .iter()
            .filter(|d| d.code == DIAG_UNDEF_SSA)
            .map(|d| interner.text(d.sym).to_string())
            .collect()
    }

    #[test]
    fn function_arguments_are_defined() {
        let src = "func.func @f(%arg0: memref<4xf32>, %i: index) {\n  %v = memref.load %arg0[%i] : memref<4xf32>\n  return\n}\n";
        assert!(undefined(src).is_empty());
    }

    #[test]
    fn gpu_launch_ids_and_buffers_are_defined() {
        let src = "func.func @f(%a: memref<64xf32>) {\n  %c1 = arith.constant 1 : index\n  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1) threads(%tx, %ty, %tz) in (%sx = %c1, %sy = %c1, %sz = %c1) workgroup(%t : memref<32xf32, #gpu.address_space<workgroup>>) {\n    %i = arith.addi %bx, %tx : index\n    %v = memref.load %a[%i] : memref<64xf32>\n    memref.store %v, %t[%tx] : memref<32xf32, #gpu.address_space<workgroup>>\n    gpu.terminator\n  }\n  return\n}\n";
        assert!(undefined(src).is_empty(), "{:?}", undefined(src));
    }

    #[test]
    fn forall_induction_variables_are_defined() {
        let src = "func.func @f(%a: memref<8xf32>) {\n  scf.forall (%i) in (8) {\n    %v = memref.load %a[%i] : memref<8xf32>\n  }\n  return\n}\n";
        assert!(undefined(src).is_empty(), "{:?}", undefined(src));
    }

    // Triton types nest braces in angle brackets: `#ttg.slice<{dim = 1, ..}>`.
    #[test]
    fn braces_inside_types_stay_balanced() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../public/samples/triton-coalesce.trace.txt");
        let trace = std::fs::read_to_string(path).unwrap();
        for dump in trace.split("// -----// IR Dump").skip(1) {
            let ir: String = dump.lines().skip(1).collect::<Vec<_>>().join("\n");
            let mut interner = Interner::new();
            let mut ast = Ast::new();
            parse(&ir, &mut interner, &mut ast);
            let lines: Vec<u32> = ast.diags.iter().map(|d| d.line).collect();
            assert!(lines.is_empty(), "diagnostics on lines {:?}", lines);
        }
    }

    #[test]
    fn a_value_used_before_any_definition_is_still_reported() {
        let src = "func.func @f(%a: index) {\n  %b = arith.addi %a, %missing : index\n  return\n}\n";
        assert_eq!(undefined(src), vec!["%missing"]);
    }

    // (resolved text, file, line, col, flags) per node.
    fn locs(src: &str) -> Vec<(String, String, u32, u32, u32)> {
        let mut interner = Interner::new();
        let mut ast = Ast::new();
        parse(src, &mut interner, &mut ast);
        assert_eq!(ast.locs.len(), ast.nodes.len());
        ast.locs
            .iter()
            .map(|l| {
                let text = |id| if id == NONE { String::new() } else { interner.text(id).to_string() };
                (text(l.text), text(l.file), l.line, l.col, l.flags)
            })
            .collect()
    }

    #[test]
    fn inline_locations_are_read_per_op() {
        let src = "func.func @f(%a: index) {\n  %b = arith.addi %a, %a : index loc(\"k.mojo\":4:9)\n  return loc(unknown)\n}\n";
        let l = locs(src);
        assert_eq!(l[0].0, "");
        assert_eq!(l[1].0, "");
        assert_eq!(l[2], ("\"k.mojo\":4:9".into(), "k.mojo".into(), 4, 9, LOC_FLAG_HAS));
        assert_eq!(l[3], ("unknown".into(), String::new(), 0, 0, LOC_FLAG_HAS | LOC_FLAG_UNKNOWN));
    }

    #[test]
    fn aliases_resolve_even_when_defined_after_use() {
        let src = "func.func @f(%a: index) {\n  %b = arith.addi %a, %a : index loc(#loc2)\n  return\n}\n#loc1 = loc(\"k.mojo\":7:3)\n#loc2 = loc(callsite(#loc1 at \"main.mojo\":1:1))\n";
        let l = locs(src);
        let (text, file, line, col, flags) = &l[2];
        assert_eq!(text, "callsite(\"k.mojo\":7:3 at \"main.mojo\":1:1)");
        assert_eq!((file.as_str(), *line, *col), ("k.mojo", 7, 3));
        assert_eq!(*flags, LOC_FLAG_HAS | LOC_FLAG_CALLSITE);
    }

    #[test]
    fn a_region_ops_location_follows_its_closing_brace() {
        let src = "func.func @f(%n: index) {\n  scf.for %i = %n to %n step %n {\n    scf.yield\n  } loc(\"k.mojo\":2:3)\n  return\n}\n";
        let l = locs(src);
        // module, func, scf.for, scf.yield, return
        assert_eq!(l[2].1, "k.mojo");
        assert_eq!((l[2].2, l[2].3), (2, 3));
        assert_eq!(l[3].0, "");
    }

    #[test]
    fn locations_do_not_add_edges_or_diagnostics() {
        let with = "func.func @f(%a: index) {\n  %b = arith.addi %a, %a : index loc(\"k\":1:1)\n  return\n}\n";
        let without = "func.func @f(%a: index) {\n  %b = arith.addi %a, %a : index\n  return\n}\n";
        let count = |src: &str| {
            let mut interner = Interner::new();
            let mut ast = Ast::new();
            parse(src, &mut interner, &mut ast);
            (ast.nodes.len(), ast.edges.len(), ast.diags.len())
        };
        assert_eq!(count(with), count(without));
    }

    #[test]
    fn ir_without_locations_has_empty_ones() {
        assert!(locs("func.func @f() {\n  return\n}\n").iter().all(|l| l.4 == 0));
    }
}
