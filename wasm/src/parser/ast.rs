use crate::abi::*;
use crate::intern::{Interner, SymId};
use crate::lexer::{Lexer, Tok, Token};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Node {
    pub label: SymId,
    pub kind: u32,
    pub parent: u32,
    pub depth: u32,
    pub line: u32,
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
}

impl Ast {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new(), diags: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.diags.clear();
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

pub fn parse(src: &str, interner: &mut Interner, ast: &mut Ast) {
    ast.clear();
    interner.clear();

    let bytes = src.as_bytes();
    let toks = Lexer::new(bytes).tokenize();
    let text = |t: &Token| -> &str { &src[t.start as usize..t.end as usize] };

    let root_label = interner.intern("module");
    ast.nodes.push(Node { label: root_label, kind: KIND_MODULE, parent: NONE, depth: 0, line: 0 });

    let mut stack: Vec<u32> = vec![0];
    let mut defs: HashMap<SymId, u32> = HashMap::with_capacity(1024);
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
                if stack.len() > 1 {
                    stack.pop();
                } else {
                    ast.diags.push(Diag { code: DIAG_UNBALANCED, line: toks[i].line, sym: root_label });
                }
                i += 1;
                
                // `} loc(...)` and the trailing types of a region op (`} -> tensor<..>`,
                // `} : (..) -> ..`) belong to the op that owns the region.
                if toks
                    .get(i)
                    .is_some_and(|t| matches!(t.kind, Tok::Arrow | Tok::Colon) || (t.kind == Tok::Ident && text(t) == "loc"))
                {
                    while i < toks.len() && !matches!(toks[i].kind, Tok::Newline | Tok::Eof) {
                        i += 1;
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
        });

        // `%i = ..` inside an op (`scf.for %i = ..`, `iter_args(%acc = ..)`)
        // defines a region argument rather than using a value.
        let binds = |k: usize| rhs[k].kind == Tok::Ssa && rhs.get(k + 1).is_some_and(|t| t.kind == Tok::Equal);
        let operand_start = if kind == KIND_BLOCK { 0 } else { 1 };
        for (k, t) in rhs.iter().enumerate().skip(operand_start) {
            if t.kind != Tok::Ssa || binds(k) {
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

        let bound = (0..rhs.len()).filter(|&k| binds(k)).map(|k| &rhs[k]);
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
}