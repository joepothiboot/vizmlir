// Source locations: the `loc(...)` MLIR prints after an op (or after a
// region's closing brace) when run with `-mlir-print-debuginfo`, and the
// `#loc3 = loc(...)` aliases it refers to.
//
// A location is kept as its resolved text (aliases substituted, so
// `loc(#loc3)` reads as `loc("kernel.mojo":12:5)`) plus the first
// `"file":line:col` found in it. For a call site, `callsite(callee at caller)`,
// that first one is the callee: where the op was written.

use crate::abi::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocInfo {
    pub flags: u32,
    pub line: u32,
    pub col: u32,
    pub file: u32,
    pub text: u32,
}

impl LocInfo {
    pub const NONE: LocInfo = LocInfo { flags: 0, line: 0, col: 0, file: NONE, text: NONE };
}

// Alias chains are short in practice; the cap only stops a cycle
// (`#a = loc(#b)`, `#b = loc(#a)`).
const MAX_DEPTH: u32 = 8;

fn is_alias_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.' | '-')
}

/// Replaces each `#name` that has an alias with the alias text, recursively.
pub fn resolve(raw: &str, aliases: &HashMap<&str, &str>) -> String {
    resolve_at(raw.trim(), aliases, 0)
}

fn resolve_at(raw: &str, aliases: &HashMap<&str, &str>, depth: u32) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    let mut in_string = false;
    while let Some(c) = rest.chars().next() {
        if c == '"' {
            in_string = !in_string;
        }
        if c == '#' && !in_string {
            let end = rest[1..].find(|ch: char| !is_alias_char(ch)).map_or(rest.len(), |n| n + 1);
            let name = &rest[..end];
            match aliases.get(name) {
                Some(text) if depth < MAX_DEPTH => out.push_str(&resolve_at(text.trim(), aliases, depth + 1)),
                _ => out.push_str(name),
            }
            rest = &rest[end..];
            continue;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// The first `"file":line:col` in a resolved location, as (file, line, col).
pub fn first_file_line_col(text: &str) -> Option<(&str, u32, u32)> {
    let mut from = 0;
    while let Some(open) = text[from..].find('"').map(|n| n + from) {
        let close = text[open + 1..].find('"').map(|n| n + open + 1)?;
        let tail = &text[close + 1..];
        if let Some(tail) = tail.strip_prefix(':') {
            let digits = |s: &str| s.bytes().take_while(u8::is_ascii_digit).count();
            let nl = digits(tail);
            if nl > 0 {
                if let Some(after) = tail[nl..].strip_prefix(':') {
                    let nc = digits(after);
                    if nc > 0 {
                        let line = tail[..nl].parse().ok()?;
                        let col = after[..nc].parse().ok()?;
                        return Some((&text[open + 1..close], line, col));
                    }
                }
            }
        }
        from = close + 1;
    }
    None
}

pub fn flags_of(text: &str) -> u32 {
    let t = text.trim();
    if t.is_empty() {
        return 0;
    }
    let kind = if t.starts_with("callsite") {
        LOC_FLAG_CALLSITE
    } else if t.starts_with("fused") {
        LOC_FLAG_FUSED
    } else if t.starts_with("unknown") {
        LOC_FLAG_UNKNOWN
    } else {
        0
    };
    LOC_FLAG_HAS | kind
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aliases<'a>(pairs: &[(&'a str, &'a str)]) -> HashMap<&'a str, &'a str> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn resolves_nested_aliases() {
        let a = aliases(&[("#loc1", "\"k.mojo\":3:7"), ("#loc2", "callsite(#loc1 at #loc3)"), ("#loc3", "\"main.mojo\":9:1")]);
        assert_eq!(resolve("#loc2", &a), "callsite(\"k.mojo\":3:7 at \"main.mojo\":9:1)");
    }

    #[test]
    fn leaves_unknown_aliases_and_hashes_in_strings() {
        let a = aliases(&[]);
        assert_eq!(resolve("#loc9", &a), "#loc9");
        assert_eq!(resolve("\"a#b.mojo\":1:2", &a), "\"a#b.mojo\":1:2");
    }

    #[test]
    fn alias_cycles_stop() {
        let a = aliases(&[("#a", "#b"), ("#b", "#a")]);
        assert!(resolve("#a", &a).starts_with('#'));
    }

    #[test]
    fn finds_the_first_file_line_col() {
        assert_eq!(first_file_line_col("\"k.mojo\":12:5"), Some(("k.mojo", 12, 5)));
        assert_eq!(
            first_file_line_col("callsite(\"k.mojo\":3:7 at \"main.mojo\":9:1)"),
            Some(("k.mojo", 3, 7))
        );
        assert_eq!(first_file_line_col("fused[\"a\", \"b\":4:2]"), Some(("b", 4, 2)));
        assert_eq!(first_file_line_col("unknown"), None);
        assert_eq!(first_file_line_col("\"just a name\""), None);
    }

    #[test]
    fn classifies_locations() {
        assert_eq!(flags_of(""), 0);
        assert_eq!(flags_of("unknown"), LOC_FLAG_HAS | LOC_FLAG_UNKNOWN);
        assert_eq!(flags_of("callsite(x at y)"), LOC_FLAG_HAS | LOC_FLAG_CALLSITE);
        assert_eq!(flags_of("fused[a, b]"), LOC_FLAG_HAS | LOC_FLAG_FUSED);
        assert_eq!(flags_of("\"f\":1:1"), LOC_FLAG_HAS);
    }
}
