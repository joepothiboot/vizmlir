#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tok {
    Eof, Newline, Ssa, Block, Symbol, Attr, Type, Ident, Number, Str,
    LParen, RParen, LBrace, RBrace, LBrack, RBrack, Lt, Gt,
    Comma, Colon, Equal, Arrow, Other,
}

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub kind: Tok,
    pub start: u32,
    pub end: u32,
    pub line: u32,
}

#[inline]
fn is_ident_body(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'$' | b'-')
}

pub struct Lexer<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0, line: 1 }
    }

    pub fn tokenize(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while self.pos < self.bytes.len() {
            let start = self.pos;
            let line = self.line;
            let byte = self.bytes[self.pos];
            let kind = match byte {
                b' ' | b'\t' | b'\r' => { self.pos += 1; continue; }
                b'\n' => { self.pos += 1; self.line += 1; Tok::Newline }
                b'%' => { self.scan_prefixed(); Tok::Ssa }
                b'^' => { self.scan_prefixed(); Tok::Block }
                b'@' => { self.scan_prefixed(); Tok::Symbol }
                b'#' => { self.scan_prefixed(); Tok::Attr }
                b'!' => { self.scan_prefixed(); Tok::Type }
                b'"' => { self.scan_string(); Tok::Str }
                b'(' => { self.pos += 1; Tok::LParen }
                b')' => { self.pos += 1; Tok::RParen }
                b'{' => { self.pos += 1; Tok::LBrace }
                b'}' => { self.pos += 1; Tok::RBrace }
                b'[' => { self.pos += 1; Tok::LBrack }
                b']' => { self.pos += 1; Tok::RBrack }
                b'<' => { self.pos += 1; Tok::Lt }
                b'>' => { self.pos += 1; Tok::Gt }
                b',' => { self.pos += 1; Tok::Comma }
                b':' => { self.pos += 1; Tok::Colon }
                b'=' => { self.pos += 1; Tok::Equal }
                b'-' if self.bytes.get(self.pos + 1) == Some(&b'>') => { self.pos += 2; Tok::Arrow }
                b'0'..=b'9' | b'.' => { self.scan_number(); Tok::Number }
                c if is_ident_body(c) => { self.scan_ident(); Tok::Ident }
                _ => { self.pos += 1; Tok::Other }
            };
            tokens.push(Token { kind, start: start as u32, end: self.pos as u32, line });
        }
        tokens.push(Token { kind: Tok::Eof, start: self.pos as u32, end: self.pos as u32, line: self.line });
        tokens
    }

    fn scan_prefixed(&mut self) {
        self.pos += 1;
        while self.pos < self.bytes.len() && is_ident_body(self.bytes[self.pos]) { self.pos += 1; }
    }

    fn scan_ident(&mut self) {
        while self.pos < self.bytes.len() && is_ident_body(self.bytes[self.pos]) { self.pos += 1; }
    }

    fn scan_number(&mut self) {
        while self.pos < self.bytes.len() && (self.bytes[self.pos].is_ascii_digit() || self.bytes[self.pos] == b'.') { self.pos += 1; }
    }

    fn scan_string(&mut self) {
        self.pos += 1;
        while self.pos < self.bytes.len() {
            let byte = self.bytes[self.pos];
            self.pos += 1;
            if byte == b'\\' { self.pos = self.pos.saturating_add(1); }
            else if byte == b'"' { break; }
            else if byte == b'\n' { self.line += 1; }
        }
    }
}
