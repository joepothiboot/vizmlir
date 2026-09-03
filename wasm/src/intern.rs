use std::collections::HashMap;

pub type SymId = u32;

pub struct Interner {
    map: HashMap<Box<str>, SymId>,
    spans: Vec<(u32, u32)>,
    pool: Vec<u8>,
}

impl Interner {
    pub fn new() -> Self {
        Self { map: HashMap::with_capacity(1024), spans: Vec::with_capacity(1024), pool: Vec::with_capacity(64 * 1024) }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.spans.clear();
        self.pool.clear();
    }

    pub fn intern(&mut self, s: &str) -> SymId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let off = self.pool.len() as u32;
        self.pool.extend_from_slice(s.as_bytes());
        let id = self.spans.len() as SymId;
        self.spans.push((off, s.len() as u32));
        self.map.insert(s.into(), id);
        id
    }

    #[inline]
    pub fn span(&self, id: SymId) -> (u32, u32) {
        self.spans.get(id as usize).copied().unwrap_or((0, 0))
    }

    pub fn text(&self, id: SymId) -> &str {
        let (o, l) = self.span(id);
        std::str::from_utf8(&self.pool[o as usize..(o + l) as usize]).unwrap_or("")
    }

    #[inline]
    pub fn pool(&self) -> &[u8] { &self.pool }

    #[inline]
    pub fn len(&self) -> usize { self.spans.len() }
}