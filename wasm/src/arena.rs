pub struct Arena {
    buf: Vec<u64>,
    head: usize,
    oom: bool,
}

impl Arena {
    pub fn with_words(words: usize) -> Self {
        Self { buf: vec![0u64; words], head: 0, oom: false }
    }

    #[inline]
    pub fn capacity(&self) -> usize { self.buf.len() * 8 }

    #[inline]
    pub fn used(&self) -> usize { self.head }

    #[inline]
    pub fn is_oom(&self) -> bool { self.oom }

    pub fn reset(&mut self) {
        self.head = 0;
        self.oom = false;
    }

    pub fn alloc(&mut self, bytes: usize, align: usize) -> Option<usize> {
        debug_assert!(align.is_power_of_two() && align <= 8);
        let start = (self.head + align - 1) & !(align - 1);
        let end = start.checked_add(bytes)?;
        if end > self.capacity() {
            self.oom = true;
            return None;
        }
        self.head = end;
        Some(start)
    }

    #[inline]
    fn base(&self) -> *mut u8 { self.buf.as_ptr() as *mut u8 }

    #[inline]
    pub fn addr(&self, off: usize) -> u32 { (self.base() as usize + off) as u32 }

    #[inline]
    pub fn u32s(&mut self, off: usize, len: usize) -> &mut [u32] {
        unsafe { core::slice::from_raw_parts_mut(self.base().add(off) as *mut u32, len) }
    }

    #[inline]
    pub fn f32s(&mut self, off: usize, len: usize) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.base().add(off) as *mut f32, len) }
    }

    #[inline]
    pub fn bytes(&mut self, off: usize, len: usize) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.base().add(off), len) }
    }
}