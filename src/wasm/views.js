export class MemoryViews {
  constructor(memory) {
    this.memory = memory;
    this._buffer = null;
    this.u8 = null;
    this.u32 = null;
    this.f32 = null;
    this.refresh();
  }

  refresh() {
    const buf = this.memory.buffer;
    if (buf !== this._buffer) {
      this._buffer = buf;
      this.u8 = new Uint8Array(buf);
      this.u32 = new Uint32Array(buf);
      this.f32 = new Float32Array(buf);
    }
    return this;
  }

  #check(ptr, byteLen, align) {
    if (ptr % align !== 0) {
      throw new Error(`misaligned pointer 0x${ptr.toString(16)} (needs ${align}-byte alignment)`);
    }
    if (ptr + byteLen > this._buffer.byteLength) {
      throw new Error(`view 0x${ptr.toString(16)}+${byteLen} exceeds linear memory`);
    }
  }

  u32At(ptr, len) {
    this.refresh();
    if (len === 0) return EMPTY_U32;
    this.#check(ptr, len * 4, 4);
    return this.u32.subarray(ptr >>> 2, (ptr >>> 2) + len);
  }

  f32At(ptr, len) {
    this.refresh();
    if (len === 0) return EMPTY_F32;
    this.#check(ptr, len * 4, 4);
    return this.f32.subarray(ptr >>> 2, (ptr >>> 2) + len);
  }

  u8At(ptr, len) {
    this.refresh();
    if (len === 0) return EMPTY_U8;
    this.#check(ptr, len, 1);
    return this.u8.subarray(ptr, ptr + len);
  }

  writeBytes(ptr, bytes) {
    this.refresh();
    this.#check(ptr, bytes.length, 1);
    this.u8.set(bytes, ptr);
    return bytes.length;
  }
}

const EMPTY_U8 = new Uint8Array(0);
const EMPTY_U32 = new Uint32Array(0);
const EMPTY_F32 = new Float32Array(0);