// Watch a local file (typically an mlir-opt trace) and report each change.
// Uses the File System Access API (Chromium browsers). The browser has no
// change events for files, so poll lastModified/size, which is cheap.

const POLL_MS = 1000;

export const canWatchFiles = typeof window.showOpenFilePicker === "function";

export class FileWatcher {
  constructor(onChange) {
    this.onChange = onChange;
    this.handle = null;
    this.stamp = "";
    this.timer = 0;
  }

  get watching() {
    return this.timer !== 0;
  }

  /** Ask the user for a file, then watch it. Returns the handle or null. */
  async pick() {
    let handle;
    try {
      [handle] = await window.showOpenFilePicker({
        types: [
          {
            description: "MLIR or mlir-opt trace",
            accept: { "text/plain": [".mlir", ".txt", ".log"] },
          },
        ],
      });
    } catch {
      return null; // Picker dismissed.
    }
    await this.start(handle);
    return handle;
  }

  /** Whether reading `handle` needs a user gesture first (after a reload). */
  static async needsPermission(handle) {
    return (await handle.queryPermission?.({ mode: "read" })) !== "granted";
  }

  /** Call from a click handler to regain access to a remembered handle. */
  static async requestPermission(handle) {
    return (await handle.requestPermission?.({ mode: "read" })) === "granted";
  }

  async start(handle) {
    this.stop();
    this.handle = handle;
    this.stamp = "";
    await this.#poll();
    this.timer = setInterval(() => this.#poll(), POLL_MS);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = 0;
  }

  async #poll() {
    let file;
    try {
      file = await this.handle.getFile();
    } catch (error) {
      // Deleted, moved or permission revoked: stop rather than spin.
      this.stop();
      this.onChange(null, error);
      return;
    }
    const stamp = `${file.lastModified}:${file.size}`;
    if (stamp === this.stamp) return;
    this.stamp = stamp;
    this.onChange({ name: file.name, text: await file.text() });
  }
}
