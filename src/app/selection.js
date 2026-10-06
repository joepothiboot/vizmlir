const EMPTY = Object.freeze({ pass: -1, node: -1, line: -1 });

export function createSelection() {
  let state = EMPTY;
  const listeners = new Set();

  return {
    get state() {
      return state;
    },
    select(patch) {
      const picked = Object.keys(patch).filter((key) => key in EMPTY);
      if (!picked.length) return;
      state = Object.freeze({ ...state, ...patch });
      for (const listener of [...listeners]) listener(state, picked);
    },
    clear(...keys) {
      state = Object.freeze({
        ...state,
        ...Object.fromEntries(
          keys.filter((k) => k in EMPTY).map((k) => [k, -1]),
        ),
      });
    },
    subscribe(listener) {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
}
