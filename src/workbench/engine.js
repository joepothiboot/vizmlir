import { MlirEngine, STATUS } from "../ir/index.js";

export const engine = await MlirEngine.load(
  `${import.meta.env.BASE_URL}mlir_core.wasm`,
);

export function parse(text) {
  const t0 = performance.now();
  const status = engine.parse(text);

  return {
    status,
    ms: performance.now() - t0,
    snapshot: status === STATUS.OK ? engine.snapshot() : null,
  };
}
