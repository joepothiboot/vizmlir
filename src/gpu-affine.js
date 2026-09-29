// Writes an SSA index as a linear (affine) expression over the values that
// vary across a launch: thread ids (tx, ty, tz), block ids (bx, by, bz), the
// iteration count of each enclosing scf.for (`iter:%iv`), and kernel
// arguments whose value is unknown (`arg:%name`). Launch sizes and constants
// fold in as numbers. Anything else (a product of two varying values, a
// division of one, a loaded value) is not affine and returns null.
//
// An expression is { c, t } where `c` is the constant term and `t` maps each
// variable to its coefficient: c + Σ t[v]·v.

const THREAD_IDS = ["tx", "ty", "tz"];
const BLOCK_IDS = ["bx", "by", "bz"];
const ID_OPS = {
  "gpu.thread_id": "t",
  "gpu.block_id": "b",
  "gpu.block_dim": "bd",
  "gpu.grid_dim": "gd",
};
const FOLD = {
  "arith.divsi": (a, b) => (b ? Math.trunc(a / b) : null),
  "arith.divui": (a, b) => (b ? Math.floor(a / b) : null),
  "arith.floordivsi": (a, b) => (b ? Math.floor(a / b) : null),
  "arith.ceildivsi": (a, b) => (b ? Math.ceil(a / b) : null),
  "arith.remsi": (a, b) => (b ? a % b : null),
  "arith.remui": (a, b) => (b ? a % b : null),
  "arith.minsi": Math.min,
  "arith.minui": Math.min,
  "arith.maxsi": Math.max,
  "arith.maxui": Math.max,
  "arith.andi": (a, b) => a & b,
  "arith.ori": (a, b) => a | b,
  "arith.xori": (a, b) => a ^ b,
  "arith.shrui": (a, b) => a >>> b,
  "arith.shrsi": (a, b) => a >> b,
};
const CASTS = new Set([
  "arith.index_cast",
  "arith.index_castui",
  "arith.extsi",
  "arith.extui",
  "arith.trunci",
]);

export const constant = (c) => ({ c, t: new Map() });
const variable = (v) => ({ c: 0, t: new Map([[v, 1]]) });
export const isConstant = (e) => e.t.size === 0;

export function add(a, b, sign = 1) {
  const t = new Map(a.t);
  for (const [v, k] of b.t) {
    const sum = (t.get(v) ?? 0) + sign * k;
    if (sum) t.set(v, sum);
    else t.delete(v);
  }
  return { c: a.c + sign * b.c, t };
}

export function scale(e, k) {
  if (!k) return constant(0);
  return { c: e.c * k, t: new Map([...e.t].map(([v, x]) => [v, x * k])) };
}

// `name` as an affine expression, or null with the value that stopped it in
// `ctx.stuck`. `env` holds the launch sizes (bdx..bdz, gdx..gdz, null when
// unknown) and `args`, known kernel argument values by name. Each loop met on
// the way is recorded in `ctx.loops` as `iter:%iv` → its trip count (null
// when its bounds are not constants), and in `ctx.ivs` as `iter:%iv` →
// { name, lb, step } so formatAffine() can print the loop variable itself.
export function linearize(name, defs, env, ctx = {}, depth = 0) {
  ctx.loops ??= new Map();
  ctx.ivs ??= new Map();
  const stuck = (at = name) => {
    ctx.stuck ??= at;
    return null;
  };
  if (/^-?\d+$/.test(name)) return constant(Number(name));
  if (env.args?.has(name)) {
    const value = env.args.get(name);
    return value === null ? variable(`arg:${name}`) : constant(value);
  }
  const def = defs.get(name);
  if (!def || depth > 200) return stuck();
  const operand = (i) => linearize(def.operands[i], defs, env, ctx, depth + 1);
  const size = (key) => (env[key] === null || env[key] === undefined ? stuck() : constant(env[key]));

  if (def.op === "env") {
    if (THREAD_IDS.includes(def.rest) || BLOCK_IDS.includes(def.rest)) return variable(def.rest);
    return size(def.rest);
  }
  if (def.op === "loop") {
    // iv = lb + step·k for k in [0, trip).
    const [lb, ub, step] = [0, 1, 2].map((i) => (def.operands[i] ? operand(i) : null));
    if (!lb || !step || !isConstant(step) || step.c <= 0) return stuck();
    const iter = `iter:${name}`;
    const trip =
      ub && isConstant(ub) && isConstant(lb) ? Math.max(0, Math.ceil((ub.c - lb.c) / step.c)) : null;
    ctx.loops.set(iter, trip);
    ctx.ivs.set(iter, { name, lb, step: step.c });
    return add(lb, scale(variable(iter), step.c));
  }
  if (def.op === "arith.constant") {
    const match = /^(-?\d+)\b(?!\.)/.exec(def.rest.trim());
    return match ? constant(Number(match[1])) : stuck();
  }
  if (ID_OPS[def.op]) {
    const dim = /^\s*([xyz])\b/.exec(def.rest)?.[1];
    if (!dim) return stuck();
    const key = `${ID_OPS[def.op]}${dim}`;
    return def.op === "gpu.thread_id" || def.op === "gpu.block_id" ? variable(key) : size(key);
  }
  if (CASTS.has(def.op) && def.operands.length === 1) return operand(0);
  if (def.operands.length !== 2) return stuck();

  const a = operand(0);
  const b = a && operand(1);
  if (!a || !b) return null;
  switch (def.op) {
    case "arith.addi":
      return add(a, b);
    case "arith.subi":
      return add(a, b, -1);
    case "arith.muli":
      if (isConstant(a)) return scale(b, a.c);
      if (isConstant(b)) return scale(a, b.c);
      return stuck();
    case "arith.shli":
      return isConstant(b) && b.c >= 0 && b.c < 31 ? scale(a, 2 ** b.c) : stuck();
  }
  if (FOLD[def.op] && isConstant(a) && isConstant(b)) {
    const value = FOLD[def.op](a.c, b.c);
    return value === null ? stuck() : constant(value);
  }
  return stuck();
}

// An expression as text, largest coefficient first: "32768·bx + 1024·tx +
// 32·by + ty". Iteration counts are written back as their loop variable
// (k = (iv - lb) / step) when `ivs` (from linearize's ctx) allows it.
export function formatAffine(e, ivs = new Map()) {
  let expr = constant(e.c);
  for (const [v, k] of e.t) {
    const loop = ivs.get(v);
    if (loop && k % loop.step === 0) {
      const per = k / loop.step;
      expr = add(add(expr, scale(variable(loop.name), per)), scale(loop.lb, per), -1);
    } else {
      expr = add(expr, scale(variable(loop ? `iter(${loop.name})` : v.replace(/^arg:/, "")), k));
    }
  }
  const terms = [...expr.t]
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([v, k]) => (k === 1 ? v : k === -1 ? `-${v}` : `${k}·${v}`));
  if (expr.c || !terms.length) terms.push(String(expr.c));
  return terms.join(" + ").replace(/\+ -/g, "- ");
}
