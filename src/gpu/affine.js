// Writes an SSA index as a linear (affine) expression over the values that
// vary across a launch: thread ids (tx, ty, tz), block ids (bx, by, bz), the
// iteration count of each enclosing scf.for (`iter:%iv`), and kernel
// arguments whose value is unknown (`arg:%name`). Launch sizes and constants
// fold in as numbers, and affine.apply maps and affine.load index expressions
// are expanded. Anything else (a product of two varying values, a division of
// one, a loaded value) is not affine and returns null.
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

// ---- Affine expressions ------------------------------------------------------
//
// The right-hand side of an affine_map (`d0 * 32 + s0`, `d0 floordiv 4`) or an
// affine.load index (`%i * 2 + %j`), as a tree: { num } | { leaf } | { op, a, b }.
// A leaf is a dimension or symbol (`d0`, `s1`) or an SSA name (`%i`, including
// `symbol(%n)`). Returns null when the text is not an affine expression.
export function parseAffine(text) {
  const tokens = text.match(/%[\w$.-]+|[ds]\d+|\d+|floordiv|ceildiv|mod|symbol|[-+*()]/g);
  if (!tokens || tokens.join("").length !== text.replace(/\s+/g, "").length) return null;
  let at = 0;
  const peek = () => tokens[at];
  const take = () => tokens[at++];
  function primary() {
    const t = take();
    if (t === undefined) throw new Error("end");
    if (t === "(") {
      const e = sum();
      if (take() !== ")") throw new Error(")");
      return e;
    }
    if (t === "-") return { op: "*", a: { num: -1 }, b: primary() };
    if (t === "symbol") {
      if (take() !== "(") throw new Error("(");
      const leaf = take();
      if (take() !== ")") throw new Error(")");
      return { leaf };
    }
    if (/^\d+$/.test(t)) return { num: Number(t) };
    if (/^(%|[ds]\d)/.test(t)) return { leaf: t };
    throw new Error(t);
  }
  function product() {
    let e = primary();
    while (["*", "floordiv", "ceildiv", "mod"].includes(peek())) e = { op: take(), a: e, b: primary() };
    return e;
  }
  function sum() {
    let e = product();
    while (peek() === "+" || peek() === "-") e = { op: take(), a: e, b: product() };
    return e;
  }
  try {
    const e = sum();
    return at === tokens.length ? e : null;
  } catch {
    return null;
  }
}

// `affine_map<(d0, d1)[s0] -> (d0 * 32 + s0)>` → { dims, syms, expr } for a
// single-result map, else null.
export function parseAffineMap(text) {
  const m = /affine_map<\(([^)]*)\)(?:\[([^\]]*)\])?\s*->\s*\((.*)\)>\s*$/.exec(text.trim());
  if (!m || /,(?![^(]*\))/.test(m[3])) return null;
  const names = (list) => (list ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const expr = parseAffine(m[3]);
  return expr ? { dims: names(m[1]), syms: names(m[2]), expr } : null;
}

const FLOOR = (a, b) => (b ? Math.floor(a / b) : null);
const NUMERIC = {
  "+": (a, b) => a + b,
  "-": (a, b) => a - b,
  "*": (a, b) => a * b,
  floordiv: FLOOR,
  ceildiv: (a, b) => (b ? Math.ceil(a / b) : null),
  mod: (a, b) => (b ? ((a % b) + b) % b : null),
};

// Folds `tree` bottom-up: `leaf(name)` gives a leaf's value, `num(n)` a
// number's, and `combine(op, a, b)` two children's; any null stops it.
export function foldAffine(tree, { leaf, num, combine }) {
  if ("num" in tree) return num(tree.num);
  if ("leaf" in tree) return leaf(tree.leaf);
  const a = foldAffine(tree.a, { leaf, num, combine });
  const b = a === null ? null : foldAffine(tree.b, { leaf, num, combine });
  return b === null ? null : combine(tree.op, a, b);
}

// `tree` on numbers.
export const evaluateAffine = (tree, leaf) =>
  foldAffine(tree, { leaf, num: (n) => n, combine: (op, a, b) => NUMERIC[op](a, b) });

// `tree` on linear expressions: products need a constant side, and
// divisions and remainders need both.
export const linearAffine = (tree, leaf) =>
  foldAffine(tree, {
    leaf,
    num: (n) => constant(n),
    combine(op, a, b) {
      if (op === "+") return add(a, b);
      if (op === "-") return add(a, b, -1);
      if (op === "*") return isConstant(a) ? scale(b, a.c) : isConstant(b) ? scale(a, b.c) : null;
      if (isConstant(a) && isConstant(b)) {
        const value = NUMERIC[op](a.c, b.c);
        return value === null ? null : constant(value);
      }
      return null;
    },
  });

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
  // An affine.load index such as `%i * 2 + %j`.
  if (!/^%[\w$.-]+$/.test(name)) {
    const tree = parseAffine(name);
    return tree ? linearAffine(tree, (leaf) => linearize(leaf, defs, env, ctx, depth + 1)) ?? stuck() : stuck();
  }
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
  if (def.op === "affine.apply" && def.map) {
    const names = [...def.map.dims, ...def.map.syms];
    const e = linearAffine(def.map.expr, (leaf) => {
      const k = names.indexOf(leaf);
      return k < 0 ? null : operand(k);
    });
    return e ?? stuck();
  }
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
