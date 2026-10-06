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

export function parseAffine(text) {
  const tokens = text.match(
    /%[\w$.-]+|[ds]\d+|\d+|floordiv|ceildiv|mod|symbol|[-+*()]/g,
  );

  if (!tokens || tokens.join("").length !== text.replace(/\s+/g, "").length) {
    return null;
  }

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

    while (["*", "floordiv", "ceildiv", "mod"].includes(peek())) {
      e = { op: take(), a: e, b: primary() };
    }

    return e;
  }

  function sum() {
    let e = product();

    while (peek() === "+" || peek() === "-") {
      e = { op: take(), a: e, b: product() };
    }

    return e;
  }

  try {
    const e = sum();

    return at === tokens.length ? e : null;
  } catch {
    return null;
  }
}

export function parseAffineMap(text) {
  const m = /affine_map<\(([^)]*)\)(?:\[([^\]]*)\])?\s*->\s*\((.*)\)>\s*$/.exec(
    text.trim(),
  );

  if (!m || /,(?![^(]*\))/.test(m[3])) return null;

  const names = (list) =>
    (list ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

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

export function foldAffine(tree, { leaf, num, combine }) {
  if ("num" in tree) return num(tree.num);
  if ("leaf" in tree) return leaf(tree.leaf);

  const a = foldAffine(tree.a, { leaf, num, combine });
  const b = a === null ? null : foldAffine(tree.b, { leaf, num, combine });

  return b === null ? null : combine(tree.op, a, b);
}

export const evaluateAffine = (tree, leaf) =>
  foldAffine(tree, {
    leaf,
    num: (n) => n,
    combine: (op, a, b) => NUMERIC[op](a, b),
  });

export const linearAffine = (tree, leaf) =>
  foldAffine(tree, {
    leaf,
    num: (n) => constant(n),
    combine(op, a, b) {
      if (op === "+") return add(a, b);
      if (op === "-") return add(a, b, -1);

      if (op === "*") {
        if (isConstant(a)) return scale(b, a.c);
        if (isConstant(b)) return scale(a, b.c);

        return null;
      }

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

export function linearize(name, defs, env, ctx = {}, depth = 0) {
  ctx.loops ??= new Map();
  ctx.ivs ??= new Map();

  const stuck = (at = name) => {
    ctx.stuck ??= at;

    return null;
  };

  if (/^-?\d+$/.test(name)) return constant(Number(name));

  if (!/^%[\w$.-]+$/.test(name)) {
    const tree = parseAffine(name);

    return tree
      ? (linearAffine(tree, (leaf) =>
          linearize(leaf, defs, env, ctx, depth + 1),
        ) ?? stuck())
      : stuck();
  }

  if (env.args?.has(name)) {
    const value = env.args.get(name);

    return value === null ? variable(`arg:${name}`) : constant(value);
  }

  const def = defs.get(name);
  if (!def || depth > 200) return stuck();

  const operand = (i) => linearize(def.operands[i], defs, env, ctx, depth + 1);

  const size = (key) =>
    env[key] === null || env[key] === undefined ? stuck() : constant(env[key]);

  if (def.op === "env") {
    if (THREAD_IDS.includes(def.rest) || BLOCK_IDS.includes(def.rest)) {
      return variable(def.rest);
    }

    return size(def.rest);
  }

  if (def.op === "loop") {
    const [lb, ub, step] = [0, 1, 2].map((i) =>
      def.operands[i] ? operand(i) : null,
    );

    if (!lb || !step || !isConstant(step) || step.c <= 0) return stuck();

    const iter = `iter:${name}`;

    const trip =
      ub && isConstant(ub) && isConstant(lb)
        ? Math.max(0, Math.ceil((ub.c - lb.c) / step.c))
        : null;

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

    return def.op === "gpu.thread_id" || def.op === "gpu.block_id"
      ? variable(key)
      : size(key);
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
      return isConstant(b) && b.c >= 0 && b.c < 31
        ? scale(a, 2 ** b.c)
        : stuck();
  }

  if (FOLD[def.op] && isConstant(a) && isConstant(b)) {
    const value = FOLD[def.op](a.c, b.c);

    return value === null ? stuck() : constant(value);
  }

  return stuck();
}

function termText(name, factor) {
  if (factor === 1) return name;
  if (factor === -1) return `-${name}`;

  return `${factor}·${name}`;
}

export function formatAffine(e, ivs = new Map()) {
  let expr = constant(e.c);

  for (const [v, k] of e.t) {
    const loop = ivs.get(v);

    if (loop && k % loop.step === 0) {
      const per = k / loop.step;

      expr = add(
        add(expr, scale(variable(loop.name), per)),
        scale(loop.lb, per),
        -1,
      );
    } else {
      expr = add(
        expr,
        scale(
          variable(loop ? `iter(${loop.name})` : v.replace(/^arg:/, "")),
          k,
        ),
      );
    }
  }

  const terms = [...expr.t]
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .map(([v, k]) => termText(v, k));

  if (expr.c || !terms.length) terms.push(String(expr.c));

  return terms.join(" + ").replace(/\+ -/g, "- ");
}
