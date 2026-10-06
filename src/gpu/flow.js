const MAX_NODES = 60;
const AXES = ["x", "y", "z"];

const ENV = {
  t: ["thread", "thread"],
  b: ["block", "block"],
  bd: ["size", "block size"],
  gd: ["size", "grid size"],
};

const ID_OPS = {
  "gpu.thread_id": "t",
  "gpu.block_id": "b",
  "gpu.block_dim": "bd",
  "gpu.grid_dim": "gd",
};

function describe(name, def) {
  if (!def) return { kind: "arg", label: name, detail: "a kernel argument" };

  if (def.op === "env") {
    const [, key, axis] = /^(bd|gd|t|b)([xyz])$/.exec(def.rest) ?? [];
    const [kind, word] = ENV[key] ?? ["arg", def.rest];

    return {
      kind,
      label: `${word} ${axis ?? ""}`.trim(),
      detail: "bound by the launch",
    };
  }

  if (ID_OPS[def.op]) {
    const [kind, word] = ENV[ID_OPS[def.op]];

    const axis =
      AXES.find((a) => new RegExp(`^\\s*${a}\\b`).test(def.rest)) ?? "";

    return { kind, label: `${word} ${axis}`.trim(), detail: def.op };
  }

  if (def.op === "arith.constant") {
    const value = /^\s*(-?[\d.]+)/.exec(def.rest)?.[1] ?? "?";

    return { kind: "const", label: value, detail: "a constant" };
  }

  if (def.op === "loop") {
    return { kind: "loop", label: `loop ${name}`, detail: "scf.for" };
  }

  return { kind: "math", label: def.op.replace(/^\w+\./, ""), detail: def.op };
}

export function kernelFlow(kernel) {
  const defs = kernel?.defs;
  const accesses = kernel?.accesses ?? [];
  if (!defs || !accesses.length) return { nodes: [], edges: [] };

  const nodes = new Map();
  const edges = [];

  const visit = (name, depth = 0) => {
    if (nodes.has(name)) return nodes.get(name);
    if (nodes.size >= MAX_NODES || depth > 40) return null;

    const def = defs.get(name);

    const node = {
      id: name,
      name,
      line: def?.line ?? kernel.line,
      text: def
        ? `${name} = ${def.op === "env" ? "(launch)" : def.op} ${def.op === "env" ? "" : def.rest}`.trim()
        : name,
      ...describe(name, def),
      rank: 0,
    };

    nodes.set(name, node);

    if (def && !["env", "arith.constant"].includes(def.op) && !ID_OPS[def.op]) {
      for (const operand of new Set(def.operands)) {
        if (operand === name) continue;

        const from = visit(operand, depth + 1);
        if (!from) continue;
        edges.push({ from: from.id, to: name });
        node.rank = Math.max(node.rank, from.rank + 1);
      }
    }

    return node;
  };

  const accessNodes = [];

  for (const access of accesses) {
    if (nodes.size >= MAX_NODES) break;

    const id = `@${access.line}`;

    const node = {
      id,
      name: access.buffer,
      kind: "access",
      label: `${access.kind} ${access.buffer}`,
      detail: `memref.${access.kind}`,
      line: access.line,
      text: `${access.kind} ${access.buffer}[${access.indices.join(", ")}]`,
      access,
      rank: 0,
    };

    nodes.set(id, node);
    accessNodes.push(node);

    for (const index of new Set(access.indices)) {
      const from = visit(index);
      if (from) edges.push({ from: from.id, to: id });
    }
  }

  const last =
    Math.max(
      0,
      ...[...nodes.values()]
        .filter((n) => n.kind !== "access")
        .map((n) => n.rank),
    ) + 1;

  for (const node of accessNodes) node.rank = last;

  return { nodes: [...nodes.values()], edges };
}

export function flowSlice(flow, line) {
  const start = flow.nodes.filter((n) => n.line === line).map((n) => n.id);
  const slice = new Set(start);
  if (!start.length) return slice;

  const walk = (ids, key, other) => {
    const queue = [...ids];

    while (queue.length) {
      const id = queue.pop();

      for (const edge of flow.edges) {
        if (edge[key] === id && !slice.has(edge[other])) {
          slice.add(edge[other]);
          queue.push(edge[other]);
        }
      }
    }
  };

  walk(start, "to", "from");
  walk(start, "from", "to");

  return slice;
}
