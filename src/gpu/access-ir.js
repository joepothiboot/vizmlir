import { evaluateAffine, parseAffine, parseAffineMap } from "./affine.js";

const DEF = /^\s*(%[\w$.-]+)(?::\d+)?\s*=\s*"?([\w.]+)"?\s*(.*)$/;

const FOR =
  /\b(?:scf|affine)\.for\s+(%[\w$.-]+)\s*=\s*(%[\w$.-]+|-?\d+)\s+to\b/;

const FOR_BOUNDS = /\bto\s+(%[\w$.-]+|-?\d+)(?:\s+step\s+(%[\w$.-]+|-?\d+))?/;

const ACCESS =
  /\b((?:memref|affine)\.(?:load|store))\s+(?:(%[\w$.-]+)\s*,\s*)?(%[\w$.-]+)\[([^\]]*)\]\s*:\s*(memref<.*>)\s*$/;

const GEP =
  /^\s*(%[\w$.-]+)\s*=\s*llvm\.getelementptr\b[^%]*(%[\w$.-]+)\[(%[\w$.-]+|-?\d+)\]\s*:\s*\(!llvm\.ptr(?:<(\d+)>)?,[^)]*\)\s*->\s*!llvm\.ptr(?:<\d+>)?,\s*(\w+)\s*$/;

const LLVM_LOAD = /^\s*(%[\w$.-]+)\s*=\s*llvm\.load\b[^%]*(%[\w$.-]+)\s*:/;
const LLVM_STORE = /^\s*llvm\.store\b[^%]*(%[\w$.-]+)\s*,\s*(%[\w$.-]+)\s*:/;

const EXTRACT =
  /^\s*(%[\w$.-]+)\s*=\s*llvm\.extractvalue\s+(%[\w$.-]+)\[([\d,\s]+)\]/;

const INSERT =
  /^\s*(%[\w$.-]+)\s*=\s*llvm\.insertvalue\s+(%[\w$.-]+)\s*,\s*(%[\w$.-]+)\[([\d,\s]+)\]/;

const BLOCK_HEAD = /^\s*\^([\w$.-]+)\(([^)]*)\)\s*:/;
const BRANCH = /\b(?:cf|llvm)\.(?:br|cond_br)\b/;
const TARGET = /\^([\w$.-]+)(?:\(([^)]*)\))?/g;

const COMPARE =
  /\b(?:arith\.cmpi\s+(?:slt|ult)\s*,|llvm\.icmp\s+"(?:slt|ult)")\s*(%[\w$.-]+)\s*,\s*(%[\w$.-]+|-?\d+)/;

const ADDRESS_OF = /^\s*(%[\w$.-]+)\s*=\s*llvm\.mlir\.addressof\s+(@[\w$.-]+)/;

const ZERO_GEP =
  /^\s*(%[\w$.-]+)\s*=\s*llvm\.getelementptr\b[^%]*(%[\w$.-]+)\[0(?:\s*,\s*0)*\]/;

const MAP_ALIAS = /^\s*(#[\w$.-]+)\s*=\s*(affine_map<.*>)\s*$/;

const LLVM_OPS = {
  "llvm.add": "arith.addi",
  "llvm.sub": "arith.subi",
  "llvm.mul": "arith.muli",
  "llvm.sdiv": "arith.divsi",
  "llvm.udiv": "arith.divui",
  "llvm.srem": "arith.remsi",
  "llvm.urem": "arith.remui",
  "llvm.shl": "arith.shli",
  "llvm.lshr": "arith.shrui",
  "llvm.ashr": "arith.shrsi",
  "llvm.and": "arith.andi",
  "llvm.or": "arith.ori",
  "llvm.xor": "arith.xori",
  "llvm.sext": "arith.extsi",
  "llvm.zext": "arith.extui",
  "llvm.trunc": "arith.trunci",
};

const SREG = {
  tid: "gpu.thread_id",
  ctaid: "gpu.block_id",
  ntid: "gpu.block_dim",
  nctaid: "gpu.grid_dim",
  "workitem.id": "gpu.thread_id",
  "workgroup.id": "gpu.block_id",
};

function normalize(op, rest) {
  if (LLVM_OPS[op]) return { op: LLVM_OPS[op], rest };

  if (op === "llvm.mlir.constant") {
    const value = /^\s*\(\s*(-?\d+)\b(?!\.)/.exec(rest);
    if (value) return { op: "arith.constant", rest: `${value[1]} : index` };
  }

  const sreg =
    /^(?:nvvm\.read\.ptx\.sreg|rocdl)\.(tid|ctaid|ntid|nctaid|workitem\.id|workgroup\.id)\.([xyz])$/.exec(
      op,
    );

  if (sreg) return { op: SREG[sreg[1]], rest: sreg[2] };
  if (op === "tt.get_program_id") return { op: "gpu.block_id", rest };
  if (op === "tt.get_num_programs") return { op: "gpu.grid_dim", rest };

  return { op, rest };
}

export function affineAliases(ir) {
  const maps = new Map();

  for (const line of ir.split("\n")) {
    const alias = MAP_ALIAS.exec(line);
    if (alias) maps.set(alias[1], alias[2]);
  }

  return maps;
}

const LAUNCH_IDS = {
  blocks: ["bx", "by", "bz"],
  threads: ["tx", "ty", "tz"],
};

const LAUNCH_SIZES = {
  blocks: ["gdx", "gdy", "gdz"],
  threads: ["bdx", "bdy", "bdz"],
};

const ID_OPS = {
  "gpu.thread_id": "t",
  "gpu.block_id": "b",
  "gpu.block_dim": "bd",
  "gpu.grid_dim": "gd",
};

const BINARY = {
  "arith.addi": (a, b) => a + b,
  "arith.subi": (a, b) => a - b,
  "arith.muli": (a, b) => a * b,
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
  "arith.shli": (a, b) => a << b,
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

export function buildDefs(lines, firstLine = 1, maps = new Map()) {
  const defs = new Map();

  for (const [i, line] of lines.entries()) {
    const at = firstLine + i;
    const loop = FOR.exec(line);

    if (loop) {
      const bounds = FOR_BOUNDS.exec(line.slice(loop.index));

      const operands = bounds
        ? [loop[2], bounds[1], bounds[2] ?? "1"]
        : [loop[2]];

      defs.set(loop[1], { op: "loop", operands, rest: "", line: at });

      const carried = /\biter_args\s*\(([^)]*)\)/.exec(line)?.[1] ?? "";

      for (const pair of carried.matchAll(/(%[\w$.-]+)\s*=\s*(%[\w$.-]+)/g)) {
        defs.set(pair[1], {
          op: "iter",
          operands: [pair[2]],
          rest: "",
          line: at,
        });
      }
    }

    if (/\bgpu\.launch\b(?!_)/.test(line) || /^\s*threads\(/.test(line)) {
      for (const [keyword, ids] of Object.entries(LAUNCH_IDS)) {
        const match = new RegExp(
          `\\b${keyword}\\(([^)]*)\\)\\s+in\\s+\\(([^)]*)\\)`,
        ).exec(line);

        if (!match) continue;

        match[1].split(",").forEach((name, i) =>
          defs.set(name.trim(), {
            op: "env",
            operands: [],
            rest: ids[i],
            line: at,
          }),
        );

        match[2].split(",").forEach((part, i) => {
          const name = part.split("=")[0].trim();

          defs.set(name, {
            op: "env",
            operands: [],
            rest: LAUNCH_SIZES[keyword][i],
            line: at,
          });
        });
      }
    }

    const def = DEF.exec(line);
    if (!def || defs.has(def[1])) continue;

    const { op, rest } = normalize(def[2], def[3]);

    const entry = {
      op,
      operands: rest.match(/%[\w$.-]+/g) ?? [],
      rest,
      line: at,
    };

    if (op === "affine.apply") {
      const alias = /^\s*(#[\w$.-]+)/.exec(rest)?.[1];

      entry.map = parseAffineMap(
        alias
          ? (maps.get(alias) ?? "")
          : (/affine_map<.*>(?=\s*\()/.exec(rest)?.[0] ?? ""),
      );
    }

    defs.set(def[1], entry);
  }

  branchLoops(lines, firstLine, defs);

  return defs;
}

function branchLoops(lines, firstLine, defs) {
  const params = new Map();
  const incoming = new Map();
  const heads = [];

  lines.forEach((line, i) => {
    const head = BLOCK_HEAD.exec(line);

    if (head) {
      const names = head[2].split(",").map((part) => part.split(":")[0].trim());
      params.set(head[1], names);
      heads.push({ names, line: firstLine + i });
    }
  });

  for (const line of lines) {
    if (!BRANCH.test(line)) continue;

    for (const target of line.matchAll(TARGET)) {
      const names = params.get(target[1]);
      if (!names || !target[2]) continue;

      target[2]
        .split(":")[0]
        .split(",")
        .map((v) => v.trim())
        .forEach((value, k) => {
          if (!names[k]) return;
          if (!incoming.has(names[k])) incoming.set(names[k], []);
          incoming.get(names[k]).push(value);
        });
    }
  }

  const bounds = new Map();

  for (const line of lines) {
    const compare = COMPARE.exec(line);
    if (compare) bounds.set(compare[1], compare[2]);
  }

  for (const { names, line } of heads) {
    for (const name of names) {
      const values = incoming.get(name) ?? [];
      if (defs.has(name) || values.length !== 2 || !bounds.has(name)) continue;

      const back = values.findIndex((v) => {
        const d = defs.get(v);

        return (
          d?.op === "arith.addi" &&
          d.operands.length === 2 &&
          d.operands.includes(name)
        );
      });

      if (back < 0) continue;

      const next = defs.get(values[back]);

      const step =
        next.operands[0] === name ? next.operands[1] : next.operands[0];

      defs.set(name, {
        op: "loop",
        operands: [values[1 - back], bounds.get(name), step],
        rest: "",
        line,
      });
    }
  }
}

export function evaluate(name, defs, env, trace = {}, depth = 0) {
  if (/^-?\d+$/.test(name)) return Number(name);

  if (!/^%[\w$.-]+$/.test(name)) {
    const tree = parseAffine(name);

    if (!tree) {
      trace.stuck ??= name;

      return null;
    }

    return evaluateAffine(tree, (leaf) =>
      evaluate(leaf, defs, env, trace, depth + 1),
    );
  }

  if (env.args?.has(name)) {
    const value = env.args.get(name);
    if (value === null) trace.stuck ??= name;

    return value;
  }

  const def = defs.get(name);

  if (!def || depth > 200) {
    trace.stuck ??= name;

    return null;
  }

  const operand = (i) => evaluate(def.operands[i], defs, env, trace, depth + 1);
  if (def.op === "env") return env[def.rest] ?? null;
  if (def.op === "loop") return operand(0);

  if (def.op === "arith.constant") {
    const match = /^(-?\d+)\b(?!\.)/.exec(def.rest.trim());
    if (match) return Number(match[1]);
  } else if (ID_OPS[def.op]) {
    const dim = /^\s*([xyz])\b/.exec(def.rest)?.[1];
    if (dim) return env[`${ID_OPS[def.op]}${dim}`] ?? null;
  } else if (BINARY[def.op] && def.operands.length === 2) {
    const a = operand(0);
    const b = a === null ? null : operand(1);
    if (a !== null && b !== null) return BINARY[def.op](a, b);

    return null;
  } else if (CASTS.has(def.op) && def.operands.length === 1) {
    return operand(0);
  } else if (def.op === "affine.apply" && def.map) {
    const names = [...def.map.dims, ...def.map.syms];

    const value = evaluateAffine(def.map.expr, (leaf) => {
      const k = names.indexOf(leaf);

      return k < 0 ? null : operand(k);
    });

    if (value !== null) return value;
  }

  trace.stuck ??= name;

  return null;
}

export function findAccesses(lines, firstLine = 1) {
  const accesses = [];
  const loops = [];
  const geps = new Map();
  const extracts = new Map();
  const inserts = new Map();
  const aliases = new Map();

  for (const line of lines) {
    const address = ADDRESS_OF.exec(line);
    if (address) aliases.set(address[1], address[2]);

    const zero = ZERO_GEP.exec(line);
    if (zero) aliases.set(zero[1], zero[2]);

    const gep = GEP.exec(line);

    if (gep) {
      geps.set(gep[1], {
        base: gep[2],
        index: gep[3],
        space: gep[4] ?? "",
        element: gep[5],
      });
    }

    const extract = EXTRACT.exec(line);

    if (extract) {
      extracts.set(extract[1], {
        from: extract[2],
        at: extract[3].replace(/\s+/g, ""),
      });
    }

    const insert = INSERT.exec(line);

    if (insert) {
      inserts.set(insert[1], {
        value: insert[2],
        into: insert[3],
        at: insert[4].replace(/\s+/g, ""),
      });
    }
  }

  const root = (name, depth = 0) => {
    if (depth > 64) return name;
    if (aliases.has(name)) return root(aliases.get(name), depth + 1);

    const extract = extracts.get(name);
    if (!extract) return name;

    for (let s = extract.from, n = 0; inserts.has(s) && n < 64; n++) {
      const insert = inserts.get(s);
      if (insert.at === extract.at) return root(insert.value, depth + 1);
      s = insert.into;
    }

    return name;
  };

  const llvmAccess = (line, kind, value, pointer) => {
    const gep = geps.get(pointer);
    if (!gep) return;

    accesses.push({
      line,
      kind,
      value,
      buffer: root(gep.base),
      indices: [gep.index],
      type: `memref<?x${gep.element}${gep.space ? `, ${gep.space}` : ""}>`,
      inLoop: false,
    });
  };

  lines.forEach((line, i) => {
    if (line.trim() === "") return;

    const indent = /^\s*/.exec(line)[0].length;
    while (loops.length && indent <= loops.at(-1)) loops.pop();
    if (FOR.test(line)) loops.push(indent);

    const load = LLVM_LOAD.exec(line);
    if (load) return llvmAccess(firstLine + i, "load", null, load[2]);

    const store = LLVM_STORE.exec(line);
    if (store) return llvmAccess(firstLine + i, "store", store[1], store[2]);

    const match = ACCESS.exec(line);
    if (!match) return;

    accesses.push({
      line: firstLine + i,
      kind: match[1].endsWith("load") ? "load" : "store",
      value: match[2] ?? null,
      buffer: match[3],
      indices: match[4]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      type: match[5],
      inLoop: loops.length > (FOR.test(line) ? 1 : 0),
    });
  });

  return accesses;
}
