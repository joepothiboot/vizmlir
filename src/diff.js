// Compare parser snapshots while treating SSA names as unstable identifiers.
const SSA_NAME = /%[-A-Za-z0-9_.$]+/g;

export function normalizeLabel(label) {
  return label.replace(SSA_NAME, '%?');
}

export function copySnapshot(snapshot) {
  return {
    nodeCount: snapshot.nodeCount,
    nodes: Array.from({ length: snapshot.nodeCount }, (_, index) => ({
      kind: snapshot.kindOf(index),
      parent: snapshot.parentOf(index),
      label: snapshot.labelOf(index),
    })),
  };
}

export function diffSnapshots(before, after) {
  const beforeOps = snapshotOperations(before);
  const afterOps = snapshotOperations(after);
  const rows = [];
  const used = new Set();

  for (const operation of afterOps) {
    const exact = beforeOps.findIndex((candidate, index) =>
      !used.has(index) && candidate.signature === operation.signature
    );
    if (exact >= 0) {
      used.add(exact);
      continue;
    }

    const changed = beforeOps.findIndex((candidate, index) =>
      !used.has(index) && candidate.kind === operation.kind && candidate.parent === operation.parent
    );
    if (changed >= 0) {
      used.add(changed);
      rows.push({ type: 'changed', before: beforeOps[changed], after: operation });
    } else {
      rows.push({ type: 'added', after: operation });
    }
  }

  beforeOps.forEach((operation, index) => {
    if (!used.has(index)) rows.push({ type: 'removed', before: operation });
  });

  return rows;
}

function snapshotOperations(snapshot) {
  if (!snapshot) return [];
  if (snapshot.nodes) return snapshot.nodes.map((node, index) => ({
    ...node,
    index,
    signature: `${node.kind}:${normalizeLabel(node.label)}`,
  }));
  return Array.from({ length: snapshot.nodeCount }, (_, index) => ({
    index,
    kind: snapshot.kindOf(index),
    parent: snapshot.parentOf(index),
    label: snapshot.labelOf(index),
    signature: `${snapshot.kindOf(index)}:${normalizeLabel(snapshot.labelOf(index))}`,
  }));
}