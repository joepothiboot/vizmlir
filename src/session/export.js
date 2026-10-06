export function download(filename, content, type = "text/plain") {
  const blob =
    content instanceof Blob ? content : new Blob([content], { type });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function slug(text) {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "vizmlir"
  );
}

const TYPE_LABEL = { added: "added", removed: "removed", changed: "modified" };
const GLYPH = { added: "+", removed: "−", changed: "~" };

export function diffRecords(rows, parentOf) {
  return rows.map((row) => {
    const op = row.after ?? row.before;

    const record = {
      type: TYPE_LABEL[row.type],
      op: op.label,
      parent: parentOf(row, op.parent) || null,
    };

    if (row.type === "changed") record.was = row.before.label;

    return record;
  });
}

export function diffToJSON(title, records) {
  return JSON.stringify({ title, changes: records }, null, 2);
}

export function diffToPatch(title, records) {
  const lines = [`--- baseline`, `+++ current  (${title})`];
  let context;

  for (const record of records) {
    const parent = record.parent ?? "top level";

    if (parent !== context) {
      lines.push(`@@ ${parent} @@`);
      context = parent;
    }

    if (record.type === "added") lines.push(`+ ${record.op}`);
    else if (record.type === "removed") lines.push(`- ${record.op}`);
    else lines.push(`- ${record.was}`, `+ ${record.op}`);
  }

  return [...lines, ""].join("\n");
}

function recordContext(record, cell) {
  if (record.was) return `was \`${cell(record.was)}\``;
  if (record.parent) return `in \`${cell(record.parent)}\``;

  return "top level";
}

export function diffToMarkdown(title, records) {
  const cell = (text) => (text ?? "").replace(/\|/g, "\\|");
  const lines = [`## ${title}`, ""];

  if (!records.length) {
    return [...lines, "No structural changes.", ""].join("\n");
  }

  lines.push("| | Operation | Detail |", "|---|---|---|");

  for (const record of records) {
    const glyph = GLYPH[record.type === "modified" ? "changed" : record.type];

    const detail = recordContext(record, cell);

    lines.push(`| ${glyph} | \`${cell(record.op)}\` | ${detail} |`);
  }

  return [...lines, ""].join("\n");
}
