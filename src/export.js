// File downloads and diff serialisation. Everything is generated locally.

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

/** A filesystem-friendly stem, e.g. "pass-3-after-cse". */
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

/** Plain rows for export: { type, op, was?, parent }. */
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

export function diffToMarkdown(title, records) {
  const cell = (text) => (text ?? "").replace(/\|/g, "\\|");
  const lines = [`## ${title}`, ""];
  if (!records.length)
    return [...lines, "No structural changes.", ""].join("\n");
  lines.push("| | Operation | Detail |", "|---|---|---|");
  for (const record of records) {
    const glyph = GLYPH[record.type === "modified" ? "changed" : record.type];
    const detail = record.was
      ? `was \`${cell(record.was)}\``
      : record.parent
        ? `in \`${cell(record.parent)}\``
        : "top level";
    lines.push(`| ${glyph} | \`${cell(record.op)}\` | ${detail} |`);
  }
  return [...lines, ""].join("\n");
}
