function emptyText(names, missing) {
  const list = missing.join(", ");

  if (names.length && missing.length) {
    return `The IR also refers to ${list}. Add the file to see those lines.`;
  }

  if (names.length) return "";

  if (missing.length) {
    return `This IR refers to ${list}. Add the file here, or drop it on this tab, to see its lines beside the ops.`;
  }

  return "Add a source file here, or drop it on this tab. Ops are matched to it through the loc(...) the compiler printed.";
}

export function createSourceView(root, { onPick, onAdd }) {
  const select = root.querySelector(".src-file");
  const add = root.querySelector(".src-add");
  const input = root.querySelector(".src-input");
  const list = root.querySelector(".src-lines");
  const empty = root.querySelector(".src-empty");

  let sources = {};
  let missing = [];
  let file = "";
  let counts = new Map();
  let marks = new Set();

  function draw() {
    const names = Object.keys(sources);
    select.hidden = names.length < 2;

    select.replaceChildren(
      ...names.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;

        return option;
      }),
    );

    if (!sources[file]) file = names[0] ?? "";
    select.value = file;

    empty.hidden = missing.length === 0 && names.length > 0;

    empty.textContent = emptyText(names, missing);

    const rows = file ? sources[file].replace(/\n$/, "").split("\n") : [];

    list.replaceChildren(
      ...rows.map((text, i) => {
        const row = document.createElement("div");
        row.className = "src-line";
        row.dataset.line = String(i + 1);

        const n = document.createElement("span");
        n.className = "src-n";
        n.textContent = String(i + 1);

        const ops = document.createElement("span");
        ops.className = "src-ops";

        const t = document.createElement("span");
        t.className = "src-t";
        t.textContent = text;
        row.append(n, ops, t);

        return row;
      }),
    );

    paint();
  }

  function paint() {
    for (const row of list.children) {
      const line = Number(row.dataset.line);
      const count = counts.get(line) ?? 0;
      row.classList.toggle("has-ops", count > 0);
      row.classList.toggle("hit", marks.has(line));
      row.querySelector(".src-ops").textContent = count ? String(count) : "";

      row.title = count
        ? `${count} op${count === 1 ? "" : "s"} from this line`
        : "";
    }
  }

  list.addEventListener("click", (e) => {
    const row = e.target.closest(".src-line");
    if (row && file) onPick(file, Number(row.dataset.line));
  });

  select.addEventListener("change", () => {
    file = select.value;
    draw();
  });

  add.addEventListener("click", () => input.click());

  input.addEventListener("change", () => {
    if (input.files.length) onAdd([...input.files]);
    input.value = "";
  });

  root.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
  });

  root.addEventListener("drop", (e) => {
    if (!e.dataTransfer?.files?.length) return;
    e.preventDefault();
    onAdd([...e.dataTransfer.files]);
  });

  return {
    setSources(next, notLoaded = []) {
      sources = next;
      missing = notLoaded;
      draw();
    },
    setCounts(next) {
      counts = next;
      paint();
    },
    mark(name, lines) {
      if (name && sources[name] && name !== file) {
        file = name;
        draw();
      }

      marks = new Set(lines);
      paint();

      const first = [...list.children].find((r) =>
        marks.has(Number(r.dataset.line)),
      );

      first?.scrollIntoView?.({ block: "nearest" });
    },
    get file() {
      return file;
    },
    show(name) {
      if (sources[name]) {
        file = name;
        draw();
      }
    },
  };
}
