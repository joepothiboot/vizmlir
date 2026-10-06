const MAX_RESULTS = 60;

export function fuzzyScore(query, text) {
  if (!query) return 0;

  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const direct = t.indexOf(q);
  if (direct >= 0) return direct;

  let score = 100;
  let from = 0;

  for (const ch of q) {
    const at = t.indexOf(ch, from);
    if (at < 0) return null;
    score += at - from;
    from = at + 1;
  }

  return score;
}

function itemScore(query, item) {
  const text = fuzzyScore(query, item.text);
  if (text !== null) return text;

  const hint = item.hint ? fuzzyScore(query, item.hint) : null;

  return hint === null ? null : 1000 + hint;
}

export function filterItems(items, query) {
  return items
    .map((item, order) => ({
      item,
      order,
      score: itemScore(query.trim(), item),
    }))
    .filter((entry) => entry.score !== null)
    .sort((a, b) => a.score - b.score || a.order - b.order)
    .slice(0, MAX_RESULTS)
    .map((entry) => entry.item);
}

export class CommandPalette {
  constructor(dialog, getItems) {
    this.dialog = dialog;
    this.input = dialog.querySelector("input");
    this.list = dialog.querySelector("[role=listbox]");
    this.getItems = getItems;
    this.items = [];
    this.results = [];
    this.cursor = 0;

    this.input.addEventListener("input", () => this.#render());
    this.input.addEventListener("keydown", (e) => this.#onKey(e));

    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) dialog.close();
    });
  }

  open(query = "") {
    this.items = this.getItems();
    this.input.value = query;
    this.dialog.showModal();
    this.input.focus();
    this.#render();
  }

  #onKey(e) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!this.results.length) return;

      const step = e.key === "ArrowDown" ? 1 : -1;

      this.#move(
        (this.cursor + step + this.results.length) % this.results.length,
      );
    } else if (e.key === "Enter") {
      e.preventDefault();
      this.#choose(this.cursor);
    }
  }

  #choose(index) {
    const item = this.results[index];
    if (!item) return;
    this.dialog.close();
    item.run();
  }

  #move(index) {
    this.cursor = index;

    [...this.list.children].forEach((row, i) =>
      row.setAttribute("aria-selected", String(i === index)),
    );

    this.list.children[index]?.scrollIntoView({ block: "nearest" });
  }

  #render() {
    this.results = filterItems(this.items, this.input.value);

    if (!this.results.length) {
      const empty = document.createElement("li");
      empty.className = "empty";
      empty.textContent = "No matches";
      this.list.replaceChildren(empty);

      return;
    }

    this.list.replaceChildren(
      ...this.results.map((item, index) => {
        const row = document.createElement("li");
        row.setAttribute("role", "option");

        const group = document.createElement("span");
        group.className = "group";
        group.textContent = item.group;

        const text = document.createElement("span");
        text.className = "text";
        text.textContent = item.text;
        row.append(group, text);

        if (item.hint) {
          const hint = document.createElement("span");
          hint.className = "hint";
          hint.textContent = item.hint;
          row.append(hint);
        }

        row.addEventListener("click", () => this.#choose(index));

        row.addEventListener("mousemove", () => {
          if (this.cursor !== index) this.#move(index);
        });

        return row;
      }),
    );

    this.#move(0);
  }
}
