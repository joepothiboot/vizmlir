export function createScrubber(
  root,
  { onSelect, preview, label, schedule } = {},
) {
  const range = root.querySelector(".scrub-range");
  const ticks = root.querySelector(".scrub-ticks");
  const tip = root.querySelector(".scrub-preview");
  const text = root.querySelector(".scrub-label");
  const prev = root.querySelector(".scrub-prev");
  const next = root.querySelector(".scrub-next");

  const later =
    schedule ??
    ((run) =>
      typeof requestAnimationFrame === "function"
        ? requestAnimationFrame(run)
        : run());

  let count = 0;
  let previewed = -1;
  let marks = new Set();
  let opMarks = new Map();
  let pending = false;
  const value = () => Number(range.value) - 1;

  const at = (index) => {
    const f = count > 1 ? index / (count - 1) : 0;

    return `calc(var(--thumb) / 2 + (100% - var(--thumb)) * ${f})`;
  };

  function describe(index) {
    const valuetext = label?.(index) ?? `Pass ${index + 1} of ${count}`;
    range.setAttribute("aria-valuetext", valuetext);
    if (text) text.textContent = valuetext;
    if (prev) prev.disabled = index <= 0;
    if (next) next.disabled = index >= count - 1;
  }

  function showPreview(index) {
    if (!tip || !preview || index < 0 || index >= count) return;
    previewed = index;

    const { title, lines = [] } = preview(index);
    const head = document.createElement("strong");
    head.textContent = title;

    tip.replaceChildren(
      head,
      ...lines.map((line) => {
        const span = document.createElement("span");
        span.textContent = line;

        return span;
      }),
    );

    tip.style.left = at(index);
    tip.hidden = false;
  }

  function hidePreview() {
    if (!tip) return;
    previewed = -1;
    tip.hidden = true;
  }

  function commit({ now = false } = {}) {
    describe(value());

    if (now) {
      pending = false;
      onSelect(value());

      return;
    }

    if (pending) return;
    pending = true;

    later(() => {
      if (!pending) return;
      pending = false;
      onSelect(value());
    });
  }

  function step(delta) {
    const target = Math.max(0, Math.min(count - 1, value() + delta));
    if (target === value()) return;
    range.value = String(target + 1);
    commit({ now: true });
    if (previewed >= 0) showPreview(target);
  }

  range.addEventListener("input", () => {
    commit();

    if (previewed >= 0 || document.activeElement === range) {
      showPreview(value());
    }
  });

  range.addEventListener("change", () => {
    if (pending) commit({ now: true });
  });

  range.addEventListener("focus", () => showPreview(value()));
  range.addEventListener("blur", hidePreview);

  range.addEventListener("pointermove", (e) => {
    const box = range.getBoundingClientRect();
    if (!box.width || count < 1) return;

    const f = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width));
    showPreview(Math.round(f * (count - 1)));
  });

  range.addEventListener("pointerleave", () => {
    if (document.activeElement === range) showPreview(value());
    else hidePreview();
  });

  prev?.addEventListener("click", () => step(-1));
  next?.addEventListener("click", () => step(1));

  return {
    setPasses(passes) {
      count = passes.length;
      range.min = "1";
      range.max = String(Math.max(1, count));
      root.style.setProperty("--count", String(count));

      ticks?.replaceChildren(
        ...passes.map((pass, i) => {
          const tick = document.createElement("span");

          tick.className = [
            pass.failed ? "failed" : "",
            marks.has(i) ? "bp" : "",
            opMarks.get(i) ?? "",
          ]
            .filter(Boolean)
            .join(" ");

          tick.style.left = at(i);

          if (pass.share !== null && pass.share !== undefined) {
            tick.style.setProperty("--share", String(pass.share));
          }

          return tick;
        }),
      );

      hidePreview();
    },
    setOpMarks(list) {
      opMarks = new Map(list.map(({ pass, kind }) => [pass, kind]));

      [...(ticks?.children ?? [])].forEach((tick, i) => {
        tick.classList.toggle("op", opMarks.get(i) === "op");
        tick.classList.toggle("life", opMarks.get(i) === "life");
      });
    },
    setMarks(indices) {
      const hit = new Set(indices);
      marks = hit;

      [...(ticks?.children ?? [])].forEach((tick, i) =>
        tick.classList.toggle("bp", hit.has(i)),
      );
    },
    setValue(index) {
      pending = false;
      range.value = String(index + 1);

      [...(ticks?.children ?? [])].forEach((tick, i) =>
        tick.classList.toggle("current", i === index),
      );

      describe(index);
      if (previewed >= 0) showPreview(previewed);
    },
    refreshPreview(index) {
      if (previewed >= 0 && (index === undefined || index === previewed)) {
        showPreview(previewed);
      }
    },
    step,
    get value() {
      return value();
    },
  };
}
