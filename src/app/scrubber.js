// The pass scrubber: a range slider over the passes of a trace (1 … N) that
// replaces a row of cards, so a long trace costs one line of height. Hovering
// the track, or focusing the slider, previews a pass (its name, IR size, how
// many ops it changed) without loading it; the arrow keys, Home / End and a
// drag select one.
//
// Markup: `root` holds input[type=range].scrub-range, .scrub-ticks,
// .scrub-preview, .scrub-label and optional .scrub-prev / .scrub-next buttons.

/**
 * @param {HTMLElement} root
 * @param {{
 *   onSelect: (index: number) => void,
 *   preview?: (index: number) => { title: string, lines?: string[] },
 *   label?: (index: number) => string,
 *   schedule?: (run: () => void) => void,
 * }} options
 *   `schedule` coalesces the selections a drag fires (one per animation
 *   frame by default).
 */
export function createScrubber(root, { onSelect, preview, label, schedule } = {}) {
  const range = root.querySelector(".scrub-range");
  const ticks = root.querySelector(".scrub-ticks");
  const tip = root.querySelector(".scrub-preview");
  const text = root.querySelector(".scrub-label");
  const prev = root.querySelector(".scrub-prev");
  const next = root.querySelector(".scrub-next");
  const later =
    schedule ??
    ((run) =>
      typeof requestAnimationFrame === "function" ? requestAnimationFrame(run) : run());

  let count = 0;
  let previewed = -1;
  let marks = new Set();
  let pending = false;
  const value = () => Number(range.value) - 1;

  // Where pass `index` sits along the track, as a CSS length: the thumb's
  // centre travels from half a thumb in to half a thumb from the end.
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

  // Selects pass `index` right away (keys, buttons) or on the next frame
  // (drags fire one input event per pixel).
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
    if (previewed >= 0 || document.activeElement === range) showPreview(value());
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
    // Sets the passes: one entry per pass, `{ failed, share }`, where share
    // (0 … 1) is its time against the slowest, or null when untimed.
    setPasses(passes) {
      count = passes.length;
      range.min = "1";
      range.max = String(Math.max(1, count));
      root.style.setProperty("--count", String(count));
      ticks?.replaceChildren(
        ...passes.map((pass, i) => {
          const tick = document.createElement("span");
          tick.className = [pass.failed ? "failed" : "", marks.has(i) ? "bp" : ""]
            .filter(Boolean)
            .join(" ");
          tick.style.left = at(i);
          if (pass.share !== null && pass.share !== undefined)
            tick.style.setProperty("--share", String(pass.share));
          return tick;
        }),
      );
      hidePreview();
    },
    // Marks the passes where a breakpoint hits (indices), replacing the last
    // marks. They survive `setValue` and are cleared by `setPasses`.
    setMarks(indices) {
      const hit = new Set(indices);
      marks = hit;
      [...(ticks?.children ?? [])].forEach((tick, i) =>
        tick.classList.toggle("bp", hit.has(i)),
      );
    },
    // Shows pass `index` as selected without calling onSelect.
    setValue(index) {
      pending = false;
      range.value = String(index + 1);
      [...(ticks?.children ?? [])].forEach((tick, i) =>
        tick.classList.toggle("current", i === index),
      );
      describe(index);
      if (previewed >= 0) showPreview(previewed);
    },
    // Redraws the open preview (its data arrived later).
    refreshPreview(index) {
      if (previewed >= 0 && (index === undefined || index === previewed))
        showPreview(previewed);
    },
    step,
    get value() {
      return value();
    },
  };
}
