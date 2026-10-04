// The field guide: every chapter on one scrolling page, grouped into parts,
// with a sticky contents list that follows the chapter being read.
// Chapters are the <article class="lesson"> elements in #docs-view.

export function bindGuide(root) {
  const lessons = [...root.querySelectorAll(".lesson")];
  const toc = root.querySelector("#guide-toc");
  const map = root.querySelector("#course-map");
  const key = (lesson) => lesson.id.replace(/^lesson-/, "");
  let current = -1;

  const tocButtons = [];
  let group = "";
  for (const [index, lesson] of lessons.entries()) {
    if (lesson.dataset.group !== group) {
      group = lesson.dataset.group;
      const heading = document.createElement("li");
      heading.className = "group";
      heading.textContent = group;
      toc.append(heading);
      // The same part heading in the page itself (the first part is the
      // welcome, which introduces itself).
      if (index > 0) {
        const part = document.createElement("div");
        part.className = "guide-part";
        part.textContent = group;
        lesson.before(part);
      }
    }
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = lesson.dataset.title;
    button.addEventListener("click", () => go(index));
    item.append(button);
    toc.append(item);
    tocButtons.push(button);
  }

  // The welcome page's course map: every chapter after the first two.
  map?.replaceChildren(
    ...lessons.slice(2).map((lesson, i) => {
      const button = document.createElement("button");
      button.type = "button";
      const label = document.createElement("span");
      label.className = "n";
      label.textContent =
        lesson.dataset.group === "Lessons"
          ? `Lesson ${i + 1}`
          : lesson.dataset.group;
      button.append(label, lesson.dataset.title);
      button.addEventListener("click", () => go(i + 2));
      return button;
    }),
  );

  root.addEventListener("click", (e) => {
    const link = e.target.closest("[data-lesson-link]");
    if (!link) return;
    e.preventDefault();
    const index = lessons.findIndex(
      (lesson) => key(lesson) === link.dataset.lessonLink,
    );
    if (index >= 0) go(index);
  });

  function mark(index) {
    if (index === current) return;
    current = index;
    tocButtons.forEach((button, i) =>
      button.setAttribute("aria-current", i === index ? "step" : "false"),
    );
    // Keep the active entry visible by scrolling the list alone:
    // scrollIntoView would also stop the page's own smooth scroll.
    const button = tocButtons[index];
    for (const box of [toc, toc.parentElement]) {
      const outer = box.getBoundingClientRect();
      const inner = button.getBoundingClientRect();
      if (inner.left < outer.left) box.scrollLeft -= outer.left - inner.left;
      if (inner.right > outer.right)
        box.scrollLeft += inner.right - outer.right;
      if (inner.top < outer.top) box.scrollTop -= outer.top - inner.top;
      if (inner.bottom > outer.bottom)
        box.scrollTop += inner.bottom - outer.bottom;
    }
  }

  function go(index) {
    if (index < 0 || index >= lessons.length) return;
    mark(index);
    lessons[index].scrollIntoView({ block: "start" });
  }

  // The chapter being read is the last one whose top has passed the upper
  // third of the view; at the very bottom it is the last chapter.
  function spy() {
    const line = root.getBoundingClientRect().top + root.clientHeight / 3;
    let index = 0;
    for (const [i, lesson] of lessons.entries())
      if (lesson.getBoundingClientRect().top <= line) index = i;
    if (root.scrollTop + root.clientHeight >= root.scrollHeight - 2)
      index = lessons.length - 1;
    mark(index);
  }
  let frame = 0;
  root.addEventListener("scroll", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(spy);
  });

  mark(0);
  return { step: (delta) => go(current + delta) };
}
