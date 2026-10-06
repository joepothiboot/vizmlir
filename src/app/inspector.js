export function createInspector(
  root,
  { toggle = null, closeButton = null, onShow, onChange, tab: initial } = {},
) {
  const tabs = [...root.querySelectorAll('[role="tab"][data-tab]')];

  const panels = new Map(
    [...root.querySelectorAll('[role="tabpanel"][data-panel]')].map((panel) => [
      panel.dataset.panel,
      panel,
    ]),
  );

  const names = tabs.map((button) => button.dataset.tab);
  let active = names.includes(initial) ? initial : names[0];
  let returnTo = null;

  const visibleTabs = () => tabs.filter((button) => !button.hidden);

  function render() {
    for (const button of tabs) {
      const on = button.dataset.tab === active;
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
    }

    for (const [name, panel] of panels) panel.hidden = name !== active;
  }

  function changed() {
    onChange?.({ open: !root.hidden, tab: active });
  }

  function setOpen(open) {
    root.hidden = !open;
    toggle?.setAttribute("aria-expanded", String(open));

    root.dispatchEvent(
      new CustomEvent("inspector-toggle", { detail: { open } }),
    );
  }

  function open({ from } = {}) {
    if (from) returnTo = from;

    const wasOpen = !root.hidden;
    if (!wasOpen) setOpen(true);

    if (!wasOpen) {
      onShow?.(active);
      changed();
    }
  }

  function show(name, { from, open: reveal = true } = {}) {
    if (!panels.has(name)) return;

    const switched = name !== active;
    active = name;
    render();

    if (reveal && root.hidden) {
      open({ from });

      return;
    }

    if (from) returnTo = from;
    if (switched && !root.hidden) onShow?.(active);
    if (switched) changed();
  }

  function focusBack() {
    const target = returnTo?.isConnected ? returnTo : null;
    target?.focus?.({ preventScroll: true });
    if (!target || document.activeElement !== target) toggle?.focus?.();
  }

  function close({ restoreFocus = true } = {}) {
    if (root.hidden) return false;

    const hadFocus = root.contains(document.activeElement);
    setOpen(false);
    if (restoreFocus || hadFocus) focusBack();
    changed();

    return true;
  }

  function toggleTab(name, options) {
    if (!root.hidden && active === name) close();
    else show(name, options);
  }

  function handleEscape() {
    return close();
  }

  for (const button of tabs) {
    button.addEventListener("click", () => show(button.dataset.tab));
  }

  root.addEventListener("keydown", (e) => {
    const target = e.target;

    if (e.key === "Escape") {
      if (target.matches?.("input, textarea, select")) return;
      e.preventDefault();
      e.stopPropagation();
      close();

      return;
    }

    if (target.getAttribute?.("role") !== "tab") return;

    const shown = visibleTabs();
    const at = shown.indexOf(target);

    const next = {
      ArrowRight: at + 1,
      ArrowLeft: at - 1,
      Home: 0,
      End: shown.length - 1,
    }[e.key];

    if (next === undefined) return;
    e.preventDefault();
    e.stopPropagation();

    const button = shown[(next + shown.length) % shown.length];
    show(button.dataset.tab);
    button.focus();
  });

  toggle?.addEventListener("click", () => {
    if (root.hidden) {
      open({ from: toggle });
      root.querySelector('[role="tab"][aria-selected="true"]')?.focus();
    } else {
      close();
    }
  });

  closeButton?.addEventListener("click", () => close());

  render();
  toggle?.setAttribute("aria-expanded", String(!root.hidden));

  return {
    open,
    close,
    show,
    toggleTab,
    handleEscape,
    get isOpen() {
      return !root.hidden;
    },
    get active() {
      return active;
    },
  };
}
