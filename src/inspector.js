// The inspector: a non-modal drawer beside the visualization that holds every
// detail view (the pass diff, the explained line, timing, buffers, op counts,
// symbol history) as tabs, so the code and the picture stay in sight while
// you read them. Selecting something opens it; Escape closes it and returns
// focus to what was selected.
//
// Markup: `root` holds buttons [role=tab][data-tab=name] and panels
// [role=tabpanel][data-panel=name]; `toggle` is the button that reopens it.

/**
 * @param {HTMLElement} root
 * @param {{
 *   toggle?: HTMLElement | null,
 *   closeButton?: HTMLElement | null,
 *   onShow?: (tab: string) => void,
 *   onChange?: (state: { open: boolean, tab: string }) => void,
 *   tab?: string,
 * }} options
 */
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
    root.dispatchEvent(new CustomEvent("inspector-toggle", { detail: { open } }));
  }

  // Opens the drawer (without moving focus). `from` is the element that was
  // selected, where focus goes back on close.
  function open({ from } = {}) {
    if (from) returnTo = from;
    const wasOpen = !root.hidden;
    if (!wasOpen) setOpen(true);
    if (!wasOpen) {
      onShow?.(active);
      changed();
    }
  }

  // Shows tab `name`, opening the drawer unless `open: false`.
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

  // Shows `name`, or closes the drawer when it already shows it.
  function toggleTab(name, options) {
    if (!root.hidden && active === name) close();
    else show(name, options);
  }

  // For the page's own Escape handling: closes the drawer if open.
  function handleEscape() {
    return close();
  }

  for (const button of tabs) {
    button.addEventListener("click", () => show(button.dataset.tab));
  }
  // Arrow keys move between tabs (WAI-ARIA tabs pattern).
  root.addEventListener("keydown", (e) => {
    const target = e.target;
    if (e.key === "Escape") {
      // Leave inputs to the page (Escape there leaves the field first).
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
      // Keyboard users land on the active tab.
      root.querySelector('[role="tab"][aria-selected="true"]')?.focus();
    } else close();
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
