import { SVG_NS } from "./constants.js";

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;

  return node;
}

export function svg(tag, attrs, title) {
  const node = document.createElementNS(SVG_NS, tag);

  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }

  if (title) {
    const titleNode = document.createElementNS(SVG_NS, "title");
    titleNode.textContent = title;
    node.append(titleNode);
  }

  return node;
}
