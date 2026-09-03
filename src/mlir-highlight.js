const tokenPattern = /\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|@[A-Za-z0-9_.$-]+|%[A-Za-z0-9_.$-]+|\b(?:module|func|func\.func|return|ins|outs|in|out|attributes|loc|dense|true|false)\b|\b(?:tensor|memref|vector|index|i[0-9]+|f(?:16|32|64)|bf16)\b|\b(?:[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)\b|\b[A-Za-z_][A-Za-z0-9_.-]*\b/g;

const keywords = new Set(['module', 'func', 'func.func', 'return', 'ins', 'outs', 'in', 'out', 'attributes', 'loc', 'dense', 'true', 'false']);
const types = new Set(['tensor', 'memref', 'vector', 'index', 'bf16']);

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function tokenClass(token) {
  if (token.startsWith('//')) return 'syntax-comment';
  if (token.startsWith('"') || token.startsWith("'")) return 'syntax-string';
  if (token.startsWith('%')) return 'syntax-ssa';
  if (token.startsWith('@')) return 'syntax-symbol';
  if (keywords.has(token)) return 'syntax-keyword';
  if (types.has(token) || /^[if][0-9]+$/.test(token)) return 'syntax-type';
  if (/^[0-9]/.test(token)) return 'syntax-number';
  if (token.includes('.')) return 'syntax-operation';
  return 'syntax-identifier';
}

export function highlightMlir(source) {
  let output = '';
  let cursor = 0;
  for (const match of source.matchAll(tokenPattern)) {
    const token = match[0];
    const start = match.index;
    output += escapeHtml(source.slice(cursor, start));
    output += `<span class="${tokenClass(token)}">${escapeHtml(token)}</span>`;
    cursor = start + token.length;
  }
  return output + escapeHtml(source.slice(cursor)) + '\n';
}

export function bindHighlighting(textarea, layer) {
  const sync = () => {
    layer.innerHTML = highlightMlir(textarea.value);
    layer.scrollTop = textarea.scrollTop;
    layer.scrollLeft = textarea.scrollLeft;
  };
  textarea.addEventListener('input', sync);
  textarea.addEventListener('scroll', sync);
  textarea.addEventListener('focus', () => layer.classList.add('focused'));
  textarea.addEventListener('blur', () => layer.classList.remove('focused'));
  sync();
}
