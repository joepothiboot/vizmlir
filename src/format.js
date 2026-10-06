export function direction(change, flat = "same") {
  if (change > 0) return "up";
  if (change < 0) return "down";

  return flat;
}

export function signOf(change) {
  if (change > 0) return "+";
  if (change < 0) return "−";

  return "±";
}

export function signedCount(change) {
  return change > 0 ? `+${change}` : `−${-change}`;
}
