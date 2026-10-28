// Canvas drawing for mosaics. Same look as engine/render.py: bevelled brick, round stud, dark gap.
const Bricks = (() => {
  const GAP = "#161618";
  const tiles = new Map();

  const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const mix = (c, t, f) => `rgb(${c.map((v, i) => Math.round(v + (t[i] - v) * f)).join(",")})`;
  const lum = c => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

  function tile(hex, cell) {
    const key = hex + cell;
    if (tiles.has(key)) return tiles.get(key);
    const ss = 3, s = cell * ss;
    const cv = document.createElement("canvas");
    cv.width = cv.height = s;
    const g = cv.getContext("2d");
    const c = rgb(hex);
    g.fillStyle = GAP; g.fillRect(0, 0, s, s);
    const gap = Math.max(ss, Math.round(s * 0.045)), r = s * 0.08, e = ss * 1.5;
    const rr = (x, y, w, h, fill) => { g.fillStyle = fill; g.beginPath(); g.roundRect(x, y, w, h, r); g.fill(); };
    rr(gap, gap, s - 2 * gap, s - 2 * gap, mix(c, [0, 0, 0], 0.22));
    rr(gap, gap, s - 2 * gap - e, s - 2 * gap - e, mix(c, [255, 255, 255], 0.18));
    rr(gap + e, gap + e, s - 2 * (gap + e), s - 2 * (gap + e), `rgb(${c})`);
    const cx = s / 2, rad = s * 0.29, off = s * 0.045;
    const circle = (x, y, rd, fill) => { g.fillStyle = fill; g.beginPath(); g.arc(x, y, rd, 0, Math.PI * 2); g.fill(); };
    circle(cx + off, cx + off * 1.4, rad, mix(c, [0, 0, 0], 0.38));
    circle(cx, cx, rad, mix(c, [255, 255, 255], 0.10));
    circle(cx + off * 0.4, cx + off * 0.4, rad * 0.86, `rgb(${c})`);
    g.strokeStyle = "rgba(255,255,255,.6)"; g.lineWidth = Math.max(ss, rad * 0.16);
    g.beginPath(); g.arc(cx, cx, rad - g.lineWidth, Math.PI * 0.9, Math.PI * 1.6); g.stroke();
    const out = document.createElement("canvas");
    out.width = out.height = cell;
    const og = out.getContext("2d");
    og.imageSmoothingQuality = "high";
    og.drawImage(cv, 0, 0, cell, cell);
    tiles.set(key, out);
    return out;
  }

  // view: "bricks" | "flat" | "numbers". hexOf: code -> "#RRGGBB".
  function draw(canvas, grid, hexOf, view, cells = null) {
    const n = grid.length;
    const cell = view === "numbers" ? Math.max(18, Math.ceil(900 / n)) : Math.max(8, Math.ceil(900 / n));
    const size = n * cell;
    if (canvas.width !== size) { canvas.width = canvas.height = size; cells = null; }
    canvas.dataset.cell = cell;
    const g = canvas.getContext("2d");
    const list = cells || grid.flatMap((row, y) => row.map((_, x) => [y, x]));
    for (const [y, x] of list) drawCell(g, grid[y][x], hexOf, view, x, y, cell);
    if (view === "numbers") drawLines(g, n, cell);
  }

  function drawCell(g, code, hexOf, view, x, y, cell) {
    const hex = hexOf(code);
    if (view === "bricks") { g.drawImage(tile(hex, cell), x * cell, y * cell); return; }
    g.fillStyle = hex; g.fillRect(x * cell, y * cell, cell, cell);
    if (view === "numbers") {
      g.fillStyle = lum(rgb(hex)) > 140 ? "#000" : "#fff";
      g.font = `600 ${Math.round(cell * 0.46)}px system-ui, sans-serif`;
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(code.slice(1), x * cell + cell / 2, y * cell + cell / 2 + 1);
    }
  }

  function drawLines(g, n, cell) {
    for (let i = 0; i <= n; i++) {
      const major = i % 8 === 0;
      g.strokeStyle = major ? "#111" : "rgba(0,0,0,.35)";
      g.lineWidth = major ? 3 : 1;
      const p = i * cell;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, n * cell); g.moveTo(0, p); g.lineTo(n * cell, p); g.stroke();
    }
  }

  return { draw };
})();
