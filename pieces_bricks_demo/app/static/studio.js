// Mosaic Studio: photo -> crop -> generate -> edit bricks -> live counts -> download.
const $ = id => document.getElementById(id);
const S = {
  palette: [], byCode: {},
  image: null, img: null, crop: null, size: 48,
  result: null, grid: null, original: null,
  view: "bricks", brush: "C01", undo: [],
};

// ------------------------------------------------------------- utils
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.blob();
}
function busy(on, text = "Working…") { $("busy").hidden = !on; $("busyText").textContent = text; }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => t.hidden = true, 3500); }
const hexOf = code => S.byCode[code].hex;

// ------------------------------------------------------------- init
async function init() {
  const pal = await api("/api/palette");
  S.palette = pal.colors;
  pal.colors.forEach(c => S.byCode[c.code] = c);
  buildPalette();
  buildSizes();

  const samples = await api("/api/samples");
  $("samples").innerHTML = samples.map(n =>
    `<button title="${n}" data-n="${n}"><img src="/api/samples/${encodeURIComponent(n)}/thumb.jpg" alt="${n}"></button>`).join("");
  $("samples").onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll(".samples button").forEach(x => x.classList.toggle("on", x === b));
    loadImage(() => api(`/api/samples/${encodeURIComponent(b.dataset.n)}`, { method: "POST" }));
  };
  $("file").onchange = e => {
    const f = e.target.files[0]; if (!f) return;
    document.querySelectorAll(".samples button").forEach(x => x.classList.remove("on"));
    loadImage(async () => {
      const blob = await shrinkPhoto(f);
      const fd = new FormData(); fd.append("file", blob, f.name);
      const meta = await api("/api/detect", { method: "POST", body: fd });
      return { ...meta, blob, name: f.name, display_url: URL.createObjectURL(blob) };
    });
  };
  $("generate").onclick = generate;
  $("autoCrop").onclick = () => setCrop(S.image.suggested_crop);
  $("fullCrop").onclick = () => { const s = Math.min(S.image.width, S.image.height); setCrop([(S.image.width - s) / 2, (S.image.height - s) / 2, s]); };
  $("zoom").oninput = e => zoomTo(+e.target.value);
  document.querySelectorAll("#views button").forEach(b => b.onclick = () => {
    S.view = b.dataset.v;
    document.querySelectorAll("#views button").forEach(x => x.classList.toggle("on", x === b));
    redraw();
  });
  $("showOrig").onchange = e => $("origFig").hidden = !e.target.checked;
  $("undo").onclick = undo;
  $("reset").onclick = () => { if (!confirm("Discard all brick edits?")) return; S.grid = S.original.map(r => [...r]); S.undo = []; afterEdit(); redraw(); };
  $("download").onclick = download;
  bindCrop();
  bindPaint();

  // /?sample=p5_man_colour_bg.png&size=48 opens straight into a result (handy for demo links)
  const q = new URLSearchParams(location.search);
  if (q.get("size")) $("sizes").querySelector(`[data-n="${q.get("size")}"]`)?.click();
  if (q.get("sample")) $("samples").querySelector(`[data-n="${q.get("sample")}"]`)?.click();
}

function buildPalette() {
  $("palette").innerHTML = S.palette.map(c =>
    `<button class="swatch" style="background:${c.hex}" data-c="${c.code}" title="${c.code} ${c.name}"></button>`).join("");
  $("palette").onclick = e => { const b = e.target.closest(".swatch"); if (b) setBrush(b.dataset.c); };
  setBrush("C01");
}
function setBrush(code) {
  S.brush = code;
  document.querySelectorAll(".swatch").forEach(s => s.classList.toggle("on", s.dataset.c === code));
  $("brushName").textContent = `${code} ${S.byCode[code].name}`;
}
function buildSizes() {
  $("sizes").innerHTML = [32, 48, 64, 96].map(n => `<button data-n="${n}" class="${n === S.size ? "on" : ""}">${n}×${n}</button>`).join("");
  $("sizes").onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    S.size = +b.dataset.n;
    document.querySelectorAll("#sizes button").forEach(x => x.classList.toggle("on", x === b));
  };
}

// Uploads are capped at 2000 px (as an oriented JPEG) before leaving the browser: serverless
// hosts limit request bodies to ~4.5 MB, and the engine works at 1152 px anyway.
const MAX_UPLOAD_SIDE = 2000;
async function shrinkPhoto(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const k = Math.min(1, MAX_UPLOAD_SIDE / Math.max(bmp.width, bmp.height));
  const cv = document.createElement("canvas");
  cv.width = Math.round(bmp.width * k); cv.height = Math.round(bmp.height * k);
  const g = cv.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(bmp, 0, 0, cv.width, cv.height);
  return new Promise(res => cv.toBlob(res, "image/jpeg", 0.92));
}

// ------------------------------------------------------------- crop
async function loadImage(fetcher) {
  busy(true, "Detecting faces…");
  try {
    if (S.image?.blob) URL.revokeObjectURL(S.image.display_url);
    S.image = await fetcher();
    S.img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = S.image.display_url; });
    $("cropCard").hidden = $("genCard").hidden = false;
    setCrop(S.image.suggested_crop);
    await generate();
  } catch (e) { toast(e.message); }
  finally { busy(false); }
}

function setCrop([x, y, s]) {
  const W = S.image.width, H = S.image.height;
  s = Math.max(16, Math.min(s, W, H));
  S.crop = [Math.round(Math.min(Math.max(0, x), W - s)), Math.round(Math.min(Math.max(0, y), H - s)), Math.round(s)];
  const max = Math.min(W, H), min = Math.max(48, max * 0.08);
  $("zoom").value = Math.round(1000 * (max - S.crop[2]) / (max - min));
  drawCrop();
}
function zoomTo(v) {
  const max = Math.min(S.image.width, S.image.height), min = Math.max(48, max * 0.08);
  const s = max - (max - min) * v / 1000;
  const [x, y, old] = S.crop;
  setCrop([x + old / 2 - s / 2, y + old / 2 - s / 2, s]);
}

function drawCrop() {
  const cv = $("cropCanvas"), img = S.img;
  const maxW = cv.parentElement.clientWidth, k = Math.min(maxW / img.width, 360 / img.height);
  cv.width = Math.round(img.width * k); cv.height = Math.round(img.height * k);
  cv._k = k * img.width / S.image.width;  // canvas px per original px
  const g = cv.getContext("2d"), [x, y, s] = S.crop.map(v => v * cv._k);
  g.drawImage(img, 0, 0, cv.width, cv.height);
  g.fillStyle = "rgba(0,0,0,.55)";
  g.beginPath(); g.rect(0, 0, cv.width, cv.height); g.rect(x, y, s, s); g.fill("evenodd");
  g.strokeStyle = "rgba(255,255,255,.35)"; g.lineWidth = 1;
  for (const f of S.image.faces) { const [fx, fy, fw, fh] = f.box.map(v => v * cv._k); g.strokeRect(fx, fy, fw, fh); }
  g.strokeStyle = "#fff"; g.lineWidth = 2; g.strokeRect(x + 1, y + 1, s - 2, s - 2);
  g.strokeStyle = "rgba(255,255,255,.4)"; g.lineWidth = 1;
  for (const t of [1, 2]) { g.beginPath(); g.moveTo(x + s * t / 3, y); g.lineTo(x + s * t / 3, y + s); g.moveTo(x, y + s * t / 3); g.lineTo(x + s, y + s * t / 3); g.stroke(); }
}

function bindCrop() {
  const cv = $("cropCanvas");
  let start = null;
  cv.onpointerdown = e => { cv.setPointerCapture(e.pointerId); cv.classList.add("drag"); start = [e.clientX, e.clientY, ...S.crop]; };
  cv.onpointermove = e => {
    if (!start) return;
    const k = cv._k * cv.clientWidth / cv.width;
    setCrop([start[2] + (e.clientX - start[0]) / k, start[3] + (e.clientY - start[1]) / k, start[4]]);
  };
  cv.onpointerup = cv.onpointercancel = () => { start = null; cv.classList.remove("drag"); };
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    zoomTo(Math.min(1000, Math.max(0, +$("zoom").value - Math.sign(e.deltaY) * 40)));
  }, { passive: false });
  window.addEventListener("resize", () => S.img && drawCrop());
}

// ------------------------------------------------------------- generate
async function generate() {
  if (!S.image) return;
  busy(true, `Building ${S.size}×${S.size} mosaic…`);
  try {
    // Stateless: each request carries the photo itself, or the sample's name.
    const fd = new FormData();
    fd.append("params", JSON.stringify({ sample: S.image.sample || null, size: S.size, crop: S.crop, dither: $("dither").checked }));
    if (!S.image.sample) fd.append("file", S.image.blob, S.image.name);
    const r = await api("/api/convert", { method: "POST", body: fd });
    S.result = r;
    S.grid = r.grid.map(row => [...row]);
    S.original = r.grid.map(row => [...row]);
    S.undo = [];
    $("orig").src = r.crop_image;
    $("empty").hidden = true; $("result").hidden = false; $("countsCard").hidden = false;
    afterEdit(); redraw();
  } catch (e) { toast(e.message); }
  finally { busy(false); }
}

function redraw(cells = null) {
  Bricks.draw($("mosaic"), S.grid, hexOf, S.view, S.view === "numbers" ? null : cells);
  const n = S.grid.length;
  $("caption").textContent = `${n} × ${n} · ${(n * n).toLocaleString()} bricks · ${S.result.settings.dither ? "dithered" : "no dithering"}`;
}

// ------------------------------------------------------------- editing
function cellAt(e) {
  const cv = $("mosaic"), r = cv.getBoundingClientRect(), n = S.grid.length;
  const x = Math.floor((e.clientX - r.left) / r.width * n), y = Math.floor((e.clientY - r.top) / r.height * n);
  return x >= 0 && y >= 0 && x < n && y < n ? [y, x] : null;
}
function bindPaint() {
  const cv = $("mosaic");
  let stroke = null;
  const paint = e => {
    const c = cellAt(e); if (!c) return;
    const [y, x] = c, old = S.grid[y][x];
    if (old === S.brush) return;
    stroke.push([y, x, old]);
    S.grid[y][x] = S.brush;
    redraw([[y, x]]);
    afterEdit(S.brush);
  };
  cv.onpointerdown = e => {
    if (!S.grid) return;
    if (e.altKey) { const c = cellAt(e); if (c) setBrush(S.grid[c[0]][c[1]]); return; }
    cv.setPointerCapture(e.pointerId);
    stroke = []; paint(e);
  };
  cv.onpointermove = e => stroke && paint(e);
  cv.onpointerup = cv.onpointercancel = () => { if (stroke?.length) S.undo.push(stroke); stroke = null; afterEdit(); };
  document.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); undo(); } });
}
function undo() {
  const stroke = S.undo.pop(); if (!stroke) return;
  for (const [y, x, old] of stroke.reverse()) S.grid[y][x] = old;
  redraw(stroke.map(([y, x]) => [y, x]));
  afterEdit();
}

function afterEdit(flash) {
  const counts = {};
  for (const row of S.grid) for (const c of row) counts[c] = (counts[c] || 0) + 1;
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const total = rows.reduce((s, [, q]) => s + q, 0);
  $("total").textContent = total.toLocaleString();
  $("used").textContent = rows.length;
  $("counts").innerHTML = rows.map(([code, q]) => {
    const c = S.byCode[code];
    return `<tr data-c="${code}" class="${code === flash ? "flash" : ""}"><td><div class="sw" style="background:${c.hex}"></div></td>
      <td class="code">${code}</td><td>${c.name}</td><td>${q}</td></tr>`;
  }).join("");
  $("counts").onclick = e => { const tr = e.target.closest("tr"); if (tr) setBrush(tr.dataset.c); };
  const edited = S.grid.reduce((s, row, y) => s + row.filter((c, x) => c !== S.original[y][x]).length, 0);
  $("undo").disabled = !S.undo.length;
  $("reset").disabled = !edited;
}

async function download() {
  busy(true, "Rendering pack…");
  try {
    const blob = await api("/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid: S.grid, settings: S.result.settings, source: S.result.source }),
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `mosaic_${S.grid.length}x${S.grid.length}.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (e) { toast(e.message); }
  finally { busy(false); }
}

init().catch(e => toast(e.message));
