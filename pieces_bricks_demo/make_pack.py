"""Build the client pack PDF from the results chosen in config/pack.yaml.

  .venv/bin/python make_pack.py            # -> output/client_pack/PiecesBricks_Demo_Results.pdf
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from engine import load_palette
from engine.export import counts, read_matrix, validate

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

INK = colors.HexColor("#1d1c1a")
MUTED = colors.HexColor("#6e6a63")
LINE = colors.HexColor("#e4dfd6")
ACCENT = colors.HexColor("#c8352c")
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
FRAME_W = PAGE_W - 2 * MARGIN

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=INK, spaceAfter=4)
H3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9, leading=12.5, textColor=MUTED)
CENTER = ParagraphStyle("center", parent=SMALL, alignment=TA_CENTER)
STEP = ParagraphStyle("step", parent=BODY, leftIndent=16, firstLineIndent=-16, spaceAfter=6)

METHOD = [
    ("Find the faces.", "We locate every face in the photo automatically, along with the position of the eyes."),
    ("Crop a square.", "The mosaic is always square, so we frame the people: head and shoulders for one person, "
                       "the whole group for families."),
    ("Make the faces clearer.", "We gently brighten shadows on skin, which would otherwise turn into dark patches, "
                                "and sharpen the eyes so they stay visible as separate bricks."),
    ("Calm the background.", "Busy backgrounds are smoothed into flatter areas so they don't compete with the faces."),
    ("Shrink to the board.", "The photo is reduced to 48 × 48 squares. Each square takes the average colour of "
                             "its area of the photo."),
    ("Choose the brick colour.", "For each square we pick the closest of your 40 colours, using a colour-difference "
                                 "formula designed to match human vision. On faces only skin tones are allowed, so "
                                 "there are no green or grey patches on skin."),
    ("Count every brick.", "The counts always add up to exactly 2,304 bricks, and the same photo with the same "
                           "settings always gives exactly the same mosaic."),
]


def _img(path: Path, width: float, max_height: float | None = None) -> Image:
    w, h = PILImage.open(path).size
    height = width * h / w
    if max_height and height > max_height:
        width, height = max_height * w / h, max_height
    return Image(str(path), width=width, height=height)


def _jpeg(src: Path, dst: Path, max_width: int = 1400) -> Path:
    """ReportLab re-compresses PNGs poorly, so every image goes in as JPEG, capped at ~200 dpi
    for the printed width. Keeps the pack around 7 MB, small enough to email."""
    im = PILImage.open(src).convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), PILImage.LANCZOS)
    im.save(dst, "JPEG", quality=85)
    return dst


def count_table(rows: list[dict]) -> Table:
    per_col = max(12, -(-len(rows) // 3))
    cols = [rows[i:i + per_col] for i in range(0, len(rows), per_col)]
    total = sum(r["quantity"] for r in rows)

    def block(chunk):
        data = [["", "Code", "Colour", "Qty"]] + [["", r["code"], r["name"], f"{r['quantity']:,}"] for r in chunk]
        t = Table(data, colWidths=[5 * mm, 11 * mm, 27 * mm, 12 * mm], rowHeights=5.4 * mm)
        style = [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("TEXTCOLOR", (1, 1), (1, -1), MUTED),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]
        for i, r in enumerate(chunk, start=1):
            style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(r["hex"])))
            style.append(("BOX", (0, i), (0, i), 0.4, colors.HexColor("#00000033")))
        t.setStyle(TableStyle(style))
        return t

    blocks = [block(c) for c in cols]
    outer = Table([blocks], colWidths=[FRAME_W / 3] * len(blocks), hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    footer = Paragraph(f"<b>Total: {total:,} bricks</b> in {len(rows)} colours", BODY)
    return KeepTogether([outer, Spacer(1, 3 * mm), footer])


def photo_section(entry: dict, idx: int, tmp: Path, palette) -> list:
    d = OUTPUT / entry["result"]
    doc = read_matrix(d / "matrix.json")
    validate(doc, palette)
    rows = counts(doc["grid"], palette)
    n = doc["size"]
    assert sum(r["quantity"] for r in rows) == n * n
    pid, _, variant = entry["result"].rsplit("_", 2)
    style = "Smooth finish (no dithering)" if variant == "nodither" else "Textured finish (dithering)"

    story = [
        Paragraph(f"{idx}. {entry['title']}", H2),
        Paragraph(f"{n} × {n} board · {n * n:,} bricks · {len(rows)} colours · {style}", SMALL),
        Spacer(1, 4 * mm),
        _img(_jpeg(d / "preview.png", tmp / f"{entry['result']}_preview.jpg"), FRAME_W,
             PAGE_H - 2 * MARGIN - 52 * mm),
        Spacer(1, 3 * mm),
    ]
    if entry.get("note"):
        story.append(Paragraph(entry["note"], CENTER))
    story.append(PageBreak())

    story += [Paragraph(f"{idx}. {entry['title']}: comparison and brick list", H3)]
    compare = OUTPUT / f"{pid}_{n}_compare.png"
    if compare.exists():
        story += [_img(_jpeg(compare, tmp / f"{pid}_{n}_compare.jpg"), FRAME_W),
                  Paragraph("Left: the cropped photo. Middle: smooth finish. Right: textured finish.", CENTER),
                  Spacer(1, 5 * mm)]
    story += [Paragraph("Bricks needed", H3), count_table(rows), PageBreak()]
    return story


def build(plan_path: Path, out_pdf: Path) -> Path:
    plan = yaml.safe_load(plan_path.read_text())
    palette = load_palette()
    tmp = out_pdf.parent / "_img"
    tmp.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().strftime("%-d %B %Y")
    entries = plan["photos"]

    story = [
        Spacer(1, 8 * mm),
        Paragraph(plan.get("title", "Brick mosaic demo"), H1),
        Paragraph(f"{plan.get('client', '')}<br/>Prepared by {plan.get('prepared_by', '')} · {today}", SMALL),
        Spacer(1, 8 * mm),
        _img(_jpeg(OUTPUT / entries[0]["result"] / "preview.png", tmp / "cover.jpg"), FRAME_W * 0.78),
        Spacer(1, 8 * mm),
        Paragraph("We converted your photos into brick mosaics using only your 40 colours (C01–C40). For each photo "
                  "you'll find the mosaic preview, a side-by-side comparison with the original, and the exact number "
                  "of bricks of each colour.", BODY),
        PageBreak(),

        Paragraph("How your photos become mosaics", H2),
        Spacer(1, 2 * mm),
        *[Paragraph(f"<b>{i}. {head}</b> {text}", STEP) for i, (head, text) in enumerate(METHOD, 1)],
        Paragraph("Two finishes", H3),
        Paragraph("<b>Smooth</b> (no dithering) uses solid areas of colour. Faces look clean and calm; we recommend "
                  "it for portraits. <b>Textured</b> (dithering) mixes neighbouring colours to suggest in-between "
                  "shades. Gradients look smoother from a distance, but faces get speckled. Both are shown for every "
                  "photo.", BODY),
        Paragraph("What comes with every mosaic", H3),
        Paragraph("• a brick-style preview, as the finished board will look<br/>"
                  "• a numbered building grid: every square shows its colour number, with row and column numbers<br/>"
                  "• the exact brick list per colour, always adding up to the full board<br/>"
                  "• the mosaic as a data file, so it can be edited, reordered or reproduced exactly", BODY),
        Paragraph("Bigger boards for group photos", H3),
        Paragraph("On a 48 × 48 board a face in a family photo is only 6–7 bricks wide. For the two family photos we "
                  "also show a 96 × 96 version, where every face becomes recognisable.", BODY),
        PageBreak(),
    ]
    for i, e in enumerate(entries, 1):
        story += photo_section(e, i, tmp, palette)

    story += [Paragraph("Next steps and a few questions", H2), Spacer(1, 2 * mm),
              Paragraph("We'd be glad to walk you through these results live and convert any other photo you'd like to "
                        "see.", BODY)]
    if plan.get("confirmed"):
        story += [Paragraph("Confirmed with you", H3)]
        story += [Paragraph(f"• {c}", STEP) for c in plan["confirmed"]]
    if plan.get("open_questions"):
        story += [Paragraph("A few answers that will help us fine-tune the system", H3)]
        story += [Paragraph(f"<b>{i}.</b> {q}", STEP) for i, q in enumerate(plan["open_questions"], 1)]

    def on_page(c, d):
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(ACCENT)
        c.drawString(MARGIN, PAGE_H - 11 * mm, "PIECES & BRICKS")
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 11 * mm, "Brick mosaic demo")
        c.drawRightString(PAGE_W - MARGIN, 10 * mm, f"{d.page}")
        c.restoreState()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title=plan.get("title"),
                            author=plan.get("prepared_by"), subject=plan.get("client"))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    # Record exactly which results and settings went into this pack, for reproducing them live.
    manifest = {"built": today, "palette_version": palette.version,
                "results": [{"result": e["result"], **{k: read_matrix(OUTPUT / e["result"] / "matrix.json")[k]
                                                        for k in ("size", "settings", "source")}} for e in entries]}
    (out_pdf.parent / "pack_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "config" / "pack.yaml"))
    ap.add_argument("--out", default=str(OUTPUT / "client_pack" / "PiecesBricks_Demo_Results.pdf"))
    args = ap.parse_args()
    pdf = build(Path(args.plan), Path(args.out))
    print(f"Wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
