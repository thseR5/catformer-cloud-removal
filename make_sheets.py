"""Tile the exported test PNGs into contact sheets: 10 locations per sheet, one row each
(3 cloudy inputs, prediction, ground truth). Usage: python make_sheets.py <test_pngs dir> <out dir>"""
import csv, os, sys

from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
PER_SHEET, TILE, PAD, LABEL_W, HEAD = 10, 256, 6, 150, 34
COLS = [("input_t1.png", "Input t1"), ("input_t2.png", "Input t2"), ("input_t3.png", "Input t3"),
        ("prediction.png", "Prediction"), ("ground_truth.png", "Ground truth")]
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)

rows = list(csv.DictReader(open(os.path.join(src, "index.csv"))))
os.makedirs(out, exist_ok=True)
n_sheets = (len(rows) + PER_SHEET - 1) // PER_SHEET
W = LABEL_W + len(COLS) * (TILE + PAD) + PAD
for s in range(n_sheets):
    chunk = rows[s * PER_SHEET:(s + 1) * PER_SHEET]
    H = HEAD + len(chunk) * (TILE + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for c, (_, title) in enumerate(COLS):
        d.text((LABEL_W + PAD + c * (TILE + PAD) + TILE // 2, HEAD // 2), title, fill="black", font=font, anchor="mm")
    for r, row in enumerate(chunk):
        y = HEAD + PAD + r * (TILE + PAD)
        clouds = row["input_cloud_frac"].split()
        d.text((8, y + 20), f"patch {int(row['patch']):03d}", fill="black", font=font)
        d.text((8, y + 50), f"PSNR {float(row['psnr']):.2f} dB", fill="black", font=small)
        d.text((8, y + 74), f"MAE {float(row['mae']):.4f}", fill="black", font=small)
        d.text((8, y + 104), "input cloud:", fill="dimgray", font=small)
        d.text((8, y + 124), " / ".join(f"{float(c):.0%}" for c in clouds), fill="dimgray", font=small)
        for c, (fname, _) in enumerate(COLS):
            sheet.paste(Image.open(os.path.join(src, row["folder"], fname)).convert("RGB"),
                        (LABEL_W + PAD + c * (TILE + PAD), y))
    first, last = int(chunk[0]["patch"]), int(chunk[-1]["patch"])
    sheet.save(os.path.join(out, f"sheet_{s + 1:02d}_patches_{first:03d}-{last:03d}.png"), optimize=True)
print("sheets:", n_sheets, "->", out)
