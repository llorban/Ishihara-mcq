#!/usr/bin/env python3
"""
Generate 700x300 "Ishihara-style" MCQ plates and zip them.
Save as generate_ishihara_mcq.py and run: python generate_ishihara_mcq.py

Dependencies:
    pip install pillow numpy

Outputs:
    ./ishihara_mcq_output/  -> PNG files (mcq_plate_1.png ...)
    ./ishihara_mcq_output.zip -> ZIP archive of the PNGs
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, random, textwrap, zipfile
import argparse

# ---------------- Config ----------------
WIDTH, HEIGHT = 700, 300
DOT_PITCH = 14
JITTER = 6
MIN_DOT, MAX_DOT = 8, 16

OUT_DIR = "ishihara_mcq_output"
ZIP_NAME = OUT_DIR + ".zip"
# palettes: foreground (figure) and background (surround)
PALETTES = {
    "classic": {"fg":[(200,30,30),(220,60,60),(180,40,40)], "bg":[(100,160,60),(120,180,80),(80,140,50)]},
    "blue_orange": {"fg":[(200,150,60),(220,170,80)], "bg":[(100,120,200),(120,140,220),(70,100,180)]},
    "mono": {"fg":[(60,60,160),(80,80,180)], "bg":[(200,200,200),(220,220,220),(240,240,240)]}
}

# ---------------- Helpers ----------------
def find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()

def text_mask(text, w, h, padding=20):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if FONT_PATH:
        size = int(h * 0.18)
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, width=40)
    # reduce font until it fits
    while True:
        wmax = 0; hsum = 0
        for ln in lines:
            bbox = draw.textbbox((0,0), ln, font=font)
            wmax = max(wmax, bbox[2]-bbox[0])
            hsum += bbox[3]-bbox[1] + 4
        if (wmax < w - 2*padding) and (hsum < h - 2*padding):
            break
        if FONT_PATH and getattr(font, "size", None) and font.size > 10:
            font = ImageFont.truetype(FONT_PATH, max(10, font.size - 2))
        else:
            break
    total_h = sum([draw.textbbox((0,0), ln, font=font)[3] - draw.textbbox((0,0), ln, font=font)[1] + 4 for ln in lines])
    y = (h - total_h)/2
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        x = (w - (bbox[2]-bbox[0]))/2
        draw.text((x, y), ln, fill=255, font=font)
        y += (bbox[3]-bbox[1]) + 4
    return mask.filter(ImageFilter.GaussianBlur(radius=1.2))

def make_mcq_plate(mcq, filename, palette="classic"):
    pal = PALETTES.get(palette, PALETTES["classic"])
    fg_cols = pal["fg"]; bg_cols = pal["bg"]
    img = Image.new("RGB", (WIDTH, HEIGHT), (240,240,240))
    draw = ImageDraw.Draw(img)
    dot_area_w = min(WIDTH, HEIGHT)
    text_area_x = dot_area_w
    figure_text = mcq.get("figure","?")
    mask = text_mask(figure_text, dot_area_w, HEIGHT)
    mask_np = np.array(mask)/255.0
    cols = int(dot_area_w / DOT_PITCH) + 3
    rows = int(HEIGHT / DOT_PITCH) + 3
    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - dot_area_w)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            if cx < -DOT_PITCH or cy < -DOT_PITCH or cx > dot_area_w + DOT_PITCH: continue
            jx = cx + random.randint(-JITTER, JITTER)
            jy = cy + random.randint(-JITTER, JITTER)
            if jx < 0 or jy < 0 or jx >= dot_area_w or jy >= HEIGHT: continue
            rr = random.randint(MIN_DOT, MAX_DOT)
            v = mask_np[int(jy), int(jx)]
            prob_fg = v
            if random.random() < 0.03:
                is_fg = not (random.random() < prob_fg)
            else:
                is_fg = (random.random() < prob_fg)
            col = random.choice(fg_cols) if is_fg else random.choice(bg_cols)
            draw.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)
    draw.ellipse([-2, -2, dot_area_w+2, HEIGHT+2], outline=(200,200,200))
    panel_w = WIDTH - text_area_x
    panel = Image.new("RGBA", (panel_w, HEIGHT), (255,255,255,230))
    pd = ImageDraw.Draw(panel)
    if FONT_PATH:
        qfont = ImageFont.truetype(FONT_PATH, 18)
        afont = ImageFont.truetype(FONT_PATH, 16)
    else:
        qfont = ImageFont.load_default()
        afont = ImageFont.load_default()
    q_lines = textwrap.wrap(mcq.get("question","Question?"), width=36)
    y = 12
    for ln in q_lines:
        pd.text((12, y), ln, font=qfont, fill=(30,30,30))
        y += qfont.getsize(ln)[1] + 6
    y += 6
    opts = mcq.get("options", ["A","B","C","D"])
    labels = ["A","B","C","D"]
    box_h = 42
    for i, opt in enumerate(opts[:4]):
        x0, y0 = 8, y
        x1, y1 = panel_w - 12, y + box_h - 6
        pd.rectangle([x0, y0, x1, y1], fill=(250,250,250), outline=(210,210,210))
        opt_lines = textwrap.wrap(f"{labels[i]}. {opt}", width=36)
        ty = y + 6
        for ln in opt_lines:
            pd.text((x0+8, ty), ln, font=afont, fill=(20,20,20))
            ty += afont.getsize(ln)[1] + 2
        y += box_h - 6 + 8
    img.paste(panel, (text_area_x, 0), panel)
    img.save(filename)

# -------------- Main --------------
def main():
    parser = argparse.ArgumentParser(description="Generate Ishihara MCQ plates")
    parser.add_argument("--count", type=int, default=6, help="How many sample plates to create")
    parser.add_argument("--out", type=str, default=OUT_DIR, help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create zip archive")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # sample MCQs; replace or extend as needed
    sample_mcqs = [
        {"figure":"12", "question":"Which number is hidden in the dot pattern?", "options":["12","8","6","0"]},
        {"figure":"74", "question":"Select the number shown in the colored dots.", "options":["71","74","77","78"]},
        {"figure":"HELLO", "question":"What word appears in the plate?", "options":["HELLO","WORLD","TEST","PLATE"]},
        {"figure":"3", "question":"Which digit is represented in the image?", "options":["3","5","9","2"]},
        {"figure":"A", "question":"Identify the letter shown in the left panel.", "options":["A","B","C","D"]},
        {"figure":"42", "question":"Which number is embedded in the design?", "options":["24","42","48","62"]},
    ]
    # ensure enough entries
    while len(sample_mcqs) < args.count:
        sample_mcqs.append({"figure":str(random.randint(0,99)), "question":"Which number is shown?", "options":["12","34","56","78"]})

    out_files = []
    for i in range(args.count):
        mcq = sample_mcqs[i]
        fname = os.path.join(args.out, f"mcq_plate_{i+1:02d}.png")
        make_mcq_plate(mcq, fname, palette="classic" if i%2==0 else "blue_orange")
        out_files.append(fname)
        print("Wrote:", fname)

    if args.zip:
        zipname = args.out + ".zip"
        with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in out_files:
                zf.write(p, arcname=os.path.basename(p))
        print("Wrote ZIP:", zipname)
    print("Done. Open the output folder in your file manager to view PNGs.")

if __name__ == "__main__":
    main()
