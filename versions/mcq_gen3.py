#!/usr/bin/env python3
"""
Improved Ishihara-style MCQ plate generator (700x300)
Two-pass rendering: background pass + foreground overlay for clear figure.

Usage:
    python mcq_ishihara_fixed.py --count 6 --zip

Dependencies:
    pip install pillow numpy
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, random, textwrap, zipfile, argparse

# ---------------- Config / Tuning knobs ----------------
WIDTH, HEIGHT = 700, 300

# dot grid pitch (smaller -> more dots)
DOT_PITCH = 14

# jitter of dot centers (px)
JITTER = 4

# base dot size (px)
BASE_MIN_DOT = 8
BASE_MAX_DOT = 14

# foreground dots are multiplied by this to make figure pop
FG_SIZE_MULT = 1.25  # increase if figure is faint

# mask threshold (0..1) above which foreground dots are placed
MASK_THRESHOLD = 0.35  # increase to make figure crisper (less bleed)

# densities: factor of how many dots to place per grid cell (0..1)
BG_DENSITY = 0.95   # 1.0 nearly fills grid
FG_DENSITY = 0.85   # how many foreground circles are drawn where mask>threshold

OUT_DIR = "ishihara_mcq_output_fixed"

PALETTES = {
    "classic": {"fg":[(200,30,30),(220,60,60),(180,40,40)], "bg":[(100,160,60),(120,180,80),(80,140,50)]},
    "blue_orange": {"fg":[(200,150,60),(220,170,80)], "bg":[(100,120,200),(120,140,220),(70,100,180)]},
    "high_contrast": {"fg":[(220,40,40)], "bg":[(220,220,40),(200,200,80),(240,240,120)]}
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

from PIL import ImageFont, ImageDraw
def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*8, int(getattr(font,'size',12)))

def text_mask(text, w, h, padding=16):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if FONT_PATH:
        size = int(h * 0.20)
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, width=30)
    # shrink font until fits
    while True:
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw, ln, font)
            wmax = max(wmax, wln)
            hsum += hln + 4
        if (wmax < w - 2*padding) and (hsum < h - 2*padding):
            break
        if FONT_PATH and getattr(font,"size",None) and font.size > 12:
            try:
                font = ImageFont.truetype(FONT_PATH, max(12, font.size - 2))
            except Exception:
                break
        else:
            break
    # draw centered multi-line
    total_h = sum([text_size(draw, ln, font)[1] + 4 for ln in lines])
    y = (h - total_h)/2
    for ln in lines:
        wln, hln = text_size(draw, ln, font)
        x = (w - wln)/2
        draw.text((x, y), ln, fill=255, font=font)
        y += hln + 4
    # slight blur for softer edges
    return mask.filter(ImageFilter.GaussianBlur(radius=1.0))

def make_mcq_plate(mcq, filename, palette="classic"):
    pal = PALETTES.get(palette, PALETTES["classic"])
    fg_cols = pal["fg"]; bg_cols = pal["bg"]

    img = Image.new("RGB", (WIDTH, HEIGHT), (240,240,240))
    draw = ImageDraw.Draw(img)

    # left square area for dots
    dot_area_w = min(WIDTH, HEIGHT)
    text_area_x = dot_area_w

    # create mask for figure
    figure_text = mcq.get("figure","?")
    mask = text_mask(figure_text, dot_area_w, HEIGHT)
    mask_np = np.array(mask)/255.0

    # background pass: draw bg dots everywhere (with density control)
    cols = int(dot_area_w / DOT_PITCH) + 2
    rows = int(HEIGHT / DOT_PITCH) + 2
    for r in range(rows):
        for c in range(cols):
            if random.random() > BG_DENSITY:
                continue
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - dot_area_w)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            jx = cx + random.randint(-JITTER, JITTER)
            jy = cy + random.randint(-JITTER, JITTER)
            if not (0 <= jx < dot_area_w and 0 <= jy < HEIGHT):
                continue
            rr = random.randint(BASE_MIN_DOT, BASE_MAX_DOT)
            col = random.choice(bg_cols)
            draw.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # foreground pass: overlay fg dots where mask > threshold
    for r in range(rows):
        for c in range(cols):
            if random.random() > FG_DENSITY:
                continue
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - dot_area_w)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            jx = cx + random.randint(-JITTER, JITTER)
            jy = cy + random.randint(-JITTER, JITTER)
            if not (0 <= jx < dot_area_w and 0 <= jy < HEIGHT):
                continue
            # decide by deterministic mask
            v = mask_np[int(jy), int(jx)]
            if v <= MASK_THRESHOLD:
                continue
            # foreground dot slightly larger
            rr = int(random.randint(BASE_MIN_DOT, BASE_MAX_DOT) * FG_SIZE_MULT)
            col = random.choice(fg_cols)
            draw.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # subtle border for dot area
    draw.ellipse([-2, -2, dot_area_w+2, HEIGHT+2], outline=(200,200,200))

    # text panel on right (unchanged)
    panel_w = WIDTH - text_area_x
    panel = Image.new("RGBA", (panel_w, HEIGHT), (255,255,255,240))
    pd = ImageDraw.Draw(panel)
    if FONT_PATH:
        try:
            qfont = ImageFont.truetype(FONT_PATH, 18)
            afont = ImageFont.truetype(FONT_PATH, 16)
        except Exception:
            qfont = ImageFont.load_default()
            afont = ImageFont.load_default()
    else:
        qfont = ImageFont.load_default()
        afont = ImageFont.load_default()

    q_lines = textwrap.wrap(mcq.get("question","Question?"), width=36)
    y = 12
    for ln in q_lines:
        pd.text((12, y), ln, font=qfont, fill=(30,30,30))
        _, hln = text_size(pd, ln, qfont)
        y += hln + 6
    y += 6

    opts = mcq.get("options", ["A","B","C","D"])
    labels = ["A","B","C","D"]
    box_h = 46
    for i, opt in enumerate(opts[:4]):
        x0, y0 = 8, y
        x1, y1 = panel_w - 12, y + box_h - 6
        pd.rectangle([x0, y0, x1, y1], fill=(250,250,250), outline=(210,210,210))
        opt_lines = textwrap.wrap(f"{labels[i]}. {opt}", width=36)
        ty = y + 8
        for ln in opt_lines:
            pd.text((x0+8, ty), ln, font=afont, fill=(20,20,20))
            _, lnh = text_size(pd, ln, afont)
            ty += lnh + 4
        y += box_h - 6 + 8

    img.paste(panel, (text_area_x, 0), panel)
    img.save(filename)

# -------------- Main --------------
def main():
    parser = argparse.ArgumentParser(description="Generate Ishihara MCQ plates (fixed)")
    parser.add_argument("--count", type=int, default=6, help="How many sample plates to create")
    parser.add_argument("--out", type=str, default=OUT_DIR, help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Create zip archive")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sample_mcqs = [
        {"figure":"12", "question":"Which number is hidden in the dot pattern?", "options":["12","8","6","0"]},
        {"figure":"74", "question":"Select the number shown in the colored dots.", "options":["71","74","77","78"]},
        {"figure":"HELLO", "question":"What word appears in the plate?", "options":["HELLO","WORLD","TEST","PLATE"]},
        {"figure":"3", "question":"Which digit is represented in the image?", "options":["3","5","9","2"]},
        {"figure":"A", "question":"Identify the letter shown in the left panel.", "options":["A","B","C","D"]},
        {"figure":"42", "question":"Which number is embedded in the design?", "options":["24","42","48","62"]},
    ]
    while len(sample_mcqs) < args.count:
        sample_mcqs.append({"figure":str(random.randint(0,99)), "question":"Which number is shown?", "options":["12","34","56","78"]})

    out_files = []
    for i in range(args.count):
        mcq = sample_mcqs[i]
        fname = os.path.join(args.out, f"mcq_plate_{i+1:02d}.png")
        make_mcq_plate(mcq, fname, palette="classic" if i%2==0 else "high_contrast")
        out_files.append(fname)
        print("Wrote:", fname)

    if args.zip:
        zipname = args.out + ".zip"
        with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in out_files:
                zf.write(p, arcname=os.path.basename(p))
        print("Wrote ZIP:", zipname)

    print("Done. Open the output folder to view PNGs.")

if __name__ == "__main__":
    main()
