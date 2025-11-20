#!/usr/bin/env python3
"""
Debug Ishihara MCQ generator (deterministic foreground mask)
Saves for each plate: mask, bg-only, fg-only, composite.

Usage:
  python mcq_debug.py --count 3 --out debug_out

Dependencies:
  pip install pillow numpy
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, random, textwrap, argparse

# ---------- Tunable params ----------
WIDTH, HEIGHT = 700, 300
DOT_PITCH = 12         # smaller => denser dots
JITTER = 2             # small jitter for bg only; fg will be placed deterministically
BG_DOT_SIZE = 10
FG_DOT_SIZE = 14
MASK_THRESHOLD = 0.25  # 0..1 threshold for mask (lower => larger figure)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf"
]
PALETTE_BG = [(100,160,60),(120,180,80),(80,140,50)]
PALETTE_FG = [(200,30,30),(220,60,60),(180,40,40)]
# ------------------------------------

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()

def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*8, getattr(font,'size',12))

def make_mask(figure_text, w, h, padding=12):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    # choose font
    if FONT_PATH:
        size = int(h * 0.22)
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    lines = textwrap.wrap(figure_text, width=18)
    # shrink until fits
    while True:
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw, ln, font)
            wmax = max(wmax, wln); hsum += hln + 4
        if (wmax < w - 2*padding) and (hsum < h - 2*padding):
            break
        if getattr(font, "size", None) and FONT_PATH and font.size > 12:
            font = ImageFont.truetype(FONT_PATH, max(12, font.size - 2))
        else:
            break
    # center
    total_h = sum([text_size(draw, ln, font)[1] + 4 for ln in lines])
    y = (h - total_h)//2
    for ln in lines:
        wln, hln = text_size(draw, ln, font)
        x = (w - wln)//2
        draw.text((x,y), ln, fill=255, font=font)
        y += hln + 4
    return mask.filter(ImageFilter.GaussianBlur(radius=1.0))

def deterministic_plate(figure_text, question_text, options, out_prefix):
    # left square dot area
    dot_area_w = min(WIDTH, HEIGHT)
    text_area_x = dot_area_w

    # 1) mask
    mask = make_mask(figure_text, dot_area_w, HEIGHT)
    mask_np = np.array(mask).astype(float)/255.0

    # create images
    bg_img = Image.new("RGB", (dot_area_w, HEIGHT), (240,240,240))
    fg_img = Image.new("RGBA", (dot_area_w, HEIGHT), (0,0,0,0))
    bg_draw = ImageDraw.Draw(bg_img)
    fg_draw = ImageDraw.Draw(fg_img)

    # grid - deterministic coverage
    cols = int(dot_area_w / DOT_PITCH) + 1
    rows = int(HEIGHT / DOT_PITCH) + 1

    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2)
            # small jitter for bg
            jx = cx + random.randint(-JITTER, JITTER)
            jy = cy + random.randint(-JITTER, JITTER)
            if not (0 <= jx < dot_area_w and 0 <= jy < HEIGHT):
                continue
            # sample mask at nominal center (deterministic)
            v = mask_np[min(int(jy), HEIGHT-1), min(int(jx), dot_area_w-1)]
            if v > MASK_THRESHOLD:
                # foreground: deterministic placement
                rr = FG_DOT_SIZE
                col = PALETTE_FG[(r+c) % len(PALETTE_FG)]
                fg_draw.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=col, outline=None)
            else:
                # background dot
                rr = BG_DOT_SIZE
                col = PALETTE_BG[(r+c) % len(PALETTE_BG)]
                bg_draw.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # composite left panel
    composite_left = Image.alpha_composite(bg_img.convert("RGBA"), fg_img)

    # assemble full image with right panel text (simple)
    full = Image.new("RGB", (WIDTH, HEIGHT), (255,255,255))
    full.paste(composite_left, (0,0))
    pd = ImageDraw.Draw(full)
    # question text
    try:
        qfont = ImageFont.truetype(FONT_PATH, 18) if FONT_PATH else ImageFont.load_default()
        afont = ImageFont.truetype(FONT_PATH, 16) if FONT_PATH else ImageFont.load_default()
    except Exception:
        qfont = ImageFont.load_default(); afont = ImageFont.load_default()
    # write the question at right
    x0 = text_area_x + 12
    y0 = 12
    for ln in textwrap.wrap(question_text, width=40):
        pd.text((x0, y0), ln, font=qfont, fill=(20,20,20))
        y0 += text_size(pd, ln, qfont)[1] + 6
    y0 += 8
    labels = ["A","B","C","D"]
    for i,opt in enumerate(options[:4]):
        box_y0 = y0; box_y1 = y0 + 44
        pd.rectangle([x0-6, box_y0-4, WIDTH-12, box_y1], fill=(245,245,245), outline=(210,210,210))
        pd.text((x0, box_y0), f"{labels[i]}. {opt}", font=afont, fill=(10,10,10))
        y0 = box_y1 + 10

    # Save files: mask, bg, fg, composite
    mask.save(out_prefix + "_mask.png")
    bg_img.save(out_prefix + "_bg.png")
    fg_img.convert("RGB").save(out_prefix + "_fg.png")
    full.save(out_prefix + "_composite.png")
    print("Saved:", out_prefix + "_composite.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out", type=str, default="debug_out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    samples = [
        ("12","Which number is hidden in the dot pattern?", ["12","8","6","0"]),
        ("74","Select the number shown in the colored dots.", ["71","74","77","78"]),
        ("HELLO","What word appears in the plate?", ["HELLO","WORLD","TEST","PLATE"]),
    ]
    for i in range(min(args.count, len(samples))):
        fig, q, opts = samples[i]
        prefix = os.path.join(args.out, f"mcq_plate_{i+1:02d}")
        deterministic_plate(fig, q, opts, prefix)

if __name__ == "__main__":
    main()
