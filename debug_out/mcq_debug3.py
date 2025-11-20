#!/usr/bin/env python3
"""
Tuned debug generator: denser background + guaranteed larger mask text.

Usage:
  python3 mcq_debug_tuned2.py --count 3 --out debug_out

Dependencies:
  pip install pillow numpy
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, random, textwrap, argparse

# ---------------- Tunables you can change ----------------
WIDTH, HEIGHT = 700, 300

# Make grid denser (smaller pitch -> more dots). Previously ~12; use 8 for ~ (12/8)^2 ≈ 2.25x density.
DOT_PITCH = 8

# small jitter
JITTER = 2

# background dot size (smaller since denser grid)
BG_DOT_SIZE = 5

# foreground dot size (figure)
FG_DOT_SIZE = 16

# multiple full background passes (overlays) to increase dot count and reduce visible gaps
BG_PASSES = 4   # set to 4 to approximate ~4x more bg dots

# Mask font multiplier (requested x3)
MASK_FONT_MULTIPLIER = 3

# mask inclusion threshold (0..1)
MASK_THRESHOLD = 0.30

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf"
]

PALETTE_BG = [(100,160,60),(120,180,80),(80,140,50)]
PALETTE_FG = [(200,30,30),(220,60,60),(180,40,40)]
# --------------------------------------------------------

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

def make_mask(figure_text, w, h, padding=10):
    """
    Create a mask where the figure_text is rendered much larger initially
    (honouring MASK_FONT_MULTIPLIER) and only reduced if it absolutely must.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # desired large starting size (base) then multiplied
    base_size = int(h * 0.22)
    desired_size = max(14, int(base_size * MASK_FONT_MULTIPLIER))

    # try desired size: if too large, reduce only as necessary (not to tiny)
    size = desired_size
    if FONT_PATH:
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    lines = textwrap.wrap(figure_text, width=20)
    # now reduce only if it fails to fit (instead of aggressive shrinking)
    while True:
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw, ln, font)
            wmax = max(wmax, wln); hsum += hln + 4
        if (wmax <= w - 2*padding) and (hsum <= h - 2*padding):
            break
        # step down font by a small amount (gentler than before)
        if getattr(font, "size", None) and FONT_PATH and font.size > 20:
            size = max(20, font.size - 4)
            try:
                font = ImageFont.truetype(FONT_PATH, size)
            except Exception:
                font = ImageFont.load_default()
                break
        else:
            break

    # center-draw lines
    total_h = sum([text_size(draw, ln, font)[1] + 4 for ln in lines])
    y = (h - total_h) // 2
    for ln in lines:
        wln, hln = text_size(draw, ln, font)
        x = (w - wln) // 2
        draw.text((x, y), ln, fill=255, font=font)
        y += hln + 4

    return mask.filter(ImageFilter.GaussianBlur(radius=1.0))

def deterministic_plate(figure_text, question_text, options, out_prefix):
    dot_area_w = min(WIDTH, HEIGHT)
    text_area_x = dot_area_w

    mask = make_mask(figure_text, dot_area_w, HEIGHT)
    mask_np = np.array(mask).astype(float)/255.0

    # create bg and fg images
    bg_img = Image.new("RGB", (dot_area_w, HEIGHT), (240,240,240))
    fg_img = Image.new("RGBA", (dot_area_w, HEIGHT), (0,0,0,0))

    # background passes - multiple overlays to increase density
    for pass_i in range(BG_PASSES):
        bg_draw = ImageDraw.Draw(bg_img)
        # offset the grid slightly each pass to avoid exact overlap
        offset_x = (pass_i * (DOT_PITCH // 3)) % DOT_PITCH
        offset_y = (pass_i * (DOT_PITCH // 5)) % DOT_PITCH
        cols = int(dot_area_w / DOT_PITCH) + 2
        rows = int(HEIGHT / DOT_PITCH) + 2
        for r in range(rows):
            for c in range(cols):
                cx = int(c * DOT_PITCH + DOT_PITCH/2 + offset_x - (cols*DOT_PITCH - dot_area_w)/2)
                cy = int(r * DOT_PITCH + DOT_PITCH/2 + offset_y - (rows*DOT_PITCH - HEIGHT)/2)
                jx = cx + random.randint(-JITTER, JITTER)
                jy = cy + random.randint(-JITTER, JITTER)
                if not (0 <= jx < dot_area_w and 0 <= jy < HEIGHT):
                    continue
                rr = BG_DOT_SIZE
                col = PALETTE_BG[(r + c + pass_i) % len(PALETTE_BG)]
                bg_draw.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # foreground pass: place dots deterministically where mask>threshold
    fg_draw = ImageDraw.Draw(fg_img)
    cols = int(dot_area_w / DOT_PITCH) + 2
    rows = int(HEIGHT / DOT_PITCH) + 2
    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - dot_area_w)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            if not (0 <= cx < dot_area_w and 0 <= cy < HEIGHT):
                continue
            # sample mask at center
            v = mask_np[min(int(cy), HEIGHT-1), min(int(cx), dot_area_w-1)]
            if v <= MASK_THRESHOLD:
                continue
            rr = FG_DOT_SIZE
            col = PALETTE_FG[(r+c) % len(PALETTE_FG)]
            fg_draw.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=col, outline=None)

    composite_left = Image.alpha_composite(bg_img.convert("RGBA"), fg_img)

    # assemble right panel with question text and options
    full = Image.new("RGB", (WIDTH, HEIGHT), (255,255,255))
    full.paste(composite_left, (0,0))
    pd = ImageDraw.Draw(full)

    try:
        qfont = ImageFont.truetype(FONT_PATH, 18) if FONT_PATH else ImageFont.load_default()
        afont = ImageFont.truetype(FONT_PATH, 16) if FONT_PATH else ImageFont.load_default()
    except Exception:
        qfont = ImageFont.load_default(); afont = ImageFont.load_default()

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

    # Save diagnostic files
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
