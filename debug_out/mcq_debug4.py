#!/usr/bin/env python3
"""
Final debug generator:
- Binary-search font to maximize mask text size
- Denser background, multiple passes
- Explicit palette checks / debug prints for FG color
Saves: *_mask.png, *_bg.png, *_fg.png, *_composite.png

Run:
  python3 mcq_debug_final.py --out debug_out --count 3

Dependencies:
  pip install pillow numpy
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, random, textwrap, argparse, sys

# --------- Tunables ----------
WIDTH, HEIGHT = 700, 300
DOT_PITCH = 8           # denser grid -> more dots
JITTER = 2
BG_DOT_SIZE = 5
FG_DOT_SIZE = 18        # make FG larger to pop
BG_PASSES = 4
MASK_THRESHOLD = 0.30
# Try these palettes; they're explicit RGB tuples
PALETTE_BG = [(200,200,80),(180,180,60),(220,220,100)]
PALETTE_FG = [(200,40,40),(220,80,80),(180,30,30)]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf"
]
OUT_DIR_DEFAULT = "debug_out"
# ----------------------------

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()
if FONT_PATH:
    print("Using font:", FONT_PATH)
else:
    print("No TTF found in candidates; using PIL default (may be small).", file=sys.stderr)

def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*8, getattr(font,'size',12))

def largest_fitting_font(lines, w, h, padding=12, max_size_hint=None):
    """
    Find largest font size that fits all lines into (w,h) with padding.
    Binary search between 8 and max_size_cap.
    """
    draw_tmp = ImageDraw.Draw(Image.new("L",(w,h)))
    lo = 8
    hi = max_size_hint if max_size_hint else int(h * 1.2)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(FONT_PATH, mid) if FONT_PATH else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw_tmp, ln, font)
            wmax = max(wmax, wln)
            hsum += hln + 4
        if (wmax <= w - 2*padding) and (hsum <= h - 2*padding):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    # return font object for best size
    try:
        return ImageFont.truetype(FONT_PATH, best) if FONT_PATH else ImageFont.load_default(), best
    except Exception:
        return ImageFont.load_default(), best

def make_mask(figure_text, w, h, padding=10):
    """
    Render the mask using the largest font that fits. Returns PIL L image mask.
    """
    lines = textwrap.wrap(figure_text, width=20)
    # aim for a generous max size (height*0.9) as upper bound for search
    font, chosen_size = largest_fitting_font(lines, w, h, padding=padding, max_size_hint=int(h*0.9))
    # debug print
    print(f"Chosen mask font size: {chosen_size}")
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    total_h = sum([text_size(draw, ln, font)[1] + 4 for ln in lines])
    y = (h - total_h)//2
    for ln in lines:
        wln, hln = text_size(draw, ln, font)
        x = (w - wln)//2
        draw.text((x,y), ln, fill=255, font=font)
        y += hln + 4
    return mask.filter(ImageFilter.GaussianBlur(radius=1.0))

def deterministic_plate(figure_text, question_text, options, out_prefix):
    dot_area_w = min(WIDTH, HEIGHT)
    text_area_x = dot_area_w

    # 1) mask
    mask = make_mask(figure_text, dot_area_w, HEIGHT)
    mask_np = np.array(mask).astype(float)/255.0
    mask.save(out_prefix + "_mask.png")
    print("Saved mask:", out_prefix + "_mask.png")

    # 2) dense background (multiple passes)
    bg_img = Image.new("RGB", (dot_area_w, HEIGHT), (240,240,240))
    for pass_i in range(BG_PASSES):
        bg_draw = ImageDraw.Draw(bg_img)
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
    bg_img.save(out_prefix + "_bg.png")
    print("Saved bg:", out_prefix + "_bg.png")

    # 3) foreground where mask > threshold (deterministic)
    fg_img = Image.new("RGBA", (dot_area_w, HEIGHT), (0,0,0,0))
    fg_draw = ImageDraw.Draw(fg_img)
    cols = int(dot_area_w / DOT_PITCH) + 2
    rows = int(HEIGHT / DOT_PITCH) + 2
    # debug palette print
    print("FG palette sample:", PALETTE_FG[0])
    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - dot_area_w)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            if not (0 <= cx < dot_area_w and 0 <= cy < HEIGHT):
                continue
            v = mask_np[min(int(cy), HEIGHT-1), min(int(cx), dot_area_w-1)]
            if v <= MASK_THRESHOLD:
                continue
            rr = FG_DOT_SIZE
            col = PALETTE_FG[(r + c) % len(PALETTE_FG)]
            fg_draw.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=col + (255,), outline=None)
    # save fg-only (convert to RGB so viewer shows colors)
    fg_img.convert("RGB").save(out_prefix + "_fg.png")
    print("Saved fg:", out_prefix + "_fg.png")

    # composite and right panel text
    composite_left = Image.alpha_composite(bg_img.convert("RGBA"), fg_img)
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

    full.save(out_prefix + "_composite.png")
    print("Saved composite:", out_prefix + "_composite.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out", type=str, default=OUT_DIR_DEFAULT)
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
