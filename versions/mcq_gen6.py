#!/usr/bin/env python3
"""
generate_mcq_fullimage.py
Generates a 700x300 Ishihara-style MCQ plate where the *entire image* is the dot field
and the question + four answers are rendered into the foreground mask. Also outputs
a JSON with bounding boxes for the 4 options and the correct option index.

Usage:
    python generate_mcq_fullimage.py --out outdir --figure "N/A" \
        --question "Which is the capital of X?" \
        --option "A. One" --option "B. Two" --option "C. Three" --option "D. Four" \
        --correct 2

Outputs:
    outdir/plate.png
    outdir/plate_mask.png
    outdir/plate_bg.png
    outdir/plate_fg.png
    outdir/plate_meta.json
"""
import os, json, random, textwrap, argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

WIDTH, HEIGHT = 700, 300

# Tuning knobs
DOT_PITCH = 8        # grid spacing (smaller => denser)
JITTER = 2
BG_DOT_SIZE = 5
FG_DOT_SIZE = 18
BG_PASSES = 4
MASK_THRESHOLD = 0.30

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf"
]

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = find_font()

# safe text measuring
def text_size(draw, txt, font):
    try:
        bbox = draw.textbbox((0,0), txt, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(txt)
        except Exception:
            return (len(txt)*8, getattr(font,'size',12))

def largest_fitting_truetype(lines, w, h, padding=8):
    draw_tmp = ImageDraw.Draw(Image.new("L",(w,h)))
    if FONT_PATH is None:
        return None, 12
    lo, hi = 8, int(h*0.9)
    best = lo
    while lo <= hi:
        mid = (lo+hi)//2
        try:
            f = ImageFont.truetype(FONT_PATH, mid)
        except Exception:
            hi = mid - 1
            continue
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw_tmp, ln, f)
            wmax = max(wmax, wln); hsum += hln + 4
        if (wmax <= w-2*padding) and (hsum <= h-2*padding):
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    try:
        return ImageFont.truetype(FONT_PATH, best), best
    except Exception:
        return None, best

def build_mask_and_boxes(question, options):
    """
    Produce mask image (L) of size WIDTHxHEIGHT with question+options rendered.
    Also return list of option bounding boxes [(x0,y0,x1,y1),...]
    Layout strategy:
      - top area reserved for question (wrapped); remaining vertical space for 4 answer lines/boxes.
      - We render text directly and compute bounding boxes from textbbox.
    """
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)

    # Compose text blocks
    q_lines = textwrap.wrap(question, width=60)
    # Reserve ~40% height for question; 60% for options (split into 4)
    q_area_h = int(HEIGHT * 0.38)
    opt_area_h = HEIGHT - q_area_h - 20  # small margins
    # Find font that fits question area
    # We'll try to fit q_lines into (WIDTH - margin, q_area_h)
    q_font, q_size = largest_fitting_truetype(q_lines, WIDTH-24, q_area_h, padding=8)
    if q_font is None:
        q_font = ImageFont.load_default()
    # draw question centered in its area
    y = 10
    for ln in q_lines:
        w,h = text_size(draw, ln, q_font)
        draw.text(((WIDTH-w)//2, y), ln, fill=255, font=q_font)
        y += h + 4

    # Options: split remaining area into 4 equal rows, draw each option centered in its row
    boxes = []
    per_row_h = int(opt_area_h / 4)
    opt_font_candidate_lines = []
    for opt in options:
        # each opt text may be like "A. Answer text"
        opt_font_candidate_lines.append(textwrap.wrap(opt, width=40))
    # pick a font size that fits the tallest option row
    # try fonts sizes with same routine but for each option row; pick the minimal allowed size
    sizes = []
    chosen_font = None
    for size_test in range(28, 10, -1):  # try from big to small
        try:
            ftest = ImageFont.truetype(FONT_PATH, size_test) if FONT_PATH else ImageFont.load_default()
        except Exception:
            ftest = ImageFont.load_default()
        ok = True
        for opt in options:
            # measure wrapped height
            lines = textwrap.wrap(opt, width=40)
            hsum = 0
            for ln in lines:
                _, h = text_size(draw, ln, ftest)
                hsum += h + 2
            if hsum > per_row_h - 8:
                ok = False
                break
        if ok:
            chosen_font = ftest
            break
    if chosen_font is None:
        chosen_font = ImageFont.load_default()

    # draw options
    base_y = y + 8
    for i,opt in enumerate(options):
        row_y = base_y + i * per_row_h
        # center option text vertically in row
        lines = textwrap.wrap(opt, width=40)
        total_h = sum([text_size(draw, ln, chosen_font)[1] + 2 for ln in lines])
        inner_y = row_y + (per_row_h - total_h) // 2
        min_x = WIDTH; min_y = HEIGHT; max_x = 0; max_y = 0
        for ln in lines:
            w,h = text_size(draw, ln, chosen_font)
            x = 20  # left margin for options (could center or left-align)
            # draw left-aligned to aid positioning; easier bounding box detection
            draw.text((x, inner_y), ln, fill=255, font=chosen_font)
            # compute bbox
            min_x = min(min_x, x)
            min_y = min(min_y, inner_y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, inner_y + h)
            inner_y += h + 2
        # add small padding to bounding box
        pad = 6
        boxes.append([max(0, min_x-pad), max(0, min_y-pad), min(WIDTH, max_x+pad), min(HEIGHT, max_y+pad)])
    # soften edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    return mask, boxes

def generate_plate(outdir, question, options, correct_index):
    os.makedirs(outdir, exist_ok=True)
    mask, boxes = build_mask_and_boxes(question, options)
    mask_np = np.array(mask).astype(float)/255.0

    # background image: multiple passes to make dense field
    bg = Image.new("RGB", (WIDTH, HEIGHT), (240,240,240))
    for pass_i in range(BG_PASSES):
        draw_bg = ImageDraw.Draw(bg)
        offset_x = (pass_i * (DOT_PITCH // 3)) % DOT_PITCH
        offset_y = (pass_i * (DOT_PITCH // 5)) % DOT_PITCH
        cols = int(WIDTH / DOT_PITCH) + 2
        rows = int(HEIGHT / DOT_PITCH) + 2
        for r in range(rows):
            for c in range(cols):
                cx = int(c * DOT_PITCH + DOT_PITCH/2 + offset_x - (cols*DOT_PITCH - WIDTH)/2)
                cy = int(r * DOT_PITCH + DOT_PITCH/2 + offset_y - (rows*DOT_PITCH - HEIGHT)/2)
                jx = cx + random.randint(-JITTER, JITTER)
                jy = cy + random.randint(-JITTER, JITTER)
                if not (0 <= jx < WIDTH and 0 <= jy < HEIGHT): continue
                rr = BG_DOT_SIZE
                # alternate colors via simple pattern
                col = (180 + (r+c+pass_i)%3*10, 180 + (r+c+pass_i)%3*6, 60 + (r+c+pass_i)%3*20)
                draw_bg.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # foreground overlay where mask > threshold
    fg = Image.new("RGBA", (WIDTH, HEIGHT), (0,0,0,0))
    draw_fg = ImageDraw.Draw(fg)
    cols = int(WIDTH / DOT_PITCH) + 2
    rows = int(HEIGHT / DOT_PITCH) + 2
    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - WIDTH)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            if not (0 <= cx < WIDTH and 0 <= cy < HEIGHT): continue
            v = mask_np[min(int(cy), HEIGHT-1), min(int(cx), WIDTH-1)]
            if v <= MASK_THRESHOLD: continue
            rr = FG_DOT_SIZE
            col = (200,40,40)
            draw_fg.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=(col[0],col[1],col[2],255), outline=None)

    # composite
    composite = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")

    # save images
    composite.save(os.path.join(outdir, "plate.png"))
    mask.save(os.path.join(outdir, "plate_mask.png"))
    bg.save(os.path.join(outdir, "plate_bg.png"))
    fg.convert("RGB").save(os.path.join(outdir, "plate_fg.png"))

    meta = {
        "width": WIDTH, "height": HEIGHT,
        "options": [
            {"index": i, "bbox": boxes[i]} for i in range(len(options))
        ],
        "correct_index": correct_index,
        "question": question,
        "options_text": options
    }
    with open(os.path.join(outdir, "plate_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Wrote:", os.path.join(outdir, "plate.png"))
    print("Wrote meta:", os.path.join(outdir, "plate_meta.json"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outplate")
    p.add_argument("--question", required=True)
    p.add_argument("--option", action="append", required=True, help="Repeat for each option (4 total)")
    p.add_argument("--correct", type=int, required=True, help="0-based index of correct option")
    args = p.parse_args()
    if len(args.option) != 4:
        raise SystemExit("Please supply exactly 4 --option entries.")
    generate_plate(args.out, args.question, args.option, args.correct)
