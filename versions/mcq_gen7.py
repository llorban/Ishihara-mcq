#!/usr/bin/env python3
"""
generate_mcq_images.py

Generate 700x300 "Ishihara-style" MCQ plates (plain PNGs) where the entire image
is a dot field and the question + four options form the foreground figure.

Usage examples:
  # Use built-in sample questions (count controls number produced)
  python generate_mcq_images.py --count 12 --out plates_dir --zip

  # Use CSV file with header: question,optA,optB,optC,optD
  python generate_mcq_images.py --csv my_questions.csv --out plates_dir --zip

  # Save debug files (mask/bg/fg)
  python generate_mcq_images.py --count 6 --out plates_debug --debug

Dependencies:
  pip install pillow numpy
"""
import os, sys, json, argparse, textwrap, random, zipfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import csv

# ----------------- CONFIG -----------------
WIDTH, HEIGHT = 700, 300
DOT_PITCH = 8        # grid spacing (smaller => denser)
JITTER = 2
BG_DOT_SIZE = 3 
FG_DOT_SIZE = 9
BG_PASSES = 4
MASK_THRESHOLD = 0.30

# Font search directories (common)
FONT_SEARCH_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/Library/Fonts"),
    "C:\\Windows\\Fonts"
]
# ------------------------------------------

def find_any_ttf():
    for d in FONT_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        try:
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        # filter out obvious color/emoji fonts
                        name = f.lower()
                        if "emoji" in name or "color" in name:
                            continue
                        return os.path.join(root, f)
        except PermissionError:
            continue
    return None

def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*8, getattr(font, "size", 12))

def largest_fitting_truetype(lines, w, h, padding=8, font_path=None):
    """
    Binary-search largest font size that fits `lines` into w x h (approx).
    Returns (PIL.ImageFont, size)
    If font_path is None, falls back to PIL default (note: default may be small).
    """
    draw_tmp = ImageDraw.Draw(Image.new("L", (w, h)))
    if font_path is None:
        f = ImageFont.load_default()
        return f, getattr(f, "size", 12)
    lo, hi = 8, int(h * 1.2)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            hi = mid - 1
            continue
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw_tmp, ln, font)
            wmax = max(wmax, wln); hsum += hln + 4
        if (wmax <= w - 2*padding) and (hsum <= h - 2*padding):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    try:
        return ImageFont.truetype(font_path, best), best
    except Exception:
        f = ImageFont.load_default()
        return f, getattr(f, "size", 12)

def build_mask_and_option_boxes(question, options, font_path):
    """
    Render the question + options into a grayscale mask (L) of WIDTH x HEIGHT.
    Also compute and return option bounding boxes (x0,y0,x1,y1) in pixel coords.
    Layout:
      - top portion for question (wrapped), rest split into 4 rows for options (left-aligned).
    """
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)

    # Prepare question lines
    q_lines = textwrap.wrap(question, width=60)
    q_area_h = int(HEIGHT * 0.38)
    opt_area_h = HEIGHT - q_area_h - 20
    # find largest font for question that fits q_area_h
    q_font, q_size = largest_fitting_truetype(q_lines, WIDTH - 24, q_area_h, padding=8, font_path=font_path)
    y = 10
    for ln in q_lines:
        w, h = text_size(draw, ln, q_font)
        draw.text(((WIDTH - w)//2, y), ln, fill=255, font=q_font)
        y += h + 4

    # For options: split remaining area into 4 rows and fit options there.
    per_row_h = int(opt_area_h / 4)
    chosen_font = None
    # attempt to find a single font size that fits the tallest option row; try from large -> small
    for size_test in range(36, 8, -1):
        try:
            ftest = ImageFont.truetype(font_path, size_test) if font_path else ImageFont.load_default()
        except Exception:
            ftest = ImageFont.load_default()
        ok = True
        for opt in options:
            lines = textwrap.wrap(opt, width=40)
            hsum = 0
            for ln in lines:
                _, hln = text_size(draw, ln, ftest)
                hsum += hln + 2
            if hsum > per_row_h - 8:
                ok = False
                break
        if ok:
            chosen_font = ftest
            break
    if chosen_font is None:
        chosen_font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()

    # draw options left-aligned and compute their bboxes
    base_y = y + 8
    boxes = []
    for i, opt in enumerate(options):
        row_y = base_y + i * per_row_h
        lines = textwrap.wrap(opt, width=40)
        total_h = sum([text_size(draw, ln, chosen_font)[1] + 2 for ln in lines])
        inner_y = row_y + (per_row_h - total_h) // 2
        min_x = WIDTH; min_y = HEIGHT; max_x = 0; max_y = 0
        for ln in lines:
            w,h = text_size(draw, ln, chosen_font)
            x = 20  # left margin for options
            draw.text((x, inner_y), ln, fill=255, font=chosen_font)
            min_x = min(min_x, x); min_y = min(min_y, inner_y)
            max_x = max(max_x, x + w); max_y = max(max_y, inner_y + h)
            inner_y += h + 2
        pad = 6
        boxes.append([max(0, min_x-pad), max(0, min_y-pad), min(WIDTH, max_x+pad), min(HEIGHT, max_y+pad)])
    # soften edges for nicer fg/bg transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    return mask, boxes

def generate_plate_image(question, options, outpath, font_path=None, debug=False):
    mask, boxes = build_mask_and_option_boxes(question, options, font_path)
    mask_np = np.array(mask).astype(float) / 255.0

    # Background: multiple passes to create dense dot field
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
                col = (180 + ((r+c+pass_i) % 3)*10, 180 + ((r+c+pass_i)%3)*6, 60 + ((r+c+pass_i)%3)*20)
                draw_bg.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # Foreground overlay where mask > threshold
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
            draw_fg.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=(200,40,40,255), outline=None)

    composite = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")
    composite.save(outpath)
    if debug:
        mask.save(outpath.replace(".png", "_mask.png"))
        bg.save(outpath.replace(".png", "_bg.png"))
        fg.convert("RGB").save(outpath.replace(".png", "_fg.png"))
    return boxes

def load_questions_from_csv(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get('question') or r.get('Question') or ""
            opts = [r.get('optA') or r.get('A') or r.get('optA') or r.get('opt1') or r.get('optionA') or "",
                    r.get('optB') or r.get('B') or r.get('optB') or r.get('opt2') or r.get('optionB') or "",
                    r.get('optC') or r.get('C') or r.get('optC') or r.get('opt3') or r.get('optionC') or "",
                    r.get('optD') or r.get('D') or r.get('optD') or r.get('opt4') or r.get('optionD') or ""]
            rows.append((q, opts))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=12, help="Number of sample plates to produce when no CSV is given")
    p.add_argument("--out", default="plates_out", help="Output directory")
    p.add_argument("--zip", action="store_true", help="Create a zip file of PNGs")
    p.add_argument("--debug", action="store_true", help="Also save mask/bg/fg debug images")
    p.add_argument("--csv", help="Optional CSV file with header: question,optA,optB,optC,optD")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # find font
    font_path = find_any_ttf()
    if font_path:
        print("Using TTF font:", font_path)
    else:
        print("WARNING: No TTF found; using PIL default font (may be small). Install a TTF for best results.", file=sys.stderr)

    questions = []
    if args.csv:
        questions = load_questions_from_csv(args.csv)
        if not questions:
            print("CSV was read but no rows found. Exiting.", file=sys.stderr)
            sys.exit(1)
    else:
        # default sample items (editable)
        sample = [
            ("Which number is hidden in the pattern?", ["A. 12", "B. 8", "C. 6", "D. 0"]),
            ("Select the number shown in the colored dots.", ["A. 71", "B. 74", "C. 77", "D. 78"]),
            ("What word appears in the plate?", ["A. HELLO", "B. WORLD", "C. TEST", "D. PLATE"]),
            ("Which digit is represented?", ["A. 3", "B. 5", "C. 9", "D. 2"]),
            ("Identify the letter shown in the left panel.", ["A. A", "B. B", "C. C", "D. D"]),
            ("Which number is embedded in the design?", ["A. 24", "B. 42", "C. 48", "D. 62"]),
            ("Which option is highlighted?", ["A. RED", "B. BLUE", "C. GREEN", "D. YELLOW"]),
            ("Pick the correct label in the dots.", ["A. CAT", "B. DOG", "C. BIRD", "D. FISH"]),
            ("Which month is spelled in dots?", ["A. MAY", "B. JUNE", "C. JULY", "D. AUG"]),
            ("Which shape name appears?", ["A. CIRCLE", "B. SQUARE", "C. TRIANGLE", "D. OVAL"]),
            ("What does the plate show?", ["A. SUN", "B. MOON", "C. STAR", "D. CLOUD"]),
            ("Which letter is hidden?", ["A. X", "B. Y", "C. Z", "D. W"]),
        ]
        # limit to requested count
        questions = sample[:max(1, min(len(sample), args.count))]

    written = []
    for i, (qtext, opts) in enumerate(questions, start=1):
        name = f"plate_{i:02d}.png"
        outpath = os.path.join(args.out, name)
        print(f"Rendering {name} ...")
        boxes = generate_plate_image(qtext, opts, outpath, font_path=font_path, debug=args.debug)
        written.append(outpath)
        # print a little validation info
        print(f" Wrote {outpath}   (options bboxes: {boxes})")

    if args.zip:
        zipname = os.path.join(args.out, "plates.zip")
        with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pth in written:
                zf.write(pth, arcname=os.path.basename(pth))
        print("Wrote ZIP:", zipname)

    print("Done. Open the output folder to upload PNGs into Moodle.")

if __name__ == "__main__":
    main()
