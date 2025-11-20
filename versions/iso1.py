#!/usr/bin/env python3
# make_machine_visible_plate.py
# Requires: pillow, numpy, scipy (optional for morphology)
# pip install pillow numpy scipy

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random, os, sys
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

# Luminance conversion (sRGB linear approx)
def luminance_rgb(rgb):
    r,g,b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def clamp(v):
    return max(0, min(255, int(round(v))))

# Build an isoluminant color that changes R while keeping luminance L constant by adjusting G
def iso_color_with_new_r(base_rgb, new_r):
    r0,g0,b0 = base_rgb
    L = luminance_rgb(base_rgb)
    # solve for g: L = 0.2126*r + 0.7152*g + 0.0722*b  => g = (L - 0.2126*r - 0.0722*b)/0.7152
    g_new = (L - 0.2126*new_r - 0.0722*b0) / 0.7152
    return (clamp(new_r), clamp(g_new), clamp(b0))

def make_plate(text_question, options, outprefix="plate_machine", size=(700,300),
               font_path=None, font_size=40, micro=3):
    """
    micro = micro pattern cell size (px). Smaller -> more invisible to human.
    We'll render a background micro-checker of bgA/bgB and text area micro-checker of fgA/fgB,
    where bgA/bgB and fgA/fgB are isoluminant pairs but differ in red channel.
    """
    W,H = size
    # base neutral color (mid olive)
    base_bg = (170, 150, 120)
    # choose two background red offsets
    dr_bg = 0
    dr_bg2 = 6   # tiny red variation
    bgA = iso_color_with_new_r(base_bg, base_bg[0] + dr_bg)
    bgB = iso_color_with_new_r(base_bg, base_bg[0] + dr_bg2)

    # pick text red offsets that differ substantially in R but keep same luminance
    # pick R values +/- delta around base to ensure channels differ
    fgR1 = clamp(base_bg[0] + 80)
    fgR2 = clamp(base_bg[0] + 86)
    fgA = iso_color_with_new_r(base_bg, fgR1)
    fgB = iso_color_with_new_r(base_bg, fgR2)

    # create base image and draw micro-checker background
    img = Image.new("RGB", (W,H), bgA)
    pix = img.load()
    for y in range(H):
        for x in range(W):
            cell = ((x // micro) + (y // micro)) & 1
            pix[x,y] = bgA if cell == 0 else bgB

    draw = ImageDraw.Draw(img)
    # draw text mask in high-resolution (use big font) then fill text area with micro-checker of fg colors
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    # prepare mask: render text in white on black mask
    mask = Image.new("L", (W,H), 0)
    md = ImageDraw.Draw(mask)
    # simple layout: question at top-left, options below
    y = 10
    md.text((10,y), text_question, font=font, fill=255)
    # options on separate lines
    oy = y + font_size + 10
    for i,opt in enumerate(options):
        md.text((10, oy + i * (font_size+6)), opt, font=font, fill=255)

    # now iterate over pixels: where mask>0 => put micro-checker using fgA/fgB; else background stays
    mask_np = np.array(mask)
    for y in range(H):
        for x in range(W):
            if mask_np[y,x] > 20:  # in text area
                cell = ((x // micro) + (y // micro)) & 1
                pix[x,y] = fgA if cell == 0 else fgB
            # else leave background as already set

    out_camouflage = f"{outprefix}_camouflage.png"
    img.save(out_camouflage)
    print("Wrote:", out_camouflage)

    # --- MACHINE EXTRACTION: isolate red channel and threshold ---
    arr = np.array(img).astype(np.uint8)
    red = arr[:,:,0].astype(np.int16)
    # normalize and threshold adaptively
    thresh = np.percentile(red, 50) + 10
    mask_extract = (red >= thresh).astype(np.uint8) * 255
    if ndi is not None:
        mask_extract = ndi.binary_closing(mask_extract, structure=np.ones((3,3))).astype(np.uint8) * 255
        mask_extract = ndi.binary_fill_holes(mask_extract, structure=np.ones((3,3))).astype(np.uint8) * 255

    im_extract = Image.fromarray(mask_extract).convert("L")
    im_extract = im_extract.filter(ImageFilter.GaussianBlur(radius=1.0))
    # final binarize
    bw = np.array(im_extract)
    bw = (bw > 128).astype(np.uint8) * 255
    out_extract = f"{outprefix}_extracted_for_ocr.png"
    Image.fromarray(bw).save(out_extract)
    print("Wrote:", out_extract)

    return out_camouflage, out_extract

if __name__ == "__main__":
    # quick demo
    FONT = "/Library/Fonts/SourceCodePro-Semibold.ttf"  # change as needed
    q = "Which pattern in chimpanzees was used to explain some human patterns of intergroup violence?"
    opts = ["A. Territorial patrols and lethal raids by male coalitions",
           "B. Ritualized greeting without conflict",
           "C. Cooperative exploitation/sharing of resources",
           "D. Absence of male coalitions"]
    make_plate(q, opts, outprefix="demo_machine_visible", size=(700,300),
               font_path=FONT, font_size=28, micro=3)
