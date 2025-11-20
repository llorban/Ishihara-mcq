# find_isoluminant.py
# Usage: python3 find_isoluminant.py
import math

def srgb_to_linear(c):
    c = c/255.0
    if c <= 0.04045:
        return c/12.92
    return ((c+0.055)/1.055)**2.4

def linear_to_srgb(c_lin):
    if c_lin <= 0.0031308:
        c = 12.92 * c_lin
    else:
        c = 1.055 * (c_lin ** (1/2.4)) - 0.055
    return int(round(max(0, min(1, c)) * 255))

def luminance_rgb(rgb):
    r_lin = srgb_to_linear(rgb[0])
    g_lin = srgb_to_linear(rgb[1])
    b_lin = srgb_to_linear(rgb[2])
    return 0.2126*r_lin + 0.7152*g_lin + 0.0722*b_lin

def iso_color_with_new_r_precise(base_rgb, new_r):
    r_new_lin = srgb_to_linear(new_r)
    b_lin = srgb_to_linear(base_rgb[2])
    L = luminance_rgb(base_rgb)
    # solve for g_lin
    g_lin = (L - 0.2126*r_new_lin - 0.0722*b_lin) / 0.7152
    if g_lin < 0 or g_lin > 1:
        return None
    g_new = linear_to_srgb(g_lin)
    return (int(new_r), g_new, base_rgb[2])

if __name__ == "__main__":
    base = (170,150,120)   # change to test other bases
    print("Base:", base, "L=", luminance_rgb(base))
    results = []
    for dr in list(range(-60, 121, 5)):
        new_r = base[0] + dr
        if not (0 <= new_r <= 255): 
            continue
        p = iso_color_with_new_r_precise(base, new_r)
        if p is None: 
            continue
        deltaY = luminance_rgb(p) - luminance_rgb(base)
        results.append((dr, p, deltaY))
    # show best matches with large |dr| and small |deltaY|
    good = [r for r in results if abs(r[2]) < 0.002 and abs(r[0]) >= 20]
    good = sorted(good, key=lambda x: (-abs(x[0]), abs(x[2])))
    for dr, p, dy in good[:12]:
        print(f"dr={dr:+3d} -> {p}   ΔR={dr:+3d}  ΔY={dy:+0.6f}")
