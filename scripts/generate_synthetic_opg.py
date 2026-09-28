"""
Generate a SYNTHETIC panoramic dental radiograph (OPG) plus matching annotations.

Why synthetic: the demo must not use real patient images or scraped data.
Every structure here is drawn procedurally from textbook-average tooth sizes,
so the "precomputed inference" shipped with the demo is exact ground truth for
this image, not the output of a trained model.

Outputs
  src/data/demo/opg-synthetic.jpg      1600x800 greyscale JPEG
  src/data/demo/demo-inference.json    annotations in the MouthTwin inference schema

Run:  python3 scripts/generate_synthetic_opg.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

W, H = 1600, 800
S = 2  # supersampling factor
RNG = np.random.default_rng(7)

OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "data" / "demo"

# Average permanent-tooth dimensions (mm), per position 1..8 within a quadrant.
# width = mesiodistal crown width, crown = crown height, root = root length,
# opg_roots = number of roots that read as separate on a panoramic image.
UPPER = dict(
    width=[8.6, 6.6, 7.6, 7.0, 6.6, 10.2, 9.0, 8.4],
    crown=[10.5, 9.0, 10.0, 8.5, 7.8, 7.5, 7.0, 6.5],
    root=[13.0, 13.0, 17.0, 14.0, 14.0, 13.0, 12.5, 11.0],
    opg_roots=[1, 1, 1, 1, 1, 2, 2, 2],
)
LOWER = dict(
    width=[5.2, 5.8, 7.0, 7.0, 7.2, 11.2, 10.5, 10.0],
    crown=[9.0, 9.5, 11.0, 8.5, 8.0, 7.5, 7.0, 6.5],
    root=[12.5, 14.0, 16.0, 14.0, 14.5, 14.0, 13.0, 11.0],
    opg_roots=[1, 1, 1, 1, 1, 2, 2, 2],
)
KIND = ["incisor", "incisor", "canine", "premolar", "premolar", "molar", "molar", "molar"]
# Panoramic images compress the anterior region horizontally.
ANTERIOR_SQUEEZE = [0.85, 0.85, 0.92, 1, 1, 1, 1, 1]

PX_MM_X = 7.5  # horizontal pixels per mm
PX_MM_Y = 9.0  # vertical pixels per mm (OPGs magnify vertically more than horizontally)
GAP_MM = 0.35
BITE_GAP = 8.0  # px between upper and lower occlusal edges (bite block)
CX = W / 2


def y_occ(x: float) -> float:
    """Occlusal plane: the classic panoramic 'smile' curve."""
    u = (x - CX) / 560.0
    return 410.0 - 55.0 * u * u


def y_occ_slope(x: float) -> float:
    u = (x - CX) / 560.0
    return -2 * 55.0 * u / 560.0


# ----------------------------------------------------------------------------- geometry helpers


def chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        pts = np.empty((len(pts) * 2, 2))
        pts[0::2] = q
        pts[1::2] = r
    return pts


def crown_profile(kind: str) -> list[tuple[float, float]]:
    """Crown outline, left cervical -> occlusal -> right cervical. (x in widths, y in crown heights)."""
    if kind == "incisor":
        return [(-0.40, 1.0), (-0.50, 0.55), (-0.50, 0.15), (-0.45, 0.02), (-0.2, 0.0),
                (0.2, 0.0), (0.45, 0.02), (0.50, 0.15), (0.50, 0.55), (0.40, 1.0)]
    if kind == "canine":
        return [(-0.40, 1.0), (-0.50, 0.50), (-0.42, 0.20), (-0.15, 0.05), (0.0, 0.0),
                (0.15, 0.05), (0.42, 0.20), (0.50, 0.50), (0.40, 1.0)]
    if kind == "premolar":
        return [(-0.40, 1.0), (-0.50, 0.45), (-0.45, 0.12), (-0.25, 0.01), (0.0, 0.07),
                (0.25, 0.01), (0.45, 0.12), (0.50, 0.45), (0.40, 1.0)]
    return [(-0.42, 1.0), (-0.50, 0.40), (-0.47, 0.10), (-0.33, 0.0), (-0.15, 0.07),
            (0.02, 0.0), (0.18, 0.07), (0.34, 0.0), (0.47, 0.10), (0.50, 0.40), (0.42, 1.0)]


def root_profile(n: int) -> list[tuple[float, float]]:
    """Root outline from left cervical to right cervical. (x in widths, y as fraction of root length)."""
    if n == 1:
        return [(-0.40, 0.0), (-0.34, 0.35), (-0.20, 0.80), (-0.07, 1.0), (0.07, 1.0),
                (0.20, 0.80), (0.34, 0.35), (0.40, 0.0)]
    return [(-0.42, 0.0), (-0.42, 0.40), (-0.35, 0.85), (-0.26, 1.0), (-0.14, 0.95),
            (-0.09, 0.50), (0.0, 0.28), (0.09, 0.50), (0.14, 0.95), (0.26, 1.0),
            (0.35, 0.85), (0.42, 0.40), (0.42, 0.0)]


def pulp_profile(n: int) -> list[tuple[float, float, str]]:
    """Pulp outline. Third value marks whether y is in crown ('c') or root ('r') units."""
    if n == 1:
        return [(-0.17, 0.42, "c"), (0.17, 0.42, "c"), (0.19, 0.0, "r"), (0.05, 0.92, "r"),
                (-0.05, 0.92, "r"), (-0.19, 0.0, "r")]
    return [(-0.25, 0.45, "c"), (0.25, 0.45, "c"), (0.27, 0.12, "r"), (0.25, 0.90, "r"),
            (0.21, 0.90, "r"), (0.15, 0.20, "r"), (-0.15, 0.20, "r"), (-0.21, 0.90, "r"),
            (-0.25, 0.90, "r"), (-0.27, 0.12, "r")]


def tooth_local(kind: str, n_roots: int, w: float, ch: float, rl: float, distal_sign: float):
    """Return (outline, crown_poly, pulp) in local px coords: x across, y from occlusal (0) toward apex."""

    def bend(x: float, y: float) -> tuple[float, float]:
        # distal root curvature, increasing toward the apex
        if y > ch:
            t = (y - ch) / rl
            x += distal_sign * 0.12 * w * t * t
        return x, y

    crown = [(x * w, y * ch) for x, y in crown_profile(kind)]
    roots = [bend(x * w, ch + y * rl) for x, y in root_profile(n_roots)]
    # outline: crown (left cervical -> right cervical) then roots back (right -> left)
    outline = crown[:-1] + [crown[-1]] + list(reversed(roots))[1:-1]
    pulp = []
    for x, y, unit in pulp_profile(n_roots):
        yy = y * ch if unit == "c" else ch + y * rl
        pulp.append(bend(x * w, yy))
    return np.array(outline), np.array(crown), np.array(pulp)


def to_image(pts: np.ndarray, cx: float, y0: float, upper: bool, angle: float) -> np.ndarray:
    x = pts[:, 0]
    y = -pts[:, 1] if upper else pts[:, 1]
    ca, sa = math.cos(angle), math.sin(angle)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return np.stack([cx + xr, y0 + yr], axis=1)


# ----------------------------------------------------------------------------- canvas helpers


def blank() -> Image.Image:
    return Image.new("L", (W * S, H * S), 0)


def poly_mask(points, closed=True) -> np.ndarray:
    im = blank()
    d = ImageDraw.Draw(im)
    pts = [(float(x) * S, float(y) * S) for x, y in points]
    d.polygon(pts, fill=255)
    return np.asarray(im, dtype=np.float32) / 255.0


def line_mask(points, width_px: float) -> np.ndarray:
    im = blank()
    d = ImageDraw.Draw(im)
    pts = [(float(x) * S, float(y) * S) for x, y in points]
    d.line(pts, fill=255, width=int(width_px * S), joint="curve")
    return np.asarray(im, dtype=np.float32) / 255.0


def ellipse_mask(cx, cy, rx, ry) -> np.ndarray:
    im = blank()
    d = ImageDraw.Draw(im)
    d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=255)
    return np.asarray(im, dtype=np.float32) / 255.0


def blur(a: np.ndarray, sigma_px: float) -> np.ndarray:
    return ndimage.gaussian_filter(a, sigma_px * S)


def smooth_curve(pts, n=200):
    """Catmull-Rom through control points."""
    pts = np.asarray(pts, dtype=float)
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    segs = len(pts) - 1
    for i in range(segs):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        for t in np.linspace(0, 1, max(2, n // segs), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.array(out)


# ----------------------------------------------------------------------------- layout


def layout_quadrant(spec, upper: bool, side: int):
    """side = -1 for viewer-left (patient right), +1 for viewer-right (patient left)."""
    quadrant = {(True, -1): 1, (True, 1): 2, (False, 1): 3, (False, -1): 4}[(upper, side)]
    teeth = []
    edge = CX + side * (GAP_MM * PX_MM_X / 2)
    for i in range(8):
        w = spec["width"][i] * ANTERIOR_SQUEEZE[i] * PX_MM_X
        cx = edge + side * w / 2
        edge = cx + side * (w / 2 + GAP_MM * PX_MM_X)
        occ = y_occ(cx) + (-BITE_GAP / 2 if upper else BITE_GAP / 2)
        angle = math.atan(y_occ_slope(cx)) * 0.9
        # posterior teeth tip slightly toward the midline on OPGs
        angle += side * (0.02 * i if upper else -0.015 * i)
        teeth.append(dict(
            fdi=quadrant * 10 + i + 1,
            kind=KIND[i],
            upper=upper,
            cx=cx,
            y0=occ,
            angle=angle,
            w=w,
            ch=spec["crown"][i] * PX_MM_Y,
            rl=spec["root"][i] * PX_MM_Y,
            n_roots=spec["opg_roots"][i],
            distal=side,
        ))
    return teeth


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teeth = (layout_quadrant(UPPER, True, -1) + layout_quadrant(UPPER, True, 1)
             + layout_quadrant(LOWER, False, 1) + layout_quadrant(LOWER, False, -1))

    img = np.zeros((H * S, W * S), dtype=np.float32)

    # ---- soft tissue / background haze
    yy, xx = np.mgrid[0:H * S, 0:W * S].astype(np.float32) / S
    haze = np.exp(-(((xx - CX) / 700) ** 2 + ((yy - 470) / 380) ** 2))
    img += 0.10 * haze

    # ---- skull base / zygomatic arches
    for side in (-1, 1):
        zyg = [(CX + side * 740, 150), (CX + side * 600, 140), (CX + side * 440, 150), (CX + side * 360, 175)]
        img += 0.13 * blur(line_mask(smooth_curve(zyg, 60), 26), 9)
        # articular eminence / glenoid fossa region
        img += 0.10 * blur(ellipse_mask(CX + side * 660, 130, 90, 40), 10)

    # ---- cervical spine ghosts at both edges and faint midline ghost
    for side in (-1, 1):
        img += 0.13 * blur(poly_mask([(CX + side * 760, 180), (CX + side * 800, 180),
                                      (CX + side * 800, 800), (CX + side * 750, 800)]), 18)
    img += 0.05 * blur(poly_mask([(CX - 70, 250), (CX + 70, 250), (CX + 90, 800), (CX - 90, 800)]), 35)

    # ---- maxilla
    upper_crest = lambda x: y_occ(x) - BITE_GAP / 2 - 8.5 * PX_MM_Y - 10  # noqa: E731
    xs = np.linspace(CX - 560, CX + 560, 60)
    maxilla = [(CX - 600, 175)] + [(x, upper_crest(x)) for x in xs] + [(CX + 600, 175), (CX + 520, 150), (CX - 520, 150)]
    maxilla_m = blur(poly_mask(maxilla), 6)
    img += 0.26 * maxilla_m
    tex = blur(RNG.standard_normal(img.shape).astype(np.float32), 1.6) * 1.4 + blur(RNG.standard_normal(img.shape).astype(np.float32), 4.0) * 2.5
    img += 0.05 * tex * maxilla_m

    # maxillary sinuses, nasal cavity, septum, hard palate
    for side in (-1, 1):
        img -= 0.11 * blur(ellipse_mask(CX + side * 330, 180, 150, 70), 22)
        img += 0.07 * blur(line_mask(smooth_curve([(CX + side * 190, 250), (CX + side * 330, 262),
                                                    (CX + side * 470, 240)], 40), 5), 3)
    img -= 0.09 * blur(poly_mask([(CX - 120, 40), (CX + 120, 40), (CX + 105, 175), (CX - 105, 175)]), 22)
    img += 0.08 * blur(line_mask([(CX, 45), (CX, 170)], 8), 4)
    palate = smooth_curve([(CX - 470, 196), (CX - 200, 182), (CX, 178), (CX + 200, 182), (CX + 470, 196)], 80)
    img += 0.24 * blur(line_mask(palate, 12), 4)

    # ---- mandible (left half built explicitly, right half mirrored)
    def lower_crest(x):
        return y_occ(x) + BITE_GAP / 2 + 8.0 * PX_MM_Y + 10

    crest = [(x, lower_crest(x)) for x in np.linspace(CX, CX - 470, 30)]
    left_half = crest + [
        (330, 470), (300, 400), (282, 300), (268, 205), (246, 214), (205, 250), (170, 232),
        (150, 200), (132, 166), (108, 172), (100, 205), (116, 262), (104, 380), (94, 520),
        (102, 588), (132, 628), (220, 668), (350, 708), (500, 740), (650, 758), (CX, 764),
    ]
    right_half = [(2 * CX - x, y) for x, y in left_half]
    right_half = [(2 * CX - x, y) for x, y in left_half]
    loop = left_half + list(reversed(right_half))[1:-1]  # one closed outline across the midline
    mand = poly_mask(chaikin(np.array(loop, dtype=float), 3))
    mand_soft = blur(mand, 2)
    inner = ndimage.binary_erosion(mand > 0.5, iterations=int(11 * S)).astype(np.float32)
    cortex = blur(mand - inner, 3)
    img += 0.30 * mand_soft + 0.28 * cortex
    img += 0.06 * tex * mand_soft

    # mandibular canal (dark band with bright borders)
    for side in (-1, 1):
        pts = [(CX + side * (CX - x), y) for x, y in [(222, 385), (262, 470), (330, 548), (430, 590), (540, 602), (598, 590)]]
        c = smooth_curve(pts, 120)
        img += 0.07 * blur(line_mask(c, 16), 1.5)
        img -= 0.12 * blur(line_mask(c, 9), 1.5)
        img -= 0.06 * blur(ellipse_mask(c[-1][0], c[-1][1], 9, 8), 2)

    # ---- teeth (each drawn in its own cropped window for speed)
    annotations = []
    for t in teeth:
        outline, crown, pulp = tooth_local(t["kind"], t["n_roots"], t["w"], t["ch"], t["rl"], t["distal"])
        pulp = pulp * np.array([0.8, 1.0])
        o_img = to_image(chaikin(outline, 3), t["cx"], t["y0"], t["upper"], t["angle"])
        c_img = to_image(chaikin(crown, 2), t["cx"], t["y0"], t["upper"], t["angle"])
        p_img = to_image(chaikin(pulp, 2), t["cx"], t["y0"], t["upper"], t["angle"])

        pad = 14
        x0 = int(max(0, o_img[:, 0].min() - pad)); x1 = int(min(W, o_img[:, 0].max() + pad))
        y0 = int(max(0, o_img[:, 1].min() - pad)); y1 = int(min(H, o_img[:, 1].max() + pad))
        cw, chh = (x1 - x0) * S, (y1 - y0) * S

        def local_mask(pts):
            im = Image.new("L", (cw, chh), 0)
            ImageDraw.Draw(im).polygon([((x - x0) * S, (y - y0) * S) for x, y in pts], fill=255)
            return np.asarray(im, dtype=np.float32) / 255.0

        m_tooth = local_mask(o_img)
        m_crown = local_mask(c_img) * m_tooth
        m_pulp = local_mask(p_img) * m_tooth
        mt = m_tooth > 0.5
        ring = ndimage.binary_dilation(mt, iterations=int(2.5 * S)) & ~mt
        lamina = ndimage.binary_dilation(mt, iterations=int(4.5 * S)) & ~mt & ~ring
        crown_band = ndimage.binary_dilation(m_crown > 0.5, iterations=int(6 * S))

        # thickness shading: centre of the tooth is thicker, so brighter
        dt = ndimage.distance_transform_edt(mt)
        dt = np.sqrt(dt / (dt.max() + 1e-6))
        enamel_shell = (m_crown > 0.5) & ~ndimage.binary_erosion(m_crown > 0.5, iterations=int(3.2 * S))

        win = img[y0 * S:y1 * S, x0 * S:x1 * S]
        win -= 0.09 * ndimage.gaussian_filter((ring & ~crown_band).astype(np.float32), S)
        win += 0.05 * ndimage.gaussian_filter((lamina & ~crown_band).astype(np.float32), S)
        base = 0.34 + 0.05 * RNG.random()
        tooth_val = base + 0.20 * dt + 0.35 * win
        tooth_val += 0.20 * ndimage.gaussian_filter(m_crown, 1.4 * S)
        tooth_val += 0.12 * ndimage.gaussian_filter(enamel_shell.astype(np.float32), 1.2 * S)
        tooth_val -= 0.13 * ndimage.gaussian_filter(m_pulp, 1.6 * S)
        soft = ndimage.gaussian_filter(m_tooth, 0.8 * S)
        win[:] = win * (1 - soft) + np.maximum(win, tooth_val) * soft

        o_ann = to_image(chaikin(outline, 2), t["cx"], t["y0"], t["upper"], t["angle"])
        xs_, ys_ = o_ann[:, 0], o_ann[:, 1]
        apex = to_image(np.array([[0.0, t["ch"] + t["rl"]]]), t["cx"], t["y0"], t["upper"], t["angle"])[0]
        cej = to_image(np.array([[0.0, t["ch"]]]), t["cx"], t["y0"], t["upper"], t["angle"])[0]
        annotations.append(dict(
            fdi=t["fdi"],
            bbox=[round(xs_.min() / W, 5), round(ys_.min() / H, 5),
                  round((xs_.max() - xs_.min()) / W, 5), round((ys_.max() - ys_.min()) / H, 5)],
            polygon=[[round(x / W, 5), round(y / H, 5)] for x, y in o_ann[::2]],
            occlusal=[round(t["cx"] / W, 5), round(t["y0"] / H, 5)],
            cej=[round(cej[0] / W, 5), round(cej[1] / H, 5)],
            apex=[round(apex[0] / W, 5), round(apex[1] / H, 5)],
        ))

    # ---- global imaging look: blur, noise, tone curve, vignette
    img = blur(img, 1.5)
    img += 0.18 * blur(img, 14)  # scatter glow
    img += 0.018 * RNG.standard_normal(img.shape).astype(np.float32)
    vign = np.clip(1.0 - 0.35 * (((xx - CX) / 820) ** 2 + ((yy - 400) / 520) ** 2), 0.4, 1)
    img *= vign
    img = np.clip(img, 0, 1.2)
    img = img / 1.05
    img = np.clip(img, 0, 1) ** 0.95

    out = Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)

    # side marker + synthetic notice, as a real OPG carries an R/L marker
    d = ImageDraw.Draw(out)
    try:
        f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    except OSError:
        f_big = f_small = ImageFont.load_default()
    d.text((46, 34), "R", fill=225, font=f_big)
    d.text((W - 360, H - 38), "SYNTHETIC DEMO IMAGE - NOT A PATIENT", fill=150, font=f_small)

    out.save(OUT_DIR / "opg-synthetic.jpg", quality=86, optimize=True, progressive=True)

    # occlusal curve in normalised coords: y = a x^2 + b x + c
    xs_n = np.linspace(0.15, 0.85, 50)
    ys_n = np.array([y_occ(x * W) / H for x in xs_n])
    a, b, c = np.polyfit(xs_n, ys_n, 2)

    doc = dict(
        schemaVersion="1.0",
        source="precomputed-demo",
        notes="Ground-truth geometry of a procedurally generated synthetic OPG. Not produced by a trained model.",
        image=dict(width=W, height=H, modality="OPG", synthetic=True),
        occlusalCurve=dict(a=round(a, 6), b=round(b, 6), c=round(c, 6)),
        teeth=sorted(annotations, key=lambda t: t["fdi"]),
    )
    (OUT_DIR / "demo-inference.json").write_text(json.dumps(doc, indent=1))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
