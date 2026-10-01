"""
Population statistics of how each FDI tooth looks on a normal OPG: long-axis angle, relative length and how far its
crown sits from the jaw's occlusal curve. The app compares a patient's teeth with these numbers and transfers only
the *differences* (a tipped molar, an impacted canine sitting deep in the bone, a short root) onto the reference
3D anatomy. Using population means rather than the CBCT's own projection removes the OPG's built-in projection tilt.

Input:  dentex/proc/{mask/*.png, meta.json} (634 clinician-labelled DENTEX OPGs, 768x384, values 0..32)
Output: ../src/data/toothPoseStats.json

The feature extraction here is mirrored exactly in src/features/pipeline/model/toothPose.ts.
"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "dentex" / "proc"
OUT = Path(__file__).resolve().parent.parent / "src" / "data" / "toothPoseStats.json"
W, H = 768, 384


def cls_to_fdi(c):
    return ((c - 1) // 8 + 1) * 10 + (c - 1) % 8 + 1


def tooth_axis(xs, ys, sx, upper):
    """xs, ys pixel coords; sx = horizontal stretch so that pixels are square in the original image."""
    px = xs * sx
    py = ys.astype(np.float64)
    cx, cy = px.mean(), py.mean()
    dx, dy = px - cx, py - cy
    sxx, syy, sxy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    width = 4 * np.sqrt(max(1e-6, tr / 2 - np.sqrt(max(0.0, tr * tr / 4 - det))))
    ux, uy = np.cos(ang), np.sin(ang)
    if uy < 0:
        ux, uy = -ux, -uy  # point down the image
    t = dx * ux + dy * uy
    t0, t1 = np.quantile(t, 0.01), np.quantile(t, 0.99)
    L = t1 - t0
    # bulk near each end (crown is wider than the root)
    band = 0.3 * L
    lo = (t <= t0 + band).sum() / max(band, 1e-6)
    hi = (t >= t1 - band).sum() / max(band, 1e-6)
    tilt_from_vertical = abs(np.arctan2(ux, uy))  # 0 = vertical
    # end lying lower on the image (t1) is the crown for upper teeth, the apex for lower teeth
    crown_hi = upper
    if tilt_from_vertical > np.deg2rad(55):
        crown_hi = hi > lo
    tc, ta = (t1, t0) if crown_hi else (t0, t1)
    C = (cx + ux * tc, cy + uy * tc)
    A = (cx + ux * ta, cy + uy * ta)
    ax, ay = A[0] - C[0], A[1] - C[1]
    # angle of the crown->apex direction from the jaw's normal apex direction (down for lower, up for upper);
    # positive = apex toward image +x
    theta = np.arctan2(ax, ay) if not upper else np.arctan2(ax, -ay)
    return {"C": C, "A": A, "L": float(L), "theta": float(theta), "elong": float(L / width)}


def robust_curve(pts):
    """y = a x^2 + b x + c through crown points, dropping outliers (impacted / unerupted teeth)."""
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    deg = 2 if len(x) >= 6 else 1 if len(x) >= 3 else 0
    keep = np.ones(len(x), bool)
    for _ in range(3):
        coef = np.polyfit(x[keep], y[keep], deg) if deg else np.array([np.median(y[keep])])
        r = y - np.polyval(coef, x)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r) < max(3.0 * 1.4826 * mad, 6.0)
        if keep.sum() < max(3, deg + 1):
            keep[:] = True
            break
    return coef


def image_features(lab, sx):
    out = {}
    for c in np.unique(lab):
        if c == 0:
            continue
        m = lab == c
        cc, k = ndi.label(m)
        if k == 0:
            continue
        sizes = ndi.sum(m, cc, range(1, k + 1))
        if sizes.max() < 180:
            continue
        ys, xs = np.nonzero(cc == 1 + int(sizes.argmax()))
        fdi = cls_to_fdi(int(c))
        out[fdi] = tooth_axis(xs, ys, sx, fdi < 30)
    for upper in (True, False):
        jaw = {f: v for f, v in out.items() if (f < 30) == upper}
        if not jaw:
            continue
        base = [v["C"] for f, v in jaw.items() if f % 10 != 8]
        coef = robust_curve(base if len(base) >= 4 else [v["C"] for v in jaw.values()])
        for f, v in jaw.items():
            x0 = v["C"][0]
            d = v["C"][1] - np.polyval(coef, x0)
            v["dv"] = float(-d if upper else d)  # + = deeper (toward the apex)
            # apex direction a tooth standing perpendicular to the local occlusal curve would have (same convention
            # as theta). Head position (chin up / down) bends the curve and fans the teeth; this lets us remove it.
            slope = (np.polyval(coef, x0 + 1) - np.polyval(coef, x0 - 1)) / 2
            v["ctheta"] = float(np.arctan(slope) if upper else -np.arctan(slope))
    return out


def main():
    meta = json.load(open(ROOT / "meta.json"))
    feats = []
    for m in meta:
        lab = np.array(Image.open(ROOT / "mask" / f"{m['stem']}.png"))
        sx = (m["w"] / m["h"]) / (W / H)
        feats.append(image_features(lab, sx))
    fdis = sorted({f for fe in feats for f in fe})
    meanL = {f: np.median([fe[f]["L"] for fe in feats if f in fe]) for f in fdis}
    for _ in range(4):  # per-image scale k, then per-tooth mean length in scale-free units
        ks = []
        for fe in feats:
            r = [fe[f]["L"] / meanL[f] for f in fe]
            ks.append(float(np.median(r)) if r else 1.0)
        meanL = {f: float(np.median([fe[f]["L"] / k for fe, k in zip(feats, ks) if f in fe])) for f in fdis}
    # how much of a tooth's extra tilt is explained by the curve's extra slope (pooled over all teeth)
    med = lambda f, key: float(np.median([fe[f][key] for fe in feats if f in fe and key in fe[f]]))
    cth = {f: med(f, "ctheta") for f in fdis}
    th = {f: med(f, "theta") for f in fdis}
    X, Y = [], []
    for fe in feats:
        for f, v in fe.items():
            if "ctheta" in v and f % 10 != 8:
                X.append(v["ctheta"] - cth[f]); Y.append(v["theta"] - th[f])
    X, Y = np.array(X), np.array(Y)
    ok = np.abs(Y) < 0.5
    beta = float((X[ok] * Y[ok]).sum() / (X[ok] ** 2).sum())
    print("curve-slope beta", round(beta, 3), flush=True)
    stats = {}
    for f in fdis:
        rows = [(fe[f], k) for fe, k in zip(feats, ks) if f in fe]
        th = np.array([r["theta"] - beta * (r.get("ctheta", cth[f]) - cth[f]) for r, _ in rows])
        rl = np.array([r["L"] / k / meanL[f] for r, k in rows])
        dv = np.array([r["dv"] / k for r, k in rows])
        def rob(a):
            med = float(np.median(a))
            return med, float(1.4826 * np.median(np.abs(a - med)))
        stats[str(f)] = {
            "n": len(rows), "L": round(meanL[f], 3), "ctheta": round(cth[f], 4), "elong": round(float(np.median([r["elong"] for r, _ in rows])), 3),
            "theta": [round(v, 4) for v in rob(th)], "relL": [round(v, 4) for v in rob(rl)], "dv": [round(v, 3) for v in rob(dv)],
        }
    json.dump({"source": "DENTEX quadrant-enumeration, 634 clinician-labelled OPGs (CC BY-NC-SA 4.0)", "grid": [W, H], "beta": round(beta, 4), "teeth": stats},
              open(OUT, "w"), indent=1)
    for f in fdis:
        s = stats[str(f)]
        print(f, s["n"], "L", s["L"], "theta(deg) %.1f±%.1f" % tuple(np.rad2deg(s["theta"])), "relL", s["relL"], "dv", s["dv"])


if __name__ == "__main__":
    main()
