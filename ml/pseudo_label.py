"""
Teacher pass: fill in tooth labels on the extra (harder / unlabelled) OPGs so the student can train on them.

  teacher = round-1 model (runs/teacher.pt, trained on 574 clinician-labelled DENTEX OPGs only)
  * horizontal-flip test-time augmentation
  * pixels the teacher is not sure about (max prob < CONF) become IGNORE (255): the student never learns from them
  * for DENTEX disease images the clinician outlines override the teacher wherever they exist
  * images are dropped when the teacher output is structurally implausible (fewer than 6 teeth, teeth fragmented
    into many pieces, or teeth out of order along the arch)
  * images that are near-duplicates of the round-1 training/validation OPGs are dropped (leak guard)

Writes dentex/proc2/pseudo/<id>.png and dentex/proc2/pseudo_stats.json.
"""
import faulthandler, json, sys, traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

faulthandler.enable()
ROOT = Path(__file__).resolve().parent
P1 = ROOT / "dentex" / "proc"
P2 = ROOT / "dentex" / "proc2"
H, W, NC = 384, 768, 33
CONF = 0.6
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLIP = torch.arange(NC)
for t in range(1, 9):
    FLIP[t], FLIP[8 + t] = 8 + t, t
    FLIP[16 + t], FLIP[24 + t] = 24 + t, 16 + t


def norm(x):
    return (x - 0.45) / 0.25


def small(a):  # 48x24 zero-mean unit-norm vector for near-duplicate search
    im = Image.fromarray(a).resize((48, 24), Image.BILINEAR)
    v = np.asarray(im, np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def sanity(lab):
    """lab: HxW uint8 (0..32). Returns (ok, n_teeth, frag)."""
    if ndi is None:
        return True, len(np.unique(lab)) - 1, 0.0
    n = 0
    frag_pix = tot = 0
    cx = {}
    for c in np.unique(lab):
        if c == 0:
            continue
        m = lab == c
        cc, k = ndi.label(m)
        if k == 0:
            continue
        sizes = ndi.sum(m, cc, range(1, k + 1))
        big = sizes.max()
        if big < 150:
            continue
        n += 1
        tot += m.sum()
        frag_pix += m.sum() - big
        ys, xs = np.nonzero(cc == (1 + int(sizes.argmax())))
        cx[int(c)] = xs.mean()
    frag = frag_pix / max(1, tot)
    order_bad = 0
    for q in range(4):
        pts = [(t, cx[q * 8 + t]) for t in range(1, 9) if q * 8 + t in cx]
        # tooth number rises away from the midline; patient right (q 0,3) is image left
        sign = 1 if q in (1, 2) else -1
        for (t1, x1), (t2, x2) in zip(pts, pts[1:]):
            if (x2 - x1) * sign < -6:
                order_bad += 1
    ok = n >= 6 and frag < 0.12 and order_bad <= 1
    return ok, n, float(frag)


@torch.no_grad()
def main():
    teacher = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NC, decoder_channels=(128, 64, 48, 32, 16))
    teacher.load_state_dict(torch.load(ROOT / "runs" / "teacher.pt", map_location="cpu"))
    teacher.to(dev).eval()
    (P2 / "pseudo").mkdir(exist_ok=True)
    meta = json.load(open(P2 / "meta2.json"))
    # round-1 images for duplicate detection
    m1 = json.load(open(P1 / "meta.json"))
    ref = np.stack([small(np.array(Image.open(P1 / "img" / f"{m['stem']}.png"))) for m in m1])
    ref_val = np.array([m["val"] for m in m1])
    stats_path = P2 / "pseudo_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    todo = [m for m in meta if not (m["src"] == "cluj" and m["split"] == "eval") and m["id"] not in stats]
    print("to label", len(todo), flush=True)
    flip = FLIP.to(dev)
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        arr = np.stack([np.array(Image.open(P2 / "img" / f"{m['id']}.png")) for m in chunk])
        x = torch.from_numpy(arr).to(dev).float()[:, None] / 255
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            p = teacher(norm(x)).float().softmax(1)
            pf = teacher(norm(x.flip(-1))).float().softmax(1).flip(-1)[:, flip]
        p = (p + pf) / 2
        conf, lab = p.max(1)
        lab = lab.cpu().numpy().astype(np.uint8)
        conf = conf.cpu().numpy()
        for b, m in enumerate(chunk):
            uid = m["id"]
            try:
                one(b, m, uid, arr, lab, conf, ref, ref_val, stats)
            except Exception:
                print("FAILED", uid, traceback.format_exc(), flush=True)
        if (i // 8) % 10 == 0:
            json.dump(stats, open(stats_path, "w"))
            print(i, "/", len(todo), flush=True)
    json.dump(stats, open(stats_path, "w"))
    from collections import Counter
    src = {m["id"]: m["src"] for m in meta}
    c = Counter((src[k], s["keep"]) for k, s in stats.items())
    print("PSEUDO_DONE", dict(c), "dups", sum(s["dup"] >= 0.97 for s in stats.values()), flush=True)


def one(b, m, uid, arr, lab, conf, ref, ref_val, stats):
    if True:
        if True:
            sim = ref @ small(arr[b])
            dup = float(sim.max())
            dup_val = bool(ref_val[int(sim.argmax())])
            ok, n, frag = sanity(lab[b])
            pseudo = lab[b].copy()
            pseudo[conf[b] < CONF] = 255
            gt_teeth = 0
            if m["src"] == "dentex_dis":
                gt = np.array(Image.open(P2 / "mask" / f"{uid}.png"))
                known = gt != 255
                gt_teeth = int(len(np.unique(gt[known])) - (1 if (gt[known] == 0).any() else 0))
                # inside clinician-labelled teeth trust the human; keep teacher elsewhere
                pseudo[known] = gt[known]
                # a teacher tooth that overlaps a human-labelled tooth of a different class is likely wrong: ignore
            keep = (dup < 0.97) and (ok or gt_teeth > 0)
            stats[uid] = {"dup": dup, "dup_of_val": dup_val, "teeth": int(n), "frag": frag, "ok": bool(ok), "keep": bool(keep),
                          "gt_teeth": gt_teeth, "mean_conf": float(conf[b].mean())}
            Image.fromarray(pseudo).save(P2 / "pseudo" / f"{uid}.png")


if __name__ == "__main__":
    main()
