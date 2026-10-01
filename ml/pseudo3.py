"""
Round-3 teacher pass. Teacher = round-2 model (runs2/best.pt), horizontal-flip TTA.

  STS unlabelled: teacher labels, pixels with confidence < 0.6 ignored, implausible results dropped.
  STS labelled:   the dentists' tooth-vs-background mask is the truth for WHERE teeth are; the teacher only
                  supplies the FDI number inside it (unsure pixels ignored). Outside the mask = background.
  AKU:            every human tooth polygon gets one FDI number: the teacher's most likely number among those
                  allowed by the human tooth type (e.g. a "premolar" can only be x4 / x5). Polygons where the
                  teacher disagrees with the type, implants, and duplicate numbers are ignored.

Writes dentex/proc3/pseudo/<id>.png (0..32, 255 = ignore), dentex/proc3/fg/<id>.png (0/1, 255 = unknown),
dentex/proc3/stats3.json.
"""
import faulthandler, json, sys, traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

from pseudo_label import sanity, FLIP as FLIP33

faulthandler.enable()
ROOT = Path(__file__).resolve().parent
P3 = ROOT / "dentex" / "proc3"
H, W, NT, NR = 384, 768, 33, 4
CONF = 0.6
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED = {1: [1], 2: [2], 3: [3], 4: [4, 5], 5: [6, 7, 8]}


def classes_for(t):
    return [q * 8 + n for q in range(4) for n in ALLOWED[t]]


def norm(x):
    return (x - 0.45) / 0.25


@torch.no_grad()
def teacher_probs(model, arr):
    x = torch.from_numpy(arr).to(dev).float()[:, None] / 255
    flip = FLIP33.to(dev)
    with torch.autocast("cuda", enabled=dev.type == "cuda"):
        p = model(norm(x)).float()[:, :NT].softmax(1)
        pf = model(norm(x.flip(-1))).float()[:, :NT].softmax(1).flip(-1)[:, flip]
    return ((p + pf) / 2)


def aku_labels(p, inst, types):
    """p: 33xHxW probs (torch, on device); inst: HxW instance ids; types: list (instance i+1 -> type)."""
    conf, lab = p.max(0)
    lab = lab.cpu().numpy().astype(np.uint8); conf = conf.cpu().numpy()
    out = np.zeros((H, W), np.uint8)
    out[(inst == 0) & (lab > 0) & (conf >= 0.8)] = 255  # teacher sure there is a tooth the annotators skipped
    fg = (inst > 0).astype(np.uint8)
    pn = p.cpu().numpy()
    picks = []
    for k, t in enumerate(types, start=1):
        m = inst == k
        if m.sum() < 30:
            continue
        if t == 6:  # implant: not a tooth class
            out[m] = 255; fg[m] = 255
            continue
        mass = pn[:, m].sum(1)
        tooth_mass = mass[1:].sum() + 1e-6
        allowed = classes_for(t)
        best = max(allowed, key=lambda c: mass[c])
        share = mass[best] / tooth_mass
        picks.append((best, share, k))
    used = {}
    for c, share, k in sorted(picks, key=lambda r: -r[1]):
        m = inst == k
        if share < 0.25 or c in used:
            out[m] = 255
        else:
            out[m] = c; used[c] = k
    return out, fg, len(used), len(picks)


@torch.no_grad()
def main():
    model = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT + NR, decoder_channels=(128, 64, 48, 32, 16))
    model.load_state_dict(torch.load(ROOT / "runs2" / "best.pt", map_location="cpu"))
    model.to(dev).eval()
    (P3 / "pseudo").mkdir(exist_ok=True); (P3 / "fg").mkdir(exist_ok=True)
    meta = json.load(open(P3 / "meta3.json"))
    aku = json.load(open(P3 / "aku" / "meta.json"))
    sp = P3 / "stats3.json"
    stats = json.load(open(sp)) if sp.exists() else {}

    # only the unlabelled STS OPGs that train3.py will use (same capped, seeded subset) - saves disk and time
    import os, random
    cap = int(os.environ.get("MT_STS_U", "1500"))
    su = [m["id"] for m in meta if m["src"] == "sts_u"]
    random.Random(3).shuffle(su)
    skip = set(su[cap:])
    todo = [m for m in meta if m["split"] == "train" and m["id"] not in stats and m["id"] not in skip]
    print("STS to label", len(todo), flush=True)
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        arr = np.stack([np.array(Image.open(P3 / "img" / f"{m['id']}.png")) for m in chunk])
        p = teacher_probs(model, arr)
        conf, lab = p.max(1)
        lab = lab.cpu().numpy().astype(np.uint8); conf = conf.cpu().numpy()
        for b, m in enumerate(chunk):
            try:
                uid = m["id"]
                ok, n, frag = sanity(lab[b])
                pseudo = lab[b].copy()
                pseudo[conf[b] < CONF] = 255
                if m["src"] == "sts_l":
                    gt = np.array(Image.open(P3 / "bin" / f"{uid}.png")) > 0
                    pseudo[~gt] = 0
                    pseudo[gt & (pseudo == 0)] = 255
                    fg = gt.astype(np.uint8)
                else:
                    fg = np.full((H, W), 255, np.uint8)
                Image.fromarray(pseudo).save(P3 / "pseudo" / f"{uid}.png")
                Image.fromarray(fg).save(P3 / "fg" / f"{uid}.png")
                stats[uid] = {"src": m["src"], "teeth": int(n), "frag": frag, "ok": bool(ok), "keep": bool(ok),
                              "mean_conf": float(conf[b].mean())}
            except Exception:
                print("FAILED", m["id"], traceback.format_exc(), flush=True)
        if (i // 8) % 20 == 0:
            json.dump(stats, open(sp, "w")); print(i, "/", len(todo), flush=True)
    json.dump(stats, open(sp, "w"))

    for r in aku:
        uid = r["id"]
        if uid in stats:
            continue
        img = np.array(Image.open(P3 / "aku" / "img" / f"{uid}.png"))
        inst = np.array(Image.open(P3 / "aku" / "inst" / f"{uid}.png"))
        p = teacher_probs(model, img[None])[0]
        lab, fg, n_ok, n_all = aku_labels(p, inst, r["types"])
        Image.fromarray(lab).save(P3 / "pseudo" / f"{uid}.png")
        Image.fromarray(fg).save(P3 / "fg" / f"{uid}.png")
        stats[uid] = {"src": "aku", "split": r["split"], "teeth": n_ok, "teeth_polygons": n_all, "keep": n_ok >= 4 and r["split"] == "train"}
    json.dump(stats, open(sp, "w"))
    from collections import Counter
    print("PSEUDO3_DONE", dict(Counter((s["src"], s["keep"]) for s in stats.values())), flush=True)
    a = [s for s in stats.values() if s["src"] == "aku"]
    print("AKU polygons numbered", sum(s["teeth"] for s in a), "of", sum(s["teeth_polygons"] for s in a), flush=True)


if __name__ == "__main__":
    main()
