"""
MouthTwin round 3: same network, more varied adult OPGs. Starts from the round-2 weights.

New training data (on top of everything round 2 used):
  * STS-2D-Tooth adult, 750 OPGs with dentist-checked tooth masks (+ teacher FDI numbers inside) and 2,650 unlabelled
    OPGs (teacher labels). Different country and scanners from DENTEX.
  * AKU, 192 OPGs with a human outline for every tooth, many partially edentulous / restored mouths
    (FDI from teacher votes constrained by the human tooth type).
  Human tooth-vs-background masks get an extra loss term (P(tooth) = 1 - P(background)).

Held-out tests (never trained on):
  E-val 60 DENTEX (full FDI), D-hold 150 hard DENTEX teeth, Cluj 433 restorations (as round 2),
  STS-hold 100 OPGs (tooth-vs-background Dice, human masks), AKU-hold 50 OPGs (tooth-vs-background Dice and
  tooth-TYPE accuracy per human outline - the human labels have no FDI numbers).
"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

import train2 as T2
from train2 import H, W, NT, NR, augment, norm, tooth_loss, rest_loss, eval_teeth, eval_rest, predict

ROOT = T2.ROOT
P3 = ROOT / "dentex" / "proc3"
OUT = ROOT / "runs3"
OUT.mkdir(exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
STEPS = int(os.environ.get("MT_STEPS", "375"))
BS = int(os.environ.get("MT_BS", "8"))
dev = T2.dev
FGK, FGV = 1 << 4, 1 << 5  # fg-known / fg-value bits packed into the restoration bit field (survive augmentation)
TYPE_OF = {n: (1 if n == 1 else 2 if n == 2 else 3 if n == 3 else 4 if n in (4, 5) else 5) for n in range(1, 9)}


def rd(p):
    return np.array(Image.open(p))


def build3():
    """Training rows are loaded lazily from disk (paths only); held-out sets are kept in memory."""
    P1, P2 = T2.P1, T2.P2
    rows = []  # (img, label, rest or None, fg or None, tw, rw, sw)
    ex, ey, hx, hy, cx, cr = [], [], [], [], [], []
    for m in json.load(open(P1 / "meta.json")):
        img, msk = P1 / "img" / f"{m['stem']}.png", P1 / "mask" / f"{m['stem']}.png"
        if m["val"]:
            ex.append(rd(img)); ey.append(rd(msk))
        else:
            rows.append((img, msk, None, None, 1.0, 0.0, 3.0))
    n_e = len(rows)
    stats2 = json.load(open(P2 / "pseudo_stats.json"))
    for m in json.load(open(P2 / "meta2.json")):
        uid = m["id"]
        if m["src"] == "cluj" and m["split"] == "eval":
            cx.append(rd(P2 / "img" / f"{uid}.png")); cr.append(rd(P2 / "rest" / f"{uid}.png")); continue
        st = stats2.get(uid)
        if st is None or st["dup"] >= 0.97:
            continue
        if m["src"] == "dentex_dis" and m["split"] == "hold":
            hx.append(rd(P2 / "img" / f"{uid}.png")); hy.append(rd(P2 / "mask" / f"{uid}.png")); continue
        if not st["keep"]:
            continue
        img, lab = P2 / "img" / f"{uid}.png", P2 / "pseudo" / f"{uid}.png"
        if m["src"] == "cluj":
            rows.append((img, lab, P2 / "rest" / f"{uid}.png", None, 0.5, 1.0, 1.0))
        elif m["src"] == "dentex_dis":
            rows.append((img, lab, None, None, 1.0, 0.0, 1.5))
        else:
            rows.append((img, lab, None, None, 0.5, 0.0, 1.0))
    n2 = len(rows)
    stats = json.load(open(P3 / "stats3.json"))
    meta = json.load(open(P3 / "meta3.json"))
    sx, sb = [], []
    cap = int(os.environ.get("MT_STS_U", "1500"))
    su = [m["id"] for m in meta if m["src"] == "sts_u"]
    random.Random(3).shuffle(su)
    skip = set(su[cap:])
    for m in meta:
        uid = m["id"]
        if uid in skip:
            continue
        if m["split"] == "hold":
            sx.append(rd(P3 / "img" / f"{uid}.png")); sb.append(rd(P3 / "bin" / f"{uid}.png")); continue
        st = stats.get(uid)
        if not st or not st["keep"]:
            continue
        w = (0.8, 1.2) if m["src"] == "sts_l" else (0.5, 0.8)
        rows.append((P3 / "img" / f"{uid}.png", P3 / "pseudo" / f"{uid}.png", None, P3 / "fg" / f"{uid}.png", w[0], 0.0, w[1]))
    ax, ai, at = [], [], []
    for r in json.load(open(P3 / "aku" / "meta.json")):
        uid = r["id"]
        if r["split"] == "hold":
            ax.append(rd(P3 / "aku" / "img" / f"{uid}.png")); ai.append(rd(P3 / "aku" / "inst" / f"{uid}.png")); at.append(r["types"]); continue
        st = stats.get(uid)
        if not st or not st["keep"]:
            continue
        rows.append((P3 / "aku" / "img" / f"{uid}.png", P3 / "pseudo" / f"{uid}.png", None, P3 / "fg" / f"{uid}.png", 1.0, 0.0, 2.0))
    print("train", len(rows), "(clinician-labelled", n_e, ", round-2 rows", n2, ", round-3 rows", len(rows) - n2, ")  E-val", len(ex),
          " D-hold", len(hx), " Cluj-eval", len(cx), " STS-hold", len(sx), " AKU-hold", len(ax), flush=True)
    S = lambda a: torch.from_numpy(np.stack(a)) if a else None
    return {
        "rows": rows, "tw": torch.tensor([r[4] for r in rows]), "rw": torch.tensor([r[5] for r in rows]), "sw": torch.tensor([r[6] for r in rows]),
        "ex": S(ex), "ey": S(ey), "hx": S(hx), "hy": S(hy), "cx": S(cx), "cr": S(cr),
        "sx": S(sx), "sb": S(sb), "ax": S(ax), "ai": S(ai), "at": at,
    }


_POOL = None


def load_rows(rows, idx):
    """Reads one batch from disk: image, tooth label, restoration bits (+ fg bits)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(4)

    def one(i):
        img, lab, rest, fg, *_ = rows[i]
        r = rd(rest) if rest is not None else np.zeros((H, W), np.uint8)
        if fg is not None:
            f = rd(fg)
            r = r | np.where(f == 255, 0, FGK + FGV * (f == 1)).astype(np.uint8)
        return rd(img), rd(lab), r

    out = list(_POOL.map(one, idx))
    return tuple(torch.from_numpy(np.stack([o[k] for o in out])) for k in range(3))


def fg_loss(lt, r):
    known = (r & FGK) > 0
    if not known.any():
        return lt.sum() * 0
    tgt = ((r & FGV) > 0).float()
    p_tooth = 1 - lt.softmax(1)[:, 0]
    bce = F.binary_cross_entropy(p_tooth.clamp(1e-5, 1 - 1e-5), tgt, reduction="none")
    bce = (bce * known).sum() / known.sum()
    inter = (p_tooth * tgt * known).sum(); den = ((p_tooth + tgt) * known).sum()
    return bce + (1 - (2 * inter + 1) / (den + 1))


@torch.no_grad()
def eval_fg(model, xv, bv):
    model.eval()
    i = u = 0.0
    per = []
    for k in range(0, len(xv), 4):
        x = xv[k:k + 4].to(dev).float()[:, None] / 255
        pt, _ = predict(model, x)
        pf = pt.argmax(1) > 0
        g = bv[k:k + 4].to(dev) > 0
        for b in range(len(g)):
            ii = (pf[b] & g[b]).sum().item(); uu = pf[b].sum().item() + g[b].sum().item()
            per.append(2 * ii / max(1, uu)); i += ii; u += uu
    model.train()
    return {"tooth_dice": 2 * i / max(1, u), "worst_quartile_dice": float(np.mean(sorted(per)[:max(1, len(per) // 4)])), "n_images": len(per)}


@torch.no_grad()
def eval_aku(model, xv, iv, types):
    """Per human tooth outline: predicted FDI (majority inside) has the right tooth type; plus tooth-vs-bg Dice."""
    model.eval()
    ok = tot = found = 0
    i = u = 0.0
    for k in range(0, len(xv), 4):
        x = xv[k:k + 4].to(dev).float()[:, None] / 255
        pt, _ = predict(model, x)
        pred = pt.argmax(1).cpu().numpy()
        for b in range(len(pred)):
            inst = iv[k + b].numpy(); ty = types[k + b]
            g = (inst > 0) & ~np.isin(inst, [j + 1 for j, t in enumerate(ty) if t == 6])
            pf = pred[b] > 0
            i += (pf & g).sum(); u += pf.sum() + g.sum()
            for j, t in enumerate(ty, start=1):
                if t == 6:
                    continue
                m = inst == j
                if m.sum() < 30:
                    continue
                tot += 1
                inside = pred[b][m]; inside = inside[inside > 0]
                if len(inside) < 0.3 * m.sum():
                    continue
                found += 1
                c = int(np.bincount(inside).argmax())
                ok += int(TYPE_OF[(c - 1) % 8 + 1] == t)
    model.train()
    return {"tooth_dice": 2 * i / max(1, u), "tooth_found": found / max(1, tot), "type_acc": ok / max(1, tot), "n_teeth": tot}


def main():
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    D = build3()
    model = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT + NR, decoder_channels=(128, 64, 48, 32, 16)).to(dev)
    model.load_state_dict(torch.load(ROOT / "runs2" / "best.pt", map_location="cpu"))
    print("initialised from round-2 weights", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=EPOCHS * STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    probs = D["sw"] / D["sw"].sum()
    hist, best = [], -1.0
    t0 = time.time()

    def evaluate(rec):
        e = eval_teeth(model, D["ex"], D["ey"]); rec["E_val"] = e
        parts = [e["fdi_mean_dice"]]
        if D["hx"] is not None:
            h = eval_teeth(model, D["hx"], D["hy"], partial=True); rec["D_hold"] = h; parts.append(h["fdi_mean_dice"])
        if D["sx"] is not None:
            s = eval_fg(model, D["sx"], D["sb"]); rec["STS_hold"] = s; parts.append(s["tooth_dice"])
        if D["ax"] is not None:
            a = eval_aku(model, D["ax"], D["ai"], D["at"]); rec["AKU_hold"] = a; parts.append(0.5 * (a["tooth_dice"] + a["type_acc"]))
        return float(np.mean(parts))

    rec0 = {"epoch": 0}
    best = evaluate(rec0); rec0["score"] = best
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), OUT / "best.pt")
    hist.append(rec0); print(json.dumps(rec0), flush=True)
    for ep in range(EPOCHS):
        tot = 0.0
        for s in range(STEPS):
            idx = torch.multinomial(probs, BS, replacement=True)
            bx, by, br = load_rows(D["rows"], idx.tolist())
            x = bx.to(dev).float()[:, None] / 255
            y = by.to(dev).long(); r = br.to(dev).long()
            tw = D["tw"][idx].to(dev); rw = D["rw"][idx].to(dev)
            x, y, r = augment(x, y, r)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                out = model(norm(x))
            out = out.float()
            loss = tooth_loss(out[:, :NT], y, tw) + rest_loss(out[:, NT:], r & 15, rw) + 0.5 * fg_loss(out[:, :NT], r)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        rec = {"epoch": ep + 1, "train_loss": tot / STEPS, "time_min": (time.time() - t0) / 60}
        if (ep + 1) % 3 == 0 or ep == EPOCHS - 1:
            score = evaluate(rec)
            if D["cx"] is not None and (ep + 1) % 6 == 0:
                rec["Cluj_eval"] = eval_rest(model, D["cx"], D["cr"])
            rec["score"] = score
            if score > best:
                best = score
                torch.save(model.state_dict(), OUT / "best.pt")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        hist.append(rec)
        json.dump(hist, open(OUT / "history.json", "w"), indent=1)
        print(json.dumps(rec), flush=True)
    print("TRAINING_DONE best score", best, flush=True)


if __name__ == "__main__":
    main()
