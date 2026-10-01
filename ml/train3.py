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
    D = T2.build()
    # round-2 training images: tooth-vs-background is known wherever the tooth label is known
    xs, ys, rs, tw, rw, sw = [], [], [], [], [], []
    stats = json.load(open(P3 / "stats3.json"))
    meta = json.load(open(P3 / "meta3.json"))
    sx, sb = [], []
    # cap the unlabelled STS share (RAM: every image is kept in memory)
    cap = int(os.environ.get("MT_STS_U", "1500"))
    su = [m["id"] for m in meta if m["src"] == "sts_u"]
    random.Random(3).shuffle(su)
    skip = set(su[cap:])
    for m in meta:
        if m["id"] in skip:
            continue
        uid = m["id"]
        if m["split"] == "hold":
            sx.append(rd(P3 / "img" / f"{uid}.png")); sb.append(rd(P3 / "bin" / f"{uid}.png")); continue
        st = stats.get(uid)
        if not st or not st["keep"]:
            continue
        y = rd(P3 / "pseudo" / f"{uid}.png"); fg = rd(P3 / "fg" / f"{uid}.png")
        r = np.where(fg == 255, 0, FGK + FGV * (fg == 1)).astype(np.uint8)
        xs.append(rd(P3 / "img" / f"{uid}.png")); ys.append(y); rs.append(r); rw.append(0.0)
        if m["src"] == "sts_l":
            tw.append(0.8); sw.append(1.2)
        else:
            tw.append(0.5); sw.append(0.8)
    aku = json.load(open(P3 / "aku" / "meta.json"))
    ax, ai, at = [], [], []
    for r in aku:
        uid = r["id"]
        if r["split"] == "hold":
            ax.append(rd(P3 / "aku" / "img" / f"{uid}.png")); ai.append(rd(P3 / "aku" / "inst" / f"{uid}.png")); at.append(r["types"]); continue
        st = stats.get(uid)
        if not st or not st["keep"]:
            continue
        y = rd(P3 / "pseudo" / f"{uid}.png"); fg = rd(P3 / "fg" / f"{uid}.png")
        rr = np.where(fg == 255, 0, FGK + FGV * (fg == 1)).astype(np.uint8)
        xs.append(rd(P3 / "aku" / "img" / f"{uid}.png")); ys.append(y); rs.append(rr); rw.append(0.0); tw.append(1.0); sw.append(2.0)
    print("round-3 extra train", len(xs), " STS-hold", len(sx), " AKU-hold", len(ax), flush=True)
    if xs:
        Tn = lambda a: torch.from_numpy(np.stack(a))
        D["x"] = torch.cat([D["x"], Tn(xs)]); D["y"] = torch.cat([D["y"], Tn(ys)])
        D["r"] = torch.cat([D["r"], Tn(rs)]); D["tw"] = torch.cat([D["tw"], torch.tensor(tw)])
        D["rw"] = torch.cat([D["rw"], torch.tensor(rw)]); D["sw"] = torch.cat([D["sw"], torch.tensor(sw)])
    D["sx"] = torch.from_numpy(np.stack(sx)) if sx else None
    D["sb"] = torch.from_numpy(np.stack(sb)) if sb else None
    D["ax"] = torch.from_numpy(np.stack(ax)) if ax else None
    D["ai"] = torch.from_numpy(np.stack(ai)) if ai else None
    D["at"] = at
    return D


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
    torch.save(model.state_dict(), OUT / "best.pt")
    hist.append(rec0); print(json.dumps(rec0), flush=True)
    for ep in range(EPOCHS):
        tot = 0.0
        for s in range(STEPS):
            idx = torch.multinomial(probs, BS, replacement=True)
            x = D["x"][idx].to(dev).float()[:, None] / 255
            y = D["y"][idx].to(dev).long(); r = D["r"][idx].to(dev).long()
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
        hist.append(rec)
        json.dump(hist, open(OUT / "history.json", "w"), indent=1)
        print(json.dumps(rec), flush=True)
    print("TRAINING_DONE best score", best, flush=True)


if __name__ == "__main__":
    main()
