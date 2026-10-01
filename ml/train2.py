"""
MouthTwin round 2: one U-Net, two jobs.

  * 33-way tooth map (0 background, 1..32 FDI): trained on the 574 clinician-labelled DENTEX OPGs (weight 3x),
    DENTEX disease OPGs (clinician outlines + teacher fill), and teacher-labelled hard / unlabelled OPGs
    (DENTEX unlabelled + Cluj restoration dataset).
  * 4 restoration maps (implant, prosthetic restoration, filling, root-canal treatment): trained on the Cluj OPGs
    (bounding-box labels), masked out for every other image.

Evaluation (all on images the model never trained on):
  E-val    60 DENTEX OPGs, full clinician labels (same set as round 1, directly comparable)
  D-hold   up to 150 DENTEX disease OPGs: hard teeth (caries, periapical lesions, impacted). Teacher (round 1) scored too.
  Cluj     333 OPGs incl. 180 from other clinics: restoration detection
"""
import json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

ROOT = Path(__file__).resolve().parent
P1 = ROOT / "dentex" / "proc"
P2 = ROOT / "dentex" / "proc2"
OUT = ROOT / "runs2"
OUT.mkdir(exist_ok=True)
H, W = 384, 768
NT, NR = 33, 4
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
STEPS = int(os.environ.get("MT_STEPS", "375"))
BS = int(os.environ.get("MT_BS", "8"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1); random.seed(1); np.random.seed(1)

FLIP = torch.arange(256)
for t in range(1, 9):
    FLIP[t], FLIP[8 + t] = 8 + t, t
    FLIP[16 + t], FLIP[24 + t] = 24 + t, 16 + t


def rd(p):
    return np.array(Image.open(p))


def build():
    """Returns dict of train tensors and eval sets."""
    xs, ys, rs, tw, rw, sw = [], [], [], [], [], []
    m1 = json.load(open(P1 / "meta.json"))
    ex, ey = [], []
    for m in m1:
        img = rd(P1 / "img" / f"{m['stem']}.png"); msk = rd(P1 / "mask" / f"{m['stem']}.png")
        if m["val"]:
            ex.append(img); ey.append(msk)
        else:
            xs.append(img); ys.append(msk); rs.append(np.zeros((H, W), np.uint8)); tw.append(1.0); rw.append(0.0); sw.append(3.0)
    n_e = len(xs)
    stats = json.load(open(P2 / "pseudo_stats.json"))
    meta2 = json.load(open(P2 / "meta2.json"))
    hx, hy = [], []
    cx, cr = [], []
    for m in meta2:
        uid = m["id"]
        if m["src"] == "cluj" and m["split"] == "eval":
            cx.append(rd(P2 / "img" / f"{uid}.png")); cr.append(rd(P2 / "rest" / f"{uid}.png")); continue
        st = stats.get(uid)
        if st is None or st["dup"] >= 0.97:
            continue
        if m["src"] == "dentex_dis" and m["split"] == "hold":
            hx.append(rd(P2 / "img" / f"{uid}.png")); hy.append(rd(P2 / "mask" / f"{uid}.png")); continue
        if not st["keep"]:
            continue
        xs.append(rd(P2 / "img" / f"{uid}.png")); ys.append(rd(P2 / "pseudo" / f"{uid}.png"))
        if m["src"] == "cluj":
            rs.append(rd(P2 / "rest" / f"{uid}.png")); rw.append(1.0); tw.append(0.5); sw.append(1.0)
        else:
            rs.append(np.zeros((H, W), np.uint8)); rw.append(0.0)
            if m["src"] == "dentex_dis":
                tw.append(1.0); sw.append(1.5)
            else:
                tw.append(0.5); sw.append(1.0)
    print("train", len(xs), "(clinician-labelled", n_e, ")  E-val", len(ex), " D-hold", len(hx), " Cluj-eval", len(cx), flush=True)
    T = lambda a: torch.from_numpy(np.stack(a))
    return {
        "x": T(xs), "y": T(ys), "r": T(rs), "tw": torch.tensor(tw), "rw": torch.tensor(rw), "sw": torch.tensor(sw),
        "ex": T(ex), "ey": T(ey), "hx": T(hx) if hx else None, "hy": T(hy) if hy else None,
        "cx": T(cx) if cx else None, "cr": T(cr) if cr else None,
    }


def augment(x, y, r):
    """x: B,1,H,W float 0..1 ; y: B,H,W long (255 = ignore); r: B,H,W long (bit field)."""
    B, d = x.shape[0], x.device
    flip = torch.rand(B, device=d) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(flip[:, None, None], FLIP.to(d)[y.flip(-1)], y)
    r = torch.where(flip[:, None, None], r.flip(-1), r)
    s = torch.empty(B, device=d).uniform_(0.85, 1.15)
    rot = torch.empty(B, device=d).uniform_(-8, 8) * math.pi / 180
    tx = torch.empty(B, device=d).uniform_(-0.07, 0.07)
    ty = torch.empty(B, device=d).uniform_(-0.09, 0.09)
    sh = torch.empty(B, device=d).uniform_(-0.05, 0.05)
    a = W / H
    th = torch.zeros(B, 2, 3, device=d)
    th[:, 0, 0] = torch.cos(rot) / s
    th[:, 0, 1] = (-torch.sin(rot) + sh) / s / a
    th[:, 1, 0] = torch.sin(rot) / s * a
    th[:, 1, 1] = torch.cos(rot) / s
    th[:, 0, 2] = tx
    th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # ignore outside the source frame (reflection padding would invent teeth with no label)
    inside = F.grid_sample(torch.ones_like(x), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] > 0.5
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    r = F.grid_sample(r[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    g = torch.empty(B, 1, 1, 1, device=d).uniform_(0.65, 1.5)
    x = x.clamp(1e-4, 1) ** g
    c = torch.empty(B, 1, 1, 1, device=d).uniform_(0.7, 1.35)
    b = torch.empty(B, 1, 1, 1, device=d).uniform_(-0.12, 0.12)
    x = (x - 0.5) * c + 0.5 + b
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=d).uniform_(0, 0.035)
    if random.random() < 0.35:
        x = F.avg_pool2d(x, 3, 1, 1)
    if random.random() < 0.3:  # low-quality scanner: down-up sample
        k = random.choice([2, 3])
        x = F.interpolate(F.interpolate(x, scale_factor=1 / k, mode="bilinear"), size=(H, W), mode="bilinear")
    y = torch.where(inside, y, torch.full_like(y, 255))
    return x.clamp(0, 1), y, r


def norm(x):
    return (x - 0.45) / 0.25


def tooth_loss(lt, y, sw):
    """lt: B,33,H,W ; y: B,H,W (255 ignore) ; sw: B per-sample weights."""
    valid = (y != 255)
    yy = torch.where(valid, y, torch.zeros_like(y))
    wts = torch.ones(NT, device=lt.device); wts[0] = 0.5
    ce = F.cross_entropy(lt, yy, weight=wts, reduction="none")
    ce = (ce * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)
    p = lt.softmax(1)
    oh = F.one_hot(yy, NT).permute(0, 3, 1, 2).float() * valid[:, None]
    pv = p * valid[:, None]
    ww = sw[:, None, None, None]
    inter = (pv * oh * ww).sum((0, 2, 3))
    den = ((pv + oh) * ww).sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    dice = 1 - (2 * inter + 1) / (den + 1)
    return (ce * sw).sum() / sw.sum().clamp(min=1e-6) + dice[present].mean()


def rest_loss(lr, rbits, rw):
    """lr: B,4,H,W logits ; rbits: B,H,W long ; rw: B (1 for images with restoration labels)."""
    if rw.sum() == 0:
        return lr.sum() * 0
    tgt = torch.stack([((rbits >> k) & 1).float() for k in range(NR)], 1)
    pos_w = torch.tensor([4.0, 3.0, 3.0, 4.0], device=lr.device)[None, :, None, None]
    bce = F.binary_cross_entropy_with_logits(lr, tgt, pos_weight=pos_w, reduction="none").mean((2, 3))  # B,4
    bce = (bce * rw[:, None]).sum() / (rw.sum() * NR)
    p = lr.sigmoid() * rw[:, None, None, None]
    t = tgt * rw[:, None, None, None]
    inter = (p * t).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    dice = 1 - (2 * inter + 1) / (den + 1)
    return bce * 5 + (dice[present].mean() if present.any() else 0)


def comps(mask):
    if ndi is None:
        return [mask], 1
    cc, k = ndi.label(mask)
    return cc, k


@torch.no_grad()
def predict(model, x, tta=False):
    with torch.autocast("cuda", enabled=dev.type == "cuda"):
        o = model(norm(x))
        if tta:
            of = model(norm(x.flip(-1)))
    o = o.float()
    if tta:
        of = of.float().flip(-1)
        pt = (o[:, :NT].softmax(1) + of[:, :NT][:, FLIP[:NT].to(dev)].softmax(1)) / 2
        pr = (o[:, NT:].sigmoid() + of[:, NT:].sigmoid()) / 2
        return pt, pr
    return o[:, :NT].softmax(1), o[:, NT:].sigmoid()


@torch.no_grad()
def eval_teeth(model, xv, yv, partial=False):
    """Per GT tooth: numbering correct (majority predicted label inside GT tooth), pixel recall, IoU. partial: 255 = unlabelled."""
    model.eval()
    fg_i = fg_u = 0.0
    dice, iou, rec = [], [], []
    correct = total = 0
    per_img = []
    for i in range(0, len(xv), 4):
        x = xv[i:i + 4].to(dev).float()[:, None] / 255
        y = yv[i:i + 4].to(dev).long()
        pt, _ = predict(model, x)
        pred = pt.argmax(1)
        for b in range(len(y)):
            ic = []
            for c in range(1, NT):
                g = y[b] == c
                if g.sum() == 0:
                    continue
                pc = pred[b] == c
                dice.append((2 * (pc & g).sum() / (pc.sum() + g.sum())).item())
                iou.append(((pc & g).sum() / (pc | g).sum()).item())
                rec.append(((pc & g).sum() / g.sum()).item())
                inside = pred[b][g]
                inside = inside[inside > 0]
                total += 1
                ok = len(inside) > 0 and torch.mode(inside).values.item() == c
                correct += int(ok); ic.append(dice[-1])
            if not partial:
                gf, pf = y[b] > 0, pred[b] > 0
                fg_i += (gf & pf).sum().item(); fg_u += gf.sum().item() + pf.sum().item()
            per_img.append(float(np.mean(ic)) if ic else 1.0)
    model.train()
    r = {"fdi_mean_dice": float(np.mean(dice)), "fdi_mean_iou": float(np.mean(iou)), "tooth_recall": float(np.mean(rec)),
         "fdi_numbering_acc": correct / max(1, total), "n_teeth": total, "worst_quartile_dice": float(np.mean(sorted(per_img)[:max(1, len(per_img) // 4)]))}
    if not partial:
        r["tooth_dice"] = 2 * fg_i / max(1, fg_u)
    return r


@torch.no_grad()
def eval_rest(model, xv, rv, thr=0.5):
    model.eval()
    names = ["implant", "prosthetic_restoration", "filling", "root_canal_treatment"]
    agg = {n: {"i": 0.0, "u": 0.0, "gt_c": 0, "gt_hit": 0, "pr_c": 0, "pr_hit": 0} for n in names}
    for i in range(0, len(xv), 4):
        x = xv[i:i + 4].to(dev).float()[:, None] / 255
        r = rv[i:i + 4].to(dev).long()
        _, pr = predict(model, x)
        for k, n in enumerate(names):
            gt = ((r >> k) & 1).bool(); pd = pr[:, k] > thr
            a = agg[n]
            a["i"] += (gt & pd).sum().item(); a["u"] += (gt | pd).sum().item()
            if ndi is None:
                continue
            for b in range(len(gt)):
                g, p = gt[b].cpu().numpy(), pd[b].cpu().numpy()
                cg, kg = ndi.label(g); cp, kp = ndi.label(p)
                for j in range(1, kg + 1):
                    m = cg == j
                    if m.sum() < 60:
                        continue
                    a["gt_c"] += 1; a["gt_hit"] += int((p & m).sum() / m.sum() >= 0.3)
                for j in range(1, kp + 1):
                    m = cp == j
                    if m.sum() < 60:
                        continue
                    a["pr_c"] += 1; a["pr_hit"] += int((g & m).sum() / m.sum() >= 0.3)
    model.train()
    out = {}
    for n, a in agg.items():
        out[n] = {"iou": a["i"] / max(1, a["u"]), "recall": a["gt_hit"] / max(1, a["gt_c"]), "precision": a["pr_hit"] / max(1, a["pr_c"]),
                  "n_gt": a["gt_c"]}
    return out


def main():
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    D = build()
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", in_channels=1, classes=NT + NR,
                     decoder_channels=(128, 64, 48, 32, 16)).to(dev)
    t = ROOT / "runs" / "teacher.pt"
    if t.exists():  # start from round-1 weights (tooth head copied, restoration head fresh)
        sd = torch.load(t, map_location="cpu")
        cur = model.state_dict()
        for k, v in sd.items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
            elif k in cur and v.shape[0] == NT:
                cur[k][:NT] = v
        model.load_state_dict(cur)
        print("initialised from round-1 weights", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=EPOCHS * STEPS, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    N = len(D["x"])
    probs = D["sw"] / D["sw"].sum()
    hist, best = [], -1.0
    t0 = time.time()
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
            loss = tooth_loss(out[:, :NT], y, tw) + rest_loss(out[:, NT:], r, rw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        rec = {"epoch": ep + 1, "train_loss": tot / STEPS, "time_min": (time.time() - t0) / 60}
        if (ep + 1) % 3 == 0 or ep == EPOCHS - 1:
            e = eval_teeth(model, D["ex"], D["ey"])
            rec["E_val"] = e
            score = e["fdi_mean_dice"]
            if D["hx"] is not None:
                h = eval_teeth(model, D["hx"], D["hy"], partial=True)
                rec["D_hold"] = h
                score = 0.5 * score + 0.5 * h["fdi_mean_dice"]
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
