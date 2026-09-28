"""
MouthTwin tooth segmentation: 33 classes (0 = background, 1..32 = FDI teeth) on panoramic X-rays.
Data: DENTEX quadrant-enumeration subset (CC BY-NC-SA 4.0). Trains on GPU if available.

class index = (quadrant - 1) * 8 + tooth, e.g. 36 -> (3-1)*8+6 = 22
"""
import json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dentex" / "proc"
OUT = ROOT / "runs"
OUT.mkdir(exist_ok=True)
H, W = 384, 768
NC = 33
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 70
BS = int(__import__("os").environ.get("MT_BS", "8"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); random.seed(0); np.random.seed(0)

# horizontal flip swaps patient right/left: quadrant 1<->2, 4<->3
FLIP = torch.arange(NC)
for t in range(1, 9):
    FLIP[t], FLIP[8 + t] = 8 + t, t
    FLIP[16 + t], FLIP[24 + t] = 24 + t, 16 + t


def load(split_val):
    meta = json.load(open(DATA / "meta.json"))
    xs, ys, names = [], [], []
    for m in meta:
        if m["val"] != split_val:
            continue
        xs.append(np.array(Image.open(DATA / "img" / f"{m['stem']}.png"), dtype=np.uint8))
        ys.append(np.array(Image.open(DATA / "mask" / f"{m['stem']}.png"), dtype=np.uint8))
        names.append(m["stem"])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), names


def augment(x, y):
    """x: B,1,H,W float 0..1 ; y: B,H,W long. GPU-side augmentation."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(flip[:, None, None], FLIP.to(x.device)[y.flip(-1)], y)
    # affine: scale, rotation, translation
    s = torch.empty(B, device=x.device).uniform_(0.88, 1.12)
    r = torch.empty(B, device=x.device).uniform_(-6, 6) * math.pi / 180
    tx = torch.empty(B, device=x.device).uniform_(-0.06, 0.06)
    ty = torch.empty(B, device=x.device).uniform_(-0.08, 0.08)
    a = (W / H)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = torch.cos(r) / s
    theta[:, 0, 1] = -torch.sin(r) / s / a
    theta[:, 1, 0] = torch.sin(r) / s * a
    theta[:, 1, 1] = torch.cos(r) / s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
    # intensity: gamma, contrast, brightness, noise, blur
    g = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.7, 1.4)
    x = x.clamp(1e-4, 1) ** g
    c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.75, 1.3)
    b = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    x = ((x - 0.5) * c + 0.5 + b)
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0, 0.03)
    if random.random() < 0.3:
        x = F.avg_pool2d(x, 3, 1, 1)
    return x.clamp(0, 1), y


def norm(x):
    return (x - 0.45) / 0.25


def dice_loss(logits, y):
    p = logits.softmax(1)
    oh = F.one_hot(y, NC).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0
    d = 1 - (2 * inter + 1) / (den + 1)
    return d[present].mean()


@torch.no_grad()
def evaluate(model, xv, yv):
    model.eval()
    fg_i = fg_u = 0.0
    vloss = []
    cls_dice = []
    correct = total = 0
    for i in range(0, len(xv), 4):
        x = xv[i:i + 4].to(dev).float()[:, None] / 255
        y = yv[i:i + 4].to(dev).long()
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(norm(x))
        vloss.append((F.cross_entropy(logits.float(), y) + dice_loss(logits.float(), y)).item())
        pred = logits.argmax(1)
        pf, yf = pred > 0, y > 0
        fg_i += (pf & yf).sum().item()
        fg_u += pf.sum().item() + yf.sum().item()
        for b in range(len(y)):
            for c in range(1, NC):
                g = y[b] == c
                if g.sum() == 0:
                    continue
                pc = pred[b] == c
                cls_dice.append((2 * (pc & g).sum() / (pc.sum() + g.sum())).item())
                # tooth numbering accuracy: majority label of predicted tooth pixels inside the GT tooth
                inside = pred[b][g]
                inside = inside[inside > 0]
                if len(inside):
                    total += 1
                    correct += int(torch.mode(inside).values.item() == c)
                else:
                    total += 1
    model.train()
    return {
        "val_loss": float(np.mean(vloss)),
        "tooth_dice": 2 * fg_i / max(1, fg_u),
        "fdi_mean_dice": float(np.mean(cls_dice)),
        "fdi_numbering_acc": correct / max(1, total),
    }


def main():
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    xt, yt, _ = load(False)
    xv, yv, vnames = load(True)
    print("train", len(xt), "val", len(xv), flush=True)
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", in_channels=1, classes=NC,
                     decoder_channels=(128, 64, 48, 32, 16)).to(dev)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(xt) / BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    wts = torch.ones(NC, device=dev); wts[0] = 0.5
    hist, best = [], -1
    xt_d, yt_d = xt.to(dev), yt.to(dev)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(xt), device=dev)
        tot = n = 0
        for i in range(0, len(xt), BS):
            idx = perm[i:i + BS]
            x, y = augment(xt_d[idx].float()[:, None] / 255, yt_d[idx].long())
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                logits = model(norm(x))
                loss = F.cross_entropy(logits, y, weight=wts) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        rec = {"epoch": ep + 1, "train_loss": tot / n, "time_min": (time.time() - t0) / 60}
        if (ep + 1) % 2 == 0 or ep == EPOCHS - 1:
            rec.update(evaluate(model, xv, yv))
            if rec["fdi_mean_dice"] > best:
                best = rec["fdi_mean_dice"]
                torch.save(model.state_dict(), OUT / "best.pt")
        hist.append(rec)
        json.dump(hist, open(OUT / "history.json", "w"), indent=1)
        print(json.dumps(rec), flush=True)
    print("TRAINING_DONE best fdi_mean_dice", best, flush=True)


if __name__ == "__main__":
    main()
