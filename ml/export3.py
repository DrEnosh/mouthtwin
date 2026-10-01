"""Export runs3/best.pt to ONNX and write the round-3 report (round 2 vs round 3 on every held-out set)."""
import json
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train3 as T
from export2 import Deploy, fdi

NT, NR, H, W, dev, ROOT, OUT = T.NT, T.NR, T.H, T.W, T.dev, T.ROOT, T.OUT


def net_from(p):
    n = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT + NR, decoder_channels=(128, 64, 48, 32, 16))
    n.load_state_dict(torch.load(p, map_location="cpu")); return n.eval()


def metrics(net, D):
    net.to(dev)
    r = {"E_val": T.eval_teeth(net, D["ex"], D["ey"])}
    if D["hx"] is not None:
        r["D_hold"] = T.eval_teeth(net, D["hx"], D["hy"], partial=True)
    if D["cx"] is not None:
        r["Cluj_eval"] = T.eval_rest(net, D["cx"], D["cr"])
    if D["sx"] is not None:
        r["STS_hold"] = T.eval_fg(net, D["sx"], D["sb"])
    if D["ax"] is not None:
        r["AKU_hold"] = T.eval_aku(net, D["ax"], D["ai"], D["at"])
    return r


def main():
    net = net_from(OUT / "best.pt")
    dep = Deploy(net).eval()
    dummy = torch.rand(1, 1, H, W)
    path = OUT / "mouthtwin-teeth.onnx"
    kw = dict(input_names=["image"], output_names=["labels", "fg", "conf", "rest"], opset_version=17)
    try:
        torch.onnx.export(dep, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(dep, dummy, path, **kw)
    print("onnx MB", path.stat().st_size / 1e6, flush=True)

    D = T.build3()
    res = {"round3": metrics(net, D), "round2": metrics(net_from(ROOT / "runs2" / "best.pt"), D)}
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    print("METRICS", json.dumps(res, indent=1), flush=True)

    hist = json.load(open(OUT / "history.json"))
    ev = [h for h in hist if "E_val" in h]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), dpi=130)
    tr = [h for h in hist if "train_loss" in h]
    ax[0].plot([h["epoch"] for h in tr], [h["train_loss"] for h in tr], color="#3f7f93")
    ax[0].set_title("Round-3 training loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.25)
    for key, sub, lab, col in [("E_val", "fdi_numbering_acc", "DENTEX numbering acc", "#6a9955"),
                               ("D_hold", "fdi_numbering_acc", "Hard teeth numbering acc", "#8f6bb3"),
                               ("D_hold", "worst_quartile_dice", "Hard teeth worst-quartile Dice", "#c9767e")]:
        pts = [h for h in ev if key in h]
        ax[1].plot([h["epoch"] for h in pts], [h[key][sub] for h in pts], label=lab, color=col)
        ax[1].axhline(res["round2"][key][sub], color=col, ls=":", lw=1)
    ax[1].set_ylim(0.5, 1); ax[1].legend(fontsize=7, loc="lower right"); ax[1].grid(alpha=.25)
    ax[1].set_title("Held-out teeth (dotted = round 2)")
    for key, sub, lab, col in [("STS_hold", "tooth_dice", "STS tooth Dice (new scanners)", "#e0b062"),
                               ("STS_hold", "worst_quartile_dice", "STS worst-quartile Dice", "#c9767e"),
                               ("AKU_hold", "type_acc", "AKU tooth-type accuracy", "#3f7f93"),
                               ("AKU_hold", "tooth_dice", "AKU tooth Dice", "#6a9955")]:
        pts = [h for h in ev if key in h]
        ax[2].plot([h["epoch"] for h in pts], [h[key][sub] for h in pts], label=lab, color=col)
        ax[2].axhline(res["round2"][key][sub], color=col, ls=":", lw=1)
    ax[2].set_ylim(0.5, 1); ax[2].legend(fontsize=7, loc="lower right"); ax[2].grid(alpha=.25)
    ax[2].set_title("New external test sets (dotted = round 2)")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png"); plt.close(fig)

    # overlays: round 2 vs round 3 on STS held-out OPGs with the lowest round-2 Dice (the hardest ones)
    rng = np.random.default_rng(3)
    pal = (rng.uniform(0.25, 1, (NT, 3)) * 255).astype(np.uint8); pal[0] = 0
    old = net_from(ROOT / "runs2" / "best.pt").to(dev); net.to(dev)

    def lab_of(model, img):
        x = img[None, None].to(dev).float() / 255
        pt, _ = T.predict(model, x)
        return pt.argmax(1)[0].cpu().numpy()

    worst = []
    for i in range(len(D["sx"])):
        l = lab_of(old, D["sx"][i]) > 0; g = D["sb"][i].numpy() > 0
        worst.append((2 * (l & g).sum() / max(1, l.sum() + g.sum()), i))
    picks = [i for _, i in sorted(worst)[:4]]
    fig, axes = plt.subplots(len(picks), 2, figsize=(16, 4.1 * len(picks)), dpi=100)
    for row, i in enumerate(picks):
        img = D["sx"][i].numpy() / 255.0
        for col, (model, name) in enumerate([(old, "round 2"), (net, "round 3")]):
            lab = lab_of(model, D["sx"][i])
            g = D["sb"][i].numpy() > 0
            d = 2 * ((lab > 0) & g).sum() / max(1, (lab > 0).sum() + g.sum())
            rgb = np.repeat(img[..., None], 3, -1); a = (lab > 0)[..., None] * 0.4
            a_ = axes[row, col]
            a_.imshow(np.clip(rgb * (1 - a) + pal[lab] / 255 * a, 0, 1)); a_.axis("off")
            a_.set_title(f"STS held-out (hard) - {name} - tooth Dice {d:.2f}", fontsize=8)
            for c in np.unique(lab):
                if c:
                    ys, xs = np.nonzero(lab == c)
                    a_.text(xs.mean(), ys.mean(), str(fdi(c)), color="white", fontsize=5, ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUT / "val_overlays.png"); plt.close(fig)
    print("REPORT3_DONE", flush=True)


if __name__ == "__main__":
    main()
