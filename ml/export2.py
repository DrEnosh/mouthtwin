"""Export runs2/best.pt to ONNX (teeth + restoration maps) and write the round-2 report."""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train2 as T

ROOT, OUT, H, W, NT, NR = T.ROOT, T.OUT, T.H, T.W, T.NT, T.NR
REST_NAMES = ["implant", "prosthetic_restoration", "filling", "root_canal_treatment"]


class Deploy(nn.Module):
    """labels (uint8 0..32), tooth probability, label confidence, restoration probabilities (uint8 0..255, 4 channels)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # x: 1x1xHxW, 0..1
        o = self.net((x - 0.45) / 0.25)
        p = o[:, :NT].softmax(1)
        conf, lab = p.max(1)
        rest = (o[:, NT:].sigmoid() * 255).round().clamp(0, 255).to(torch.uint8)
        return lab.to(torch.uint8), 1 - p[:, 0], conf, rest


def fdi(c):
    return (int(c) - 1) // 8 * 10 + 10 + (int(c) - 1) % 8 + 1


def main():
    net = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT + NR, decoder_channels=(128, 64, 48, 32, 16))
    net.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu"))
    net.eval()
    dep = Deploy(net).eval()
    dummy = torch.rand(1, 1, H, W)
    path = OUT / "mouthtwin-teeth.onnx"
    kw = dict(input_names=["image"], output_names=["labels", "fg", "conf", "rest"], opset_version=17)
    try:
        torch.onnx.export(dep, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(dep, dummy, path, **kw)
    print("onnx MB", path.stat().st_size / 1e6)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o = s.run(None, {"image": dummy.numpy()})
        with torch.no_grad():
            ref = dep(dummy)
        print("onnx label agreement", float((o[0] == ref[0].numpy()).mean()), "rest agreement", float((o[3] == ref[3].numpy()).mean()))
    except Exception as e:
        print("onnxruntime check skipped", e)

    dev = T.dev
    D = T.build()
    net.to(dev)
    res = {"new": {}, "teacher": {}}
    res["new"]["E_val"] = T.eval_teeth(net, D["ex"], D["ey"])
    if D["hx"] is not None:
        res["new"]["D_hold"] = T.eval_teeth(net, D["hx"], D["hy"], partial=True)
    if D["cx"] is not None:
        res["new"]["Cluj_eval"] = T.eval_rest(net, D["cx"], D["cr"])
        # external clinics only
        import json as _j
        meta2 = _j.load(open(T.P2 / "meta2.json"))
        ext_ids = [m["id"] for m in meta2 if m["src"] == "cluj" and m["split"] == "eval"]
        ext = torch.tensor([m["ext"] for m in meta2 if m["src"] == "cluj" and m["split"] == "eval"])
        if ext.any():
            res["new"]["Cluj_external_clinics"] = T.eval_rest(net, D["cx"][ext], D["cr"][ext])
    tp = ROOT / "runs" / "teacher.pt"
    if not tp.exists():
        tp = ROOT / "runs" / "best.pt"
    if tp.exists():
        tea = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT, decoder_channels=(128, 64, 48, 32, 16))
        tea.load_state_dict(torch.load(tp, map_location="cpu")); tea.to(dev).eval()
        res["teacher"]["E_val"] = T.eval_teeth(tea, D["ex"], D["ey"])
        if D["hx"] is not None:
            res["teacher"]["D_hold"] = T.eval_teeth(tea, D["hx"], D["hy"], partial=True)
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    print("METRICS", json.dumps(res, indent=1), flush=True)

    hist = json.load(open(OUT / "history.json"))
    ev = [h for h in hist if "E_val" in h]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), dpi=130)
    ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], color="#3f7f93")
    ax[0].set_title("Training loss (teeth CE+Dice, restorations BCE+Dice)"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.25)
    ax[1].plot([h["epoch"] for h in ev], [h["E_val"]["fdi_mean_dice"] for h in ev], label="E-val per-tooth Dice", color="#e0b062")
    ax[1].plot([h["epoch"] for h in ev], [h["E_val"]["fdi_numbering_acc"] for h in ev], label="E-val numbering acc", color="#6a9955")
    dh = [h for h in ev if "D_hold" in h]
    if dh:
        ax[1].plot([h["epoch"] for h in dh], [h["D_hold"]["fdi_mean_dice"] for h in dh], label="Hard teeth per-tooth Dice", color="#c9767e")
        ax[1].plot([h["epoch"] for h in dh], [h["D_hold"]["fdi_numbering_acc"] for h in dh], label="Hard teeth numbering acc", color="#8f6bb3")
    if "teacher" in res and "D_hold" in res["teacher"]:
        ax[1].axhline(res["teacher"]["D_hold"]["fdi_mean_dice"], color="#c9767e", ls=":", lw=1, label="round-1 model, hard teeth")
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7, loc="lower right"); ax[1].grid(alpha=.25); ax[1].set_title("Held-out tooth metrics")
    cj = [h for h in hist if "Cluj_eval" in h]
    for k, col in zip(REST_NAMES, ["#c9767e", "#e0b062", "#6a9955", "#3f7f93"]):
        if cj:
            ax[2].plot([h["epoch"] for h in cj], [h["Cluj_eval"][k]["recall"] for h in cj], color=col, label=k.replace("_", " ") + " recall")
    ax[2].set_ylim(0, 1); ax[2].legend(fontsize=7, loc="lower right"); ax[2].grid(alpha=.25)
    ax[2].set_title("Restoration detection on held-out clinics")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png"); plt.close(fig)

    # overlays
    rng = np.random.default_rng(3)
    pal = (rng.uniform(0.25, 1, (NT, 3)) * 255).astype(np.uint8); pal[0] = 0
    rcol = np.array([[255, 60, 60], [255, 210, 60], [80, 230, 80], [60, 220, 255]]) / 255

    def show(ax, img, lab, rest=None, title=""):
        rgb = np.repeat(img[..., None], 3, -1)
        a = (lab > 0)[..., None] * 0.4
        out = rgb * (1 - a) + pal[lab] / 255 * a
        if rest is not None:
            for k in range(NR):
                m = rest[k] > 127
                out[m] = out[m] * 0.55 + rcol[k] * 0.45
        ax.imshow(np.clip(out, 0, 1)); ax.set_title(title, fontsize=8); ax.axis("off")
        for c in np.unique(lab):
            if c:
                ys, xs = np.nonzero(lab == c)
                ax.text(xs.mean(), ys.mean(), str(fdi(c)), color="white", fontsize=5, ha="center", va="center")

    def run(imgs, idxs):
        outs = []
        for i in idxs:
            x = imgs[i:i + 1].to(dev).float()[:, None] / 255
            with torch.no_grad():
                lab, fg, conf, rest = dep.to(dev)(x)
            outs.append((imgs[i].numpy() / 255.0, lab[0].cpu().numpy(), rest[0].cpu().numpy()))
        return outs

    rows = []
    for i, (im, lab, rest) in enumerate(run(D["ex"], list(range(min(4, len(D["ex"])))))):
        rows.append((im, lab, None, f"DENTEX held-out #{i + 1}"))
    if D["hx"] is not None:
        for i, (im, lab, rest) in enumerate(run(D["hx"], list(range(0, len(D["hx"]), max(1, len(D["hx"]) // 5)))[:5])):
            rows.append((im, lab, None, f"Hard case (disease set) #{i + 1}"))
    if D["cx"] is not None:
        n_cls = (D["cr"] > 0).flatten(1).float().mean(1)
        pick = torch.argsort(-n_cls)[::max(1, len(n_cls) // 40)][:8]
        for i, (im, lab, rest) in zip(pick.tolist(), run(D["cx"], pick.tolist())):
            rows.append((im, lab, rest, f"Complex / restored case (Cluj) #{i}"))
    n = len(rows)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(16, 4.1 * ((n + 1) // 2)), dpi=105)
    for a, r in zip(axes.flat, rows):
        show(a, *r[:3], title=r[3])
    for a in list(axes.flat)[n:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "val_overlays.png"); plt.close(fig)
    print("REPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
