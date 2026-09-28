"""Export best.pt to ONNX for the browser, and write the report: metrics, loss plot, overlays."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dentex" / "proc"
OUT = ROOT / "runs"
H, W, NC = 384, 768, 33


class Deploy(nn.Module):
    """Normalises input and returns compact outputs: labels (uint8), tooth probability, label confidence."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # x: 1x1xHxW, 0..1
        logits = self.net((x - 0.45) / 0.25)
        p = logits.softmax(1)
        conf, lab = p.max(1)
        fg = 1 - p[:, 0]
        return lab.to(torch.uint8), fg, conf


def fdi(c):
    return (int(c) - 1) // 8 * 10 + 10 + (int(c) - 1) % 8 + 1


def main():
    net = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NC, decoder_channels=(128, 64, 48, 32, 16))
    net.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu"))
    net.eval()
    dep = Deploy(net).eval()
    dummy = torch.rand(1, 1, H, W)
    onnx_path = OUT / "mouthtwin-teeth.onnx"
    kw = dict(input_names=["image"], output_names=["labels", "fg", "conf"], opset_version=17)
    try:
        torch.onnx.export(dep, dummy, onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(dep, dummy, onnx_path, **kw)
    print("onnx MB", onnx_path.stat().st_size / 1e6)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        o = s.run(None, {"image": dummy.numpy()})
        with torch.no_grad():
            ref = dep(dummy)
        print("onnx label agreement", float((o[0] == ref[0].numpy()).mean()))
    except Exception as e:
        print("onnxruntime check skipped", e)

    hist = json.load(open(OUT / "history.json"))
    ev = [h for h in hist if "fdi_mean_dice" in h]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
    ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train loss", color="#3f7f93")
    ax[0].plot([h["epoch"] for h in ev], [h["val_loss"] for h in ev], label="val loss", color="#c9767e")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("CE + Dice loss"); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[0].set_title("Loss")
    for k, col, lab in [("tooth_dice", "#3f7f93", "tooth vs background Dice"), ("fdi_mean_dice", "#e0b062", "per-tooth (FDI) Dice"),
                        ("fdi_numbering_acc", "#6a9955", "FDI numbering accuracy")]:
        ax[1].plot([h["epoch"] for h in ev], [h[k] for h in ev], label=lab, color=col)
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("epoch"); ax[1].legend(loc="lower right"); ax[1].grid(alpha=.25)
    ax[1].set_title("Held-out validation (60 OPGs)")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png"); plt.close(fig)

    best = max(ev, key=lambda h: h["fdi_mean_dice"])
    json.dump(best, open(OUT / "metrics.json", "w"), indent=1)
    print("BEST", json.dumps(best))

    # overlays on held-out images
    meta = [m for m in json.load(open(DATA / "meta.json")) if m["val"]][:6]
    rng = np.random.default_rng(3)
    palette = (rng.uniform(0.25, 1, (NC, 3)) * 255).astype(np.uint8)
    palette[0] = 0
    fig, axes = plt.subplots(len(meta), 2, figsize=(14, 3.6 * len(meta)), dpi=110)
    for r, m in enumerate(meta):
        img = np.array(Image.open(DATA / "img" / f"{m['stem']}.png"), dtype=np.float32) / 255
        gt = np.array(Image.open(DATA / "mask" / f"{m['stem']}.png"))
        with torch.no_grad():
            lab, fg, conf = dep(torch.from_numpy(img)[None, None])
        pr = lab[0].numpy()
        for c, (mask, title) in enumerate([(gt, "Clinician labels (DENTEX)"), (pr, "MouthTwin model")]):
            rgb = np.repeat(img[..., None], 3, -1)
            col = palette[mask] / 255
            a = (mask > 0)[..., None] * 0.45
            axes[r, c].imshow(rgb * (1 - a) + col * a)
            axes[r, c].set_title(f"{title} · {m['stem']}", fontsize=9)
            axes[r, c].axis("off")
            for cl in np.unique(mask):
                if cl == 0:
                    continue
                ys, xs = np.nonzero(mask == cl)
                axes[r, c].text(xs.mean(), ys.mean(), str(fdi(cl)), color="white", fontsize=6, ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUT / "val_overlays.png"); plt.close(fig)
    print("REPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
