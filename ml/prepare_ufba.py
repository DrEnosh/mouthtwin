"""
Round-4 data: UFBA-425 (CC BY 4.0), 425 panoramic X-rays with a human polygon mask per FDI-numbered tooth.
The 512x512 images are 2:1 panoramas squeezed to a square; they are stretched back to 768x384.
Writes dentex/proc4/ufba/{img,pseudo,fg}/<id>.png and meta.json (split: train / hold).
pseudo = FDI class (0..32), exact human labels. 40 images are held out for an honest numbering test.
"""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
U = ROOT / "ufba"
OUT = ROOT / "dentex" / "proc4" / "ufba"
H, W = 384, 768
for d in ("img", "pseudo", "fg"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
imgs = {p.stem: p for p in (U / "segmentation_xrays" / "segmentation_xrays").glob("*.jpg")}
mdirs = {}
for d in (U / "polygon_masks" / "polygon_masks").iterdir():
    if d.is_dir():
        for f in d.glob("*_11.ome.tiff"):
            mdirs[f.name[: -len("_11.ome.tiff")]] = d
ids = sorted(set(imgs) & set(mdirs))
print("images", len(imgs), "mask sets", len(mdirs), "matched", len(ids), flush=True)
rnd = random.Random(4)
hold = set(rnd.sample(ids, 40))
meta = []
for uid in ids:
    d = mdirs[uid]
    lab = np.zeros((512, 512), np.uint8)
    masks = []
    for q in range(1, 5):
        for n in range(1, 9):
            f = d / f"{uid}_{q}{n}.ome.tiff"
            if f.exists():
                m = np.array(Image.open(f)) > 0
                if m.shape == (512, 512) and m.sum() > 20:
                    masks.append(((q - 1) * 8 + n, m))
    if len(masks) < 4:
        continue
    for c, m in sorted(masks, key=lambda t: -t[1].sum()):  # small teeth drawn last
        lab[m] = c
    im = Image.open(imgs[uid]).convert("L").resize((W, H), Image.BILINEAR)
    lab = np.array(Image.fromarray(lab).resize((W, H), Image.NEAREST))
    im.save(OUT / "img" / f"{uid}.png")
    Image.fromarray(lab).save(OUT / "pseudo" / f"{uid}.png")
    Image.fromarray((lab > 0).astype(np.uint8)).save(OUT / "fg" / f"{uid}.png")
    meta.append({"id": uid, "split": "hold" if uid in hold else "train", "teeth": len(masks)})
json.dump(meta, open(OUT / "meta.json", "w"))
print("UFBA_DONE", len(meta), "train", sum(m["split"] == "train" for m in meta), "hold", sum(m["split"] == "hold" for m in meta), flush=True)
