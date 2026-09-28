"""
Download the DENTEX quadrant-enumeration subset (634 OPGs, per-tooth FDI polygons) and rasterise
training pairs. Only the needed zip members are fetched (HTTP range requests).

DENTEX: Hamamci et al., MICCAI 2023. CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike).
https://huggingface.co/datasets/ibrahimhamamci/DENTEX
"""
import io, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip"
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "dentex"
ANN_MEMBER = "training_data/quadrant_enumeration/train_quadrant_enumeration.json"
OUT = ROOT / "proc"
H, W = 384, 768


def fdi_class(q, t):
    return (q - 1) * 8 + t  # 1..32, 0 = background


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    ann_path = ROOT / ANN_MEMBER
    if not ann_path.exists():
        with RemoteZip(URL) as z:
            z.extract(ANN_MEMBER, ROOT)
    d = json.load(open(ann_path))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    images = sorted(d["images"], key=lambda im: im["id"])
    random.Random(42).shuffle(images)
    val_ids = {im["id"] for im in images[:60]}
    for sub in ("img", "mask", "demo"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    meta = []

    def work(chunk):
        z = RemoteZip(URL)
        for im in chunk:
            stem = Path(im["file_name"]).stem
            rec = {"stem": stem, "val": im["id"] in val_ids, "w": im["width"], "h": im["height"]}
            if (OUT / "mask" / f"{stem}.png").exists():
                meta.append(rec)
                continue
            for attempt in range(3):
                try:
                    raw = z.read(f"training_data/quadrant_enumeration/xrays/{im['file_name']}")
                    break
                except Exception as e:  # transient network error
                    print("retry", stem, e, flush=True)
                    z = RemoteZip(URL)
            else:
                continue
            img = Image.open(io.BytesIO(raw)).convert("L")
            sx, sy = W / img.width, H / img.height
            img.resize((W, H), Image.LANCZOS).save(OUT / "img" / f"{stem}.png")
            if rec["val"]:
                s = 1600 / img.width
                img.resize((1600, round(img.height * s)), Image.LANCZOS).save(OUT / "demo" / f"{stem}.jpg", quality=88)
            mask = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(mask)
            for a in sorted(anns.get(im["id"], []), key=lambda a: -a["area"]):
                seg = a["segmentation"]
                if isinstance(seg, str):
                    seg = json.loads(seg)
                cls = fdi_class(a["category_id_1"] + 1, a["category_id_2"] + 1)
                for poly in seg:
                    pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
                    if len(pts) >= 3:
                        dr.polygon(pts, fill=cls)
            mask.save(OUT / "mask" / f"{stem}.png")
            meta.append(rec)
            print("ok", stem, flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, [images[i::4] for i in range(4)]))
    json.dump(sorted(meta, key=lambda m: m["stem"]), open(OUT / "meta.json", "w"))
    print("DONE", len(meta), "val", sum(m["val"] for m in meta), flush=True)


if __name__ == "__main__":
    main()
