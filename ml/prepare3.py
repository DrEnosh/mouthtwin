"""
Round-3 data: more varied adult OPGs.

  * STS-2D-Tooth (Wang et al., Scientific Data 12:117, 2025; Zenodo 10.5281/zenodo.10597292, CC BY 4.0;
    HF mirror MedOtter/STS-2D-Tooth): adult subset only. 850 OPGs with dentist-checked tooth-vs-background
    masks and 2,650 unlabelled OPGs from a different country / scanner than DENTEX. The child subset is not used:
    the model has no classes for baby teeth, so mixed dentition would be taught wrongly.
  * AKU (see prepare_aku.py), already converted to dentex/proc3/aku by prepare_aku.py.

Writes dentex/proc3/{img,bin}/<id>.png (768x384) and dentex/proc3/meta3.json.
"""
import io, json, random, sys, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
P3 = ROOT / "dentex" / "proc3"
H, W = 384, 768
HF = "https://huggingface.co/datasets/MedOtter/STS-2D-Tooth/resolve/main/data/"
SPLITS = {"a_pxi_labeled": "sts_l", "a_pxi_unlabeled": "sts_u"}


def fetch(name, dst):
    if dst.exists() and dst.stat().st_size > 1e6:
        return
    print("downloading", name, flush=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(HF + name, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    tmp.replace(dst)


def main():
    import pyarrow.parquet as pq
    (P3 / "img").mkdir(parents=True, exist_ok=True); (P3 / "bin").mkdir(exist_ok=True); (P3 / "raw").mkdir(exist_ok=True)
    meta = []
    for split, tag in SPLITS.items():
        pf = P3 / "raw" / f"{split}.parquet"
        fetch(f"{split}-00000-of-00001.parquet", pf)
        f = pq.ParquetFile(pf)
        n = 0
        for batch in f.iter_batches(batch_size=64, columns=["image", "mask", "sample_id"]):
            for row in batch.to_pylist():
                uid = f"{tag}_{row['sample_id']}".replace("/", "_").replace(" ", "_")
                rec = {"id": uid, "src": tag, "split": "train"}
                if not (P3 / "img" / f"{uid}.png").exists():
                    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L").resize((W, H), Image.BICUBIC)
                    im.save(P3 / "img" / f"{uid}.png")
                    if row.get("mask"):
                        m = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])).convert("L")) > 0
                        Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST).save(P3 / "bin" / f"{uid}.png")
                meta.append(rec); n += 1
        print(split, n, flush=True)
    # hold out 100 of the labelled STS OPGs for testing tooth-vs-background
    lab = [m for m in meta if m["src"] == "sts_l"]
    random.Random(5).shuffle(lab)
    for m in lab[:100]:
        m["split"] = "hold"
    json.dump(meta, open(P3 / "meta3.json", "w"))
    print("PREP3_DONE", len(meta), flush=True)


if __name__ == "__main__":
    main()
