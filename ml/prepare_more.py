"""
Round-2 data for MouthTwin: harder / more varied OPGs and restoration labels.

  * Cluj OPG condition dataset (Muresanu, Hedesiu, Iacob et al., Diagnostics 2024; Zenodo 15487430,
    CC BY 4.0, "non-commercial research and educational purposes"): 1,808 real clinic OPGs with YOLO boxes.
    We use four of its 14 classes: 0 implant, 1 prosthetic restoration, 2 obturation (filling),
    3 endodontic treatment (root canal filling). Mirror: HF ismaelportog/Panoramic_Radiographs_for_Dental_Condition.
  * DENTEX quadrant-enumeration-disease (705 OPGs; only diseased teeth are outlined + numbered) and the
    1,571 unlabelled DENTEX OPGs (CC BY-NC-SA 4.0). They provide the difficult cases: caries, impacted,
    periapical lesions. Their missing tooth labels are filled by the teacher model in pseudo_label.py,
    and the human outlines always override the model where they exist.

Writes dentex/proc2/{img,mask,rest}/*.png (768x384) + meta2.json.
"""
import io, json, os, random, sys, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw
from remotezip import RemoteZip

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "dentex"
OUT = ROOT / "proc2"
H, W = 384, 768
CLUJ = "https://huggingface.co/datasets/ismaelportog/Panoramic_Radiographs_for_Dental_Condition/resolve/main/"
CLUJ_API = "https://huggingface.co/api/datasets/ismaelportog/Panoramic_Radiographs_for_Dental_Condition"
DZIP = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip"
DIS_JSON = "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
REST_CLASSES = {0: 1, 1: 2, 2: 4, 3: 8}  # yolo class -> bit  (implant, prosthetic, filling, root canal)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as e:
            print("retry", url[-50:], e, flush=True)
    raise RuntimeError(url)


def cluj(meta):
    files = [f["rfilename"] for f in json.loads(get(CLUJ_API))["siblings"]]
    labels = [f for f in files if "/labels/" in f and f.endswith(".txt")]

    def one(lab):
        parts = lab.split("/")  # train/labels/1001.txt | test_alte_cabinete/Ext-validation/labels/7.txt
        split_dir = parts[0]
        stem = Path(lab).stem
        base = lab.rsplit("/labels/", 1)[0]
        cand = [f"{base}/images/{stem}.jpg", f"{base}/images/{stem}.png"]
        ipath = next((c for c in cand if c in set_files), None)
        if ipath is None:
            return
        uid = f"cj_{split_dir}_{stem}"
        rec = {"id": uid, "src": "cluj", "split": "eval" if split_dir in ("valid", "test_alte_cabinete", "test") else "train",
               "ext": split_dir == "test_alte_cabinete"}
        if (OUT / "rest" / f"{uid}.png").exists():
            meta.append(rec); return
        img = Image.open(io.BytesIO(get(CLUJ + ipath))).convert("L").resize((W, H), Image.LANCZOS)
        rest = np.zeros((H, W), np.uint8)
        rows = [l.split() for l in get(CLUJ + lab).decode().splitlines() if l.strip()]
        for r in rows:
            c = int(r[0])
            if c not in REST_CLASSES:
                continue
            x, y, w, h = (float(v) for v in r[1:5])
            x0, x1 = max(0, round((x - w / 2) * W)), min(W, round((x + w / 2) * W))
            y0, y1 = max(0, round((y - h / 2) * H)), min(H, round((y + h / 2) * H))
            rest[y0:y1, x0:x1] |= REST_CLASSES[c]
        img.save(OUT / "img" / f"{uid}.png")
        Image.fromarray(rest).save(OUT / "rest" / f"{uid}.png")
        meta.append(rec)

    set_files = set(files)
    lim = int(os.environ.get("LIMIT", "0"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, labels[:lim] if lim else labels))
    print("cluj", len(meta), flush=True)


def dentex_extra(meta):
    ann_path = ROOT / DIS_JSON
    if not ann_path.exists():
        with RemoteZip(DZIP) as z:
            z.extract(DIS_JSON, ROOT)
    d = json.load(open(ann_path))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    dis = sorted(d["images"], key=lambda im: im["id"])
    random.Random(7).shuffle(dis)
    hold = {im["id"] for im in dis[:150]}
    with RemoteZip(DZIP) as z:
        unl = sorted(n for n in z.namelist() if n.startswith("training_data/unlabelled/xrays/") and n.endswith(".png"))

    jobs = [("dis", im) for im in dis] + [("unl", n) for n in unl]
    lim = int(os.environ.get("LIMIT", "0"))
    if lim:
        jobs = jobs[:lim // 2] + jobs[-lim // 2:]

    def work(chunk):
        z = RemoteZip(DZIP)
        for kind, item in chunk:
            if kind == "dis":
                im = item; stem = Path(im["file_name"]).stem
                uid = f"dd_{stem}"; member = f"training_data/quadrant-enumeration-disease/xrays/{im['file_name']}"
                rec = {"id": uid, "src": "dentex_dis", "split": "hold" if im["id"] in hold else "train"}
            else:
                stem = Path(item).stem; uid = f"du_{stem}"; member = item
                rec = {"id": uid, "src": "dentex_unl", "split": "train"}
            if (OUT / "img" / f"{uid}.png").exists():
                meta.append(rec); continue
            for attempt in range(3):
                try:
                    raw = z.read(member); break
                except Exception as e:
                    print("retry", uid, e, flush=True); z = RemoteZip(DZIP)
            else:
                continue
            img = Image.open(io.BytesIO(raw)).convert("L")
            sx, sy = W / img.width, H / img.height
            img.resize((W, H), Image.LANCZOS).save(OUT / "img" / f"{uid}.png")
            if kind == "dis":
                mask = Image.new("L", (W, H), 255)  # 255 = unknown
                dr = ImageDraw.Draw(mask)
                for a in sorted(anns.get(im["id"], []), key=lambda a: -a["area"]):
                    seg = a["segmentation"]
                    if isinstance(seg, str):
                        seg = json.loads(seg)
                    cls = a["category_id_1"] * 8 + a["category_id_2"] + 1
                    for poly in seg:
                        pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
                        if len(pts) >= 3:
                            dr.polygon(pts, fill=cls)
                mask.save(OUT / "mask" / f"{uid}.png")
            meta.append(rec)
            print("ok", uid, flush=True)

    n = 4
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(work, [jobs[i::n] for i in range(n)]))


def main():
    for sub in ("img", "mask", "rest"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    meta = []
    cluj(meta)
    dentex_extra(meta)
    json.dump(sorted(meta, key=lambda m: m["id"]), open(OUT / "meta2.json", "w"))
    from collections import Counter
    print("DONE", Counter((m["src"], m["split"]) for m in meta), flush=True)


if __name__ == "__main__":
    main()
