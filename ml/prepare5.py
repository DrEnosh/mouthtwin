"""
Round-5 data (all CC0 / CC BY 4.0). Teacher = round-4 model (runs4b/best.pt), flip TTA.

  khurshid  NOT USED in the end (kept for the record): most Roboflow copies are zoomed crops, not whole OPGs.
            Khurshid/Trachoo OPG restoration set (Mendeley mdvs6mjgf2, CC BY 4.0): polygons for crown, filling, post,
            root-canal-treated tooth, bridge, implant on 640x640 Roboflow exports. Roboflow made augmented copies (crops,
            90-degree turns); per source image we keep the one copy the teacher reads as a whole OPG (>= 14 teeth, sane
            layout). Numeric-only file names (probably Tufts Dental Database images, which have their own terms) are
            skipped. val+test images -> KH-hold (restoration masks for testing).
  dopg      DentalOPG-1550 (Mendeley rtt726b26d, CC BY 4.0): box labels are partial, so only the images are used
            (teacher teeth labels, no restoration loss). Patients under 16 (age in the file name) are skipped.
            File names contain patient names: ids are hashed and names are never stored.
  kairouan  Univ. Kairouan panoramic set part 1 (Mendeley 73n3kz2k4k, CC BY 4.0): 107 OPGs with a human outline per
            tooth (no numbers) -> tooth-vs-background truth + teacher FDI numbers inside, like STS labelled.
  hitl      Humans in the Loop teeth set (Kaggle, CC0; images Zenodo 4457648, CC BY 4.0): 598 OPGs, a human polygon per
            tooth with Universal numbers 1-32 -> FDI. 60 OPGs -> HITL-hold (full FDI test).

Writes dentex/proc5/{img,pseudo,fg,rest}/<id>.png and meta5.json.
"""
import hashlib, json, os, random, re, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
from pseudo_label import sanity
from pseudo3 import teacher_probs, NT, NR

ROOT = Path(__file__).resolve().parent
D5 = ROOT / "data5"
P5 = ROOT / "dentex" / "proc5"
H, W = 384, 768
CONF = 0.6
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for d in ("img", "pseudo", "fg", "rest"):
    (P5 / d).mkdir(parents=True, exist_ok=True)


def hid(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def gray(path):
    return Image.open(path).convert("L").resize((W, H), Image.BILINEAR)


def teach(arrs):
    p = teacher_probs(model, np.stack(arrs))
    conf, lab = p.max(1)
    return lab.cpu().numpy().astype(np.uint8), conf.cpu().numpy()


def save(uid, img, pseudo, fg=None, rest=None):
    Image.fromarray(img).save(P5 / "img" / f"{uid}.png")
    Image.fromarray(pseudo).save(P5 / "pseudo" / f"{uid}.png")
    if fg is not None:
        Image.fromarray(fg).save(P5 / "fg" / f"{uid}.png")
    if rest is not None:
        Image.fromarray(rest).save(P5 / "rest" / f"{uid}.png")


def poly_mask(polys, sx, sy):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for pts in polys:
        if len(pts) >= 3:
            d.polygon([(x * sx, y * sy) for x, y in pts], fill=1)
    return np.array(im) > 0


# ---------------------------------------------------------------- Khurshid (restoration polygons)
KH_BIT = {0: 2, 1: 4, 2: 8, 3: 8, 4: 2, 5: 1}  # crown, filling, post->RCT, RCT tooth->RCT, bridge->prosthetic, implant


def khurshid(meta):
    base = D5 / "khurshid" / "Dataset_x" / "Dataset"
    by = {}
    for split in ("train", "val", "test"):
        for f in sorted((base / split / "images").glob("*")):
            stem = re.sub(r"_(jpe?g|png|bmp)\..*$", "", f.name, flags=re.I)
            by.setdefault(stem, []).append((split, f))
    kept = skipped_num = rejected = 0
    for stem, files in by.items():
        if stem.isdigit():
            skipped_num += 1
            continue
        imgs = [np.array(gray(f)) for _, f in files]
        lab, conf = teach(imgs)
        best, score = None, -1
        for k, (split, f) in enumerate(files):
            ok, n, frag = sanity(lab[k])
            s = n + float(conf[k].mean())
            if ok and n >= 14 and s > score:
                best, score = k, s
        if best is None:
            rejected += 1
            continue
        split, f = files[best]
        rest = np.zeros((H, W), np.uint8)
        lf = f.parent.parent / "labels" / (f.stem + ".txt")
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            c = int(p[0]); v = list(map(float, p[1:]))
            m = poly_mask([list(zip(v[0::2], v[1::2]))], W, H)
            rest[m] |= KH_BIT[c]
        pseudo = lab[best].copy(); pseudo[conf[best] < CONF] = 255
        uid = "kh_" + hid(stem)
        save(uid, imgs[best], pseudo, rest=rest)
        meta.append({"id": uid, "src": "khurshid", "split": "hold" if split in ("val", "test") else "train"})
        kept += 1
    print("khurshid kept", kept, "numeric-name skipped", skipped_num, "not a whole OPG", rejected, flush=True)


# ---------------------------------------------------------------- DentalOPG-1550 (images only)
def dopg(meta):
    base = D5 / "dopg1550" / "yolo_x"
    kept = young = bad = 0
    files = [f for s in ("train", "valid", "test") for f in sorted((base / s / "images").glob("*.jpg"))]
    for i in range(0, len(files), 8):
        chunk = []
        for f in files[i:i + 8]:
            m = re.search(r"-(\d+)\s*[yY]", f.name)
            if m and int(m.group(1)) < 16:
                young += 1
                continue
            lf = f.parent.parent / "labels" / (f.stem + ".txt")
            if lf.exists() and any(l.split()[:1] == ["8"] for l in lf.read_text().splitlines()):  # 'Mixed Dentition'
                young += 1
                continue
            chunk.append(f)
        if not chunk:
            continue
        imgs = [np.array(gray(f)) for f in chunk]
        lab, conf = teach(imgs)
        for k, f in enumerate(chunk):
            ok, n, frag = sanity(lab[k])
            if not ok or n < 14:
                bad += 1
                continue
            pseudo = lab[k].copy(); pseudo[conf[k] < CONF] = 255
            uid = "dopg_" + hid(f.name)
            save(uid, imgs[k], pseudo)
            meta.append({"id": uid, "src": "dopg", "split": "train"})
            kept += 1
    print("dopg kept", kept, "under 16 / mixed dentition skipped", young, "teacher rejected", bad, flush=True)


# ---------------------------------------------------------------- Kairouan (human tooth outlines, no numbers)
def kairouan(meta):
    base = D5 / "kairouan"
    j = json.load(open(base / "annotations.json"))
    kept = 0
    for v in j["_via_img_metadata"].values():
        f = base / v["filename"]
        if not f.exists():
            continue
        w0, h0 = Image.open(f).size
        polys = []
        for r in v["regions"]:
            s = r["shape_attributes"]
            if "all_points_x" in s:
                polys.append(list(zip(s["all_points_x"], s["all_points_y"])))
        if len(polys) < 8:
            continue
        fgm = poly_mask(polys, W / w0, H / h0)
        img = np.array(gray(f))
        lab, conf = teach([img])
        pseudo = lab[0].copy(); pseudo[conf[0] < CONF] = 255
        pseudo[~fgm] = 0
        pseudo[fgm & (pseudo == 0)] = 255
        uid = "kai_" + hid(v["filename"])
        save(uid, img, pseudo, fg=fgm.astype(np.uint8))
        meta.append({"id": uid, "src": "kairouan", "split": "train"})
        kept += 1
    print("kairouan kept", kept, flush=True)


# ---------------------------------------------------------------- Humans in the Loop (Universal-numbered polygons)
def uni_to_cls(n):
    """Universal 1..32 -> our class (q-1)*8+t. 1=FDI 18 ... 8=11, 9=21 ... 16=28, 17=38 ... 24=31, 25=41 ... 32=48."""
    if 1 <= n <= 8:
        q, t = 1, 9 - n
    elif 9 <= n <= 16:
        q, t = 2, n - 8
    elif 17 <= n <= 24:
        q, t = 3, 25 - n
    else:
        q, t = 4, n - 24
    return (q - 1) * 8 + t


def hitl(meta):
    anns = sorted(a for a in (D5 / "hitl").rglob("ann/*.json") if "Teeth Segmentation PNG" not in str(a))
    if not anns:
        print("hitl: no annotation files found", flush=True)
        return
    rnd = random.Random(5)
    hold = set(a.name for a in rnd.sample(anns, min(60, len(anns))))
    kept = 0
    for a in anns:
        j = json.load(open(a))
        img_path = a.parent.parent / "img" / a.name[:-5]
        if not img_path.exists():
            continue
        h0, w0 = j["size"]["height"], j["size"]["width"]
        lab = np.zeros((H, W), np.uint8)
        objs = []
        for o in j["objects"]:
            try:
                n = int(o["classTitle"])
            except ValueError:
                continue
            if o.get("geometryType") != "polygon" or not 1 <= n <= 32:
                continue
            m = poly_mask([o["points"]["exterior"]], W / w0, H / h0)
            objs.append((uni_to_cls(n), m))
        if len(objs) < 8:
            continue
        for c, m in sorted(objs, key=lambda t: -t[1].sum()):
            lab[m] = c
        img = np.array(gray(img_path))
        uid = "hitl_" + hid(a.name)
        save(uid, img, lab, fg=(lab > 0).astype(np.uint8))
        meta.append({"id": uid, "src": "hitl", "split": "hold" if a.name in hold else "train"})
        kept += 1
    print("hitl kept", kept, "hold", sum(1 for m in meta if m["src"] == "hitl" and m["split"] == "hold"), flush=True)


if __name__ == "__main__":
    model = smp.Unet("mobilenet_v2", encoder_weights=None, in_channels=1, classes=NT + NR, decoder_channels=(128, 64, 48, 32, 16))
    model.load_state_dict(torch.load(ROOT / "runs4b" / "best.pt", map_location="cpu"))
    model.to(dev).eval()
    only = sys.argv[1:] or ["dopg", "kairouan", "hitl"]  # khurshid: Roboflow copies are mostly zoomed crops, dropped
    mp = P5 / "meta5.json"
    meta = [m for m in (json.load(open(mp)) if mp.exists() else []) if m["src"] not in only]
    with torch.no_grad():
        for name in only:
            globals()[name](meta)
    json.dump(meta, open(mp, "w"))
    from collections import Counter
    print("PREP5_DONE", dict(Counter((m["src"], m["split"]) for m in meta)), flush=True)
