"""
AKU OPG dataset (Data in Brief 2024; Zenodo 10.5281/zenodo.10538750, CC BY 4.0 on Zenodo / CC BY-NC 4.0 in the
article; GitHub Niihhaa/Dataset): 250 permanent-dentition OPGs with a human-drawn polygon for every tooth, labelled by
tooth TYPE (central / lateral incisor, canine, premolar, molar) plus implants. No FDI numbers: pseudo3.py assigns
them from the round-2 model's votes, constrained to the human type and outline.

Writes dentex/proc3/aku/{img,inst}/<id>.png (768x384; inst = instance id 1..N) and dentex/proc3/aku/meta.json
(per image: instance types; 50 images held out for testing).
Usage: python prepare_aku.py <path to the cloned dataset folder> [out root]
"""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".") / "dentex" / "proc3" / "aku"
H, W = 384, 768
TYPES = {"central incisor": 1, "lateral incisor": 2, "canine": 3, "premolar": 4, "molar": 5, "implant": 6}


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True); (OUT / "inst").mkdir(parents=True, exist_ok=True)
    recs = []
    for js in sorted(SRC.glob("*/ann*/*.json")):
        d = json.load(open(js))
        folder = js.parent.parent
        name = Path(d["imagePath"].replace("\\", "/")).name
        ip = folder / "OPGs" / name
        if not ip.exists():
            cands = [p for p in (folder / "OPGs").iterdir() if p.stem == Path(name).stem]
            if not cands:
                print("missing image", js); continue
            ip = cands[0]
        uid = f"aku_{folder.name.replace(' ', '')}_{Path(name).stem}"
        img = Image.open(ip).convert("L")
        sx, sy = W / img.width, H / img.height
        img.resize((W, H), Image.LANCZOS).save(OUT / "img" / f"{uid}.png")
        inst = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(inst)
        types = []
        for s in d["shapes"]:
            t = TYPES.get(s["label"].strip().lower())
            if t is None or len(s["points"]) < 3 or len(types) >= 250:
                continue
            types.append(t)
            dr.polygon([(x * sx, y * sy) for x, y in s["points"]], fill=len(types))
        inst.save(OUT / "inst" / f"{uid}.png")
        recs.append({"id": uid, "types": types, "w": img.width, "h": img.height})
    random.Random(11).shuffle(recs)
    for i, r in enumerate(recs):
        r["split"] = "hold" if i < 50 else "train"
    json.dump(recs, open(OUT / "meta.json", "w"))
    print("AKU images", len(recs), "teeth", sum(len(r["types"]) for r in recs))


if __name__ == "__main__":
    main()
