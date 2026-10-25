"""Round-5 downloads (all CC0 / CC BY 4.0). Saves into data5/<name>/ and unpacks. Safe to rerun."""
import json, tarfile, zipfile, urllib.request, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent / "data5"
ROOT.mkdir(exist_ok=True)
MD = "https://data.mendeley.com/public-files/datasets/"
FILES = {
    "hitl": [("hitl.tar", "https://www.dropbox.com/scl/fi/mzmaw1s3h8cjnl2vakbvz/teeth-segmentation-on-dental-x-ray-images-DatasetNinja.tar?rlkey=oeaa9wicuj929zr6q0v6kgjc9&dl=1")],
    "dopg1550": [("yolo.zip", MD + "rtt726b26d/files/4b33a007-dc93-4b95-b4a1-0c608459093a/file_downloaded")],
    "khurshid": [("Dataset.zip", MD + "mdvs6mjgf2/files/1f223672-71ef-4e43-be6c-9a7e5ec7e798/file_downloaded"),
                 ("Readme.txt", MD + "mdvs6mjgf2/files/bb054bce-9955-4cc3-9e98-c0f61de7a61c/file_downloaded")],
}
UA = {"User-Agent": "Mozilla/5.0 (MouthTwin research download)"}


def get(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        print("have", dst.name, flush=True); return
    print("downloading", dst, flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(str(dst) + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    Path(str(dst) + ".part").rename(dst)
    print("  ", dst.stat().st_size // (1 << 20), "MB", flush=True)


def kairouan():
    out = ROOT / "kairouan"; out.mkdir(exist_ok=True)
    api = "https://data.mendeley.com/public-api/datasets/73n3kz2k4k/files?folder_id={}&version=3"
    def walk(fid, sub):
        try:
            items = json.load(urllib.request.urlopen(urllib.request.Request(api.format(fid), headers=UA), timeout=60))
        except Exception as e:
            print("kairouan listing failed", e, flush=True); return
        for it in items:
            if it.get("content_details"):
                d = out / sub; d.mkdir(parents=True, exist_ok=True)
                get(it["content_details"]["download_url"], d / it["filename"])
        fapi = "https://data.mendeley.com/public-api/datasets/73n3kz2k4k/folders/3"
        try:
            folders = json.load(urllib.request.urlopen(urllib.request.Request(fapi, headers=UA), timeout=60))
        except Exception:
            folders = []
        return folders
    folders = walk("root", "")
    for fo in folders or []:
        walk(fo["id"], fo.get("name", fo["id"]))


for name, files in FILES.items():
    d = ROOT / name; d.mkdir(exist_ok=True)
    for fn, url in files:
        try:
            get(url, d / fn)
        except Exception as e:
            print("FAILED", name, fn, e, flush=True)
try:
    kairouan()
except Exception as e:
    print("FAILED kairouan", e, flush=True)
for p in ROOT.rglob("*"):
    if p.suffix in (".zip", ".tar") and not (p.parent / (p.stem + "_x")).exists():
        print("unpacking", p, flush=True)
        try:
            if p.suffix == ".zip":
                zipfile.ZipFile(p).extractall(p.parent / (p.stem + "_x"))
            else:
                tarfile.open(p).extractall(p.parent / (p.stem + "_x"))
        except Exception as e:
            print("FAILED unpack", p, e, flush=True)
print("GET5_DONE", flush=True)
