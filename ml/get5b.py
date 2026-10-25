"""Humans-in-the-Loop teeth set: Dataset Ninja (Dropbox) first; if that link is disabled, Kaggle CLI (uses your own kaggle.json if present)."""
import subprocess, sys, tarfile, urllib.request
from pathlib import Path
D = Path(__file__).resolve().parent / "data5" / "hitl"
D.mkdir(parents=True, exist_ok=True)
URL = "https://www.dropbox.com/scl/fi/mzmaw1s3h8cjnl2vakbvz/teeth-segmentation-on-dental-x-ray-images-DatasetNinja.tar?rlkey=oeaa9wicuj929zr6q0v6kgjc9&dl=1"
tar = D / "hitl.tar"
ok = False
try:
    with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r:
        head = r.read(512)
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            print("dropbox link disabled", flush=True)
        else:
            with open(tar, "wb") as f:
                f.write(head)
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            tarfile.open(tar).extractall(D / "ninja")
            ok = True
            print("HITL_NINJA_OK", flush=True)
except Exception as e:
    print("dropbox failed", e, flush=True)
if not ok:
    py = sys.executable
    subprocess.run([py, "-m", "pip", "install", "-q", "kaggle"], check=False)
    r = subprocess.run([py, "-m", "kaggle", "datasets", "download", "humansintheloop/teeth-segmentation-on-dental-x-ray-images",
                        "-p", str(D / "kaggle"), "--unzip"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    print("HITL_KAGGLE_OK" if r.returncode == 0 else "HITL_KAGGLE_FAILED", flush=True)
