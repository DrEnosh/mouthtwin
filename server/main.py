"""
MouthTwin inference server (stub).

The browser app posts the OPG to /infer and expects an InferenceResult (schema.py).
Images are processed in memory and never written to disk.

Run:  uvicorn main:app --port 8000
"""
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from schema import ImageMeta, InferenceResult, StageReport

app = FastAPI(title="MouthTwin inference", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])

MAX_BYTES = 25 * 1024 * 1024


def run_model(image: Image.Image):
    """
    Replace this with real inference, for example:
      1. YOLOv8-seg (or your OralGuard detector + a numbering head) → per-tooth masks and FDI labels
      2. masks → polygons (cv2.findContours), normalised by image size
      3. occlusal point per tooth: midpoint of the mask edge nearest the occlusal plane
    Return a list of schema.ToothDetection.
    """
    raise NotImplementedError


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/infer", response_model=InferenceResult)
async def infer(image: UploadFile = File(...)):
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 25 MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        raise HTTPException(415, "Could not decode image")

    try:
        teeth = run_model(img)
    except NotImplementedError:
        raise HTTPException(501, "No model is configured on this server yet")

    return InferenceResult(
        image=ImageMeta(width=img.width, height=img.height, fileName=image.filename),
        teeth=teeth,
        stages=[
            StageReport(id="preprocess", label="Uploading to inference server", method="Decoded in memory, not stored"),
            StageReport(id="detect", label="Detecting teeth", method="Model output"),
            StageReport(id="number", label="Assigning FDI numbers", method="Model output"),
            StageReport(id="segment", label="Segmenting tooth outlines", method="Model output"),
            StageReport(id="map", label="Mapping teeth to the arch", method="Panoramic position → average arch template"),
            StageReport(id="visualize", label="Building interactive model", method="Procedural template geometry per tooth type"),
        ],
    )
