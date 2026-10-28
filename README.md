# MouthTwin

Upload a dental panoramic X-ray (OPG) and get an interactive 3D model of the dental arch in your browser.

> **Research prototype, not a medical device.** MouthTwin is an educational proof of concept. It must not be used for diagnosis, treatment planning or any clinical decision. A 2D image cannot give true patient-specific 3D anatomy; that needs 3D imaging such as CBCT. Anything beyond tooth position, outline and radiopaque restorations is inferred or generic.

## What it does

- **Run the demo.** A synthetic OPG goes through a short analysis animation, grows into 3D and bends back into an arch.
- **Upload your own OPG.** JPEG, PNG, WebP or BMP. The file is validated and processed in your browser. Nothing is uploaded or stored.
- **Workspace views.** Image, Model and Split. Click a tooth to isolate it and fly the camera to it. The inspector shows FDI, Universal and Palmer notation, the tooth outline from the X-ray, and how each pipeline stage was produced.
- **Layers.** Teeth, jaw bone, nerve canals, maxillary sinuses, restorations (crowns, fillings, root canals, implants) and an optional X-ray relief.
- **Audience modes.** Clinical, Anatomy and Patient. They change how much is said, not what is true.

## How it works

1. **Read the X-ray.** A U-Net (MobileNetV2 encoder, about 4.2M parameters) runs in the browser through ONNX Runtime Web. It outlines every tooth, assigns its FDI number and flags four kinds of dental work: implants, crowns and caps, fillings, and root-canal fillings.
2. **Fit real anatomy.** Teeth, jaw bone, nerve canals and sinuses come from one expert-labelled CBCT (ToothFairy2). Each tooth is compared with the same tooth on 634 expert-labelled OPGs and adjusted for tilt, depth, length and crown-only outlines (for example developing wisdom teeth). Head-position fanning is removed first.
3. **Draw the dental work.** Metallic caps, filling patches, bright root canals, implant screws and bridges are added where the X-ray shows them.

What comes straight from the image: which teeth exist, their outline, length, tilt, depth in the bone, and which carry a radiopaque restoration. What is generic: crown and root shape, the arch, the bone, and how a cap, filling or implant looks. Depth is never measured. Details are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Model results

The model went through four shipped rounds of training. Numbers below are from held-out OPGs that were never used for training; each round is compared on the same sets.

| Held-out test | Round 3 | Round 4 (current) |
| --- | --- | --- |
| UFBA, 40 OPGs: mean per-tooth Dice | 0.840 | **0.852** |
| UFBA: FDI numbering accuracy | 96.3% | 96.3% |
| STS, 100 OPGs from new scanners: tooth-vs-background Dice | 0.91 | 0.91 |
| AKU, 50 restored or partially edentulous OPGs: tooth Dice | 0.88 | 0.88 |
| DENTEX, 60 OPGs: FDI numbering accuracy | 97.1% | 96.9% |
| DENTEX: mean per-tooth Dice | 0.834 | 0.835 |

Restoration detection on 220 held-out Cluj OPGs (per region, app thresholds):

| Restoration | Recall / precision (round 4) |
| --- | --- |
| Root-canal filling | 0.94 / 0.81 |
| Implant | 0.92 / 0.85 |
| Crown or cap | 0.81 / 0.67 |
| Filling | 0.85 / 0.71 |

The restoration thresholds were tuned on these same 220 OPGs, so treat those numbers as slightly optimistic. Rounds 1 and 2 and the full comparison tables are in `ml/results/`.

**Limits.** These numbers measure agreement with clinician outlines on public data. They are not clinical validation. Crowded, rotated or heavily restored mouths still produce wrong or missing tooth numbers, and crown and filling calls are sometimes wrong. Children's OPGs are not supported because the model has no baby-tooth classes.

A round-5 experiment (human-in-the-loop labels plus DentalOPG-1550 and Kairouan data) was run but not shipped. Round 4 is the live model.

## Run it locally

You need Node 20 or newer (tested on 22). Python 3.10+ is only needed to regenerate the demo image.

```bash
npm install
npm run dev            # http://localhost:5173
```

| Command | What it does |
| --- | --- |
| `npm run build` | Static site in `dist/` |
| `npm run build:single` | One self-contained HTML file in `dist-single/` |
| `npm run preview` | Serve `dist/` locally |
| `npm run typecheck` | TypeScript check |
| `npm run demo:generate` | Regenerate the synthetic demo OPG (`pip install numpy scipy pillow` first) |

### Optional: connect your own inference server

Copy `.env.example` to `.env` and set `VITE_INFERENCE_URL`. Uploads are then sent to `POST {url}/infer` and the privacy note on the landing page changes to say so.

1. Implement `run_model()` in `server/main.py` so it returns an `InferenceResult` (see `server/schema.py`).
2. Run `cd server && pip install -r requirements.txt && uvicorn main:app --port 8000`.
3. Set `VITE_INFERENCE_URL=http://localhost:8000` and restart `npm run dev`.

### Deploy

The build is static and `base` is `./`, so it works from any sub-path: GitHub Pages (publish `dist/`), Netlify (`npm run build`, publish `dist`) or Vercel (Vite preset).

### Keyboard

`1` Image, `2` Model, `3` Split, `R` reset camera, `N` FDI numbers, `Esc` clear selection.

## Repository layout

```
src/        React + Three.js app (landing, imaging, pipeline, 3D visualization)
public/     ONNX model, reference anatomy (GLB), ONNX Runtime WASM
ml/         Training and export scripts, per-round results (see ml/README.md)
server/     Optional FastAPI inference server
scripts/    Demo OPG generator, reference-anatomy builder
docs/       Architecture notes
```

## Data credits and licences

- **DENTEX**: Hamamci et al., MICCAI 2023, https://huggingface.co/datasets/ibrahimhamamci/DENTEX (CC BY-NC-SA 4.0).
- **Cluj OPG condition dataset**: Mureșanu, Hedeșiu, Iacob et al., Diagnostics 2024, https://doi.org/10.5281/zenodo.15487430 (CC BY 4.0, used for non-commercial research and education).
- **STS-2D-Tooth**: Wang et al., Scientific Data 12:117 (2025), https://doi.org/10.5281/zenodo.10597292 (CC BY 4.0 on Zenodo; the paper states CC BY-NC-ND).
- **AKU OPG dataset**: Data in Brief 2024, https://doi.org/10.5281/zenodo.10538750 (CC BY 4.0 on Zenodo; CC BY-NC 4.0 in the article).
- **UFBA-425**: 425 panoramic X-rays with FDI tooth masks, https://figshare.com/articles/dataset/UFBA-425/29827475 (CC BY 4.0), from the authors of OralBBNet (https://github.com/devichand579/Instance_seg_teeth).
- **Reference 3D anatomy**: one healthy dentition (case ToothFairy2F_027) from ToothFairy2, Bolelli et al., University of Modena and Reggio Emilia, https://ditto.ing.unimore.it/toothfairy2/ (CC BY-SA 4.0). Only the label volume is used; `scripts/build_reference_anatomy.py` turns it into `public/models/reference-anatomy.glb`.

AU-OPG was not used because it has no published licence.

Because DENTEX is share-alike, the trained weights (`public/models/mouthtwin-teeth.onnx`) and the demo X-ray are shared under **CC BY-NC-SA 4.0**: non-commercial use only, with attribution.

## Author

Dr. Enosh A Paulson. [LinkedIn](https://linkedin.com/in/dr-enosh) · [GitHub](https://github.com/DrEnosh)
