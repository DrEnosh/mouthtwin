# MouthTwin

Turn a dental panoramic X-ray into an interactive, 3D-style model of the arch.

> MouthTwin is an experimental educational proof of concept. It is not intended for diagnosis, treatment planning, or clinical decision-making. Visualizations derived from 2D dental images may be conceptual or inferred. True patient-specific 3D anatomical information requires appropriate 3D imaging such as CBCT.

## What works now

- **Run MouthTwin**: a synthetic OPG goes through a 5-second analysis animation, grows into 3D and bends back into an arch.
- **3D anatomy fitted to your X-ray**: real teeth (crowns and roots), mandible, maxilla, nerve canals and sinuses from an expert-labelled CBCT (ToothFairy2). The tooth model decides which teeth are present, and each tooth's length and tilt are adjusted to match the OPG. The older brightness relief is still available as the "X-ray relief" layer.
- **Upload OPG**: JPEG, PNG, WebP or BMP. The file is validated and processed in your browser. It is not uploaded or stored.
- **Workspace**: Image, Model and Split views. Click any tooth to isolate it and fly the camera to it. The inspector shows FDI, Universal and Palmer notation, the tooth's outline in the OPG, and how each pipeline stage was produced.
- **Layers**: teeth, jaw bone, nerve canals, maxillary sinuses, X-ray relief.
- **Modes**: Clinical, Anatomy, Patient.

## Tooth model

Uploads and the demo run a U-Net (MobileNetV2 encoder, 4.2M parameters) in the browser with ONNX Runtime Web. It outlines every tooth and assigns its FDI number, and the 3D relief is built from its masks.

Trained on the DENTEX quadrant-enumeration set (574 OPGs for training, 60 held out). Results on the 60 held-out OPGs:

| Metric | Value |
| --- | --- |
| Tooth vs background Dice | 0.909 |
| Mean per-tooth (FDI class) Dice | 0.834 |
| FDI numbering accuracy (per clinician-labelled tooth) | 96.4% |

Curves and example overlays are in `ml/results/`. Training code is in `ml/`.

**Data credit and license.** DENTEX: Hamamci et al., "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays", MICCAI 2023. https://huggingface.co/datasets/ibrahimhamamci/DENTEX. Released under CC BY-NC-SA 4.0, so the trained weights (`public/models/mouthtwin-teeth.onnx`) and the demo X-ray are shared under the same license: non-commercial use only, with attribution.

**Reference 3D anatomy.** One full healthy dentition (case ToothFairy2F_027) from ToothFairy2 (Bolelli et al., University of Modena and Reggio Emilia, https://ditto.ing.unimore.it/toothfairy2/), CC BY-SA 4.0. Only the label volume is used. `scripts/build_reference_anatomy.py` turns it into `public/models/reference-anatomy.glb`.

These numbers describe agreement with clinician outlines on public data. They are not clinical validation. See `docs/ARCHITECTURE.md` for what the 3D can and can't show.

## Run it

Requirements: Node 20+ (tested on 22). Python 3.10+ only if you want to regenerate the demo image.

```bash
npm install
npm run dev            # http://localhost:5173
```

Build:

```bash
npm run build          # static site in dist/
npm run build:single   # one self-contained HTML file in dist-single/
npm run preview        # serve dist/ locally
```

Regenerate the synthetic demo OPG and its annotations:

```bash
pip install numpy scipy pillow
npm run demo:generate
```

## Environment variables

Copy `.env.example` to `.env`.

| Variable | Default | Effect |
| --- | --- | --- |
| `VITE_INFERENCE_URL` | unset | When set, uploads are sent to `POST {url}/infer` (see `server/`). The privacy line on the landing page changes to say so. |

## Where things go

| Asset | Location |
| --- | --- |
| Demo OPG | `src/data/demo/opg-synthetic.jpg` (bundled). Swap in any de-identified OPG you have rights to. |
| Demo inference | `src/data/demo/demo-inference.json`, same schema as `src/types/inference.ts` |
| Segmentation masks | Convert to polygons in the inference JSON (`teeth[].polygon`, normalised 0..1) |
| Model weights | `server/models/` |
| Training data | `data/{images,masks,annotations,metadata}/` (keep out of the app bundle) |
| Icons | Inline SVG in `src/components/` |
| 3D assets | Not needed today. Teeth are procedural (`src/features/visualization/toothGeometry.ts`). Mean meshes from Teeth3DS could go in `public/models/`. |

## Connect a real model

1. Implement `run_model()` in `server/main.py` so it returns `InferenceResult` (see `server/schema.py`).
2. `cd server && pip install -r requirements.txt && uvicorn main:app --port 8000`
3. Set `VITE_INFERENCE_URL=http://localhost:8000` and restart `npm run dev`.

## Deploy

The build is static. `base` is `./`, so it works from any sub-path.

- **GitHub Pages**: push `dist/` to a `gh-pages` branch, or use the Pages action with `npm run build`.
- **Netlify**: build command `npm run build`, publish directory `dist`.
- **Vercel**: framework preset Vite.

## Keyboard

`1` Image · `2` Model · `3` Split · `R` reset camera · `N` FDI numbers · `Esc` clear selection

## Recording a demo

Click **Run MouthTwin** and let it play (about 8 s from click to rotating arch). Then click tooth 36 and switch to Split. **Replay** in the top bar reruns the transformation. For a still frame of the transformation, run `__mouthtwin.freeze(0.5)` in the browser console.
