# MouthTwin

Turn a dental panoramic X-ray into an interactive, 3D-style model of the arch.

> MouthTwin is an experimental educational proof of concept. It is not intended for diagnosis, treatment planning, or clinical decision-making. Visualizations derived from 2D dental images may be conceptual or inferred. True patient-specific 3D anatomical information requires appropriate 3D imaging such as CBCT.

## What works now

- **Run MouthTwin**: a synthetic OPG goes through a 5-second analysis animation, grows into 3D and bends back into an arch.
- **3D anatomy fitted to your X-ray**: real teeth (crowns and roots), mandible, maxilla, nerve canals and sinuses from an expert-labelled CBCT (ToothFairy2). The tooth model decides which teeth are present. Each tooth is then compared with the same tooth on 634 expert-labelled OPGs (`ml/tooth_pose_stats.py`): extra tilt, a crown sitting deeper or higher than the bite line (impacted / unerupted teeth), a short root and crown-only outlines (developing wisdom teeth) are copied onto the 3D tooth. Fanning caused by head position on the OPG is removed first. Ambient occlusion gives the arch real depth. The older brightness relief is still available as the "X-ray relief" layer.
- **Upload OPG**: JPEG, PNG, WebP or BMP. The file is validated and processed in your browser. It is not uploaded or stored.
- **Workspace**: Image, Model and Split views. Click any tooth to isolate it and fly the camera to it. The inspector shows FDI, Universal and Palmer notation, the tooth's outline in the OPG, and how each pipeline stage was produced.
- **Layers**: teeth, jaw bone, nerve canals, maxillary sinuses, crowns/fillings/root canals/implants, X-ray relief.
- **Modes**: Clinical, Anatomy, Patient.

## Tooth and restoration model

Uploads and the demo run one U-Net (MobileNetV2 encoder, about 4.2M parameters) in the browser with ONNX Runtime Web. It outlines every tooth, assigns its FDI number, and also flags four kinds of work: **implants, crowns/caps (prosthetic restorations), fillings and root-canal fillings**. The app turns those into 3D: metallic caps on the crown, a filling patch, a bright filled canal in the root, an implant screw, and a bar between crowns that look joined (bridge). Everything is decided from the picture only. The material, the exact shape and the quality of the work are not known, and the 3D shape of a cap or canal is generic.

**Round 2** adds harder data and a multi-task head:

- DENTEX quadrant-enumeration (634 OPGs, clinician tooth outlines) plus 705 DENTEX disease OPGs (partial tooth labels) and 1,571 unlabelled OPGs. The round-1 model pseudo-labels the extra OPGs; pixels it is unsure about are ignored, implausible results and near-duplicates of held-out images are dropped.
- Cluj dental-condition OPG set (1,808 OPGs with boxes for implants, prosthetic restorations, fillings, root-canal treatment) for the restoration planes.
- Heavy augmentation (tilt, shear, contrast, noise, blur, resolution loss) so noisy and awkward OPGs behave better.

**Round 3** adds 2,400 more varied adult OPGs: STS-2D-Tooth (different country and scanners, dentist-checked tooth masks + 1,500 unlabelled) and AKU (many partially edentulous, bridged and crowned mouths, a human outline for every tooth). Children's OPGs are not used: the model has no classes for baby teeth.

**Round 4 (current)** adds UFBA-425 (377 usable OPGs with a human mask for every FDI-numbered tooth, many with braces; 40 kept as a new held-out test) and the remaining STS unlabelled OPGs (2,606 used instead of 1,500), about 7,600 training OPGs in all. It starts from round 3 with half the learning rate, samples the Cluj restoration OPGs more often, and picks its checkpoint on teeth *and* restorations together. A first attempt (round 4a) picked on teeth only and lost filling detection (IoU 0.46 to 0.40), so it was not shipped.

Results on held-out OPGs (never used for training). Each round starts from the previous one and is compared on the same sets:

| Held-out test | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | --- | --- | --- | --- |
| UFBA, 40 OPGs (new, human FDI masks): mean per-tooth Dice | - | - | 0.840 | **0.852** |
| UFBA: worst-quartile Dice | - | - | 0.676 | **0.702** |
| UFBA: FDI numbering accuracy | - | - | 96.3% | 96.3% |
| STS, 100 OPGs from new scanners: tooth-vs-background Dice | - | 0.76 | 0.91 | 0.91 |
| STS: worst-quartile Dice | - | 0.70 | 0.88 | 0.87 |
| AKU, 50 restored / partially edentulous OPGs: tooth Dice | - | 0.80 | 0.88 | 0.88 |
| AKU: tooth type correct (incisor / canine / premolar / molar) | - | 96.8% | 97.5% | 97.4% |
| DENTEX, 60 OPGs: FDI numbering accuracy | 96.4% | 97.2% | 97.1% | 96.9% |
| DENTEX: mean per-tooth Dice | 0.834 | 0.839 | 0.834 | 0.835 |
| DENTEX hard teeth: numbering accuracy | 92.0% | 94.5% | 93.5% | 93.3% |
| DENTEX hard teeth: worst-quartile Dice | 0.574 | 0.651 | 0.608 | **0.629** |

Round 3 was much better than round 2 on X-rays that do not look like DENTEX (which is what a random upload usually is). Round 4 is a small step: tooth outlines are tighter on the new UFBA X-rays and on the hardest teeth, and implant calls are cleaner; tooth numbering did not move (changes of 0.2 points are one to three teeth). The model is close to what this network size and these labels can give.

| Restoration found on 220 held-out Cluj OPGs (per region, box labels, app thresholds) | Round 3 recall / precision | Round 4 recall / precision |
| --- | --- | --- |
| Root-canal filling | 0.94 / 0.81 | 0.94 / 0.81 |
| Implant | 0.92 / 0.84 | 0.92 / 0.85 |
| Crown / cap | 0.84 / 0.64 | 0.81 / 0.67 |
| Filling | 0.84 / 0.70 | 0.85 / 0.71 |

For round 4 the crown/cap threshold was raised (0.8 to 0.9) to keep its precision; at the old threshold round 4 calls more crowns that are not there (precision 0.58). The thresholds were tuned on these same 220 OPGs, so treat these restoration numbers as slightly optimistic.

Read these honestly: complex OPGs are better, not solved. Crowded, rotated or heavily restored mouths still produce wrong or missing tooth numbers, and crown and filling calls will sometimes be wrong (they are trained from loose boxes, and the app also checks that the pixels are actually bright). AU-OPG, which has much more crown detail, was not used because it has no published licence.

Curves and example overlays are in `ml/results/` (round 3 in `ml/results/round3/`, round 4 in `ml/results/round4/`). Training code is in `ml/`.

**Data credit and license.** DENTEX: Hamamci et al., "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays", MICCAI 2023, https://huggingface.co/datasets/ibrahimhamamci/DENTEX (CC BY-NC-SA 4.0). Cluj OPG condition dataset: Mureșanu, Hedeșiu, Iacob et al., Diagnostics 2024, https://doi.org/10.5281/zenodo.15487430 (CC BY 4.0, used for non-commercial research and education). STS-2D-Tooth: Wang et al., Scientific Data 12:117 (2025), https://doi.org/10.5281/zenodo.10597292 (CC BY 4.0; the paper states CC BY-NC-ND). AKU OPG dataset: Data in Brief 2024, https://doi.org/10.5281/zenodo.10538750 (CC BY 4.0 on Zenodo; CC BY-NC 4.0 in the article). UFBA-425: 425 annotated panoramic X-rays with FDI tooth masks, https://figshare.com/articles/dataset/UFBA-425/29827475 (CC BY 4.0), from the authors of OralBBNet (https://github.com/devichand579/Instance_seg_teeth). Because DENTEX is share-alike, the trained weights (`public/models/mouthtwin-teeth.onnx`) and the demo X-ray are shared under CC BY-NC-SA 4.0: non-commercial use only, with attribution.

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
