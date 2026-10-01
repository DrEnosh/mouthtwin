# MouthTwin architecture

MouthTwin takes a panoramic dental X-ray (OPG) and turns it into an interactive, 3D-style model of the dental arch. It is an educational prototype. It is not a diagnostic tool.

## The one idea the whole product rests on

A panoramic machine rotates around the head and images a curved slab of tissue that follows the dental arch. The flat OPG is that arch **unrolled**. MouthTwin rolls it back up in three steps:

1. **Read the X-ray.** A U-Net runs in the browser (ONNX Runtime Web) and outlines every tooth with its FDI number. The same network has four extra output maps for what dentists have put in the mouth: implants, prosthetic restorations (crowns, caps, bridges), fillings and root-canal fillings. `postprocess.ts` turns the tooth map into per-tooth detections. `restorations.ts` combines the restoration maps with the radiopacity inside each tooth outline into per-tooth findings.
2. **Fit real anatomy.** Teeth, jaw bone, nerve canals and sinuses come from one expert-labelled CBCT (ToothFairy2). Teeth the OPG does not show are hidden; the others are scaled to the OPG length and tilted to the OPG angle (`AnatomyModel.tsx`).
3. **Draw the dental work.** Where the X-ray shows it, the 3D tooth gets a metallic crown cap, an occlusal filling patch, a translucent root with a glowing canal filling, an implant screw (with a crown if one is seen) or a bar joining bridged crowns (`restorationGeometry.ts`).

What is exactly from the image: which teeth exist, their outline, length, tilt, and which of them carry a radiopaque restoration or canal filling. What is generic: crown and root shape, the arch, the bone, and how a cap, filling, canal or implant looks (the X-ray cannot tell material or exact shape). Depth is never measured. The UI says so.

The older "X-ray relief" layer (the image extruded by brightness, `Relief.tsx`) is still available as an optional layer.

## A. Product architecture

| Screen | Purpose |
| --- | --- |
| Landing | What this is, **Run MouthTwin** (demo) and **Upload OPG**. Drag-and-drop anywhere. |
| Processing | About 5 s: scan sweep, detection brackets, FDI numbers, outlines drawing on, occlusal curve. The stage list shows how each stage was actually produced. |
| Workspace | Image / Model / Split views, layer rail, tooth inspector (including what the X-ray shows on the tooth), lens bar (Anatomy, Teeth only, X-ray relief, Cross section*, Timeline*). |

\* Cross section is planned for Phase 5 and Timeline for Phase 6.

Audience modes change how much is said, not what is true:

- **Clinical**: FDI / Universal / Palmer notation, image metadata, per-stage method, reference sizes.
- **Anatomy**: every anatomical layer switched on.
- **Patient**: plain language, fewer numbers, a reminder that the model is not a scan of their teeth.

## B. Technical architecture

```
Image (File or bundled demo)
  │  loadImage.ts          type / size / dimension / aspect checks, decode in the browser
  ▼
InferenceProvider.run()    → InferenceResult (types/inference.ts)
  ├─ demoProvider          precomputed geometry for the synthetic OPG
  ├─ heuristicProvider     preprocess → occlusal plane → 32-tooth template (in-browser CV)
  └─ remoteProvider        POST /infer to your own model server (VITE_INFERENCE_URL)
  ▼
archLayout.ts              OPG x → arc length on the template arch; flat pose ↔ arch pose
  ▼
Scene (React Three Fiber)  Tooth × N, Jaw layers, OpgWrap, CameraRig
```

Rules the code follows:

- Every result carries a `source` (`precomputed-demo`, `heuristic-template` or `model`), and every stage carries a plain-language `method`. The UI shows both.
- `score` on a detection may only be set by a trained model. When it is missing, the UI shows "Experimental · no model score".
- The pipeline contract is the JSON schema in `src/types/inference.ts`, mirrored in `server/schema.py`. Swapping in a real model means returning that JSON; the viewer does not change.

### Why these choices

- **Vite + React + TypeScript**: static output, so it deploys to GitHub Pages or Netlify like PanoSlice, and a single-file build is available for sharing.
- **React Three Fiber + drei**: declarative scene, `CameraControls` for smooth fly-to, `Html` for FDI chips.
- **Image relief** (custom shader: flat ↔ arch morph and grow on the GPU, per-vertex FDI id for highlight/isolate), picked through an invisible low-poly proxy so hover stays cheap. Procedural average teeth remain as a reference layer.
- **zustand**: small shared store; per-frame values (reveal progress, screen anchors) live outside React state in `visualization/runtime.ts`.

## C. UI / UX architecture

```
┌ TopBar: wordmark · case · imaging · Experimental · [Image|Model|Split] · Replay · New image ┐
├ LeftRail ─────┬ Stage (image pane | model pane, animated widths) ─────────┬ Inspector ─────┤
│ Anatomy layers│  OPG + SVG overlay      3D canvas + viewport controls       │ none: chart,   │
│ Display toggles│ (hover/select synced)  (hover/select synced)              │ imaging, stages│
│ Mode          │           └── split-view link line ──┘                     │ tooth: detail  │
├ BottomBar: Anatomy · Surface · Roots · Cross section (soon) · Timeline (soon) · disclaimer ┤
```

Below 1024 px the rail and inspector become drawers. Keyboard: `1/2/3` views, `R` reset, `N` numbers, `Esc` clear.

Design tokens are in `src/index.css`: reading-room near-black, a single pale film-blue accent (`#86c5d8`), enamel ivory for teeth, IBM Plex Sans and Mono.

## D. Dataset and model strategy

The tooth model is trained on public data (see `ml/README.md`):

| Job | Data | Licence |
| --- | --- | --- |
| Tooth outlines + FDI numbers | DENTEX quadrant-enumeration (634 OPGs, full clinician labels) | CC BY-NC-SA 4.0 |
| Harder cases: caries, periapical lesions, impacted teeth | DENTEX disease set (705 OPGs, only diseased teeth outlined; clinician outlines kept, the rest filled by the round-1 model) and 1,571 unlabelled DENTEX OPGs (teacher labels, uncertain pixels ignored) | CC BY-NC-SA 4.0 |
| Crowns, fillings, root canals, implants | Cluj panoramic condition dataset (1,808 clinic OPGs, boxes) | CC BY 4.0, non-commercial research and education |
| 3D anatomy | ToothFairy2 CBCT label volume (one case) | CC BY-SA 4.0 |

Restoration labels are bounding boxes, so the restoration maps are box-like. Per tooth, a crown is only reported when the model flags it and the crown pixels are clearly denser than the tooth's usual grey; a filling when only part of the crown is; a root-canal filling when the root is flagged.

Not used: AU-OPG (HF `YSFF/AU-OPG`, 901 OPGs with per-tooth crown/implant/root-canal labels) has no licence stated. It would help crowns most; use it only with the authors' permission.

Where assets go:

```
public/models/mouthtwin-teeth.onnx      tooth + restoration model (browser)
public/models/reference-anatomy.glb     CBCT-derived anatomy
server/models/                          weights for the optional remote server
ml/                                     training code and results
```

## E. MVP plan

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Shell, landing, upload + validation, demo mode, processing reveal, workspace, 3D arch | Done |
| 2 | Hover, select, isolate, camera fly-to, FDI chips, odontogram | Done |
| 3 | Layers: teeth, bone, nerve canals, sinuses, restorations | Done (real CBCT reference meshes) |
| 4 | Split view, image ↔ model mapping, link line | Done |
| 5 | Cross-section: slice plane through the selected tooth plus a 2D section panel with a depth slider | Next |
| 6 | Timeline: two demo timepoints, aligned OPGs, before/after slider, "visual comparison" labelling | Planned |
| 7 | Polish: end card for recordings, tuned easing, sound-free 30 s auto-demo | Planned |
| 8 | Tooth model in the browser (DENTEX), real CBCT anatomy fitted to the OPG | Done |
| 9 | Crowns, fillings, root canals, implants, bridges detected and drawn | Done |
| 10 | Patient-specific 3D shape (needs CBCT or a learned shape prior) | Not started |

## F. Folder structure

```
src/
  app/                      App shell, error boundary, stage routing
  components/               Wordmark, disclaimer
  data/                     FDI reference table, demo image + demo inference JSON
  features/
    landing/                Landing page
    processing/             Processing animation (runs inference in parallel)
    imaging/                OPG view with SVG annotation overlay
    pipeline/               loadImage, errors, providers, stages/ (preprocess, occlusal plane, template)
    teeth/                  2D tooth silhouettes shared with the template stage
    visualization/          arch layout, procedural geometry, Scene, Tooth, Jaw, OpgWrap, CameraRig
    workspace/              TopBar, LeftRail, Inspector, BottomBar, LinkLine
  store/                    zustand store
  types/                    inference schema
scripts/generate_synthetic_opg.py   synthetic demo OPG + annotations
server/                     FastAPI stub with the same schema
docs/ARCHITECTURE.md
```

## Known limitations

- A single OPG has no reliable depth, and magnification varies across the image. Arch shape, tooth depth, root shape and bone are reference values from one CBCT.
- Restoration findings come from a model trained on boxes from one clinic's data. Crown vs filling is decided by how much of the crown is radiopaque; material, margins and quality are not judged. Bridges are inferred when neighbouring crowns form one radiopaque block. Crowded, rotated or overlapping teeth, poor-quality X-rays and unusual anatomy still cause errors, see the measured numbers in the README.
- Primary and mixed dentition are not supported.
- DICOM input is not supported. Export the OPG as JPEG or PNG.
