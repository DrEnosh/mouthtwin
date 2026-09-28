# MouthTwin architecture

MouthTwin takes a panoramic dental X-ray (OPG) and turns it into an interactive, 3D-style model of the dental arch. It is an educational prototype. It is not a diagnostic tool.

## The one idea the whole product rests on

A panoramic machine rotates around the head and images a curved slab of tissue (the focal trough) that follows the dental arch. The flat OPG is that arch **unrolled**. MouthTwin rolls it back up, using the uploaded image itself as the model:

1. `reliefAnalysis.ts` finds tooth pixels (brightest structures around the occlusal plane, kept only if they reach it) and bone pixels, directly in the uploaded image.
2. Thickness is estimated: teeth get thicker toward their middle (distance from the tooth edge) and where they are brighter. Bone thickness follows brightness.
3. `Relief.tsx` extrudes both into closed shells, front and back, and bends them onto an average arch curve. Outline vertices are moved to the sub-pixel edge so outlines aren't stair-stepped.
4. The X-ray is painted onto the surfaces, so fillings, canals and trabecular pattern stay visible.

What is exactly from the image: every outline (crowns, roots, gaps, restorations, bone margins) and the surface shading. What is estimated: thickness (an OPG has no depth) and the arch curve (population average). Average tooth shapes are still available as a "Reference shapes" layer.

## A. Product architecture

| Screen | Purpose |
| --- | --- |
| Landing | What this is, **Run MouthTwin** (demo) and **Upload OPG**. Drag-and-drop anywhere. |
| Processing | About 5 s: scan sweep, detection brackets, FDI numbers, outlines drawing on, occlusal curve. The stage list shows how each stage was actually produced. |
| Workspace | Image / Model / Split views, layer rail, tooth inspector, lens bar (Anatomy, Surface, Roots, Cross section*, Timeline*). |

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

What is real today: file validation, image decoding, contrast check, occlusal-plane estimation (row-intensity minima plus a quadratic fit), template placement and the whole visualization layer.

What is simulated: tooth detection, numbering and segmentation. For the demo they come from the generator's ground truth. For uploads they come from the template, which assumes a full adult dentition.

Candidates for Phase 8 (check each licence first; several are research-only or non-commercial):

| Need | Candidates |
| --- | --- |
| Tooth instance segmentation + FDI numbering on OPGs | DENTEX (MICCAI 2023), Tufts Dental Database, OdontoAI / O²PR, UFBA-UESC dental images |
| Detection backbone | Your OralGuard YOLOv8 detector → add a numbering head, or YOLOv8-seg / Mask R-CNN fine-tuned on DENTEX quadrant-enumeration labels |
| 3D tooth shape priors | Teeth3DS (3DTeethSeg '22 intraoral scans): replace the lathe templates with per-type mean meshes |
| Mandibular canal and 3D bone | ToothFairy / ToothFairy2 (CBCT): canal and jaw priors. Patient-specific 3D still needs CBCT. |

Place real assets like this:

```
server/models/             model weights (.pt / .onnx), loaded by server/main.py
data/images/               training OPGs          (not bundled with the app)
data/masks/                per-tooth masks
data/annotations/          COCO or DENTEX JSON
data/metadata/             per-image metadata (no identifiers)
src/data/demo/             the demo OPG + demo-inference.json bundled into the app
```

## E. MVP plan

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Shell, landing, upload + validation, demo mode, processing reveal, workspace, 3D arch | Done |
| 2 | Hover, select, isolate, camera fly-to, FDI chips, odontogram | Done |
| 3 | Layers: crowns, roots, bone, gingiva, canal | Done (conceptual meshes) |
| 4 | Split view, image ↔ model mapping, link line | Done |
| 5 | Cross-section: slice plane through the selected tooth plus a 2D section panel with a depth slider | Next |
| 6 | Timeline: two demo timepoints, aligned OPGs, before/after slider, "visual comparison" labelling | Planned |
| 7 | Polish: end card for recordings, tuned easing, sound-free 30 s auto-demo | Planned |
| 8 | Real inference via `server/` (YOLOv8-seg + FDI numbering), missing-tooth handling | Planned |

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

- A single OPG has no reliable depth, and magnification varies across the image. Arch shape, tooth depth and bone are template values.
- The upload path assumes 32 teeth, centred in the image. Missing, impacted or rotated teeth are not detected until a real model is connected.
- Primary and mixed dentition are not supported.
- DICOM input is not supported yet. Export the OPG as JPEG or PNG.
