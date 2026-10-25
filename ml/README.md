# MouthTwin tooth + restoration model (training)

Trains a 37-output segmentation model on panoramic X-rays: 33-class tooth softmax (background + 32 FDI teeth) and 4 sigmoid planes
(implant, prosthetic restoration/crown, filling, root-canal treatment).

Data and licences:
- DENTEX (Hamamci et al., MICCAI 2023, https://huggingface.co/datasets/ibrahimhamamci/DENTEX), CC BY-NC-SA 4.0 - weights derived from it stay non-commercial/share-alike.
- Cluj OPG dental-condition dataset (Mureșanu, Hedeșiu, Iacob et al., Diagnostics 2024, Zenodo 10.5281/zenodo.15487430), CC BY 4.0.
- STS-2D-Tooth (Wang et al., Scientific Data 2025, Zenodo 10.5281/zenodo.10597292), CC BY 4.0 on Zenodo.
- AKU OPG dataset (Data in Brief 2024, Zenodo 10.5281/zenodo.10538750), CC BY 4.0 on Zenodo.
- UFBA-425 (https://figshare.com/articles/dataset/UFBA-425/29827475; Instance_seg_teeth / OralBBNet authors), CC BY 4.0: 425 OPGs with a human mask per FDI-numbered tooth.
- AU-OPG is deliberately not used (no licence).

Round 1 (tooth numbering only, 574 labelled OPGs):
```bash
pip install torch segmentation-models-pytorch timm remotezip pillow matplotlib onnx onnxruntime scipy
python prepare_dentex.py .
python train.py 70
python export_and_report.py
```

Round 2 (harder OPGs + restorations; needs a GPU, ~2-3 h on an RTX 4060; `run_v2.bat` does all of it on Windows):
```bash
python prepare_more.py     # Cluj + DENTEX disease/unlabelled OPGs -> dentex/proc2/
python pseudo_label.py     # round-1 teacher fills tooth labels on the extra OPGs (uncertain pixels ignored)
python train2.py 40        # multi-task student, masked losses, flip/affine/photometric augmentation
python export2.py          # ONNX with a `rest` output, metrics.json, training_curves.png, val_overlays.png
```
Round 3 (STS-2D-Tooth + AKU; `run_v3.bat`):
```bash
python prepare_aku.py && python prepare3.py .
python pseudo3.py          # round-2 teacher numbers teeth inside the human masks / tooth-type polygons
python train3.py 30        # lazy loading, extra tooth-vs-background loss on human masks
python export3.py
```

Round 4 (UFBA-425 + all 2,650 unlabelled STS OPGs; `get_ufba.bat`, then `run_v4b.bat`):
```bash
python prepare_ufba.py     # 512x512 squeezed OPGs stretched back to 768x384; 40 held out (UFBA-hold, full FDI)
python train4b.py 18       # from round 3; half LR; Cluj restoration OPGs sampled 1.5x;
                           # checkpoint chosen on 0.75 x teeth score + 0.25 x mean restoration IoU
python export4b.py
```
`train4.py` (round 4a) chose its checkpoint on teeth only and lost fillings/root canals (filling IoU 0.46 -> 0.40); it was
not shipped. Its numbers are in `results/round4/metrics_round4a_rejected.json`.

Round 5 (experiment, NOT shipped; `get5.bat`, `get5b.bat`, then `run_v5.bat`):
```bash
python get5.py && python get5b.py   # Humans-in-the-Loop (Kaggle CC0 / Zenodo 4457648 CC BY 4.0), DentalOPG-1550 and
                                    # Kairouan (Mendeley, CC BY 4.0)
python prepare5.py                  # HITL Universal 1-32 -> FDI polygons (55 held out); DentalOPG images only (box labels
                                    # are partial; patients under 16 skipped; file names contain patient names -> hashed ids);
                                    # Kairouan human tooth outlines + teacher numbers
python train5.py 24 && python export5.py
```
Result (`results/round5/`): tooth outlines on the new HITL test went from Dice 0.738 to 0.806 and hard DENTEX numbering
from 93.3% to 94.5%, but HITL numbering stayed at 91.5%, STS/UFBA/DENTEX outlines dropped slightly and implant precision
(app thresholds) fell from 0.85 to 0.79. Much of the HITL gain looks like learning that set's outline style (tooth
recall fell). Round 4 stays live. The Khurshid/Trachoo restoration set (Mendeley mdvs6mjgf2) was downloaded but not used:
its Roboflow export is mostly zoomed crops and 90-degree turns, and about 300 images look like Tufts Dental Database
images, which have their own terms.

Results (held-out OPGs) are in `results/` (`results/round3/`, `results/round4/`, `results/round5/`).
