# MouthTwin tooth + restoration model (training)

Trains a 37-output segmentation model on panoramic X-rays: 33-class tooth softmax (background + 32 FDI teeth) and 4 sigmoid planes
(implant, prosthetic restoration/crown, filling, root-canal treatment).

Data and licences:
- DENTEX (Hamamci et al., MICCAI 2023, https://huggingface.co/datasets/ibrahimhamamci/DENTEX), CC BY-NC-SA 4.0 - weights derived from it stay non-commercial/share-alike.
- Cluj OPG dental-condition dataset (Mureșanu, Hedeșiu, Iacob et al., Diagnostics 2024, Zenodo 10.5281/zenodo.15487430), CC BY 4.0.
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
Results (held-out OPGs) are in `results/`.
