# MouthTwin tooth model (training)

Trains a 33-class segmentation model (background + 32 FDI teeth) on panoramic X-rays.

Data: DENTEX quadrant-enumeration subset (634 OPGs, clinician tooth outlines with FDI numbers),
Hamamci et al., MICCAI 2023, https://huggingface.co/datasets/ibrahimhamamci/DENTEX.
License CC BY-NC-SA 4.0: non-commercial use, credit the authors, share derived work (including trained weights) under the same license.

```bash
pip install torch segmentation-models-pytorch timm remotezip pillow matplotlib onnx onnxruntime
python prepare_dentex.py .        # downloads only the needed files, writes dentex/proc/
python train.py 70                # MT_BS env var sets batch size (default 8)
python export_and_report.py       # runs/mouthtwin-teeth.onnx, training_curves.png, val_overlays.png
```
