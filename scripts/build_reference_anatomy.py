"""
Build the MouthTwin reference anatomy (GLB) from one ToothFairy2 CBCT label volume.

ToothFairy2: Bolelli et al., "Segmenting Maxillofacial Structures in CBCT Volumes", CVPR 2025 /
MICCAI ToothFairy2 challenge, https://ditto.ing.unimore.it/toothfairy2/. License CC BY-SA 4.0.

Only the expert label volume is used (no image data): per-FDI teeth, mandible, maxilla,
inferior alveolar canals and maxillary sinuses are meshed with marching cubes, smoothed,
decimated and re-oriented to the MouthTwin frame (+x patient left, +y up, +z anterior, mm,
origin at the occlusal plane centre).
"""
import sys
from pathlib import Path

import fast_simplification
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage
from skimage import measure

LABELS = sys.argv[1]
OUT = Path(sys.argv[2])

STRUCTS = {
    "mandible": (1, 2, 60000, 1.4),
    "maxilla": (2, 2, 45000, 1.4),
    "canal_L": (3, 1, 3000, 1.0),
    "canal_R": (4, 1, 3000, 1.0),
    "sinus_L": (5, 2, 6000, 1.6),
    "sinus_R": (6, 2, 6000, 1.6),
}


def to_scene(v):
    """volume (x,y,z) mm → scene: x_s = -x (patient left +), y_s = -z (up), z_s = -y (anterior)"""
    return np.stack([-v[:, 0], -v[:, 2], -v[:, 1]], axis=1)


def mesh_from_mask(mask, spacing, down, target_faces, sigma, solid=False):
    if solid:  # teeth: close pulp chambers and small gaps so the surface is one clean shell
        mask = ndimage.binary_closing(mask, iterations=1)
        mask = ndimage.binary_fill_holes(mask)
    if down > 1:
        mask = mask[::down, ::down, ::down]
    sp = np.array(spacing) * down  # spacing as (x, y, z)
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - 3, 0)
    hi = np.minimum(idx.max(0) + 4, mask.shape)
    crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    crop = np.pad(crop, 2)  # close surfaces that touch the scan's field-of-view edge
    crop = ndimage.gaussian_filter(crop, sigma)
    verts, faces, _, _ = measure.marching_cubes(crop, 0.5)
    verts = (verts - 2 + lo) * sp[::-1]  # array order is (z, y, x)
    verts = verts[:, ::-1]  # → (x, y, z) mm
    m = trimesh.Trimesh(to_scene(verts), faces[:, ::-1], process=True)
    if len(m.faces) > target_faces:
        v, f = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - target_faces / len(m.faces))
        m = trimesh.Trimesh(v, f, process=True)
    trimesh.smoothing.filter_taubin(m, iterations=10)
    # keep the largest piece (drops label specks and internal pulp surfaces)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: p.area)
    m.fix_normals()
    return m


def main():
    img = sitk.ReadImage(LABELS)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()  # (x, y, z)
    meshes = {}
    for fdi in [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)]:
        if (arr == fdi).sum() < 500:
            continue
        meshes[f"tooth_{fdi}"] = mesh_from_mask(arr == fdi, spacing, 1, 5000, 0.9, solid=True)
        print("tooth", fdi, len(meshes[f"tooth_{fdi}"].faces), flush=True)
    for name, (lab, down, faces, sigma) in STRUCTS.items():
        if (arr == lab).sum() < 500:
            continue
        meshes[name] = mesh_from_mask(arr == lab, spacing, down, faces, sigma)
        print(name, len(meshes[name].faces), flush=True)

    # origin: midline between the central incisors, occlusal plane between the arches, arch centre in depth
    teeth = [m for k, m in meshes.items() if k.startswith("tooth_")]
    upper = [m for k, m in meshes.items() if k.startswith("tooth_") and int(k[-2]) in (1, 2)]
    lower = [m for k, m in meshes.items() if k.startswith("tooth_") and int(k[-2]) in (3, 4)]
    occ_upper = np.median([m.vertices[:, 1].min() for m in upper])
    occ_lower = np.median([m.vertices[:, 1].max() for m in lower])
    y0 = (occ_upper + occ_lower) / 2
    x0 = (meshes["tooth_11"].centroid[0] + meshes["tooth_21"].centroid[0]) / 2
    allv = np.concatenate([m.vertices for m in teeth])
    z0 = (allv[:, 2].max() + allv[:, 2].min()) / 2
    shift = np.array([x0, y0, z0])

    scene = trimesh.Scene()
    for name, m in meshes.items():
        m.vertices -= shift
        scene.add_geometry(m, node_name=name, geom_name=name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scene.export(OUT)
    print("wrote", OUT, OUT.stat().st_size / 1e6, "MB; origin shift", shift.round(2))
    print("extent", scene.bounds.round(1))


if __name__ == "__main__":
    main()
