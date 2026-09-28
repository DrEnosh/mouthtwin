/**
 * Average adult arch template (units: millimetres; +z = anterior, +y = up, +x = patient's left).
 *
 * A panoramic image is, geometrically, the dental arch unrolled along the scanner's focal trough.
 * MouthTwin reverses that: each tooth's horizontal position in the OPG is mapped to an arc length
 * on this template curve. The curve itself is a population average, not the patient's arch.
 */
import * as THREE from 'three'
import { TEETH, type ToothRef } from '../../data/fdi'
import type { InferenceResult, ToothDetection } from '../../types/inference'

const UPPER_HALF: [number, number][] = [
  [0, 27], [5, 26.2], [10.5, 24], [15.5, 20.6], [19.6, 15.8], [22.8, 9.8], [25.2, 2.6],
  [27, -6], [28.4, -15], [29.4, -24.5], [30, -34], [30.3, -42],
]
const LOWER_HALF: [number, number][] = [
  [0, 24], [4, 23.4], [8.5, 21.6], [13, 18.6], [17, 14.4], [20.2, 9], [22.8, 2.4],
  [24.8, -5.6], [26.4, -14.5], [27.6, -23.5], [28.4, -32.5], [28.9, -41],
]
const INTERPROXIMAL_GAP = 0.2

class HalfArch {
  curve: THREE.CatmullRomCurve3
  length: number
  constructor(pts: [number, number][]) {
    this.curve = new THREE.CatmullRomCurve3(pts.map(([x, z]) => new THREE.Vector3(x, 0, z)), false, 'centripetal')
    this.length = this.curve.getLength()
  }
  /** Point and tangent (pointing distally) at arc length s from the midline. Extrapolates past the end. */
  at(s: number): { p: THREE.Vector3; t: THREE.Vector3 } {
    const clamped = Math.min(Math.max(s, 0), this.length)
    const u = clamped / this.length
    const p = this.curve.getPointAt(u)
    const t = this.curve.getTangentAt(u).normalize()
    if (s > this.length) p.addScaledVector(t, s - this.length)
    return { p, t }
  }
}

export const UPPER_ARCH = new HalfArch(UPPER_HALF)
export const LOWER_ARCH = new HalfArch(LOWER_HALF)
export const FOCAL_ARCH = new HalfArch(UPPER_HALF.map(([x, z], i) => [(x + LOWER_HALF[i][0]) / 2, (z + LOWER_HALF[i][1]) / 2 + 1.2]))

const UP = new THREE.Vector3(0, 1, 0)

/** Arc length (from the midline) of each tooth's centre on its own arch. */
export const ARC_CENTER = new Map<number, number>()
for (const upper of [true, false]) {
  for (const left of [true, false]) {
    const quad = TEETH.filter((t) => t.upper === upper && t.patientLeft === left).sort((a, b) => a.position - b.position)
    let s = INTERPROXIMAL_GAP / 2
    for (const t of quad) {
      ARC_CENTER.set(t.fdi, s + t.widthMm / 2)
      s += t.widthMm + INTERPROXIMAL_GAP
    }
  }
}

/** Signed arc length: negative on the patient's right (viewer's left), positive on the patient's left. */
export function signedArc(fdi: number, ref: ToothRef): number {
  const s = ARC_CENTER.get(fdi) ?? 0
  return ref.patientLeft ? s : -s
}

/**
 * Point on a full arch at a signed arc length, with a frame:
 * t = direction of increasing signed arc length (patient right → left), n = outward (buccal).
 */
export function archFrame(arch: HalfArch, signedS: number): { p: THREE.Vector3; t: THREE.Vector3; n: THREE.Vector3 } {
  const { p, t } = arch.at(Math.abs(signedS))
  let pt: THREE.Vector3
  let tt: THREE.Vector3
  if (signedS >= 0) {
    pt = p
    tt = t
  } else {
    pt = new THREE.Vector3(-p.x, p.y, p.z)
    tt = new THREE.Vector3(t.x, t.y, -t.z) // mirrored distal tangent, reversed to point toward the midline
  }
  const n = new THREE.Vector3().crossVectors(tt, UP).normalize()
  return { p: pt, t: tt, n }
}

export interface ToothPose {
  position: THREE.Vector3
  quaternion: THREE.Quaternion
  /** Scale applied to the tooth's inner group: mirrors for side (x) and arch (y). */
  mirror: THREE.Vector3
}

function occlusalOffset(ref: ToothRef): number {
  if (ref.upper) return ref.position <= 3 ? -1.8 : 0.4 // overbite on the anterior teeth
  return ref.position <= 3 ? -0.2 : -0.4
}

const LABIAL_TILT: Record<ToothRef['kind'], number> = { incisor: 0.3, canine: 0.16, premolar: 0.05, molar: -0.04 }

export function archPose(ref: ToothRef): ToothPose {
  const arch = ref.upper ? UPPER_ARCH : LOWER_ARCH
  const { p, t, n } = archFrame(arch, signedArc(ref.fdi, ref))
  const basis = new THREE.Matrix4().makeBasis(t, UP, n)
  const q = new THREE.Quaternion().setFromRotationMatrix(basis)
  const tilt = LABIAL_TILT[ref.kind] * (ref.upper ? -1 : 1)
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), tilt))
  return {
    position: new THREE.Vector3(p.x, occlusalOffset(ref), p.z),
    quaternion: q,
    mirror: new THREE.Vector3(ref.patientLeft ? 1 : -1, ref.upper ? -1 : 1, 1),
  }
}

/** Size of the OPG plane in scene units when it is laid flat in front of the camera. */
export const PANO_WIDTH = 170

export function panoHeight(result: InferenceResult): number {
  return (PANO_WIDTH * result.image.height) / result.image.width
}

export interface FlatPose {
  position: THREE.Vector3
  quaternion: THREE.Quaternion
  scale: THREE.Vector3
}

/** Pose of a tooth lying on the flat panoramic plane, sized to its footprint in the image. */
export function flatPose(ref: ToothRef, det: ToothDetection, result: InferenceResult): FlatPose {
  const ph = panoHeight(result)
  const [ox, oy] = det.occlusal
  const [, , bw, bh] = det.bbox
  const sy = (bh * ph) / (ref.crownMm + ref.rootMm)
  const sx = Math.max(0.35, (bw * PANO_WIDTH) / ref.widthMm)
  return {
    position: new THREE.Vector3((ox - 0.5) * PANO_WIDTH, (0.5 - oy) * ph, 1.5),
    quaternion: new THREE.Quaternion(),
    scale: new THREE.Vector3(sx, sy, Math.min(sx, 0.6)),
  }
}

/**
 * Piecewise-linear map from a normalised panoramic x coordinate to a signed arc length,
 * anchored on the detected upper teeth. Used to wrap the OPG image back around the arch.
 */
export function panoToArcMapper(result: InferenceResult): (xn: number) => number {
  const anchors = result.teeth
    .filter((d) => TEETH.find((t) => t.fdi === d.fdi)?.upper)
    .map((d) => {
      const ref = TEETH.find((t) => t.fdi === d.fdi) as ToothRef
      return [d.occlusal[0], signedArc(d.fdi, ref)] as [number, number]
    })
    .sort((a, b) => a[0] - b[0])
  if (anchors.length < 2) return (xn) => (xn - 0.5) * 130
  return (xn: number) => {
    let i = 0
    if (xn <= anchors[0][0]) i = 0
    else if (xn >= anchors[anchors.length - 1][0]) i = anchors.length - 2
    else while (i < anchors.length - 2 && xn > anchors[i + 1][0]) i++
    const [x0, s0] = anchors[i]
    const [x1, s1] = anchors[i + 1]
    return s0 + ((xn - x0) / (x1 - x0)) * (s1 - s0)
  }
}

/** Reference-average tooth properties interpolated along a half arch, used to size bone and gingiva. */
export function propsAlongArch(upper: boolean, s: number): { depth: number; crown: number; root: number } {
  const teeth = TEETH.filter((t) => t.upper === upper && t.patientLeft).sort((a, b) => a.position - b.position)
  const pts = teeth.map((t) => ({ s: ARC_CENTER.get(t.fdi) as number, depth: t.depthMm, crown: t.crownMm, root: t.rootMm }))
  const a = Math.abs(s)
  if (a <= pts[0].s) return pts[0]
  if (a >= pts[pts.length - 1].s) return pts[pts.length - 1]
  let i = 0
  while (a > pts[i + 1].s) i++
  const k = (a - pts[i].s) / (pts[i + 1].s - pts[i].s)
  const lerp = (x: number, y: number) => x + (y - x) * k
  return { depth: lerp(pts[i].depth, pts[i + 1].depth), crown: lerp(pts[i].crown, pts[i + 1].crown), root: lerp(pts[i].root, pts[i + 1].root) }
}

/** Distal end of the dentition on each arch (arc length). */
export const ARCH_END = {
  upper: Math.max(...TEETH.filter((t) => t.upper).map((t) => (ARC_CENTER.get(t.fdi) as number) + t.widthMm / 2)),
  lower: Math.max(...TEETH.filter((t) => !t.upper).map((t) => (ARC_CENTER.get(t.fdi) as number) + t.widthMm / 2)),
}
