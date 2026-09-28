/**
 * Turns the uploaded OPG itself into height fields for a 3D relief.
 *
 * What comes from the image: every outline (crowns, roots, gaps, restorations, bone margins),
 * the brightness pattern painted on the surface, and which pixels count as tooth or bone.
 * What is estimated: thickness. A panoramic image has no depth, so thickness is inferred from
 * brightness (denser → brighter → thicker) and from distance to the tooth edge (teeth are round).
 */
import type { InferenceResult } from '../../types/inference'

export interface ReliefData {
  gw: number
  gh: number
  /** Normalised brightness 0..1 */
  v: Float32Array<ArrayBufferLike>
  /** Tooth thickness (half-depth, scene mm) per grid point; 0 outside teeth */
  toothH: Float32Array<ArrayBufferLike>
  /** Bone thickness (half-depth) per grid point; 0 outside bone */
  boneH: Float32Array<ArrayBufferLike>
  /** Smoothed, unthresholded fields and their cut levels, used to place outline vertices sub-pixel */
  toothField: Float32Array<ArrayBufferLike>
  boneField: Float32Array<ArrayBufferLike>
  toothLevel: number
  boneLevel: number
  /** FDI id per grid point (0 = none) */
  ids: Uint8Array<ArrayBufferLike>
  toothMask: Uint8Array<ArrayBufferLike>
  occ: (xn: number) => number
}

function percentile(values: Float32Array | number[], p: number): number {
  const a = Float32Array.from(values).sort()
  return a[Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))))]
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  if (r <= 0) return src
  const tmp = new Float32Array(src.length)
  const out = new Float32Array(src.length)
  for (let y = 0; y < h; y++) {
    let acc = 0
    for (let x = -r; x <= r; x++) acc += src[y * w + Math.min(w - 1, Math.max(0, x))]
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / (2 * r + 1)
      const xo = Math.max(0, x - r)
      const xi = Math.min(w - 1, x + r + 1)
      acc += src[y * w + xi] - src[y * w + xo]
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x]
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / (2 * r + 1)
      const yo = Math.max(0, y - r)
      const yi = Math.min(h - 1, y + r + 1)
      acc += tmp[yi * w + x] - tmp[yo * w + x]
    }
  }
  return out
}

/** Two-pass chamfer distance to the nearest pixel outside the mask. */
function distanceInside(mask: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e9
  const d = new Float32Array(w * h)
  for (let i = 0; i < d.length; i++) d[i] = mask[i] ? INF : 0
  const a = 1, b = Math.SQRT2
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (!d[i]) continue
      let m = d[i]
      if (x > 0) m = Math.min(m, d[i - 1] + a)
      else m = Math.min(m, a)
      if (y > 0) {
        m = Math.min(m, d[i - w] + a)
        if (x > 0) m = Math.min(m, d[i - w - 1] + b)
        if (x < w - 1) m = Math.min(m, d[i - w + 1] + b)
      } else m = Math.min(m, a)
      d[i] = m
    }
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x
      if (!d[i]) continue
      let m = d[i]
      if (x < w - 1) m = Math.min(m, d[i + 1] + a)
      else m = Math.min(m, a)
      if (y < h - 1) {
        m = Math.min(m, d[i + w] + a)
        if (x < w - 1) m = Math.min(m, d[i + w + 1] + b)
        if (x > 0) m = Math.min(m, d[i + w - 1] + b)
      } else m = Math.min(m, a)
      d[i] = m
    }
  return d
}

function morph(mask: Uint8Array, w: number, h: number, erode: boolean): Uint8Array {
  const out = new Uint8Array(mask.length)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let hit = erode ? 1 : 0
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.min(w - 1, Math.max(0, x + dx))
          const yy = Math.min(h - 1, Math.max(0, y + dy))
          const m = mask[yy * w + xx]
          if (erode && !m) hit = 0
          if (!erode && m) hit = 1
        }
      out[y * w + x] = hit
    }
  return out
}

/** Keep connected components that are big enough and pass `keep` (e.g. reach the occlusal plane). */
function filterComponents(mask: Uint8Array, w: number, h: number, minSize: number, keep: (i: number) => boolean): Uint8Array {
  const seen = new Uint8Array(mask.length)
  const out = new Uint8Array(mask.length)
  const stack: number[] = []
  const comp: number[] = []
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue
    comp.length = 0
    stack.push(s)
    seen[s] = 1
    while (stack.length) {
      const i = stack.pop() as number
      comp.push(i)
      const x = i % w, y = (i - x) / w
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]
      for (const j of nb) if (j >= 0 && mask[j] && !seen[j]) { seen[j] = 1; stack.push(j) }
    }
    if (comp.length >= minSize && comp.some(keep)) for (const i of comp) out[i] = 1
  }
  return out
}

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

export function analyzeOpg(img: HTMLImageElement, result: InferenceResult, gw = 720): ReliefData {
  const gh = Math.max(2, Math.round((gw * img.naturalHeight) / img.naturalWidth))
  const canvas = document.createElement('canvas')
  canvas.width = gw
  canvas.height = gh
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D
  ctx.drawImage(img, 0, 0, gw, gh)
  const rgba = ctx.getImageData(0, 0, gw, gh).data
  const g = new Float32Array(gw * gh)
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = (0.2126 * rgba[j] + 0.7152 * rgba[j + 1] + 0.0722 * rgba[j + 2]) / 255

  const curve = result.occlusalCurve ?? { a: 0, b: 0, c: 0.5 }
  const occ = (xn: number) => curve.a * xn * xn + curve.b * xn + curve.c

  // contrast normalisation from the central region
  const central: number[] = []
  for (let y = Math.floor(gh * 0.15); y < gh * 0.85; y += 2)
    for (let x = Math.floor(gw * 0.1); x < gw * 0.9; x += 2) central.push(g[y * gw + x])
  const lo = percentile(central, 0.01)
  const hi = percentile(central, 0.995)
  let v: Float32Array = new Float32Array(g.length)
  for (let i = 0; i < g.length; i++) v[i] = Math.min(1, Math.max(0, (g[i] - lo) / Math.max(1e-3, hi - lo)))
  v = boxBlur(v, gw, gh, 1)

  // --- teeth: the brightest structures in the band around the occlusal plane
  const inBand = (x: number, y: number) => {
    const xn = x / gw
    const yn = y / gh
    return xn > 0.1 && xn < 0.9 && Math.abs(yn - occ(xn)) < 0.36
  }
  const band: number[] = []
  for (let y = 0; y < gh; y += 2) for (let x = 0; x < gw; x += 2) if (inBand(x, y)) band.push(v[y * gw + x])
  const tT = percentile(band, 0.7)
  let toothMask: Uint8Array = new Uint8Array(g.length)
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) if (inBand(x, y) && v[y * gw + x] > tT) toothMask[y * gw + x] = 1
  toothMask = morph(morph(toothMask, gw, gh, true), gw, gh, false)
  // teeth reach the occlusal plane; palate lines, cortices and canal borders don't
  toothMask = filterComponents(toothMask, gw, gh, Math.round(gw * gh * 0.0005), (i) => {
    const x = i % gw
    const y = (i - x) / gw
    return Math.abs(y / gh - occ(x / gw)) < 0.1
  })

  // round the teeth: thickness grows with distance from the tooth edge (smoothed to avoid terracing)
  let dist: Float32Array = distanceInside(toothMask, gw, gh)
  dist = boxBlur(boxBlur(dist, gw, gh, 2), gw, gh, 2)
  const R = gw / 72 // ≈ half a tooth width in grid pixels
  const pxToMm = 170 / gw
  let toothH: Float32Array = new Float32Array(g.length)
  for (let i = 0; i < g.length; i++) {
    if (!toothMask[i]) continue
    const round = Math.sqrt(Math.min(1, dist[i] / R))
    const soft = smoothstep(tT - 0.08, tT + 0.05, v[i])
    toothH[i] = R * pxToMm * 1.7 * round * (0.72 + 0.4 * v[i]) * (0.35 + 0.65 * soft)
  }
  toothH = boxBlur(boxBlur(toothH, gw, gh, 1), gw, gh, 1)
  const toothField = Float32Array.from(toothH)
  for (let i = 0; i < g.length; i++) if (toothH[i] < 0.04) toothH[i] = 0

  // --- bone: everything moderately radiopaque that isn't tooth
  const vb = boxBlur(v, gw, gh, 3)
  const all: number[] = []
  for (let i = 0; i < g.length; i += 3) all.push(vb[i])
  const tB = percentile(all, 0.42)
  let boneH: Float32Array = new Float32Array(g.length)
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x
      const xn = x / gw
      const yn = y / gh
      const edge = smoothstep(0.02, 0.07, xn) * smoothstep(0.02, 0.07, 1 - xn)
      const band = 1 - smoothstep(0.3, 0.4, Math.abs(yn - occ(xn))) // keeps skull base, labels and markers out
      boneH[i] = 5.5 * smoothstep(tB, tB + 0.22, vb[i]) * edge * band
    }
  boneH = boxBlur(boneH, gw, gh, 2)
  const boneField = Float32Array.from(boneH)
  for (let i = 0; i < g.length; i++) if (boneH[i] < 0.05) boneH[i] = 0

  // --- tooth ids: polygon hit, otherwise nearest detection in the same arch
  const idCanvas = document.createElement('canvas')
  idCanvas.width = gw
  idCanvas.height = gh
  const ic = idCanvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D
  for (const t of result.teeth) {
    ic.fillStyle = `rgb(${t.fdi},0,0)`
    ic.beginPath()
    if (t.polygon.length > 2) t.polygon.forEach(([x, y], k) => (k ? ic.lineTo(x * gw, y * gh) : ic.moveTo(x * gw, y * gh)))
    else ic.rect(t.bbox[0] * gw, t.bbox[1] * gh, t.bbox[2] * gw, t.bbox[3] * gh)
    ic.closePath()
    ic.fill()
  }
  const idPx = ic.getImageData(0, 0, gw, gh).data
  const ids = new Uint8Array(g.length)
  const centers = result.teeth.map((t) => ({
    fdi: t.fdi,
    cx: t.bbox[0] + t.bbox[2] / 2,
    cy: t.bbox[1] + t.bbox[3] / 2,
    upper: t.fdi < 30,
  }))
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x
      if (idPx[i * 4 + 3] === 255 && idPx[i * 4]) {
        ids[i] = idPx[i * 4]
        continue
      }
      const xn = x / gw
      const yn = y / gh
      if (Math.abs(yn - occ(xn)) > 0.4) continue
      const upper = yn < occ(xn)
      let best = 0
      let bd = Infinity
      for (const c of centers) {
        if (c.upper !== upper) continue
        const d = 9 * (xn - c.cx) ** 2 + (yn - c.cy) ** 2
        if (d < bd) { bd = d; best = c.fdi }
      }
      if (bd < 0.012) ids[i] = best
    }

  return { gw, gh, v, toothH, boneH, toothField, boneField, toothLevel: 0.04, boneLevel: 0.05, ids, toothMask, occ }
}
