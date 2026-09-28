/**
 * Turns the model's per-pixel labels into per-tooth detections:
 * largest connected region per FDI class → bbox, traced outline, occlusal/apex points, mean confidence.
 */
import type { NormPoint, ToothDetection } from '../../../types/inference'
import { classToFdi, type SegOutput } from './segModel'

const MIN_AREA = 180 // pixels at 768×384; smaller regions are noise

interface Region {
  cls: number
  pixels: number[]
  conf: number
}

function largestComponents(seg: SegOutput): Region[] {
  const { width: w, height: h, labels, conf } = seg
  const seen = new Uint8Array(labels.length)
  const best = new Map<number, Region>()
  const stack: number[] = []
  for (let s = 0; s < labels.length; s++) {
    const c = labels[s]
    if (!c || seen[s]) continue
    const pixels: number[] = []
    let cs = 0
    stack.push(s)
    seen[s] = 1
    while (stack.length) {
      const i = stack.pop() as number
      pixels.push(i)
      cs += conf[i]
      const x = i % w
      const y = (i - x) / w
      if (x > 0 && !seen[i - 1] && labels[i - 1] === c) { seen[i - 1] = 1; stack.push(i - 1) }
      if (x < w - 1 && !seen[i + 1] && labels[i + 1] === c) { seen[i + 1] = 1; stack.push(i + 1) }
      if (y > 0 && !seen[i - w] && labels[i - w] === c) { seen[i - w] = 1; stack.push(i - w) }
      if (y < h - 1 && !seen[i + w] && labels[i + w] === c) { seen[i + w] = 1; stack.push(i + w) }
    }
    const prev = best.get(c)
    if (!prev || pixels.length > prev.pixels.length) best.set(c, { cls: c, pixels, conf: cs / pixels.length })
  }
  return [...best.values()].filter((r) => r.pixels.length >= MIN_AREA)
}

/** Moore-neighbour boundary trace of a binary mask; returns the outer contour in pixel coords. */
function traceContour(mask: Uint8Array, w: number, h: number, start: number): [number, number][] {
  const at = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1
  const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
  let x = start % w
  let y = (start - x) / w
  const sx = x, sy = y
  let d = 6 // came from the left-top scan order, so start looking "up"
  const out: [number, number][] = []
  for (let guard = 0; guard < 20000; guard++) {
    out.push([x, y])
    let found = false
    for (let k = 0; k < 8; k++) {
      const nd = (d + 6 + k) % 8 // turn left first, then sweep clockwise
      const nx = x + dirs[nd][0]
      const ny = y + dirs[nd][1]
      if (at(nx, ny)) {
        x = nx
        y = ny
        d = nd
        found = true
        break
      }
    }
    if (!found || (x === sx && y === sy && out.length > 2)) break
  }
  return out
}

export function detectionsFromSegmentation(seg: SegOutput): ToothDetection[] {
  const { width: w, height: h } = seg
  const regions = largestComponents(seg)
  const mask = new Uint8Array(w * h)
  const dets: ToothDetection[] = []
  for (const r of regions) {
    const fdi = classToFdi(r.cls)
    const upper = fdi < 30
    let minX = w, maxX = 0, minY = h, maxY = 0
    let start = Infinity
    for (const i of r.pixels) {
      mask[i] = 1
      const x = i % w
      const y = (i - x) / w
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (i < start) start = i
    }
    const contour = traceContour(mask, w, h, start)
    for (const i of r.pixels) mask[i] = 0

    const step = Math.max(1, Math.floor(contour.length / 48))
    const polygon: NormPoint[] = contour.filter((_, k) => k % step === 0).map(([x, y]) => [(x + 0.5) / w, (y + 0.5) / h])

    // occlusal edge: the crown end facing the other arch; apex: the opposite end
    const band = Math.max(2, Math.round((maxY - minY) * 0.08))
    const edgeY = upper ? maxY : minY
    const apexY = upper ? minY : maxY
    const xsAt = (yy: number) => {
      let sum = 0, n = 0
      for (const i of r.pixels) {
        const y = Math.floor(i / w)
        if (Math.abs(y - yy) <= band) { sum += i % w; n++ }
      }
      return n ? sum / n : (minX + maxX) / 2
    }
    dets.push({
      fdi,
      bbox: [minX / w, minY / h, (maxX - minX + 1) / w, (maxY - minY + 1) / h],
      polygon,
      occlusal: [xsAt(edgeY) / w, edgeY / h],
      apex: [xsAt(apexY) / w, apexY / h],
      score: Math.round(r.conf * 1000) / 1000,
    })
  }
  return dets.sort((a, b) => a.fdi - b.fdi)
}

/** Occlusal plane from the model's teeth: least-squares quadratic through all occlusal points. */
export function occlusalCurveFrom(dets: ToothDetection[]): { a: number; b: number; c: number } | undefined {
  const pts = dets.map((d) => d.occlusal)
  if (pts.length < 6) return undefined
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0
  for (const [x, y] of pts) {
    const x2 = x * x
    s0++; s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2
    t0 += y; t1 += x * y; t2 += x2 * y
  }
  const m = [[s4, s3, s2, t2], [s3, s2, s1, t1], [s2, s1, s0, t0]]
  for (let i = 0; i < 3; i++) {
    for (let r = 0; r < 3; r++) {
      if (r === i || Math.abs(m[i][i]) < 1e-12) continue
      const f = m[r][i] / m[i][i]
      for (let c = i; c < 4; c++) m[r][c] -= f * m[i][c]
    }
  }
  if (m.some((row, i) => Math.abs(row[i]) < 1e-12)) return undefined
  return { a: m[0][3] / m[0][0], b: m[1][3] / m[1][1], c: m[2][3] / m[2][2] }
}
