/**
 * How each tooth sits on this OPG compared with a typical OPG.
 *
 * For every tooth: its long axis (crown end → apex), its length, and how far its crown sits from the jaw's
 * occlusal curve. These are compared with population medians from 634 clinician-labelled DENTEX OPGs
 * (ml/tooth_pose_stats.py, same feature code) so that only the differences are kept: a mesially tipped molar,
 * an impacted canine high in the bone, a short root. The 3D view applies those differences to the reference
 * CBCT teeth. The OPG's own projection tilt (lower molars normally look tipped ~30°) is in the medians, so it
 * cancels out.
 */
import STATS from '../../../data/toothPoseStats.json'
import type { NormPoint, ToothDetection, ToothPose } from '../../../types/inference'

export interface ToothAxis {
  /** crown end and apex, pixel coords on the model grid (x stretched by sx) */
  C: [number, number]
  A: [number, number]
  L: number
  theta: number
  /** length / width of the region; low for crown-only fragments (pontics, broken-down teeth) */
  elong: number
}

type Stat = { n: number; L: number; elong: number; ctheta: number; theta: [number, number]; relL: [number, number]; dv: [number, number] }
const TEETH = (STATS as unknown as { teeth: Record<string, Stat> }).teeth
const BETA = (STATS as unknown as { beta: number }).beta

/** Mirror of tooth_axis() in ml/tooth_pose_stats.py. xs/ys: pixel coords of the tooth region. */
export function toothAxis(pixels: number[], w: number, sx: number, upper: boolean): ToothAxis {
  const n = pixels.length
  let cx = 0, cy = 0
  for (const i of pixels) { const x = i % w; cx += x * sx; cy += (i - x) / w }
  cx /= n; cy /= n
  let sxx = 0, syy = 0, sxy = 0
  for (const i of pixels) {
    const x = i % w
    const dx = x * sx - cx, dy = (i - x) / w - cy
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy
  }
  const ang = 0.5 * Math.atan2(2 * sxy / n, (sxx - syy) / n)
  // minor-axis spread → width of a solid ellipse
  const tr = (sxx + syy) / n, det = (sxx * syy - sxy * sxy) / (n * n)
  const minorVar = Math.max(1e-6, tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det)))
  const width = 4 * Math.sqrt(minorVar)
  let ux = Math.cos(ang), uy = Math.sin(ang)
  if (uy < 0) { ux = -ux; uy = -uy }
  const t = new Float64Array(n)
  pixels.forEach((i, k) => { const x = i % w; t[k] = (x * sx - cx) * ux + ((i - x) / w - cy) * uy })
  const sorted = Float64Array.from(t).sort()
  const q = (p: number) => {
    // numpy linear-interpolated quantile
    const pos = p * (n - 1), lo = Math.floor(pos), hi = Math.min(n - 1, lo + 1)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
  }
  const t0 = q(0.01), t1 = q(0.99)
  const L = t1 - t0
  const band = 0.3 * L
  let lo = 0, hi = 0
  for (let k = 0; k < n; k++) { if (t[k] <= t0 + band) lo++; if (t[k] >= t1 - band) hi++ }
  let crownHi = upper
  if (Math.abs(Math.atan2(ux, uy)) > (55 * Math.PI) / 180) crownHi = hi > lo
  const tc = crownHi ? t1 : t0, ta = crownHi ? t0 : t1
  const C: [number, number] = [cx + ux * tc, cy + uy * tc]
  const A: [number, number] = [cx + ux * ta, cy + uy * ta]
  const ax = A[0] - C[0], ay = A[1] - C[1]
  const theta = upper ? Math.atan2(ax, -ay) : Math.atan2(ax, ay)
  return { C, A, L, theta, elong: L / width }
}

/** Robust polynomial fit of crown points (drops impacted / unerupted outliers). Mirror of robust_curve(). */
function robustCurve(pts: [number, number][]): (x: number) => number {
  const deg = pts.length >= 6 ? 2 : pts.length >= 3 ? 1 : 0
  let keep = pts.map(() => true)
  let f: (x: number) => number = () => 0
  const median = (a: number[]) => { const s = [...a].sort((p, q) => p - q); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2 }
  for (let it = 0; it < 3; it++) {
    const P = pts.filter((_, i) => keep[i])
    f = polyfit(P, deg)
    const r = pts.map(([x, y]) => y - f(x))
    const rk = r.filter((_, i) => keep[i])
    const mr = median(rk)
    const mad = median(rk.map((v) => Math.abs(v - mr))) + 1e-6
    keep = r.map((v) => Math.abs(v) < Math.max(3 * 1.4826 * mad, 6))
    if (keep.filter(Boolean).length < Math.max(3, deg + 1)) { keep = pts.map(() => true); break }
  }
  return f
}

function polyfit(pts: [number, number][], deg: number): (x: number) => number {
  if (deg === 0) {
    const s = pts.map((p) => p[1]).sort((a, b) => a - b)
    const m = s.length >> 1
    const c = s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
    return () => c
  }
  const k = deg + 1
  const M = Array.from({ length: k }, () => new Array(k + 1).fill(0))
  for (const [x, y] of pts) {
    const pw = Array.from({ length: k }, (_, i) => x ** i)
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) M[r][c] += pw[r] * pw[c]
      M[r][k] += pw[r] * y
    }
  }
  for (let i = 0; i < k; i++) {
    let p = i
    for (let r = i + 1; r < k; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r
    ;[M[i], M[p]] = [M[p], M[i]]
    if (Math.abs(M[i][i]) < 1e-12) return () => pts[0][1]
    for (let r = 0; r < k; r++) {
      if (r === i) continue
      const f = M[r][i] / M[i][i]
      for (let c = i; c <= k; c++) M[r][c] -= f * M[i][c]
    }
  }
  const coef = M.map((row, i) => row[k] / row[i])
  return (x) => coef.reduce((s, c, i) => s + c * x ** i, 0)
}

const median = (a: number[]) => [...a].sort((p, q) => p - q)[a.length >> 1]
/** soft dead zone: values within ±z are treated as measurement noise */
const dead = (v: number, z: number) => (Math.abs(v) <= z ? 0 : v - Math.sign(v) * z)

/**
 * Per-tooth pose relative to a typical OPG. `axes` keyed by FDI; grid w×h with horizontal stretch sx.
 * Returns poses keyed by FDI (teeth without population stats are skipped).
 */
export function fitPoses(axes: Map<number, ToothAxis>, w: number, h: number, sx: number): Map<number, ToothPose> {
  const out = new Map<number, ToothPose>()
  const known = [...axes].filter(([f]) => TEETH[String(f)])
  if (known.length < 4) return out
  const k = median(known.map(([f, a]) => a.L / TEETH[String(f)].L))
  const dv = new Map<number, number>()
  const ctheta = new Map<number, number>()
  for (const upper of [true, false]) {
    const jaw = known.filter(([f]) => (f < 30) === upper)
    if (!jaw.length) continue
    // third molars often sit off the occlusal plane; fit the curve without them when possible
    const base = jaw.filter(([f]) => f % 10 !== 8)
    const curve = robustCurve((base.length >= 4 ? base : jaw).map(([, a]) => a.C))
    for (const [f, a] of jaw) {
      const x0 = a.C[0]
      const d = a.C[1] - curve(x0)
      dv.set(f, upper ? -d : d)
      const slope = (curve(x0 + 1) - curve(x0 - 1)) / 2
      ctheta.set(f, upper ? Math.atan(slope) : -Math.atan(slope))
    }
  }
  const norm = (p: [number, number]): NormPoint => [p[0] / sx / w, p[1] / h]
  for (const [f, a] of known) {
    const s = TEETH[String(f)]
    // third molars: the population mixes erupted teeth with crown-only buds, so size them against a
    // fully formed second molar instead (a formed third molar is ~0.93 of it on OPGs)
    const third = f % 10 === 8
    const s7 = third ? TEETH[String(f - 1)] : undefined
    const refL = s7 ? 0.93 * s7.L : s.L
    const refElong = s7 ? s7.elong : s.elong
    const relL = a.L / k / refL
    // remove the fanning caused by head position: part of the tilt follows the local slope of the occlusal curve
    const curveDev = (ctheta.get(f) ?? s.ctheta) - s.ctheta
    let tiltDev = a.theta - BETA * curveDev - s.theta[0]
    tiltDev = Math.atan2(Math.sin(tiltDev), Math.cos(tiltDev))
    // a short or stubby region (crown without visible root, pontic, broken-down tooth) has no reliable long axis
    const elongRel = a.elong / refElong
    const axisOk = relL >= 0.7 && elongRel >= 0.75 && (Math.abs(tiltDev) < 1 || elongRel >= 0.9)
    if (!axisOk) tiltDev = 0
    // the reference CBCT has every tooth erupted into the occlusal plane. Third molars are often unerupted in the
    // population, so their "typical" depth is not a usable baseline: compare them with the occlusal plane itself.
    const dvBase = third ? 0 : s.dv[0]
    const dvSd = third ? 3 : s.dv[1]
    const dvRaw = (dv.get(f) ?? 0) / k - dvBase
    const depth = dvRaw / s.L
    out.set(f, {
      crown: norm(a.C),
      apex: norm(a.A),
      theta: a.theta,
      // deviations from the typical OPG, with measurement noise removed
      tilt: dead(tiltDev, 0.6 * s.theta[1]),
      length: 1 + dead(relL - 1, 0.6 * s.relL[1]),
      // much stubbier than this tooth normally is: probably only the crown is there (developing / unerupted
      // tooth, crown on a pontic)
      crownOnly: elongRel < 0.65 || relL < 0.6,
      partialRoot: third && relL < 0.85,
      depth: dead(depth, (1.2 * dvSd) / s.L),
      // how unusual, in robust standard deviations (for the UI)
      z: {
        tilt: Math.abs(tiltDev) / Math.max(1e-3, s.theta[1]),
        length: Math.abs(relL - 1) / Math.max(1e-3, s.relL[1]),
        depth: Math.abs(dvRaw) / Math.max(1e-3, dvSd),
      },
    })
  }
  return out
}

export function attachPoses(dets: ToothDetection[], poses: Map<number, ToothPose>): ToothDetection[] {
  return dets.map((d) => (poses.has(d.fdi) ? { ...d, pose: poses.get(d.fdi) } : d))
}
