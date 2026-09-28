import type { GrayImage } from './preprocess'

export interface OcclusalEstimate {
  curve: { a: number; b: number; c: number }
  samples: [number, number][]
  usedFallback: boolean
}

/**
 * Classical CV heuristic: on an OPG the gap between the upper and lower teeth is the darkest
 * horizontal band in the middle of the image. We find that minimum in several vertical strips
 * and fit the panoramic "smile" curve through them.
 */
export function estimateOcclusalPlane(g: GrayImage): OcclusalEstimate {
  const bands = [0.3, 0.37, 0.44, 0.5, 0.56, 0.63, 0.7]
  const y0 = Math.floor(g.height * 0.3)
  const y1 = Math.floor(g.height * 0.72)
  const halfBand = Math.max(2, Math.floor(g.width * 0.025))
  const samples: [number, number][] = []

  for (const bx of bands) {
    const cx = Math.floor(bx * g.width)
    const profile: number[] = []
    for (let y = y0; y < y1; y++) {
      let s = 0
      for (let x = cx - halfBand; x <= cx + halfBand; x++) s += g.data[y * g.width + x]
      profile.push(s / (2 * halfBand + 1))
    }
    const r = Math.max(2, Math.floor(g.height * 0.006))
    let best = Infinity
    let bestY = (y0 + y1) / 2
    for (let i = r; i < profile.length - r; i++) {
      let s = 0
      for (let k = -r; k <= r; k++) s += profile[i + k]
      if (s < best) {
        best = s
        bestY = y0 + i
      }
    }
    samples.push([bx, bestY / g.height])
  }

  const fit = fitQuadratic(samples)
  const median = [...samples].map((s) => s[1]).sort((a, b) => a - b)[Math.floor(samples.length / 2)]
  const valid =
    fit &&
    fit.a <= 0.05 &&
    fit.a >= -0.9 &&
    [0.2, 0.5, 0.8].every((x) => {
      const y = fit.a * x * x + fit.b * x + fit.c
      return y > 0.3 && y < 0.72
    })

  if (!valid) return { curve: { a: 0, b: 0, c: median }, samples, usedFallback: true }
  return { curve: fit, samples, usedFallback: false }
}

function fitQuadratic(pts: [number, number][]): { a: number; b: number; c: number } | null {
  // normal equations for y = a x² + b x + c
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0
  for (const [x, y] of pts) {
    const x2 = x * x
    s0 += 1; s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2
    t0 += y; t1 += x * y; t2 += x2 * y
  }
  const m = [
    [s4, s3, s2, t2],
    [s3, s2, s1, t1],
    [s2, s1, s0, t0],
  ]
  for (let i = 0; i < 3; i++) {
    let p = i
    for (let r = i + 1; r < 3; r++) if (Math.abs(m[r][i]) > Math.abs(m[p][i])) p = r
    ;[m[i], m[p]] = [m[p], m[i]]
    if (Math.abs(m[i][i]) < 1e-12) return null
    for (let r = 0; r < 3; r++) {
      if (r === i) continue
      const f = m[r][i] / m[i][i]
      for (let c = i; c < 4; c++) m[r][c] -= f * m[i][c]
    }
  }
  return { a: m[0][3] / m[0][0], b: m[1][3] / m[1][1], c: m[2][3] / m[2][2] }
}
