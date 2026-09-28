/**
 * 2D tooth silhouettes as they read on a panoramic image.
 * Mirrors scripts/generate_synthetic_opg.py so template output looks like the demo annotations.
 */
import type { ToothKind } from '../../data/fdi'

type P = [number, number]

function crownProfile(kind: ToothKind): P[] {
  switch (kind) {
    case 'incisor':
      return [[-0.4, 1], [-0.5, 0.55], [-0.5, 0.15], [-0.45, 0.02], [-0.2, 0], [0.2, 0], [0.45, 0.02], [0.5, 0.15], [0.5, 0.55], [0.4, 1]]
    case 'canine':
      return [[-0.4, 1], [-0.5, 0.5], [-0.42, 0.2], [-0.15, 0.05], [0, 0], [0.15, 0.05], [0.42, 0.2], [0.5, 0.5], [0.4, 1]]
    case 'premolar':
      return [[-0.4, 1], [-0.5, 0.45], [-0.45, 0.12], [-0.25, 0.01], [0, 0.07], [0.25, 0.01], [0.45, 0.12], [0.5, 0.45], [0.4, 1]]
    default:
      return [[-0.42, 1], [-0.5, 0.4], [-0.47, 0.1], [-0.33, 0], [-0.15, 0.07], [0.02, 0], [0.18, 0.07], [0.34, 0], [0.47, 0.1], [0.5, 0.4], [0.42, 1]]
  }
}

function rootProfile(n: number): P[] {
  if (n === 1) return [[-0.4, 0], [-0.34, 0.35], [-0.2, 0.8], [-0.07, 1], [0.07, 1], [0.2, 0.8], [0.34, 0.35], [0.4, 0]]
  return [[-0.42, 0], [-0.42, 0.4], [-0.35, 0.85], [-0.26, 1], [-0.14, 0.95], [-0.09, 0.5], [0, 0.28], [0.09, 0.5], [0.14, 0.95], [0.26, 1], [0.35, 0.85], [0.42, 0.4], [0.42, 0]]
}

export function chaikin(pts: P[], iterations = 2): P[] {
  let out = pts
  for (let k = 0; k < iterations; k++) {
    const next: P[] = []
    for (let i = 0; i < out.length; i++) {
      const a = out[i]
      const b = out[(i + 1) % out.length]
      next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]])
      next.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]])
    }
    out = next
  }
  return out
}

/**
 * Tooth outline in local pixels: x across the tooth, y from the occlusal edge (0) toward the apex.
 * `distalSign` bends the root apex away from the midline, as roots usually curve distally.
 */
export function toothOutline(kind: ToothKind, opgRoots: number, w: number, ch: number, rl: number, distalSign: number): P[] {
  const bend = ([x, y]: P): P => {
    if (y > ch) {
      const t = (y - ch) / rl
      return [x + distalSign * 0.12 * w * t * t, y]
    }
    return [x, y]
  }
  const crown = crownProfile(kind).map(([x, y]) => [x * w, y * ch] as P)
  const roots = rootProfile(opgRoots).map(([x, y]) => bend([x * w, ch + y * rl]))
  const outline = [...crown, ...roots.slice().reverse().slice(1, -1)]
  return chaikin(outline, 2)
}
