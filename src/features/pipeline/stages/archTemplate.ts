import { OPG_ORDER_LOWER, OPG_ORDER_UPPER, toothRef } from '../../../data/fdi'
import type { NormPoint, ToothDetection } from '../../../types/inference'
import { toothOutline } from '../../teeth/toothOutline2d'

/**
 * Places a full 32-tooth template along the estimated occlusal curve.
 *
 * This is NOT detection. It assumes a complete adult dentition centred in the image and
 * spaces teeth by average crown widths. The UI labels its output as template mapping.
 */

// Reference geometry of a 1600 px wide OPG (same constants as the synthetic generator).
const REF_W = 1600
const PX_MM_X = 7.5
const PX_MM_Y = 9.0
const GAP_MM = 0.35
const BITE_GAP = 8
const ANTERIOR_SQUEEZE = [0.85, 0.85, 0.92, 1, 1, 1, 1, 1]
const OPG_ROOTS = [1, 1, 1, 1, 1, 2, 2, 2]

export function fitArchTemplate(
  curve: { a: number; b: number; c: number },
  imageW: number,
  imageH: number,
): ToothDetection[] {
  const s = imageW / REF_W // px scale relative to reference
  const yOcc = (xPx: number) => {
    const xn = xPx / imageW
    return (curve.a * xn * xn + curve.b * xn + curve.c) * imageH
  }
  const slope = (xPx: number) => {
    const xn = xPx / imageW
    return ((2 * curve.a * xn + curve.b) * imageH) / imageW
  }
  const cx0 = imageW / 2
  const out: ToothDetection[] = []

  const place = (order: number[], upper: boolean) => {
    for (const side of [-1, 1] as const) {
      // walk outward from the midline on this side
      const quadrantTeeth = order.filter((fdi) => {
        const t = toothRef(fdi)
        return side === -1 ? !t.patientLeft : t.patientLeft
      })
      quadrantTeeth.sort((a, b) => toothRef(a).position - toothRef(b).position)
      let edge = cx0 + side * (GAP_MM * PX_MM_X * s) / 2
      for (const fdi of quadrantTeeth) {
        const t = toothRef(fdi)
        const i = t.position - 1
        const w = t.widthMm * ANTERIOR_SQUEEZE[i] * PX_MM_X * s
        const cx = edge + (side * w) / 2
        edge = cx + side * (w / 2 + GAP_MM * PX_MM_X * s)
        const occ = yOcc(cx) + ((upper ? -1 : 1) * BITE_GAP * s) / 2
        let angle = Math.atan(slope(cx)) * 0.9
        angle += side * (upper ? 0.02 * i : -0.015 * i)
        const ch = t.crownMm * PX_MM_Y * s
        const rl = t.rootMm * PX_MM_Y * s

        const local = toothOutline(t.kind, OPG_ROOTS[i], w, ch, rl, side)
        const ca = Math.cos(angle)
        const sa = Math.sin(angle)
        const toImg = ([x, y]: [number, number]): NormPoint => {
          const yy = upper ? -y : y
          const xr = x * ca - yy * sa
          const yr = x * sa + yy * ca
          return [(cx + xr) / imageW, (occ + yr) / imageH]
        }
        const polygon = local.filter((_, k) => k % 2 === 0).map(toImg)
        const xs = polygon.map((p) => p[0])
        const ys = polygon.map((p) => p[1])
        const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
        out.push({
          fdi,
          bbox: [minX, minY, maxX - minX, maxY - minY],
          polygon,
          occlusal: [cx / imageW, occ / imageH],
          cej: toImg([0, ch]),
          apex: toImg([0, ch + rl]),
          score: null,
        })
      }
    }
  }
  place(OPG_ORDER_UPPER, true)
  place(OPG_ORDER_LOWER, false)
  return out.sort((a, b) => a.fdi - b.fdi)
}
