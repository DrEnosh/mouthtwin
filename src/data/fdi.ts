/**
 * FDI tooth reference data.
 *
 * Dimensions are textbook population averages for permanent teeth (mm). They drive the
 * template geometry and are shown in the UI as reference values, never as measurements.
 */

export type ToothKind = 'incisor' | 'canine' | 'premolar' | 'molar'
export type Quadrant = 1 | 2 | 3 | 4

export interface ToothRef {
  fdi: number
  quadrant: Quadrant
  position: number // 1..8 from the midline
  upper: boolean
  /** Patient's left side (quadrants 2 and 3). Appears on the viewer's right. */
  patientLeft: boolean
  kind: ToothKind
  name: string
  shortName: string
  friendlyName: string
  friendlyRole: string
  universal: number
  palmer: string
  widthMm: number // mesiodistal
  depthMm: number // buccolingual
  crownMm: number
  rootMm: number
  roots: number
}

const UPPER = {
  width: [8.6, 6.6, 7.6, 7.0, 6.6, 10.2, 9.0, 8.4],
  depth: [7.0, 6.0, 8.0, 9.0, 9.0, 11.0, 11.0, 10.5],
  crown: [10.5, 9.0, 10.0, 8.5, 7.8, 7.5, 7.0, 6.5],
  root: [13.0, 13.0, 17.0, 14.0, 14.0, 13.0, 12.5, 11.0],
  roots: [1, 1, 1, 2, 1, 3, 3, 3],
}
const LOWER = {
  width: [5.2, 5.8, 7.0, 7.0, 7.2, 11.2, 10.5, 10.0],
  depth: [5.8, 6.2, 7.5, 7.5, 8.0, 10.3, 10.0, 9.8],
  crown: [9.0, 9.5, 11.0, 8.5, 8.0, 7.5, 7.0, 6.5],
  root: [12.5, 14.0, 16.0, 14.0, 14.5, 14.0, 13.0, 11.0],
  roots: [1, 1, 1, 1, 1, 2, 2, 2],
}

const KIND: ToothKind[] = ['incisor', 'incisor', 'canine', 'premolar', 'premolar', 'molar', 'molar', 'molar']
const POSITION_NAME = [
  'central incisor',
  'lateral incisor',
  'canine',
  'first premolar',
  'second premolar',
  'first molar',
  'second molar',
  'third molar',
]
const FRIENDLY_ROLE: Record<ToothKind, string> = {
  incisor: 'Front tooth with a thin edge for biting into food.',
  canine: 'The longest-rooted tooth. Its pointed tip grips and tears food.',
  premolar: 'Transition tooth with two cusps that crush food.',
  molar: 'Broad back tooth with several cusps that grind food.',
}
const PALMER_MARK: Record<Quadrant, string> = { 1: '┘', 2: '└', 3: '┌', 4: '┐' }

function universalNumber(q: Quadrant, p: number): number {
  if (q === 1) return 9 - p
  if (q === 2) return 8 + p
  if (q === 3) return 25 - p
  return 24 + p
}

function build(): ToothRef[] {
  const out: ToothRef[] = []
  for (const q of [1, 2, 3, 4] as Quadrant[]) {
    const upper = q <= 2
    const patientLeft = q === 2 || q === 3
    const spec = upper ? UPPER : LOWER
    for (let p = 1; p <= 8; p++) {
      const i = p - 1
      const kind = KIND[i]
      const arch = upper ? 'Upper' : 'Lower'
      const side = patientLeft ? 'left' : 'right'
      const zone = p <= 3 ? 'front' : p <= 5 ? 'side' : 'back'
      out.push({
        fdi: q * 10 + p,
        quadrant: q,
        position: p,
        upper,
        patientLeft,
        kind,
        name: `${arch} ${side} ${POSITION_NAME[i]}`,
        shortName: POSITION_NAME[i].replace(/^\w/, (c) => c.toUpperCase()),
        friendlyName: `${arch} ${side} ${zone} tooth`,
        friendlyRole: FRIENDLY_ROLE[kind],
        universal: universalNumber(q, p),
        palmer: `${p}${PALMER_MARK[q]}`,
        widthMm: spec.width[i],
        depthMm: spec.depth[i],
        crownMm: spec.crown[i],
        rootMm: spec.root[i],
        roots: spec.roots[i],
      })
    }
  }
  return out
}

export const TEETH: ToothRef[] = build()
export const TOOTH_BY_FDI = new Map(TEETH.map((t) => [t.fdi, t]))

export function toothRef(fdi: number): ToothRef {
  const t = TOOTH_BY_FDI.get(fdi)
  if (!t) throw new Error(`Unknown FDI number ${fdi}`)
  return t
}

/** OPG reading order: viewer-left to viewer-right, upper row then lower row. */
export const OPG_ORDER_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
export const OPG_ORDER_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
