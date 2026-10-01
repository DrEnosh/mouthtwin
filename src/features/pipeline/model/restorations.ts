/**
 * What is on each tooth: crown / cap, filling, root-canal filling, implant, bridge.
 *
 * Inputs are the restoration planes of the tooth model (implant, prosthetic restoration, filling, root-canal
 * treatment; trained on boxes from the Cluj OPG dataset) and the X-ray brightness inside each tooth outline.
 * A restoration is only reported when the model flags it AND, for crowns and fillings, the pixels are actually
 * radiopaque. Nothing here knows the material or the quality of the work.
 */
import { OPG_ORDER_LOWER, OPG_ORDER_UPPER } from '../../../data/fdi'
import type { ImplantDetection, ToothDetection, ToothRestorations } from '../../../types/inference'
import type { SegOutput } from './segModel'
import { occlusalCurveFrom } from './postprocess'

const P = 127 // probability 0.5 on the 0..255 planes
const CROWN_FRACTION = 0.46 // share of the tooth height, from the occlusal edge, counted as crown

export const fdiToClass = (fdi: number) => (Math.floor(fdi / 10) - 1) * 8 + (fdi % 10)

export interface RestorationAnalysis {
  teeth: ToothDetection[]
  implants: ImplantDetection[]
}

interface Comp {
  size: number
  minX: number
  maxX: number
  minY: number
  maxY: number
  sx: number
  sy: number
}

/** 8-connected components of a mask. Returns the label image (0 = none) and per-label stats (index = label). */
function components(mask: Uint8Array, w: number, h: number): { lab: Int32Array; comps: Comp[] } {
  const lab = new Int32Array(mask.length)
  const comps: Comp[] = [null as unknown as Comp]
  const stack: number[] = []
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || lab[s]) continue
    const id = comps.length
    const c: Comp = { size: 0, minX: w, maxX: 0, minY: h, maxY: 0, sx: 0, sy: 0 }
    stack.push(s)
    lab[s] = id
    while (stack.length) {
      const i = stack.pop() as number
      const x = i % w
      const y = (i - x) / w
      c.size++
      c.sx += x
      c.sy += y
      if (x < c.minX) c.minX = x
      if (x > c.maxX) c.maxX = x
      if (y < c.minY) c.minY = y
      if (y > c.maxY) c.maxY = y
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue
          const nx = x + dx
          const ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          const j = ny * w + nx
          if (mask[j] && !lab[j]) {
            lab[j] = id
            stack.push(j)
          }
        }
      }
    }
    comps.push(c)
  }
  return { lab, comps }
}

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]

export function analyseRestorations(seg: SegOutput, teeth: ToothDetection[]): RestorationAnalysis {
  const { width: w, height: h, labels, rest, input } = seg
  if (!rest || !input) return { teeth, implants: [] }
  const plane = (k: number) => rest.subarray(k * w * h, (k + 1) * w * h)
  const imp = plane(0)
  const prr = plane(1)
  const fil = plane(2)
  const rct = plane(3)

  // per-tooth pixel lists
  const pix = new Map<number, number[]>()
  const lum: number[] = []
  for (const d of teeth) {
    const cls = fdiToClass(d.fdi)
    const [bx, by, bw, bh] = d.bbox
    const x0 = Math.max(0, Math.floor(bx * w)), x1 = Math.min(w - 1, Math.ceil((bx + bw) * w))
    const y0 = Math.max(0, Math.floor(by * h)), y1 = Math.min(h - 1, Math.ceil((by + bh) * h))
    const list: number[] = []
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * w + x
        if (labels[i] === cls) {
          list.push(i)
          lum.push(input[i])
        }
      }
    }
    pix.set(d.fdi, list)
  }
  if (lum.length < 200) return { teeth, implants: [] }
  lum.sort((a, b) => a - b)
  const med = quantile(lum, 0.5)
  const hi = quantile(lum, 0.995)
  // "bright" = clearly denser than the tooth's usual grey; never below an absolute floor
  const bright = Math.max(0.72, med + 0.5 * (hi - med))

  // bright restoration mask (for bridge detection and the radiopacity check)
  const brightRest = new Uint8Array(w * h)
  for (let i = 0; i < brightRest.length; i++) if ((prr[i] > P || fil[i] > P) && input[i] >= bright) brightRest[i] = 1
  const { lab: bl } = components(brightRest, w, h)

  const perTooth = new Map<number, ToothRestorations>()
  const capComp = new Map<number, number>()

  for (const d of teeth) {
    const list = pix.get(d.fdi) ?? []
    if (list.length < 80) continue
    const upper = d.fdi < 30
    const [, by, , bh] = d.bbox
    const topY = by * h
    const height = bh * h
    const crownLimit = upper ? topY + height * (1 - CROWN_FRACTION) : topY + height * CROWN_FRACTION
    let crown = 0, root = 0, crownRest = 0, crownBright = 0, crownFill = 0, rootRct = 0, impPix = 0
    const compVotes = new Map<number, number>()
    for (const i of list) {
      const y = Math.floor(i / w)
      const inCrown = upper ? y >= crownLimit : y <= crownLimit
      if (imp[i] > P) impPix++
      if (inCrown) {
        crown++
        const r = prr[i] > P || fil[i] > P
        if (r) crownRest++
        if (fil[i] > P) crownFill++
        if (r && input[i] >= bright) {
          crownBright++
          const c = bl[i]
          if (c) compVotes.set(c, (compVotes.get(c) ?? 0) + 1)
        }
      } else {
        root++
        if (rct[i] > P) rootRct++
      }
    }
    const r: ToothRestorations = {}
    if (crown > 40) {
      const restFrac = crownRest / crown
      const brightFrac = crownBright / crown
      if (restFrac >= 0.3) {
        if (brightFrac >= 0.5) r.crown = { kind: 'cap', coverage: round(brightFrac) }
        else if (brightFrac >= 0.08 || crownFill / crown >= 0.5) r.crown = { kind: 'filling', coverage: round(brightFrac >= 0.08 ? brightFrac : restFrac) }
      }
    }
    if (root > 40 && rootRct / root >= 0.25) r.rootCanal = { coverage: round(rootRct / root) }
    if (impPix / list.length >= 0.25) r.implant = true
    if (r.crown?.kind === 'cap') {
      let best = 0, bestN = 0
      for (const [c, n] of compVotes) if (n > bestN) { best = c; bestN = n }
      if (best && bestN / crown >= 0.4) capComp.set(d.fdi, best)
    }
    if (Object.keys(r).length) perTooth.set(d.fdi, r)
  }

  // bridge: crowns of the same jaw that share one connected radiopaque block
  const groups = new Map<string, number[]>()
  for (const [fdi, comp] of capComp) {
    const key = `${comp}:${fdi < 30 ? 'u' : 'l'}`
    groups.set(key, [...(groups.get(key) ?? []), fdi])
  }
  let bridgeId = 0
  for (const members of groups.values()) {
    if (members.length < 2) continue
    bridgeId++
    for (const fdi of members) (perTooth.get(fdi) as ToothRestorations).bridgeId = bridgeId
  }

  const outTeeth = teeth.map((d) => (perTooth.has(d.fdi) ? { ...d, restorations: perTooth.get(d.fdi) } : { ...d, restorations: {} }))

  // implants with no tooth outline of their own
  const owned = new Set<number>()
  for (const d of teeth) for (const i of pix.get(d.fdi) ?? []) owned.add(i)
  const impMask = new Uint8Array(w * h)
  for (let i = 0; i < impMask.length; i++) if (imp[i] > P) impMask[i] = 1
  const { lab: il, comps: ic } = components(impMask, w, h)
  const inside = new Array(ic.length).fill(0)
  for (let i = 0; i < il.length; i++) if (il[i] && owned.has(i)) inside[il[i]]++
  const curve = occlusalCurveFrom(teeth)
  const centreX = (d: ToothDetection) => d.bbox[0] + d.bbox[2] / 2
  const implants: ImplantDetection[] = []
  const taken = new Set<number>()
  for (let id = 1; id < ic.length; id++) {
    const c = ic[id]
    if (c.size < 90 || inside[id] / c.size >= 0.3) continue
    const cx = c.sx / c.size / w
    const cy = c.sy / c.size / h
    const upper = curve ? cy < curve.a * cx * cx + curve.b * cx + curve.c : cy < 0.5
    const order = upper ? OPG_ORDER_UPPER : OPG_ORDER_LOWER
    const xs: (number | null)[] = order.map((f) => {
      const d = teeth.find((t) => t.fdi === f)
      return d ? centreX(d) : null
    })
    const estimate = (idx: number): number | null => {
      let l = idx - 1
      while (l >= 0 && xs[l] === null) l--
      let r = idx + 1
      while (r < xs.length && xs[r] === null) r++
      const lx = l >= 0 ? (xs[l] as number) : null
      const rx = r < xs.length ? (xs[r] as number) : null
      if (lx !== null && rx !== null) return lx + ((rx - lx) * (idx - l)) / (r - l)
      if (lx !== null) return lx + 0.028 * (idx - l)
      if (rx !== null) return rx - 0.028 * (r - idx)
      return null
    }
    let slot: number | null = null
    let bestD = 0.04
    order.forEach((f, idx) => {
      if (xs[idx] !== null || taken.has(f)) return
      const e = estimate(idx)
      if (e !== null && Math.abs(e - cx) < bestD) { bestD = Math.abs(e - cx); slot = f }
    })
    if (slot !== null) taken.add(slot)
    // radiopaque crown on the occlusal side of the fixture
    const bw = c.maxX - c.minX + 1
    const bh = c.maxY - c.minY + 1
    const wy0 = upper ? c.maxY : Math.max(0, c.minY - Math.round(bh * 0.9))
    const wy1 = upper ? Math.min(h - 1, c.maxY + Math.round(bh * 0.9)) : c.minY
    let n = 0, hit = 0
    for (let y = wy0; y <= wy1; y++) {
      for (let x = Math.max(0, c.minX - 3); x <= Math.min(w - 1, c.maxX + 3); x++) {
        n++
        const i = y * w + x
        if ((prr[i] > P || fil[i] > P) && input[i] >= bright) hit++
      }
    }
    implants.push({
      bbox: [c.minX / w, c.minY / h, bw / w, bh / h],
      slot,
      crowned: n > 0 && hit / n >= 0.3,
    })
  }
  return { teeth: outTeeth, implants }
}

const round = (x: number) => Math.round(x * 100) / 100

/** One-line plain description of what was found on a tooth ("" when nothing). */
export function describeRestorations(r?: ToothRestorations): string[] {
  if (!r) return []
  const out: string[] = []
  if (r.implant) out.push('Implant')
  if (r.crown?.kind === 'cap') out.push(r.bridgeId ? 'Crown, joined to a neighbour (bridge)' : 'Crown or cap')
  if (r.crown?.kind === 'filling') out.push('Filling')
  if (r.rootCanal) out.push('Root canal filling')
  return out
}
