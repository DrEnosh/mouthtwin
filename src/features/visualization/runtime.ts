/**
 * Small mutable singletons shared between the render loop and the DOM.
 * Updated every frame, so they live outside React state on purpose.
 */
import * as THREE from 'three'

/** Progress of the OPG → arch transformation. */
export const reveal = {
  startedAt: 0,
  /** 0..1: teeth materialise over the flat image. */
  appear: 1,
  /** 0..1: the flat panoramic layout bends into the arch. */
  morph: 1,
  running: false,
}

export const REVEAL_TIMING = { hold: 0.35, appear: 0.8, morph: 2.6 }

const easeInOutCubic = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const clamp01 = (x: number) => Math.min(1, Math.max(0, x))

export function startReveal(now: number) {
  reveal.startedAt = now
  reveal.appear = 0
  reveal.morph = 0
  reveal.running = true
}

export function tickReveal(now: number) {
  if (!reveal.running) return
  const t = (now - reveal.startedAt) / 1000
  const { hold, appear, morph } = REVEAL_TIMING
  reveal.appear = easeInOutCubic(clamp01((t - hold) / appear))
  reveal.morph = easeInOutCubic(clamp01((t - hold - appear * 0.8) / morph))
  if (reveal.morph >= 1) reveal.running = false
}

export function skipReveal() {
  reveal.appear = 1
  reveal.morph = 1
  reveal.running = false
}

/** Inner (mirrored) group of every tooth, for camera targeting and screen projection. */
export const toothObjects = new Map<number, { object: THREE.Object3D; localCenter: THREE.Vector3 }>()

/** Per-tooth anchors on the image-derived relief (arch state). Preferred over template teeth. */
export const toothAnchors = new Map<number, { center: THREE.Vector3; normal: THREE.Vector3; label: THREE.Vector3; upper: boolean }>()

export function toothWorldCenter(fdi: number, target = new THREE.Vector3()): THREE.Vector3 | null {
  const anchor = toothAnchors.get(fdi)
  if (anchor) return target.copy(anchor.center)
  const entry = toothObjects.get(fdi)
  if (!entry) return null
  entry.object.updateWorldMatrix(true, false)
  return entry.object.localToWorld(target.copy(entry.localCenter))
}

/** Screen-space anchors for the split-view link line (page pixels). */
export const linkAnchors = {
  model: { x: 0, y: 0, visible: false },
  image: { x: 0, y: 0, visible: false },
}

// Handy for recording demos from the console, e.g. __mouthtwin.freeze(0.5)
if (typeof window !== 'undefined') {
  ;(window as unknown as { __mouthtwin: unknown }).__mouthtwin = {
    reveal,
    freeze(morph: number) {
      reveal.running = false
      reveal.appear = 1
      reveal.morph = morph
    },
  }
}
