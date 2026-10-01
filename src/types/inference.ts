/**
 * The contract between any inference provider (precomputed demo, in-browser heuristic,
 * or a real model behind an API) and the visualisation layer.
 *
 * All image coordinates are normalised to 0..1 (x across the image width, y down the height)
 * so results stay valid if the image is displayed at any size.
 */

export type NormPoint = [number, number]

/** Where a result came from. The UI always shows this — never implies more than it is. */
export type InferenceSource = 'precomputed-demo' | 'heuristic-template' | 'model'

export interface ToothDetection {
  /** FDI two-digit notation, e.g. 36 = lower left first molar. */
  fdi: number
  /** [x, y, width, height], normalised. */
  bbox: [number, number, number, number]
  /** Outline polygon, normalised. Empty if the provider does not segment. */
  polygon: NormPoint[]
  /** Centre of the occlusal / incisal edge. Anchor used to map the tooth onto the 3D arch. */
  occlusal: NormPoint
  /** Cemento-enamel junction centre, if known. */
  cej?: NormPoint
  /** Root apex, if known. */
  apex?: NormPoint
  /** Model score in 0..1. Only a trained model may set this. Heuristics and demo data leave it null. */
  score?: number | null
  /** Treatments the model sees on this tooth. Absent when the provider does not look for them. */
  restorations?: ToothRestorations
  /** How this tooth sits compared with the same tooth on a typical OPG. Only from the trained model. */
  pose?: ToothPose
}

/**
 * Tooth position relative to a typical OPG (population medians from 634 clinician-labelled DENTEX OPGs).
 * Deviations already have measurement noise removed; 0 / 1 means "as usual".
 */
export interface ToothPose {
  /** crown end and root apex along the tooth's long axis, normalised image coords */
  crown: NormPoint
  apex: NormPoint
  /** long-axis angle (rad) on the image, crown→apex, from the jaw's usual apex direction; + = apex toward image right */
  theta: number
  /** extra tilt (rad) compared with the same tooth on a typical OPG, same sign as theta */
  tilt: number
  /** length compared with typical (1 = typical) */
  length: number
  /** how much deeper (+) or higher (−) the crown sits than usual, as a fraction of the tooth's length */
  depth: number
  /** only a crown-shaped region is visible (no root length), e.g. a developing or unerupted tooth or a pontic */
  crownOnly: boolean
  /** third molar clearly shorter than a formed one: root probably still developing */
  partialRoot?: boolean
  /** how unusual each measure is, in robust standard deviations */
  z: { tilt: number; length: number; depth: number }
}

/**
 * What the X-ray shows on a tooth. These come from the restoration heads of the tooth model plus a radiopacity
 * check inside the tooth outline. They describe the picture, not the material or the quality of the work.
 */
export interface ToothRestorations {
  /**
   * Bright (radiopaque) restoration on the crown. `cap`: covers most of the crown (crown / cap / large restoration).
   * `filling`: covers part of it. `coverage` is the fraction of the crown area, 0..1.
   */
  crown?: { kind: 'cap' | 'filling'; coverage: number }
  /** Root-canal filling seen in the root. `coverage` is the fraction of the root area flagged. */
  rootCanal?: { coverage: number }
  /** The tooth carries an implant fixture (the tooth outline is an implant crown). */
  implant?: boolean
  /** Same id on teeth whose crowns are joined into one radiopaque block (bridge or splinted crowns). */
  bridgeId?: number
}

/** An implant fixture with no tooth outline of its own. `slot` is the FDI position it most likely replaces. */
export interface ImplantDetection {
  /** [x, y, width, height], normalised. */
  bbox: [number, number, number, number]
  /** FDI position of the missing tooth it stands in (estimated from the neighbouring teeth), or null. */
  slot: number | null
  /** Radiopaque crown seen on top of it. */
  crowned: boolean
}

export interface ImageMeta {
  width: number
  height: number
  modality: 'OPG'
  synthetic?: boolean
  fileName?: string
  fileSizeBytes?: number
}

export type StageStatus = 'done' | 'skipped' | 'failed'

export interface StageReport {
  id: PipelineStageId
  label: string
  /** Plain description of how this stage was actually produced. */
  method: string
  status: StageStatus
}

export type PipelineStageId =
  | 'preprocess'
  | 'detect'
  | 'number'
  | 'segment'
  | 'map'
  | 'visualize'

export interface InferenceResult {
  schemaVersion: '1.0'
  source: InferenceSource
  notes?: string
  image: ImageMeta
  /** Occlusal plane as y = a·x² + b·x + c in normalised coordinates. */
  occlusalCurve?: { a: number; b: number; c: number }
  teeth: ToothDetection[]
  /** Implants the model sees where no tooth was outlined. */
  implants?: ImplantDetection[]
  stages: StageReport[]
  /** Per-pixel model output (runtime only, not part of the JSON contract). Labels are classes 0..32. */
  segmentation?: {
    width: number
    height: number
    labels: Uint8Array
    fg: Float32Array
    /** Restoration probabilities (0..255) as 4 planes: implant, prosthetic restoration, filling, root-canal treatment. */
    rest?: Uint8Array
  }
}

/** Runtime guard for JSON coming from files or a remote model. */
export function parseInferenceResult(raw: unknown): Omit<InferenceResult, 'stages'> & { stages?: StageReport[] } {
  if (!raw || typeof raw !== 'object') throw new Error('Inference result is not an object')
  const r = raw as Record<string, unknown>
  if (r.schemaVersion !== '1.0') throw new Error(`Unsupported schema version: ${String(r.schemaVersion)}`)
  if (!Array.isArray(r.teeth)) throw new Error('Inference result has no teeth array')
  for (const t of r.teeth as unknown[]) {
    const tooth = t as Partial<ToothDetection>
    if (typeof tooth.fdi !== 'number' || !Array.isArray(tooth.bbox) || !Array.isArray(tooth.occlusal)) {
      throw new Error('Malformed tooth entry in inference result')
    }
  }
  return raw as InferenceResult
}
