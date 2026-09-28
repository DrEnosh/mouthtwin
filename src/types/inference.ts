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
  stages: StageReport[]
  /** Per-pixel model output (runtime only, not part of the JSON contract). Labels are classes 0..32. */
  segmentation?: {
    width: number
    height: number
    labels: Uint8Array
    fg: Float32Array
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
