import demoJson from '../../data/demo/demo-inference.json'
import demoImageUrl from '../../data/demo/dentex-heldout-demo.jpg'
import { parseInferenceResult, type InferenceResult, type InferenceSource, type PipelineStageId, type StageReport } from '../../types/inference'
import { PipelineError } from './errors'
import type { LoadedImage } from './loadImage'
import { fitArchTemplate } from './stages/archTemplate'
import { estimateOcclusalPlane } from './stages/occlusalPlane'
import { preprocess } from './stages/preprocess'
import { detectionsFromSegmentation, occlusalCurveFrom } from './model/postprocess'
import { segmentTeeth, type SegOutput } from './model/segModel'
import { analyseRestorations } from './model/restorations'

export interface ProviderStage {
  id: PipelineStageId
  /** Shown in the processing animation. */
  label: string
  /** Shown in the clinician panel: how this stage is actually produced. */
  method: string
}

export interface InferenceProvider {
  source: InferenceSource
  label: string
  stages: ProviderStage[]
  run(image: LoadedImage): Promise<InferenceResult>
}

const report = (stages: ProviderStage[]): StageReport[] => stages.map((s) => ({ ...s, status: 'done' }))

export const DEMO_IMAGE_URL = demoImageUrl

/** Returns precomputed geometry for the bundled synthetic OPG. */
export const demoProvider: InferenceProvider = {
  source: 'precomputed-demo',
  label: 'Precomputed demo inference',
  stages: [
    { id: 'preprocess', label: 'Reading panoramic image', method: 'Bundled synthetic OPG, 1600×800' },
    { id: 'detect', label: 'Detecting teeth', method: 'Precomputed from the generator geometry (no model)' },
    { id: 'number', label: 'Assigning FDI numbers', method: 'Precomputed from the generator geometry' },
    { id: 'segment', label: 'Segmenting tooth outlines', method: 'Precomputed polygons from the generator' },
    { id: 'map', label: 'Mapping teeth to the arch', method: 'Panoramic position → average arch template' },
    { id: 'visualize', label: 'Building interactive model', method: '3D relief extruded from the image pixels' },
  ],
  async run() {
    const parsed = parseInferenceResult(demoJson)
    return { ...parsed, source: 'precomputed-demo', stages: report(this.stages) }
  },
}

/** Classical CV + arch template. Runs in the browser; the image is never uploaded. */
export const heuristicProvider: InferenceProvider = {
  source: 'heuristic-template',
  label: 'Template mapping (no trained model)',
  stages: [
    { id: 'preprocess', label: 'Normalising image', method: 'Downscale to 1024 px, luminance, contrast check' },
    { id: 'detect', label: 'Estimating the occlusal plane', method: 'Row-intensity minimum in 7 strips + quadratic fit' },
    { id: 'number', label: 'Fitting 32-tooth template', method: 'Average crown widths along the fitted curve' },
    { id: 'segment', label: 'Drawing template outlines', method: 'Type-specific template silhouettes (not segmentation)' },
    { id: 'map', label: 'Mapping teeth to the arch', method: 'Panoramic position → average arch template' },
    { id: 'visualize', label: 'Building interactive model', method: '3D relief extruded from the image pixels' },
  ],
  async run(image) {
    const gray = preprocess(image)
    const occ = estimateOcclusalPlane(gray)
    const teeth = fitArchTemplate(occ.curve, image.width, image.height)
    const stages = report(this.stages)
    if (occ.usedFallback) {
      stages[1] = { ...stages[1], method: 'Curve fit was unstable, fell back to a flat plane at the darkest row' }
    }
    return {
      schemaVersion: '1.0',
      source: 'heuristic-template',
      notes: 'Positions come from an average adult template. Assumes a complete dentition. Missing teeth are not detected.',
      image: { width: image.width, height: image.height, modality: 'OPG', fileName: image.fileName, fileSizeBytes: image.fileSizeBytes },
      occlusalCurve: occ.curve,
      teeth,
      stages,
    }
  },
}

/** Sends the image to your own inference server (see server/). Enabled by VITE_INFERENCE_URL. */
export function remoteProvider(baseUrl: string): InferenceProvider {
  const stages: ProviderStage[] = [
    { id: 'preprocess', label: 'Uploading to inference server', method: `POST ${baseUrl}/infer` },
    { id: 'detect', label: 'Detecting teeth', method: 'Model output' },
    { id: 'number', label: 'Assigning FDI numbers', method: 'Model output' },
    { id: 'segment', label: 'Segmenting tooth outlines', method: 'Model output' },
    { id: 'map', label: 'Mapping teeth to the arch', method: 'Panoramic position → average arch template' },
    { id: 'visualize', label: 'Building interactive model', method: '3D relief extruded from the image pixels' },
  ]
  return {
    source: 'model',
    label: 'Remote model',
    stages,
    async run(image) {
      const blob = await fetch(image.url).then((r) => r.blob())
      const body = new FormData()
      body.append('image', blob, image.fileName ?? 'opg.png')
      let res: Response
      try {
        res = await fetch(`${baseUrl}/infer`, { method: 'POST', body })
      } catch {
        throw new PipelineError('model-unavailable', "The inference server isn't reachable.", `Check that it is running at ${baseUrl}.`)
      }
      if (!res.ok) throw new PipelineError('model-unavailable', `The inference server returned ${res.status}.`, 'Check the server logs.')
      const parsed = parseInferenceResult(await res.json())
      if (parsed.teeth.length === 0) {
        throw new PipelineError('no-teeth', 'No teeth were detected in this image.', 'Try uploading a clearer panoramic dental image.')
      }
      return { ...parsed, source: 'model', stages: parsed.stages ?? stages.map((s) => ({ ...s, status: 'done' })) }
    },
  }
}

/**
 * Trained tooth model running in the browser (ONNX Runtime Web). Falls back to template mapping
 * if the model can't be loaded (for example in a sandbox that blocks WebAssembly downloads).
 */
export const modelProvider: InferenceProvider = {
  source: 'model',
  label: 'MouthTwin tooth model (in your browser)',
  stages: [
    { id: 'preprocess', label: 'Normalising image', method: 'Resize to 768×384, greyscale' },
    { id: 'detect', label: 'Detecting teeth', method: 'U-Net (MobileNetV2) trained on DENTEX + Cluj OPGs' },
    { id: 'number', label: 'Assigning FDI numbers', method: 'Per-pixel FDI class (32 teeth), largest region per tooth' },
    { id: 'segment', label: 'Segmenting tooth outlines', method: 'Model mask, traced outline' },
    { id: 'map', label: 'Finding crowns, root canals, implants', method: 'Restoration maps of the same model + radiopacity inside each tooth outline' },
    { id: 'visualize', label: 'Building interactive model', method: 'Real CBCT anatomy (ToothFairy2) fitted tooth by tooth, restorations drawn where found' },
  ],
  async run(image) {
    let seg: SegOutput
    try {
      seg = await segmentTeeth(image)
    } catch (e) {
      if (e instanceof PipelineError && e.code === 'model-unavailable') {
        const r = await heuristicProvider.run(image)
        return { ...r, notes: 'The tooth model could not be loaded here, so template mapping was used instead.' }
      }
      throw e
    }
    const found = detectionsFromSegmentation(seg, image.width / image.height / (seg.width / seg.height))
    const { teeth, implants, pontics } = analyseRestorations(seg, found)
    if (teeth.length < 4) {
      throw new PipelineError(
        'no-teeth',
        "We couldn't confidently process this image.",
        'The model found almost no teeth. Try uploading a clearer panoramic dental image.',
      )
    }
    return {
      schemaVersion: '1.0',
      source: 'model',
      notes: 'Tooth outlines and FDI numbers predicted by a model trained on the DENTEX dataset (CC BY-NC-SA 4.0).',
      image: { width: image.width, height: image.height, modality: 'OPG', fileName: image.fileName, fileSizeBytes: image.fileSizeBytes },
      occlusalCurve: occlusalCurveFrom(teeth) ?? estimateOcclusalPlane(preprocess(image)).curve,
      teeth,
      implants: seg.rest ? implants : undefined,
      pontics: seg.rest ? pontics : undefined,
      stages: report(this.stages),
      segmentation: { width: seg.width, height: seg.height, labels: seg.labels, fg: seg.fg, rest: seg.rest },
    }
  },
}

export function uploadProvider(): InferenceProvider {
  const url = import.meta.env.VITE_INFERENCE_URL as string | undefined
  return url ? remoteProvider(url.replace(/\/$/, '')) : modelProvider
}

export const IMAGES_LEAVE_BROWSER = Boolean(import.meta.env.VITE_INFERENCE_URL)
