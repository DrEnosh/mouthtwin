/**
 * In-browser tooth segmentation (ONNX Runtime Web, WebAssembly).
 *
 * Model: U-Net (MobileNetV2 encoder) trained on DENTEX (Hamamci et al., MICCAI 2023, CC BY-NC-SA 4.0).
 * Input: 1×1×384×768 greyscale 0..1. Outputs: per-pixel label (0 = background, 1..32 = FDI class),
 * tooth probability and label confidence. The image never leaves the browser.
 */
import type * as OrtNs from 'onnxruntime-web/wasm'
import { PipelineError } from '../errors'
import type { LoadedImage } from '../loadImage'

export const MODEL_H = 384
export const MODEL_W = 768
const BASE = import.meta.env?.BASE_URL ?? '/'
export const MODEL_URL = `${BASE}models/mouthtwin-teeth.onnx`

export interface SegOutput {
  width: number
  height: number
  labels: Uint8Array
  fg: Float32Array
  conf: Float32Array
}

let sessionPromise: Promise<{ ort: typeof OrtNs; session: OrtNs.InferenceSession }> | null = null

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-web/wasm')
      // resolve against the page, not the script (bundled scripts live in assets/)
      ort.env.wasm.wasmPaths = new URL(`${BASE}ort/`, document.baseURI).href
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
      const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
      return { ort, session }
    })().catch((e) => {
      sessionPromise = null
      throw e
    })
  }
  return sessionPromise
}

/** Start downloading the model early (e.g. when the landing page opens). */
export function preloadModel() {
  getSession().catch(() => undefined)
}

export async function segmentTeeth(image: LoadedImage): Promise<SegOutput> {
  let ort: typeof OrtNs
  let session: OrtNs.InferenceSession
  try {
    ;({ ort, session } = await getSession())
  } catch {
    throw new PipelineError('model-unavailable', "The tooth model couldn't be loaded.", 'Check your connection and reload the page.')
  }
  const canvas = document.createElement('canvas')
  canvas.width = MODEL_W
  canvas.height = MODEL_H
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D
  ctx.drawImage(image.element, 0, 0, MODEL_W, MODEL_H)
  const rgba = ctx.getImageData(0, 0, MODEL_W, MODEL_H).data
  const input = new Float32Array(MODEL_W * MODEL_H)
  for (let i = 0, j = 0; i < input.length; i++, j += 4) {
    input[i] = (0.2126 * rgba[j] + 0.7152 * rgba[j + 1] + 0.0722 * rgba[j + 2]) / 255
  }
  const feeds = { image: new ort.Tensor('float32', input, [1, 1, MODEL_H, MODEL_W]) }
  const out = await session.run(feeds)
  return {
    width: MODEL_W,
    height: MODEL_H,
    labels: out.labels.data as Uint8Array,
    fg: out.fg.data as Float32Array,
    conf: out.conf.data as Float32Array,
  }
}

/** Class index 1..32 ↔ FDI number. */
export function classToFdi(c: number): number {
  const q = Math.floor((c - 1) / 8) + 1
  const t = ((c - 1) % 8) + 1
  return q * 10 + t
}
