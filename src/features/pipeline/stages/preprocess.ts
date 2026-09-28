import { PipelineError } from '../errors'
import type { LoadedImage } from '../loadImage'

export interface GrayImage {
  data: Float32Array // luminance 0..1, row-major
  width: number
  height: number
  mean: number
  std: number
}

/** Downscale to a working resolution and convert to luminance. Runs entirely in the browser. */
export function preprocess(img: LoadedImage, maxWidth = 1024): GrayImage {
  const scale = Math.min(1, maxWidth / img.width)
  const width = Math.max(1, Math.round(img.width * scale))
  const height = Math.max(1, Math.round(img.height * scale))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new PipelineError('processing-failed', 'Your browser could not create a drawing surface.', 'Try a different browser.')
  ctx.drawImage(img.element, 0, 0, width, height)
  const rgba = ctx.getImageData(0, 0, width, height).data
  const data = new Float32Array(width * height)
  let sum = 0
  for (let i = 0, j = 0; i < data.length; i++, j += 4) {
    const v = (0.2126 * rgba[j] + 0.7152 * rgba[j + 1] + 0.0722 * rgba[j + 2]) / 255
    data[i] = v
    sum += v
  }
  const mean = sum / data.length
  let sq = 0
  for (let i = 0; i < data.length; i++) sq += (data[i] - mean) ** 2
  const std = Math.sqrt(sq / data.length)

  if (std < 0.035) {
    throw new PipelineError(
      'low-contrast',
      "We couldn't confidently process this image.",
      'It has almost no contrast. Try uploading a clearer panoramic dental image.',
    )
  }
  return { data, width, height, mean, std }
}
