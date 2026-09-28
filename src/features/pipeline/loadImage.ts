import { PipelineError } from './errors'

export interface LoadedImage {
  url: string
  width: number
  height: number
  element: HTMLImageElement
  fileName?: string
  fileSizeBytes?: number
  /** Object URLs we created and must revoke when the image is discarded. */
  revocable: boolean
}

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
export const MAX_FILE_BYTES = 25 * 1024 * 1024
const MIN_WIDTH = 600
const MIN_HEIGHT = 250
export const PANORAMIC_MIN_ASPECT = 1.4

function decode(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.decoding = 'async'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('decode'))
    img.src = url
  })
}

export async function loadImageFromUrl(url: string, fileName?: string): Promise<LoadedImage> {
  const element = await decode(url)
  return { url, element, width: element.naturalWidth, height: element.naturalHeight, fileName, revocable: false }
}

/** Validates and decodes a user file. Nothing leaves the browser. */
export async function loadImageFromFile(file: File, opts: { allowNonPanoramic?: boolean } = {}): Promise<LoadedImage> {
  if (!ACCEPTED.includes(file.type)) {
    throw new PipelineError(
      'unsupported-type',
      `${file.name} isn't a supported image type.`,
      'Upload a JPEG, PNG, WebP or BMP export of the panoramic X-ray. DICOM support is planned.',
    )
  }
  if (file.size > MAX_FILE_BYTES) {
    throw new PipelineError(
      'too-large',
      `This file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is 25 MB.`,
      'Export the OPG at a lower resolution (around 3000 px wide is plenty) and try again.',
    )
  }
  const url = URL.createObjectURL(file)
  let element: HTMLImageElement
  try {
    element = await decode(url)
  } catch {
    URL.revokeObjectURL(url)
    throw new PipelineError('decode-failed', "This image couldn't be opened.", 'The file may be damaged. Re-export it and try again.')
  }
  const width = element.naturalWidth
  const height = element.naturalHeight
  if (width < MIN_WIDTH || height < MIN_HEIGHT) {
    URL.revokeObjectURL(url)
    throw new PipelineError(
      'too-small',
      `This image is only ${width}×${height} px.`,
      `Use an image at least ${MIN_WIDTH} px wide so individual teeth are visible.`,
    )
  }
  if (!opts.allowNonPanoramic && width / height < PANORAMIC_MIN_ASPECT) {
    URL.revokeObjectURL(url)
    throw new PipelineError(
      'not-panoramic',
      `This image is ${width}×${height} px, which doesn't look like a panoramic X-ray.`,
      'OPGs are usually about twice as wide as they are tall. Periapical and bitewing images are not supported yet.',
      true,
    )
  }
  return { url, element, width, height, fileName: file.name, fileSizeBytes: file.size, revocable: true }
}
