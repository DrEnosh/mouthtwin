export type PipelineErrorCode =
  | 'unsupported-type'
  | 'too-large'
  | 'decode-failed'
  | 'too-small'
  | 'not-panoramic'
  | 'low-contrast'
  | 'no-teeth'
  | 'model-unavailable'
  | 'processing-failed'

export class PipelineError extends Error {
  code: PipelineErrorCode
  hint: string
  /** True when the user may choose to continue anyway (a soft check, not a hard failure). */
  overridable: boolean

  constructor(code: PipelineErrorCode, message: string, hint: string, overridable = false) {
    super(message)
    this.name = 'PipelineError'
    this.code = code
    this.hint = hint
    this.overridable = overridable
  }
}

export function toPipelineError(err: unknown): PipelineError {
  if (err instanceof PipelineError) return err
  return new PipelineError(
    'processing-failed',
    "We couldn't confidently process this image.",
    'Try uploading a clearer panoramic dental image.',
  )
}
