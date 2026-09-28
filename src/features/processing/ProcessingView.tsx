import { useEffect, useRef, useState } from 'react'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult } from '../../types/inference'
import { OpgImageView } from '../imaging/OpgImageView'
import { toPipelineError } from '../pipeline/errors'
import { Wordmark } from '../../components/Wordmark'

// ms at which each of the six stages completes, and when we hand over to the workspace
const STEP_DONE = [650, 1850, 2500, 3550, 4150, 4700]
const HANDOFF = 5250

export function ProcessingView() {
  const image = useMouthTwin((s) => s.image)
  const provider = useMouthTwin((s) => s.provider)
  const finish = useMouthTwin((s) => s.finishProcessing)
  const fail = useMouthTwin((s) => s.fail)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef(0)
  const resultRef = useRef<InferenceResult | null>(null)
  resultRef.current = result

  // run inference in parallel with the animation
  useEffect(() => {
    if (!image || !provider) return
    let cancelled = false
    provider
      .run(image)
      .then((r) => !cancelled && setResult(r))
      .catch((e) => !cancelled && fail(toPipelineError(e)))
    return () => {
      cancelled = true
    }
  }, [image, provider, fail])

  // animation clock; pauses after the first stage until inference has returned
  useEffect(() => {
    startRef.current = performance.now()
    let raf = 0
    let pausedAt: number | null = null
    const tick = (now: number) => {
      let t = now - startRef.current
      const ready = resultRef.current !== null
      if (!ready && t > STEP_DONE[0]) {
        pausedAt ??= now
        t = STEP_DONE[0]
      } else if (pausedAt !== null) {
        startRef.current += now - pausedAt
        pausedAt = null
        t = now - startRef.current
      }
      setElapsed(t)
      if (t < HANDOFF) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  useEffect(() => {
    if (result && elapsed >= HANDOFF) finish(result)
  }, [elapsed, result, finish])

  if (!image || !provider) return null
  const stepIndex = STEP_DONE.findIndex((ms) => elapsed < ms)
  const active = stepIndex === -1 ? STEP_DONE.length : stepIndex
  const ready = active >= STEP_DONE.length
  const shown: InferenceResult | null = result

  return (
    <div className="flex h-full flex-col bg-ground">
      <header className="flex items-center justify-between px-5 py-4 sm:px-8">
        <Wordmark />
        <span className="label">{provider.label}</span>
      </header>

      <main className="grid min-h-0 flex-1 grid-rows-[1fr_auto] gap-6 px-4 pb-6 sm:px-8 lg:grid-cols-[1fr_300px] lg:grid-rows-1 lg:gap-10">
        <section className="relative min-h-0">
          <div className="relative h-full">
            {shown ? (
              <OpgImageView
                url={image.url}
                result={shown}
                sweep={1.1}
                dimImage={active >= 4}
                scanning={active === 1}
                overlay={{ boxes: active >= 1 && active < 4, labels: active >= 2, outlines: active >= 3, curve: active >= 4 }}
              />
            ) : (
              <div className="relative h-full" style={{ containerType: 'size' }}>
                <img
                  src={image.url}
                  alt="Panoramic dental X-ray"
                  className="absolute inset-0 m-auto max-h-full max-w-full object-contain"
                />
              </div>
            )}
          </div>
        </section>

        <aside className="flex flex-col justify-center gap-5">
          <div>
            <p className="label">{ready ? 'Complete' : 'Analyzing image'}</p>
            <h2 className="mt-2 text-[22px] font-light tracking-tight text-ink">
              {ready ? 'MouthTwin ready' : 'Building your oral model…'}
            </h2>
          </div>
          <ol className="flex flex-col gap-2.5">
            {provider.stages.map((s, i) => {
              const done = i < active
              const current = i === active
              return (
                <li key={s.id} className="flex items-start gap-3" style={{ opacity: done || current ? 1 : 0.35, transition: 'opacity .4s' }}>
                  <span
                    className={`mt-[3px] grid h-4 w-4 shrink-0 place-items-center rounded-full border text-[10px] transition-colors ${
                      done ? 'border-accent bg-accent text-ground' : current ? 'border-accent text-accent' : 'border-line-strong text-faint'
                    }`}
                    aria-hidden
                  >
                    {done ? '✓' : current ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" /> : ''}
                  </span>
                  <span className="min-w-0">
                    <span className="block text-[13.5px] text-ink">{s.label}</span>
                    <span className="block font-mono text-[11px] leading-snug text-faint">{s.method}</span>
                  </span>
                </li>
              )
            })}
          </ol>
          <div className="h-px w-full overflow-hidden bg-line">
            <div
              className="h-full bg-accent transition-[width] duration-200"
              style={{ width: `${Math.min(100, (elapsed / STEP_DONE[STEP_DONE.length - 1]) * 100)}%` }}
            />
          </div>
          <p className="font-mono text-[11px] text-faint tabular">
            OPG · {image.width}×{image.height}px{shown ? ` · ${shown.teeth.length} teeth mapped` : ''}
          </p>
        </aside>
      </main>
    </div>
  )
}
