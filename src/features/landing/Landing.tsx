import { lazy, Suspense, useCallback, useRef, useState } from 'react'
import { Wordmark } from '../../components/Wordmark'
import { DISCLAIMER } from '../../components/Disclaimer'
import { useMouthTwin } from '../../store/useMouthTwin'
import { loadImageFromFile, loadImageFromUrl } from '../pipeline/loadImage'
import { toPipelineError } from '../pipeline/errors'
import { DEMO_IMAGE_URL, demoProvider, IMAGES_LEAVE_BROWSER, uploadProvider } from '../pipeline/providers'

const ArchPreview = lazy(() => import('../visualization/ArchPreview').then((m) => ({ default: m.ArchPreview })))

const PIPELINE = ['Input', 'OPG analysis', 'Tooth detection', 'Segmentation', 'Anatomical mapping', 'Digital twin']

export function Landing() {
  const begin = useMouthTwin((s) => s.beginProcessing)
  const fail = useMouthTwin((s) => s.fail)
  const error = useMouthTwin((s) => s.error)
  const pendingFile = useMouthTwin((s) => s.pendingFile)
  const clearError = useMouthTwin((s) => s.clearError)
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)

  const runDemo = useCallback(async () => {
    setBusy(true)
    try {
      const img = await loadImageFromUrl(DEMO_IMAGE_URL, 'synthetic-demo-opg.jpg')
      begin(img, demoProvider)
    } catch (e) {
      fail(toPipelineError(e))
    } finally {
      setBusy(false)
    }
  }, [begin, fail])

  const runFile = useCallback(
    async (file: File, force = false) => {
      setBusy(true)
      clearError()
      try {
        const img = await loadImageFromFile(file, { allowNonPanoramic: force })
        begin(img, uploadProvider())
      } catch (e) {
        const err = toPipelineError(e)
        fail(err, err.overridable ? file : null)
      } finally {
        setBusy(false)
      }
    },
    [begin, fail, clearError],
  )

  return (
    <div
      className="relative flex h-full flex-col overflow-y-auto overflow-x-hidden bg-ground"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files?.[0]
        if (f) runFile(f)
      }}
    >
      {/* backdrop */}
      <div className="pointer-events-none absolute inset-0">
        <div
          className="absolute inset-0"
          style={{ background: 'radial-gradient(60% 55% at 70% 45%, rgb(134 197 216 / 0.07), transparent 70%)' }}
        />
        <div className="absolute inset-y-0 right-[-10%] w-[80%] opacity-90 max-lg:inset-x-0 max-lg:right-0 max-lg:w-full max-lg:opacity-25">
          <Suspense fallback={null}>
            <ArchPreview />
          </Suspense>
        </div>
      </div>

      <header className="relative z-10 flex items-center justify-between px-5 py-5 sm:px-10">
        <Wordmark />
        <span className="rounded-full border border-caution/40 px-2.5 py-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-caution">
          Experimental prototype
        </span>
      </header>

      <main className="relative z-10 flex flex-1 items-center px-5 sm:px-10">
        <div className="max-w-[600px] py-10">
          <p className="label fade-up" style={{ animationDelay: '60ms' }}>
            Panoramic X-ray → interactive arch
          </p>
          <h1
            className="fade-up mt-4 text-[44px] font-light leading-[1.02] tracking-[-0.025em] text-ink sm:text-[64px]"
            style={{ textWrap: 'balance', animationDelay: '120ms' }}
          >
            Meet your mouth’s digital twin.
          </h1>
          <p className="fade-up mt-5 max-w-[46ch] text-[16px] leading-relaxed text-muted" style={{ animationDelay: '200ms' }}>
            Upload a dental panoramic X-ray. MouthTwin numbers each tooth, then turns the X-ray itself into a 3D arch you
            can turn, select and explore.
          </p>

          <div className="fade-up mt-8 flex flex-wrap items-center gap-3" style={{ animationDelay: '280ms' }}>
            <button
              onClick={runDemo}
              disabled={busy}
              className="group inline-flex items-center gap-2.5 rounded-md bg-accent px-5 py-3 text-[14px] font-medium text-[#061015] transition hover:bg-[#a3d6e5] disabled:opacity-60"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
                <path d="M2.5 1.5v9l8-4.5z" fill="currentColor" />
              </svg>
              Run MouthTwin
            </button>
            <button
              onClick={() => fileInput.current?.click()}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-md border border-line-strong bg-panel/60 px-5 py-3 text-[14px] text-ink backdrop-blur transition hover:border-accent/60"
            >
              Upload OPG
            </button>
            <input
              ref={fileInput}
              id="opg-upload"
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) runFile(f)
                e.target.value = ''
              }}
            />
          </div>
          <p className="fade-up mt-4 text-[12.5px] text-faint" style={{ animationDelay: '340ms' }}>
            The demo uses a synthetic panoramic image. No patient data.{' '}
            {IMAGES_LEAVE_BROWSER
              ? 'Uploads are sent to your configured inference server.'
              : 'Uploaded images are processed in your browser and are not uploaded or retained.'}
          </p>

          {error && (
            <div role="alert" className="fade-up mt-6 max-w-[520px] rounded-lg border border-caution/35 bg-[#16130d]/90 p-4 backdrop-blur">
              <p className="text-[14px] text-ink">{error.message}</p>
              <p className="mt-1 text-[13px] text-muted">{error.hint}</p>
              <div className="mt-3 flex gap-2">
                {error.overridable && pendingFile && (
                  <button
                    onClick={() => runFile(pendingFile, true)}
                    className="rounded border border-line-strong px-3 py-1.5 text-[12.5px] text-ink hover:border-accent/60"
                  >
                    Process anyway
                  </button>
                )}
                <button onClick={clearError} className="rounded px-3 py-1.5 text-[12.5px] text-muted hover:text-ink">
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="relative z-10 flex flex-col gap-3 border-t border-line px-5 py-4 sm:px-10 lg:flex-row lg:items-center lg:justify-between">
        <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[10.5px] uppercase tracking-[0.1em] text-faint">
          {PIPELINE.map((p, i) => (
            <li key={p} className="flex items-center gap-2">
              {i > 0 && <span className="text-line-strong">→</span>}
              <span className={i === PIPELINE.length - 1 ? 'text-accent' : undefined}>{p}</span>
            </li>
          ))}
        </ol>
        <p className="max-w-[70ch] text-[11.5px] leading-snug text-faint">{DISCLAIMER.short}</p>
      </footer>

      {dragging && (
        <div className="pointer-events-none absolute inset-3 z-20 grid place-items-center rounded-xl border border-dashed border-accent/70 bg-ground/80">
          <p className="text-[15px] text-ink">Drop a panoramic X-ray to build its MouthTwin</p>
        </div>
      )}
    </div>
  )
}
