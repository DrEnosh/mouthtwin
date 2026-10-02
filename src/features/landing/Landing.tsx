import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Wordmark } from '../../components/Wordmark'
import { ConsentDialog, DISCLAIMER, hasAcknowledged } from '../../components/Disclaimer'
import { toothRef } from '../../data/fdi'
import { useMouthTwin } from '../../store/useMouthTwin'
import { loadImageFromFile, loadImageFromUrl } from '../pipeline/loadImage'
import { toPipelineError } from '../pipeline/errors'
import { DEMO_IMAGE_URL, IMAGES_LEAVE_BROWSER, modelProvider, uploadProvider } from '../pipeline/providers'
import { preloadModel } from '../pipeline/model/segModel'

const ArchPreview = lazy(() => import('../visualization/ArchPreview').then((m) => ({ default: m.ArchPreview })))
const BASE = import.meta.env.BASE_URL

const STEPS = [
  {
    img: 'landing/step-opg.webp',
    title: 'Upload a panoramic X-ray',
    body: 'JPEG, PNG or WebP. The image is read in your browser and never leaves your device.',
  },
  {
    img: 'landing/step-teeth.webp',
    title: 'Every tooth is outlined and numbered',
    body: 'A segmentation model finds each tooth, gives it an FDI number and flags crowns, bridges, fillings, root canals and implants.',
  },
  {
    img: 'landing/step-3d.webp',
    title: 'Explore it as a 3D arch',
    body: 'Reference anatomy is adjusted to the X-ray: missing teeth removed, tilt and depth matched, dental work drawn on. Click any tooth for details.',
  },
]

const RESULTS: [string, string, string][] = [
  ['Tooth numbering (FDI)', '97%', '60 DENTEX X-rays'],
  ['Tooth outlines, other scanners', '0.91 Dice', '100 STS X-rays'],
  ['Tooth type in restored mouths', '97.5%', '50 AKU X-rays'],
  ['Root-canal fillings found', '94%', '220 Cluj X-rays'],
  ['Implants found', '92%', '220 Cluj X-rays'],
]

type Pending = { kind: 'demo' } | { kind: 'file'; file: File; force: boolean }

export function Landing() {
  const begin = useMouthTwin((s) => s.beginProcessing)
  const fail = useMouthTwin((s) => s.fail)
  const error = useMouthTwin((s) => s.error)
  const pendingFile = useMouthTwin((s) => s.pendingFile)
  const clearError = useMouthTwin((s) => s.clearError)
  const fileInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [consent, setConsent] = useState<Pending | null>(null)
  const [hovered, setHovered] = useState<{ fdi: number; x: number; y: number } | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [touched, setTouched] = useState(false)
  const hero = useRef<HTMLDivElement>(null)

  useEffect(() => preloadModel(), [])

  const start = useCallback(
    async (p: Pending) => {
      setBusy(true)
      clearError()
      try {
        if (p.kind === 'demo') begin(await loadImageFromUrl(DEMO_IMAGE_URL, 'DENTEX held-out OPG'), modelProvider)
        else begin(await loadImageFromFile(p.file, { allowNonPanoramic: p.force }), uploadProvider())
      } catch (e) {
        const err = toPipelineError(e)
        fail(err, p.kind === 'file' && err.overridable ? p.file : null)
      } finally {
        setBusy(false)
      }
    },
    [begin, fail, clearError],
  )
  const request = useCallback((p: Pending) => (hasAcknowledged() ? start(p) : setConsent(p)), [start])

  const sel = selected !== null ? toothRef(selected) : null
  const hov = hovered ? toothRef(hovered.fdi) : null
  const heroRect = hero.current?.getBoundingClientRect()

  return (
    <div
      className="relative h-full overflow-y-auto overflow-x-hidden bg-ground"
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
        if (f) request({ kind: 'file', file: f, force: false })
      }}
    >
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-transparent bg-ground/70 px-5 py-4 backdrop-blur-md sm:px-10">
        <Wordmark />
        <nav className="flex items-center gap-1 text-[13px] text-muted">
          <a href="#how" className="hidden rounded-md px-3 py-1.5 hover:text-ink sm:block">How it works</a>
          <a href="#results" className="hidden rounded-md px-3 py-1.5 hover:text-ink sm:block">Accuracy</a>
          <a href="#limits" className="hidden rounded-md px-3 py-1.5 hover:text-ink md:block">Limits</a>
          <span className="ml-2 inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-[11.5px] text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-caution" aria-hidden />
            Research preview
          </span>
        </nav>
      </header>

      {/* hero */}
      <section className="relative grid min-h-[calc(100svh-64px)] grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="relative z-10 flex flex-col justify-center px-5 pb-6 pt-8 sm:px-10 lg:py-16">
          <h1 className="max-w-[13ch] text-[42px] font-light leading-[1.04] tracking-[-0.03em] text-ink sm:text-[60px]" style={{ textWrap: 'balance' }}>
            See a panoramic X-ray as a 3D mouth.
          </h1>
          <p className="mt-5 max-w-[44ch] text-[16px] leading-relaxed text-muted">
            MouthTwin outlines and numbers every tooth on an OPG, spots crowns, bridges, root canals and implants, and
            shows them on a 3D jaw you can turn and explore. Built for dental students and teaching.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              onClick={() => fileInput.current?.click()}
              disabled={busy}
              className="inline-flex items-center gap-2.5 rounded-lg bg-enamel px-5 py-3 text-[14px] font-medium text-[#14100a] transition hover:bg-white disabled:opacity-60"
            >
              <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden>
                <path d="M8 11V2.5M4.5 6 8 2.5 11.5 6M2.5 10.5v3h11v-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Upload an OPG
            </button>
            <button
              onClick={() => request({ kind: 'demo' })}
              disabled={busy}
              className="inline-flex items-center gap-2 rounded-lg border border-line-strong px-5 py-3 text-[14px] text-ink transition hover:border-accent/60 disabled:opacity-60"
            >
              Try a sample X-ray
            </button>
            <input
              ref={fileInput}
              id="opg-upload"
              type="file"
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) request({ kind: 'file', file: f, force: false })
                e.target.value = ''
              }}
            />
          </div>
          <p className="mt-4 max-w-[52ch] text-[12.5px] leading-relaxed text-faint">
            {IMAGES_LEAVE_BROWSER
              ? 'Uploads are sent to your configured inference server.'
              : 'Runs entirely in your browser. Nothing is uploaded or stored.'}{' '}
            Or drop an X-ray anywhere on this page.
          </p>

          {error && (
            <div role="alert" className="mt-6 max-w-[520px] rounded-xl border border-caution/35 bg-[#16130d]/90 p-4">
              <p className="text-[14px] text-ink">{error.message}</p>
              <p className="mt-1 text-[13px] text-muted">{error.hint}</p>
              <div className="mt-3 flex gap-2">
                {error.overridable && pendingFile && (
                  <button
                    onClick={() => start({ kind: 'file', file: pendingFile, force: true })}
                    className="rounded-md border border-line-strong px-3 py-1.5 text-[12.5px] text-ink hover:border-accent/60"
                  >
                    Process anyway
                  </button>
                )}
                <button onClick={clearError} className="rounded-md px-3 py-1.5 text-[12.5px] text-muted hover:text-ink">
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>

        <div
          ref={hero}
          className="relative h-[52svh] min-h-[340px] lg:absolute lg:inset-y-0 lg:right-[-6%] lg:h-auto lg:w-[70%]"
          aria-label="Interactive tooth arch"
        >
          <div
            className="pointer-events-none absolute inset-0"
            style={{ background: 'radial-gradient(55% 55% at 55% 48%, rgb(134 197 216 / 0.08), transparent 70%)' }}
          />
          <Suspense fallback={null}>
            <ArchPreview
              hovered={hovered?.fdi ?? null}
              selected={selected}
              onHover={(fdi, x, y) => setHovered(fdi === null ? null : { fdi, x: x ?? 0, y: y ?? 0 })}
              onSelect={(f) => {
                setSelected(f)
                setTouched(true)
              }}
              interacting={(on) => on && setTouched(true)}
            />
          </Suspense>

          {hov && hovered && heroRect && hovered.fdi !== selected && (
            <div
              className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md border border-line-strong bg-panel/90 px-2.5 py-1.5 text-[12px] text-ink backdrop-blur"
              style={{ left: hovered.x - heroRect.left + 14, top: hovered.y - heroRect.top + 14 }}
            >
              <span className="mr-1.5 font-mono text-accent">{hov.fdi}</span>
              {hov.name}
            </div>
          )}

          {sel ? (
            <div className="absolute bottom-6 left-5 right-5 z-10 max-w-[340px] rounded-xl border border-line-strong bg-panel/90 p-4 backdrop-blur sm:left-auto lg:right-[12%]">
              <div className="flex items-baseline justify-between gap-3">
                <p className="text-[15px] font-medium text-ink">{sel.name}</p>
                <button onClick={() => setSelected(null)} className="text-[12px] text-faint hover:text-ink" aria-label="Close tooth details">
                  Close
                </button>
              </div>
              <p className="mt-1 text-[13px] leading-relaxed text-muted">{sel.friendlyRole}</p>
              <dl className="mt-3 grid grid-cols-3 gap-2 text-[12px]">
                {[
                  ['FDI', String(sel.fdi)],
                  ['Universal', String(sel.universal)],
                  ['Roots', String(sel.roots)],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-md bg-ground/70 px-2.5 py-1.5">
                    <dt className="text-faint">{k}</dt>
                    <dd className="font-mono text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          ) : (
            <p
              className={`pointer-events-none absolute bottom-6 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-full border border-line bg-panel/70 px-3 py-1.5 text-[12px] text-muted backdrop-blur transition-opacity duration-700 ${
                touched ? 'opacity-0' : 'opacity-100'
              }`}
            >
              Drag to turn · click a tooth
            </p>
          )}
        </div>
      </section>

      {/* how it works: a real three-step sequence */}
      <section id="how" className="border-t border-line px-5 py-16 sm:px-10 sm:py-20">
        <h2 className="max-w-[24ch] text-[28px] font-light tracking-[-0.02em] text-ink sm:text-[34px]">From one X-ray to an arch you can explore</h2>
        <ol className="mt-10 grid gap-8 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <li key={s.title} className="flex flex-col">
              <div className="aspect-[16/10] overflow-hidden rounded-xl border border-line bg-black">
                <img src={`${BASE}${s.img}`} alt="" loading="lazy" className="h-full w-full object-cover" />
              </div>
              <p className="mt-4 flex items-baseline gap-3 text-[16px] text-ink">
                <span className="font-mono text-[13px] text-accent">{i + 1}</span>
                {s.title}
              </p>
              <p className="mt-1.5 pl-[22px] text-[14px] leading-relaxed text-muted">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* honest accuracy */}
      <section id="results" className="border-t border-line px-5 py-16 sm:px-10 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div>
            <h2 className="max-w-[18ch] text-[28px] font-light tracking-[-0.02em] text-ink sm:text-[34px]">Tested on X-rays it never saw</h2>
            <p className="mt-4 max-w-[44ch] text-[14.5px] leading-relaxed text-muted">
              Scores from held-out public datasets, compared with expert labels. They describe how well the model reads these
              images, not clinical accuracy. Crowded, rotated or heavily restored mouths still go wrong, and children’s X-rays
              are not supported.
            </p>
          </div>
          <table className="w-full text-left text-[14px]">
            <thead>
              <tr className="border-b border-line text-[12.5px] text-faint">
                <th className="py-2.5 font-normal">What is measured</th>
                <th className="py-2.5 font-normal">Result</th>
                <th className="py-2.5 font-normal max-sm:hidden">Test set</th>
              </tr>
            </thead>
            <tbody>
              {RESULTS.map(([k, v, n]) => (
                <tr key={k} className="border-b border-line/70">
                  <td className="py-3 pr-4 text-ink">{k}</td>
                  <td className="py-3 pr-4 font-mono text-[13.5px] text-enamel tabular">{v}</td>
                  <td className="py-3 text-muted max-sm:hidden">{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* limits, in plain words */}
      <section id="limits" className="border-t border-line px-5 py-16 sm:px-10 sm:py-20">
        <h2 className="text-[28px] font-light tracking-[-0.02em] text-ink sm:text-[34px]">What it is, and what it is not</h2>
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-line bg-panel p-6">
            <p className="text-[15px] text-ink">It is</p>
            <ul className="mt-3 flex flex-col gap-2 text-[14px] leading-relaxed text-muted">
              <li>A way to learn tooth numbering and read OPGs in 3D</li>
              <li>A research preview of what AI can pull out of a single X-ray</li>
              <li>Private by design: your image stays on your device</li>
            </ul>
          </div>
          <div className="rounded-xl border border-caution/25 bg-[#13110c] p-6">
            <p className="text-[15px] text-ink">It is not</p>
            <ul className="mt-3 flex flex-col gap-2 text-[14px] leading-relaxed text-muted">
              <li>A medical device, or a tool for diagnosis or treatment planning</li>
              <li>A 3D scan of the patient: the jaw is reference anatomy fitted to the X-ray</li>
              <li>Always right: findings can be wrong or missed, so check them against the X-ray</li>
            </ul>
          </div>
        </div>
      </section>

      <footer className="border-t border-line px-5 py-8 sm:px-10">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <Wordmark />
          <p className="max-w-[78ch] text-[12px] leading-relaxed text-faint">
            {DISCLAIMER.short} Data: <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://huggingface.co/datasets/ibrahimhamamci/DENTEX" target="_blank" rel="noreferrer">DENTEX</a> (Hamamci et al., MICCAI 2023, CC BY-NC-SA 4.0),{' '}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://doi.org/10.5281/zenodo.15487430" target="_blank" rel="noreferrer">Cluj OPG dataset</a> (Mureșanu et al., 2024, CC BY 4.0),{' '}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://doi.org/10.5281/zenodo.10597292" target="_blank" rel="noreferrer">STS-2D-Tooth</a> (Wang et al., 2025),{' '}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://doi.org/10.5281/zenodo.10538750" target="_blank" rel="noreferrer">AKU OPG</a> (2024). Reference 3D anatomy:{' '}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-muted" href="https://ditto.ing.unimore.it/toothfairy2/" target="_blank" rel="noreferrer">ToothFairy2</a> (CC BY-SA 4.0). Model weights and sample X-ray: non-commercial use only.
          </p>
        </div>
      </footer>

      {dragging && (
        <div className="pointer-events-none fixed inset-3 z-40 grid place-items-center rounded-2xl border border-dashed border-accent/70 bg-ground/85">
          <p className="text-[15px] text-ink">Drop the panoramic X-ray to analyse it</p>
        </div>
      )}

      <ConsentDialog
        open={consent !== null}
        onCancel={() => setConsent(null)}
        onAccept={() => {
          const p = consent
          setConsent(null)
          if (p) start(p)
        }}
      />
    </div>
  )
}
