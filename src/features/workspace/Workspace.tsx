import { lazy, Suspense, useEffect, useState } from 'react'
import { useMouthTwin } from '../../store/useMouthTwin'
import { OpgImageView } from '../imaging/OpgImageView'
import { BottomBar } from './BottomBar'
import { Inspector } from './Inspector'
import { LeftRail } from './LeftRail'
import { LinkLine } from './LinkLine'
import { TopBar } from './TopBar'

const MouthTwinScene = lazy(() => import('../visualization/Scene').then((m) => ({ default: m.MouthTwinScene })))

const SOURCE_BADGE = {
  'precomputed-demo': 'Precomputed demo annotations',
  'heuristic-template': 'Template mapping · not a trained model',
  model: 'Tooth model · trained on DENTEX',
} as const

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return
      const s = useMouthTwin.getState()
      if (e.key === 'Escape') s.select(null)
      else if (e.key === 'r' || e.key === 'R') s.resetCamera()
      else if (e.key === 'n' || e.key === 'N') s.setShowNumbers(!s.showNumbers)
      else if (e.key === '1') s.setView('image')
      else if (e.key === '2') s.setView('model')
      else if (e.key === '3') s.setView('split')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}

export function Workspace() {
  const result = useMouthTwin((s) => s.result)
  const image = useMouthTwin((s) => s.image)
  const view = useMouthTwin((s) => s.view)
  const selected = useMouthTwin((s) => s.selectedFdi)
  const showMapping = useMouthTwin((s) => s.showMapping)
  const [railOpen, setRailOpen] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)
  useShortcuts()

  useEffect(() => {
    if (selected !== null) setSheetOpen(true)
  }, [selected])

  if (!result || !image) return null
  const imgW = view === 'image' ? '100%' : view === 'split' ? '50%' : '0%'
  const modelW = view === 'model' ? '100%' : view === 'split' ? '50%' : '0%'

  return (
    <div className="fade-in flex h-full flex-col bg-ground">
      <TopBar />
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {/* left rail */}
        <aside
          className={`z-30 w-[228px] shrink-0 overflow-y-auto border-r border-line bg-panel max-lg:absolute max-lg:inset-y-0 max-lg:left-0 max-lg:shadow-2xl max-lg:transition-transform ${
            railOpen ? 'max-lg:translate-x-0' : 'max-lg:-translate-x-full'
          }`}
        >
          <LeftRail />
        </aside>

        {/* stage */}
        <main className="relative flex min-w-0 flex-1 max-sm:flex-col">
          <section
            className="relative min-h-0 overflow-hidden transition-[width,height] duration-500 ease-out max-sm:!w-full"
            style={{ width: imgW, height: undefined, flexBasis: imgW }}
            aria-hidden={view === 'model'}
          >
            {view !== 'model' && (
              <div className="absolute inset-0 p-3 pt-14 sm:p-5 sm:pt-16">
                <OpgImageView
                  url={image.url}
                  result={result}
                  interactive
                  registerForLinks
                  sweep={0.6}
                  overlay={{ boxes: false, labels: showMapping, outlines: showMapping, curve: false }}
                />
              </div>
            )}
            <ViewportLabel title="Original OPG" sub={SOURCE_BADGE[result.source]} />
          </section>

          <section
            className={`relative min-h-0 overflow-hidden transition-[width] duration-500 ease-out max-sm:!w-full ${
              view === 'split' ? 'border-l border-line max-sm:border-l-0 max-sm:border-t' : ''
            }`}
            style={{ width: modelW, flexBasis: modelW }}
          >
            <div
              className="pointer-events-none absolute inset-0"
              style={{ background: 'radial-gradient(70% 60% at 50% 42%, rgb(134 197 216 / 0.06), transparent 70%)' }}
            />
            <Suspense fallback={<div className="grid h-full place-items-center text-[12px] text-faint">Loading 3D engine…</div>}>
              <MouthTwinScene result={result} imageUrl={image.url} image={image.element} animate />
            </Suspense>
            <ViewportLabel title="3D from your OPG" sub="Reference CBCT anatomy fitted to this X-ray · not a scan of this patient" />
            <ModelControls showHint={view === 'model'} />
          </section>
        </main>

        {/* inspector */}
        <aside
          className={`z-30 w-[312px] shrink-0 overflow-y-auto border-l border-line bg-panel max-lg:absolute max-lg:inset-x-0 max-lg:bottom-0 max-lg:max-h-[55%] max-lg:w-full max-lg:border-l-0 max-lg:border-t max-lg:shadow-2xl max-lg:transition-transform ${
            sheetOpen ? 'max-lg:translate-y-0' : 'max-lg:translate-y-full'
          }`}
        >
          <div className="sticky top-0 z-10 flex justify-end border-b border-line bg-panel px-2 py-1 lg:hidden">
            <button onClick={() => setSheetOpen(false)} className="px-2 py-1 text-[12px] text-muted">
              Close
            </button>
          </div>
          <Inspector />
        </aside>

        {/* compact-screen toggles */}
        <div className="absolute left-3 top-3 z-20 flex gap-2 lg:hidden">
          <button
            onClick={() => setRailOpen((o) => !o)}
            className="rounded-md border border-line-strong bg-panel/90 px-3 py-1.5 text-[12px] text-ink backdrop-blur"
          >
            {railOpen ? 'Close' : 'Layers'}
          </button>
          {!sheetOpen && (
            <button
              onClick={() => setSheetOpen(true)}
              className="rounded-md border border-line-strong bg-panel/90 px-3 py-1.5 text-[12px] text-ink backdrop-blur"
            >
              Details
            </button>
          )}
        </div>
      </div>
      <BottomBar />
      <LinkLine />
    </div>
  )
}

function ViewportLabel({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 text-right max-lg:top-14 sm:right-4 sm:top-4">
      <p className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-muted">{title}</p>
      <p className="mt-0.5 text-[11px] text-faint max-sm:hidden">{sub}</p>
    </div>
  )
}

function ModelControls({ showHint }: { showHint: boolean }) {
  const autoRotate = useMouthTwin((s) => s.autoRotate)
  const setAutoRotate = useMouthTwin((s) => s.setAutoRotate)
  const reset = useMouthTwin((s) => s.resetCamera)
  const showNumbers = useMouthTwin((s) => s.showNumbers)
  const setShowNumbers = useMouthTwin((s) => s.setShowNumbers)
  const btn = (on: boolean) =>
    `rounded px-2.5 py-1.5 text-[12px] transition ${on ? 'bg-raised text-ink' : 'text-muted hover:bg-raised hover:text-ink'}`
  return (
    <>
      <div className="absolute bottom-3 left-3 z-10 flex items-center gap-0.5 rounded-md border border-line bg-panel/85 p-0.5 backdrop-blur sm:bottom-4 sm:left-4">
        <button className={btn(false)} onClick={reset} title="Reset view (R)">
          Reset view
        </button>
        <button className={btn(autoRotate)} onClick={() => setAutoRotate(!autoRotate)} aria-pressed={autoRotate}>
          Rotate
        </button>
        <button className={btn(showNumbers)} onClick={() => setShowNumbers(!showNumbers)} aria-pressed={showNumbers} title="FDI numbers (N)">
          Numbers
        </button>
      </div>
      <p className={`pointer-events-none absolute bottom-4 right-4 z-10 hidden text-[11px] text-faint ${showHint ? 'xl:block' : ''}`}>
        Drag to rotate · Scroll to zoom · Right-drag to pan · Click a tooth
      </p>
    </>
  )
}
