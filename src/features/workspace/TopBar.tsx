import { DisclaimerNote } from '../../components/Disclaimer'
import { Wordmark } from '../../components/Wordmark'
import { useMouthTwin, type ViewMode } from '../../store/useMouthTwin'

const VIEWS: { id: ViewMode; label: string; key: string }[] = [
  { id: 'image', label: 'Image', key: '1' },
  { id: 'model', label: 'Model', key: '2' },
  { id: 'split', label: 'Split', key: '3' },
]

export function TopBar() {
  const result = useMouthTwin((s) => s.result)
  const image = useMouthTwin((s) => s.image)
  const view = useMouthTwin((s) => s.view)
  const setView = useMouthTwin((s) => s.setView)
  const replay = useMouthTwin((s) => s.replayReveal)
  const goHome = useMouthTwin((s) => s.goHome)
  if (!result) return null
  const caseName = result.image.synthetic ? 'Synthetic demo' : image?.fileName ?? 'Uploaded image'

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line bg-panel px-3 sm:gap-5 sm:px-4">
      <button onClick={goHome} className="shrink-0" aria-label="Back to start">
        <span className="sm:hidden">
          <Wordmark compact />
        </span>
        <span className="max-sm:hidden">
          <Wordmark />
        </span>
      </button>
      <div className="hidden h-5 w-px bg-line md:block" />
      <dl className="hidden min-w-0 items-center gap-5 md:flex">
        <Meta label="Case" value={caseName} />
        <Meta label="Imaging" value={`OPG · ${result.image.width}×${result.image.height}`} />
      </dl>
      <div className="hidden lg:block">
        <DisclaimerNote placement="down" />
      </div>

      <div className="ml-auto flex items-center gap-1 rounded-md border border-line bg-ground p-0.5" role="tablist" aria-label="View">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            role="tab"
            aria-selected={view === v.id}
            onClick={() => setView(v.id)}
            title={`${v.label} view (${v.key})`}
            className={`rounded px-2.5 py-1 text-[12.5px] transition sm:px-3 ${
              view === v.id ? 'bg-raised text-ink' : 'text-muted hover:text-ink'
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={replay}
          className="hidden rounded px-2.5 py-1.5 text-[12.5px] text-muted hover:bg-raised hover:text-ink sm:block"
          title="Replay the OPG → model transformation"
        >
          Replay
        </button>
        <button onClick={goHome} className="whitespace-nowrap rounded px-2.5 py-1.5 text-[12.5px] text-muted hover:bg-raised hover:text-ink">
          New image
        </button>
      </div>
    </header>
  )
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-baseline gap-2">
      <dt className="label">{label}</dt>
      <dd className="truncate text-[12.5px] text-ink tabular">{value}</dd>
    </div>
  )
}
