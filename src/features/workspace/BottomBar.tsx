import { DisclaimerNote } from '../../components/Disclaimer'
import { useMouthTwin, type Lens } from '../../store/useMouthTwin'

const LENSES: { id: Lens; label: string; soon?: string }[] = [
  { id: 'anatomy', label: 'Anatomy' },
  { id: 'teeth', label: 'Teeth only' },
  { id: 'xray', label: 'X-ray relief' },
  { id: 'section', label: 'Cross section', soon: 'Phase 5' },
  { id: 'timeline', label: 'Timeline', soon: 'Phase 6' },
]

export function BottomBar() {
  const lens = useMouthTwin((s) => s.lens)
  const setLens = useMouthTwin((s) => s.setLens)
  return (
    <footer className="flex h-11 shrink-0 items-center gap-3 border-t border-line bg-panel px-2 sm:px-4">
      <nav className="flex min-w-0 items-center gap-0.5 overflow-x-auto" aria-label="Lens">
        {LENSES.map((l) => (
          <button
            key={l.id}
            disabled={Boolean(l.soon)}
            onClick={() => setLens(l.id)}
            title={l.soon ? `Coming in ${l.soon}` : undefined}
            className={`relative flex shrink-0 items-center gap-1.5 rounded px-2.5 py-1.5 text-[12.5px] transition sm:px-3 ${
              lens === l.id ? 'text-ink' : l.soon ? 'text-faint' : 'text-muted hover:text-ink'
            }`}
          >
            {l.label}
            {l.soon && <span className="font-mono text-[9px] uppercase tracking-[0.08em] text-faint">soon</span>}
            {lens === l.id && <span className="absolute inset-x-2.5 -bottom-[5px] h-px bg-accent sm:inset-x-3" />}
          </button>
        ))}
      </nav>
      <div className="ml-auto shrink-0">
        <DisclaimerNote />
      </div>
    </footer>
  )
}
