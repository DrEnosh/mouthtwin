import { useMouthTwin, type AudienceMode, type Layers } from '../../store/useMouthTwin'

const LAYER_ROWS: { key: keyof Layers; label: string; swatch: string; tag: string }[] = [
  { key: 'teeth', label: 'Teeth', swatch: 'var(--color-enamel)', tag: 'From image' },
  { key: 'bone', label: 'Jaw bone', swatch: '#b9c2c9', tag: 'From image' },
  { key: 'template', label: 'Average tooth shapes', swatch: '#d9c6a2', tag: 'Reference' },
]

const MODES: { id: AudienceMode; label: string; hint: string }[] = [
  { id: 'clinical', label: 'Clinical', hint: 'Notation, pipeline, image data' },
  { id: 'anatomy', label: 'Anatomy', hint: 'All anatomical layers' },
  { id: 'patient', label: 'Patient', hint: 'Plain language, fewer details' },
]

export function LeftRail() {
  const layers = useMouthTwin((s) => s.layers)
  const toggle = useMouthTwin((s) => s.toggleLayer)
  const audience = useMouthTwin((s) => s.audience)
  const setAudience = useMouthTwin((s) => s.setAudience)
  const showNumbers = useMouthTwin((s) => s.showNumbers)
  const setShowNumbers = useMouthTwin((s) => s.setShowNumbers)
  const isolate = useMouthTwin((s) => s.isolate)
  const setIsolate = useMouthTwin((s) => s.setIsolate)
  const showMapping = useMouthTwin((s) => s.showMapping)
  const setShowMapping = useMouthTwin((s) => s.setShowMapping)

  return (
    <div className="flex flex-col gap-7 p-4">
      <section>
        <h3 className="label mb-3">Anatomy</h3>
        <ul className="flex flex-col gap-0.5">
          {LAYER_ROWS.map((r) => (
            <li key={r.key}>
              <label className="flex cursor-pointer items-center gap-2.5 rounded px-1.5 py-1.5 hover:bg-raised">
                <input
                  id={`layer-${r.key}`}
                  type="checkbox"
                  checked={layers[r.key]}
                  onChange={() => toggle(r.key)}
                  className="peer sr-only"
                />
                <span
                  className="grid h-3.5 w-3.5 place-items-center rounded-[3px] border border-line-strong peer-checked:border-transparent"
                  style={{ background: layers[r.key] ? r.swatch : 'transparent' }}
                  aria-hidden
                />
                <span className={`flex-1 text-[13px] ${layers[r.key] ? 'text-ink' : 'text-muted'}`}>{r.label}</span>
                <span className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-faint">{r.tag}</span>
              </label>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="label mb-3">Display</h3>
        <div className="flex flex-col gap-0.5">
          <Toggle id="opt-numbers" label="FDI numbers" checked={showNumbers} onChange={setShowNumbers} />
          <Toggle id="opt-isolate" label="Isolate selection" checked={isolate} onChange={setIsolate} />
          <Toggle id="opt-mapping" label="Show AI mapping" checked={showMapping} onChange={setShowMapping} />
        </div>
      </section>

      <section>
        <h3 className="label mb-3">Mode</h3>
        <div className="flex flex-col gap-0.5" role="radiogroup" aria-label="Mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={audience === m.id}
              onClick={() => setAudience(m.id)}
              className={`flex items-start gap-2.5 rounded px-1.5 py-1.5 text-left hover:bg-raised`}
            >
              <span
                className={`mt-[3px] grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border ${
                  audience === m.id ? 'border-accent' : 'border-line-strong'
                }`}
              >
                {audience === m.id && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}
              </span>
              <span>
                <span className={`block text-[13px] ${audience === m.id ? 'text-ink' : 'text-muted'}`}>{m.label} view</span>
                <span className="block text-[11.5px] text-faint">{m.hint}</span>
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}

function Toggle({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label htmlFor={id} className="flex cursor-pointer items-center justify-between rounded px-1.5 py-1.5 hover:bg-raised">
      <span className={`text-[13px] ${checked ? 'text-ink' : 'text-muted'}`}>{label}</span>
      <input id={id} type="checkbox" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="relative h-4 w-7 rounded-full bg-line-strong transition peer-checked:bg-accent-deep after:absolute after:left-0.5 after:top-0.5 after:h-3 after:w-3 after:rounded-full after:bg-muted after:transition peer-checked:after:translate-x-3 peer-checked:after:bg-accent" />
    </label>
  )
}
