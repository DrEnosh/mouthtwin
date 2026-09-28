import { useState } from 'react'

export const DISCLAIMER = {
  short: 'Experimental educational prototype. Not for diagnosis, treatment planning or clinical decisions.',
  full:
    'MouthTwin is an experimental educational proof of concept. It is not intended for diagnosis, treatment planning, or clinical decision-making. Visualizations derived from 2D dental images may be conceptual or inferred. True patient-specific 3D anatomical information requires appropriate 3D imaging such as CBCT.',
}

export function DisclaimerNote() {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-[11.5px] text-faint hover:text-muted"
        aria-expanded={open}
      >
        <span className="grid h-3.5 w-3.5 place-items-center rounded-full border border-caution/60 font-mono text-[9px] text-caution">
          i
        </span>
        <span className="hidden xl:inline">Experimental · not for diagnosis</span>
        <span className="xl:hidden">Not for diagnosis</span>
      </button>
      {open && (
        <div className="absolute bottom-8 right-0 z-30 w-[340px] max-w-[calc(100vw-32px)] rounded-lg border border-line-strong bg-panel-2 p-4 text-[12.5px] leading-relaxed text-muted shadow-2xl">
          {DISCLAIMER.full}
        </div>
      )}
    </div>
  )
}
