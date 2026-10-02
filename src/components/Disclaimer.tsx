import { useEffect, useRef, useState } from 'react'

/**
 * One wording for the whole product. Shown once in full before the first analysis (ConsentDialog), then as a quiet
 * line in the workspace and the footer. Plain language on purpose: it has to be understood, not just displayed.
 */
export const DISCLAIMER = {
  short: 'Research preview for education. Not a medical device and not for diagnosis or treatment.',
  points: [
    'MouthTwin is a research preview built for teaching and learning. It is not a medical device.',
    'It does not diagnose, plan treatment or replace a clinical examination. Do not use it to make decisions about a patient.',
    'The 3D view is a reference jaw from one CBCT scan, adjusted to match the X-ray. It is not a 3D scan of the person in the X-ray. Real 3D anatomy needs 3D imaging such as CBCT.',
    'Tooth numbers, crowns, root canals, implants and bridges are found automatically and can be wrong or missed.',
  ],
  full:
    'MouthTwin is a research preview for education. It is not a medical device and is not intended for diagnosis, treatment planning or clinical decisions. The 3D view is reference anatomy from one CBCT scan adjusted to the X-ray, not a scan of the patient; patient-specific 3D anatomy requires 3D imaging such as CBCT. Automatic findings can be wrong or missed.',
}

const ACK_KEY = 'mouthtwin.ack.v1'

export function hasAcknowledged(): boolean {
  try {
    return window.localStorage.getItem(ACK_KEY) === '1'
  } catch {
    return false
  }
}

function rememberAck() {
  try {
    window.localStorage.setItem(ACK_KEY, '1')
  } catch {
    /* private mode: ask again next time */
  }
}

/** Shown once before the first analysis. The person has to tick the box to continue. */
export function ConsentDialog({ open, onAccept, onCancel }: { open: boolean; onAccept: () => void; onCancel: () => void }) {
  const [checked, setChecked] = useState(false)
  const box = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (open) {
      setChecked(false)
      setTimeout(() => box.current?.focus(), 30)
    }
  }, [open])
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onCancel()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onCancel])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[#040506]/75 p-4 backdrop-blur-sm" role="presentation" onClick={onCancel}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="consent-title"
        className="w-full max-w-[520px] rounded-2xl border border-line-strong bg-panel p-6 shadow-[0_30px_80px_-20px_rgb(0_0_0/0.8)] sm:p-7"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2.5">
          <span className="grid h-6 w-6 place-items-center rounded-full bg-caution/15 text-[13px] font-medium text-caution" aria-hidden>
            !
          </span>
          <h2 id="consent-title" className="text-[18px] font-medium text-ink">
            Before you start
          </h2>
        </div>
        <ul className="mt-4 flex flex-col gap-2.5 text-[14px] leading-relaxed text-muted">
          {DISCLAIMER.points.map((p) => (
            <li key={p} className="flex gap-3">
              <span className="mt-[9px] h-1 w-1 shrink-0 rounded-full bg-faint" aria-hidden />
              <span>{p}</span>
            </li>
          ))}
        </ul>
        <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-lg border border-line bg-ground/60 p-3.5 text-[13.5px] text-ink">
          <input
            ref={box}
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[#86c5d8]"
            data-testid="consent-check"
          />
          I understand MouthTwin is for education only and I will not use it to diagnose or treat anyone.
        </label>
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onCancel} className="rounded-lg px-4 py-2.5 text-[13.5px] text-muted hover:text-ink">
            Cancel
          </button>
          <button
            disabled={!checked}
            onClick={() => {
              rememberAck()
              onAccept()
            }}
            data-testid="consent-continue"
            className="rounded-lg bg-accent px-4 py-2.5 text-[13.5px] font-medium text-[#061015] transition hover:bg-[#a3d6e5] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Continue
          </button>
        </div>
      </div>
    </div>
  )
}

/** Quiet reminder for the workspace: a short line that opens the full wording. */
export function DisclaimerNote({ placement = 'up' }: { placement?: 'up' | 'down' }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-[11.5px] text-muted hover:border-line-strong hover:text-ink"
        aria-expanded={open}
      >
        <span className="h-1.5 w-1.5 rounded-full bg-caution" aria-hidden />
        <span className="hidden sm:inline">Research preview · education only</span>
        <span className="sm:hidden">Education only</span>
      </button>
      {open && (
        <div
          className={`absolute right-0 z-30 w-[360px] max-w-[calc(100vw-32px)] rounded-xl border border-line-strong bg-panel-2 p-4 text-[12.5px] leading-relaxed text-muted shadow-2xl ${
            placement === 'up' ? 'bottom-9' : 'top-9'
          }`}
        >
          {DISCLAIMER.full}
        </div>
      )}
    </div>
  )
}
