import { useEffect, useRef, useState } from 'react'

/** Before / after X-ray: drag (or use arrow keys) to reveal the model's tooth outlines. Sweeps once when first seen. */
export function CompareSlider({ before, after }: { before: string; after: string }) {
  const box = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState(0.5)
  const [touched, setTouched] = useState(false)
  const dragging = useRef(false)

  useEffect(() => {
    const el = box.current
    if (!el || touched) return
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0
    const io = new IntersectionObserver(
      ([e]) => {
        if (!e.isIntersecting) return
        io.disconnect()
        const t0 = performance.now()
        const tick = (t: number) => {
          const u = Math.min(1, (t - t0) / 2200)
          // 0.5 → 0.08 → 0.92 → 0.5, eased
          const k = 0.5 - 0.42 * Math.sin(u * Math.PI * 2) * (1 - u * 0.15)
          setPos(k)
          if (u < 1) raf = requestAnimationFrame(tick)
        }
        raf = requestAnimationFrame(tick)
      },
      { threshold: 0.6 },
    )
    io.observe(el)
    return () => {
      io.disconnect()
      cancelAnimationFrame(raf)
    }
  }, [touched])

  const setFromX = (clientX: number) => {
    const r = box.current?.getBoundingClientRect()
    if (!r) return
    setTouched(true)
    setPos(Math.min(1, Math.max(0, (clientX - r.left) / r.width)))
  }

  return (
    <div
      ref={box}
      className="relative aspect-[1400/656] w-full touch-pan-y select-none overflow-hidden rounded-2xl border border-line bg-black"
      onPointerDown={(e) => {
        dragging.current = true
        ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
        setFromX(e.clientX)
      }}
      onPointerMove={(e) => dragging.current && setFromX(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <img src={after} alt="Panoramic X-ray with every tooth outlined and numbered by the model" className="absolute inset-0 h-full w-full object-cover" draggable={false} />
      <img
        src={before}
        alt="The same panoramic X-ray without annotations"
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 ${(1 - pos) * 100}% 0 0)` }}
        draggable={false}
      />
      <div className="pointer-events-none absolute inset-y-0 w-px bg-[#d6f1fa]/80" style={{ left: `${pos * 100}%` }} />
      <div
        role="slider"
        tabIndex={0}
        aria-label="Compare the X-ray with the model's outlines"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pos * 100)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault()
            setTouched(true)
            setPos((p) => Math.min(1, Math.max(0, p + (e.key === 'ArrowLeft' ? -0.05 : 0.05))))
          }
        }}
        className="absolute top-1/2 grid h-10 w-10 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full border border-[#d6f1fa]/70 bg-panel/80 text-ink backdrop-blur"
        style={{ left: `${pos * 100}%` }}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M6.5 5 3 9l3.5 4M11.5 5 15 9l-3.5 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <span className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/60 px-2 py-1 text-[12px] text-ink">X-ray</span>
      <span className="pointer-events-none absolute right-3 top-3 rounded-md bg-black/60 px-2 py-1 text-[12px] text-ink">Model outlines</span>
    </div>
  )
}
