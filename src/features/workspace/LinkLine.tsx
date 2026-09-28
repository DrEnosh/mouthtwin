import { useEffect, useRef } from 'react'
import { useMouthTwin } from '../../store/useMouthTwin'
import { imageOverlay } from '../imaging/OpgImageView'
import { linkAnchors } from '../visualization/runtime'

/**
 * In split view, draws a line from the active tooth in the OPG to the same tooth in the 3D model.
 * Runs on its own animation frame loop and writes attributes directly to avoid React re-renders.
 */
export function LinkLine() {
  const path = useRef<SVGPathElement>(null)
  const dotA = useRef<SVGCircleElement>(null)
  const dotB = useRef<SVGCircleElement>(null)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      raf = requestAnimationFrame(tick)
      const s = useMouthTwin.getState()
      const fdi = s.hoveredFdi ?? s.selectedFdi
      const el = imageOverlay.el
      const det = fdi !== null ? imageOverlay.result?.teeth.find((t) => t.fdi === fdi) : undefined
      const show = s.view === 'split' && el && det && linkAnchors.model.visible
      const p = path.current
      if (!p || !dotA.current || !dotB.current) return
      if (!show) {
        p.style.opacity = '0'
        dotA.current.style.opacity = '0'
        dotB.current.style.opacity = '0'
        return
      }
      const r = el.getBoundingClientRect()
      const ax = r.left + (det.bbox[0] + det.bbox[2] / 2) * r.width
      const ay = r.top + (det.bbox[1] + det.bbox[3] / 2) * r.height
      const bx = linkAnchors.model.x
      const by = linkAnchors.model.y
      const mx = (ax + bx) / 2
      p.setAttribute('d', `M${ax},${ay} C${mx},${ay} ${mx},${by} ${bx},${by}`)
      dotA.current.setAttribute('cx', String(ax))
      dotA.current.setAttribute('cy', String(ay))
      dotB.current.setAttribute('cx', String(bx))
      dotB.current.setAttribute('cy', String(by))
      p.style.opacity = '1'
      dotA.current.style.opacity = '1'
      dotB.current.style.opacity = '1'
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <svg className="pointer-events-none fixed inset-0 z-20 h-full w-full" aria-hidden>
      <path ref={path} fill="none" stroke="#86c5d8" strokeWidth={1.2} strokeDasharray="3 4" style={{ transition: 'opacity .2s' }} />
      <circle ref={dotA} r={3.5} fill="#86c5d8" />
      <circle ref={dotB} r={3.5} fill="#86c5d8" />
    </svg>
  )
}
