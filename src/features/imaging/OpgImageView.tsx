import { useEffect, useRef } from 'react'
import { toothRef } from '../../data/fdi'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult, ToothDetection } from '../../types/inference'

export const imageOverlay: { el: SVGSVGElement | null; result: InferenceResult | null } = { el: null, result: null }

export interface OverlayState {
  boxes: boolean
  labels: boolean
  outlines: boolean
  curve: boolean
}

interface Props {
  url: string
  result: InferenceResult
  overlay: OverlayState
  /** Stagger overlay reveal left→right like a detector sweep (seconds across the full width). */
  sweep?: number
  interactive?: boolean
  dimImage?: boolean
  registerForLinks?: boolean
  scanning?: boolean
}

function bracketPath([x, y, w, h]: ToothDetection['bbox'], W: number, H: number): string {
  const X = x * W, Y = y * H, BW = w * W, BH = h * H
  const k = Math.min(BW, BH) * 0.22
  return [
    `M${X},${Y + k}V${Y}H${X + k}`,
    `M${X + BW - k},${Y}H${X + BW}V${Y + k}`,
    `M${X + BW},${Y + BH - k}V${Y + BH}H${X + BW - k}`,
    `M${X + k},${Y + BH}H${X}V${Y + BH - k}`,
  ].join('')
}

export function OpgImageView({ url, result, overlay, sweep = 0, interactive = false, dimImage = false, registerForLinks = false, scanning = false }: Props) {
  const W = result.image.width
  const H = result.image.height
  const svgRef = useRef<SVGSVGElement>(null)
  const hoveredFdi = useMouthTwin((s) => s.hoveredFdi)
  const selectedFdi = useMouthTwin((s) => s.selectedFdi)
  const hover = useMouthTwin((s) => s.hover)
  const select = useMouthTwin((s) => s.select)

  useEffect(() => {
    if (!registerForLinks) return
    imageOverlay.el = svgRef.current
    imageOverlay.result = result
    return () => {
      if (imageOverlay.el === svgRef.current) imageOverlay.el = null
    }
  }, [registerForLinks, result])

  const fontSize = W * 0.0105
  const curve = result.occlusalCurve
  const curvePts =
    curve &&
    Array.from({ length: 41 }, (_, i) => {
      const x = 0.12 + (0.76 * i) / 40
      return `${x * W},${(curve.a * x * x + curve.b * x + curve.c) * H}`
    }).join(' ')

  return (
    <div className="relative h-full w-full" style={{ containerType: 'size' }}>
      <div
        className="absolute inset-0 m-auto"
        style={{ aspectRatio: `${W} / ${H}`, width: `min(100cqw, calc(100cqh * ${W / H}))` }}
      >
        <img
          src={url}
          alt="Panoramic dental X-ray"
          className="absolute inset-0 h-full w-full select-none transition-[filter] duration-700"
          style={{ filter: dimImage ? 'brightness(0.55) contrast(1.05)' : 'none' }}
          draggable={false}
        />
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="absolute inset-0 h-full w-full overflow-visible"
          onPointerLeave={() => interactive && hover(null)}
        >
          {curvePts && (
            <polyline
              points={curvePts}
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth={1}
              strokeDasharray="4 5"
              vectorEffect="non-scaling-stroke"
              style={{ opacity: overlay.curve ? 0.6 : 0, transition: 'opacity .6s ease' }}
            />
          )}
          {result.teeth.map((d) => {
            const delay = sweep ? `${(d.bbox[0] + d.bbox[2] / 2) * sweep}s` : '0s'
            const isHover = hoveredFdi === d.fdi
            const isSel = selectedFdi === d.fdi
            const someSelected = selectedFdi !== null
            const upper = toothRef(d.fdi).upper
            const labelX = (d.apex ? d.apex[0] : d.bbox[0] + d.bbox[2] / 2) * W
            const labelY = upper ? d.bbox[1] * H - fontSize * 0.9 : (d.bbox[1] + d.bbox[3]) * H + fontSize * 1.6
            const poly = d.polygon.map(([x, y]) => `${x * W},${y * H}`).join(' ')
            return (
              <g
                key={d.fdi}
                style={{ opacity: someSelected && !isSel && !isHover ? 0.45 : 1, transition: 'opacity .3s' }}
                onPointerEnter={() => interactive && hover(d.fdi)}
                onClick={(e) => {
                  if (!interactive) return
                  e.stopPropagation()
                  select(isSel ? null : d.fdi)
                }}
                className={interactive ? 'cursor-pointer' : undefined}
              >
                <path
                  d={bracketPath(d.bbox, W, H)}
                  fill="none"
                  stroke={isSel || isHover ? '#dff1f6' : 'var(--color-accent)'}
                  strokeWidth={isSel ? 1.6 : 1}
                  vectorEffect="non-scaling-stroke"
                  className={`ann-box ${overlay.boxes || isSel || isHover ? 'on' : ''}`}
                  style={{ transitionDelay: overlay.boxes ? delay : '0s', strokeOpacity: overlay.boxes ? 0.85 : 1 }}
                />
                {poly && (
                  <polygon
                    points={poly}
                    pathLength={1}
                    fill={isSel ? 'rgb(134 197 216 / 0.32)' : isHover ? 'rgb(134 197 216 / 0.2)' : 'rgb(134 197 216 / 0.07)'}
                    stroke={isSel ? '#e6f6fb' : 'var(--color-accent)'}
                    strokeWidth={isSel ? 1.8 : 1.1}
                    strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                    className={`ann-outline ${overlay.outlines || isSel || isHover ? 'on' : ''}`}
                    style={{ transitionDelay: overlay.outlines ? delay : '0s' }}
                  />
                )}
                {interactive && (
                  <rect
                    x={d.bbox[0] * W}
                    y={d.bbox[1] * H}
                    width={d.bbox[2] * W}
                    height={d.bbox[3] * H}
                    fill="transparent"
                  />
                )}
                <text
                  x={labelX}
                  y={labelY}
                  textAnchor="middle"
                  fontFamily="IBM Plex Mono, ui-monospace, monospace"
                  fontSize={fontSize}
                  fontWeight={isSel ? 600 : 500}
                  fill={isSel ? '#e6f6fb' : 'var(--color-accent)'}
                  className={`ann-label ${overlay.labels || isSel || isHover ? 'on' : ''}`}
                  style={{ transitionDelay: overlay.labels ? delay : '0s' }}
                >
                  {d.fdi}
                </text>
                {(() => {
                  const r = d.restorations
                  const tag = r ? [r.implant ? 'I' : '', r.crown?.kind === 'cap' ? 'C' : r.crown ? 'F' : '', r.rootCanal ? 'R' : ''].join('') : ''
                  return tag ? (
                    <text
                      x={labelX}
                      y={labelY + (upper ? -fontSize * 1.05 : fontSize * 1.05)}
                      textAnchor="middle"
                      fontFamily="IBM Plex Mono, ui-monospace, monospace"
                      fontSize={fontSize * 0.85}
                      fontWeight={600}
                      fill="#d9dfe7"
                      className={`ann-label ${overlay.labels || isSel || isHover ? 'on' : ''}`}
                    >
                      {tag}
                    </text>
                  ) : null
                })()}
              </g>
            )
          })}
          {(result.implants ?? []).map((im, k) => (
            <g key={`imp${k}`} className={`ann-label ${overlay.labels ? 'on' : ''}`} pointerEvents="none">
              <rect
                x={im.bbox[0] * W}
                y={im.bbox[1] * H}
                width={im.bbox[2] * W}
                height={im.bbox[3] * H}
                fill="none"
                stroke="#d9dfe7"
                strokeWidth={1}
                strokeDasharray="3 3"
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={(im.bbox[0] + im.bbox[2] / 2) * W}
                y={(im.bbox[1] + im.bbox[3]) * H + fontSize * 1.1}
                textAnchor="middle"
                fontFamily="IBM Plex Mono, ui-monospace, monospace"
                fontSize={fontSize * 0.85}
                fontWeight={600}
                fill="#d9dfe7"
              >
                {im.slot ? `I ${im.slot}` : 'I'}
              </text>
            </g>
          ))}
        </svg>
        {scanning && (
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="raster absolute inset-0" />
            <div className="scanline" />
          </div>
        )}
      </div>
    </div>
  )
}
