/** Arch glyph: fourteen tooth marks set along a parabolic arch. */
function ArchGlyph({ size = 22 }: { size?: number }) {
  const marks = Array.from({ length: 14 }, (_, i) => {
    const u = (i - 6.5) / 6.5 // -1..1
    const x = 12 + u * 9
    const y = 5 + (1 - u * u) * -0 + u * u * 12
    const angle = Math.atan2(24 * u * 0.5, 9) * (180 / Math.PI)
    return { x, y, angle, big: Math.abs(u) > 0.55 }
  })
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      {marks.map((m, i) => (
        <rect
          key={i}
          x={m.x - (m.big ? 1.1 : 0.8)}
          y={m.y - 1.6}
          width={m.big ? 2.2 : 1.6}
          height={3.2}
          rx={0.8}
          transform={`rotate(${m.angle} ${m.x} ${m.y})`}
          fill="var(--color-accent)"
          opacity={0.55 + 0.45 * (1 - Math.abs(i - 6.5) / 6.5)}
        />
      ))}
    </svg>
  )
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2">
      <ArchGlyph />
      {!compact && <span className="text-[15px] font-medium tracking-[-0.01em] text-ink">MouthTwin</span>}
    </span>
  )
}
