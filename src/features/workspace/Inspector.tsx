import { OPG_ORDER_LOWER, OPG_ORDER_UPPER, toothRef, type ToothRef } from '../../data/fdi'
import { useMouthTwin } from '../../store/useMouthTwin'
import { describeRestorations } from '../pipeline/model/restorations'
import type { InferenceResult, InferenceSource, ToothDetection, ToothPose, ToothRestorations } from '../../types/inference'

const SOURCE_TEXT: Record<InferenceSource, { title: string; body: string }> = {
  'precomputed-demo': {
    title: 'Precomputed demo annotations',
    body: 'The demo image is synthetic. Tooth outlines are the exact geometry used to draw it, not model predictions.',
  },
  'heuristic-template': {
    title: 'Template mapping · no trained model',
    body: 'Teeth are placed from an average adult template along the detected occlusal plane. Missing or displaced teeth are not detected.',
  },
  model: {
    title: 'Tooth model · runs in your browser',
    body: 'Outlines and FDI numbers are predicted by a U-Net trained on the public DENTEX dataset (Hamamci et al., MICCAI 2023, CC BY-NC-SA 4.0). The same model also flags crowns, fillings, root-canal fillings and implants (trained on the Cluj panoramic dataset, CC BY 4.0), which are drawn on the 3D teeth. The 3D teeth and jaws come from one expert-labelled ToothFairy2 CBCT (CC BY-SA 4.0), with teeth removed, lengthened or tilted to match this X-ray. Scores are the model’s own confidence, not clinical accuracy.',
  },
}

export function Inspector() {
  const result = useMouthTwin((s) => s.result)
  const selected = useMouthTwin((s) => s.selectedFdi)
  const audience = useMouthTwin((s) => s.audience)
  if (!result) return null
  const det = selected !== null ? result.teeth.find((t) => t.fdi === selected) : undefined
  return (
    <div className="flex flex-col gap-6 p-4">
      {selected !== null && det ? (
        audience === 'patient' ? (
          <PatientTooth tooth={toothRef(selected)} r={det.restorations} />
        ) : (
          <ToothDetail tooth={toothRef(selected)} det={det} result={result} />
        )
      ) : audience === 'patient' ? (
        <PatientSummary result={result} />
      ) : (
        <CaseSummary result={result} />
      )}
    </div>
  )
}

function Odontogram({ result }: { result: InferenceResult }) {
  const present = new Set(result.teeth.map((t) => t.fdi))
  const marks = new Map<number, string>()
  for (const t of result.teeth) {
    const r = t.restorations
    if (!r) continue
    const m = [r.implant ? 'I' : '', r.crown?.kind === 'cap' ? 'C' : r.crown ? 'F' : '', r.rootCanal ? 'R' : ''].join('')
    if (m) marks.set(t.fdi, m)
  }
  for (const im of result.implants ?? []) if (im.slot !== null && !present.has(im.slot)) marks.set(im.slot, 'I')
  const selected = useMouthTwin((s) => s.selectedFdi)
  const hovered = useMouthTwin((s) => s.hoveredFdi)
  const select = useMouthTwin((s) => s.select)
  const hover = useMouthTwin((s) => s.hover)
  const row = (order: number[]) => (
    <div className="grid grid-cols-16 gap-[3px]" style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}>
      {order.map((fdi, i) => {
        const on = present.has(fdi)
        const isSel = selected === fdi
        const isHov = hovered === fdi
        return (
          <button
            key={fdi}
            disabled={!on}
            onClick={() => select(isSel ? null : fdi)}
            onPointerEnter={() => hover(fdi)}
            onPointerLeave={() => hover(null)}
            title={`${on ? `${fdi} · ${toothRef(fdi).name}` : `${fdi} · not mapped`}${marks.has(fdi) ? ` · ${marks.get(fdi)}` : ''}`}
            className={`relative h-7 rounded-[3px] font-mono text-[9.5px] tabular transition ${i === 7 ? 'mr-[3px]' : ''} ${
              isSel
                ? 'bg-accent text-ground'
                : isHov
                  ? 'bg-raised text-ink ring-1 ring-accent/60'
                  : on
                    ? 'bg-panel-2 text-muted hover:text-ink'
                    : 'border border-dashed border-line-strong text-faint'
            }`}
          >
            {fdi}
            {marks.has(fdi) && (
              <span className="pointer-events-none absolute -right-px -top-1 rounded-sm bg-[#b9c3cf] px-[2px] text-[7px] font-semibold leading-[9px] text-[#101418]">
                {marks.get(fdi)}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
  return (
    <div className="flex flex-col gap-1">
      {row(OPG_ORDER_UPPER)}
      <div className="flex items-center justify-between px-0.5 font-mono text-[9px] uppercase tracking-[0.1em] text-faint">
        <span>R</span>
        <span>L</span>
      </div>
      {row(OPG_ORDER_LOWER)}
      {marks.size > 0 && (
        <p className="mt-1 font-mono text-[9.5px] leading-snug text-faint">C crown/cap · F filling · R root-canal filling · I implant</p>
      )}
    </div>
  )
}

function CaseSummary({ result }: { result: InferenceResult }) {
  const src = SOURCE_TEXT[result.source]
  return (
    <>
      <section>
        <p className="label">Selected tooth</p>
        <p className="mt-2 text-[15px] text-ink">None selected</p>
        <p className="mt-1 text-[12.5px] text-muted">Click a tooth in the model, the image or the chart below.</p>
      </section>
      <section className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between">
          <p className="label">Dental chart · FDI</p>
          <p className="font-mono text-[11px] text-muted tabular">{result.teeth.length}/32 mapped</p>
        </div>
        <Odontogram result={result} />
      </section>
      <Findings result={result} />
      <section className="flex flex-col gap-2">
        <p className="label">Imaging</p>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12.5px]">
          <dt className="text-faint">Modality</dt>
          <dd className="text-ink">Panoramic (OPG)</dd>
          <dt className="text-faint">Dimensions</dt>
          <dd className="text-ink tabular">
            {result.image.width} × {result.image.height} px
          </dd>
          <dt className="text-faint">Source</dt>
          <dd className="text-ink">{result.image.synthetic ? 'Synthetic, generated' : result.image.fileName ?? 'Upload'}</dd>
          {result.image.fileSizeBytes ? (
            <>
              <dt className="text-faint">File size</dt>
              <dd className="text-ink tabular">{(result.image.fileSizeBytes / 1024 / 1024).toFixed(2)} MB</dd>
            </>
          ) : null}
        </dl>
      </section>
      <section className="flex flex-col gap-2">
        <p className="label">Pipeline</p>
        <ol className="flex flex-col gap-2">
          {result.stages.map((s) => (
            <li key={s.id} className="flex gap-2.5 text-[12.5px]">
              <span className={s.status === 'done' ? 'text-accent' : 'text-caution'}>{s.status === 'done' ? '✓' : '–'}</span>
              <span>
                <span className="block text-ink">{s.label}</span>
                <span className="block font-mono text-[10.5px] leading-snug text-faint">{s.method}</span>
              </span>
            </li>
          ))}
        </ol>
      </section>
      <section className="rounded-md border border-line bg-panel-2 p-3">
        <p className="text-[12.5px] text-ink">{src.title}</p>
        <p className="mt-1 text-[12px] leading-relaxed text-muted">{src.body}</p>
      </section>
    </>
  )
}

function ToothCrop({ det, result, url }: { det: ToothDetection; result: InferenceResult; url: string }) {
  const W = result.image.width
  const H = result.image.height
  const [x, y, w, h] = det.bbox
  const pad = 0.35
  const vx = (x - w * pad) * W
  const vy = (y - h * 0.08) * H
  const vw = w * (1 + 2 * pad) * W
  const vh = h * 1.16 * H
  const poly = det.polygon.map(([px, py]) => `${px * W},${py * H}`).join(' ')
  return (
    <svg viewBox={`${vx} ${vy} ${vw} ${vh}`} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
      <image href={url} width={W} height={H} />
      {poly && (
        <polygon points={poly} fill="rgb(134 197 216 / 0.14)" stroke="#86c5d8" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  )
}

function ToothDetail({ tooth, det, result }: { tooth: ToothRef; det: ToothDetection; result: InferenceResult }) {
  const url = useMouthTwin((s) => s.image?.url)
  const layers = useMouthTwin((s) => s.layers)
  const select = useMouthTwin((s) => s.select)
  const src = result.stages
  const method = (id: string) => src.find((s) => s.id === id)?.method ?? '—'
  return (
    <>
      <section className="flex items-start justify-between gap-3">
        <div>
          <p className="label">Selected tooth</p>
          <p className="mt-1.5 font-mono text-[34px] font-medium leading-none text-ink tabular">#{tooth.fdi}</p>
          <p className="mt-2 text-[14px] text-ink">{tooth.name}</p>
        </div>
        <button onClick={() => select(null)} className="rounded px-2 py-1 text-[12px] text-muted hover:bg-raised hover:text-ink" aria-label="Clear selection">
          Esc
        </button>
      </section>

      <dl className="grid grid-cols-3 overflow-hidden rounded-md border border-line text-center">
        {[
          ['FDI', String(tooth.fdi)],
          ['Universal', String(tooth.universal)],
          ['Palmer', tooth.palmer],
        ].map(([k, v]) => (
          <div key={k} className="border-r border-line py-2 last:border-r-0">
            <dt className="label !text-[9.5px]">{k}</dt>
            <dd className="mt-0.5 font-mono text-[14px] text-ink">{v}</dd>
          </div>
        ))}
      </dl>

      {url && det.polygon.length > 0 && (
        <section>
          <p className="label mb-2">In the image</p>
          <div className="h-36 overflow-hidden rounded-md border border-line bg-black">
            <ToothCrop det={det} result={result} url={url} />
          </div>
        </section>
      )}

      <ToothFindings r={det.restorations} />
      {det.pose && <ToothPosition fdi={det.fdi} pose={det.pose} lengthMm={(tooth.crownMm ?? 0) + (tooth.rootMm ?? 0)} />}

      <section>
        <p className="label mb-2">Pipeline</p>
        <ul className="flex flex-col gap-1.5 text-[12.5px]">
          <Status label="Detected" detail={method('detect')} />
          <Status label="Segmentation" detail={det.polygon.length ? method('segment') : 'Not available'} ok={det.polygon.length > 0} />
          <Status label="Numbering" detail={method('number')} />
          <Status label="3D" detail={det.pose ? "Real CBCT tooth shape; tilt, depth and length moved to match this X-ray" : "Real CBCT tooth shape, length and tilt fitted to this X-ray"} />
        </ul>
      </section>

      <section>
        <p className="label mb-2">Layers</p>
        <div className="flex flex-wrap gap-1.5">
          {[
            ['Tooth (crown + roots)', layers.teeth],
            ['Jaw bone', layers.bone],
            ['Nerve canal', layers.canals],
          ].map(([name, on]) => (
            <span
              key={String(name)}
              className={`rounded border px-2 py-0.5 text-[11.5px] ${on ? 'border-accent/40 text-ink' : 'border-line text-faint'}`}
            >
              {name}
            </span>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-baseline justify-between">
          <p className="label">Reference size</p>
          <p className="text-[10.5px] text-faint">population average</p>
        </div>
        <dl className="flex flex-col gap-1.5 text-[12.5px] tabular">
          <Dim k="Crown width (MD)" v={tooth.widthMm} />
          <Dim k="Crown depth (BL)" v={tooth.depthMm} />
          <Dim k="Crown height" v={tooth.crownMm} />
          <Dim k="Root length" v={tooth.rootMm} />
        </dl>
        <p className="mt-2 text-[11.5px] leading-snug text-faint">Textbook averages for this tooth, not measured from the image. A single OPG has no reliable depth or scale.</p>
      </section>

      <section className="flex items-center justify-between rounded-md border border-line bg-panel-2 px-3 py-2.5">
        <span className="text-[12.5px] text-muted">Confidence</span>
        <span className="font-mono text-[12px] text-ink">
          {typeof det.score === 'number' ? det.score.toFixed(2) : 'Experimental · no model score'}
        </span>
      </section>
    </>
  )
}

/** Plain description of how this tooth sits compared with the same tooth on a typical OPG. */
export function describePosition(fdi: number, p: ToothPose, lengthMm: number): string[] {
  const out: string[] = []
  const q = Math.floor(fdi / 10)
  if (p.crownOnly) out.push('Only a crown-shaped outline is visible, with no root length. Drawn as a crown.')
  else if (p.partialRoot) out.push('Shorter than a fully formed third molar. Drawn with a short root.')
  const mm = Math.round(Math.abs(p.depth) * (lengthMm || 21))
  if (mm >= 2) {
    out.push(
      p.depth > 0
        ? `Crown sits about ${mm} mm ${q <= 2 ? 'above' : 'below'} the bite line of the other teeth (deeper in the bone).`
        : `Crown sits about ${mm} mm past the bite line of the other teeth.`,
    )
  }
  const deg = Math.round((Math.abs(p.tilt) * 180) / Math.PI)
  if (deg >= 5) {
    // tilt > 0: apex toward image right. Image right is mesial for quadrants 1 and 4 (patient's right side).
    const apexRight = p.tilt > 0
    const mesialRight = q === 1 || q === 4
    const crown = apexRight === mesialRight ? 'distally' : 'mesially'
    out.push(`Tipped ${crown} about ${deg}° more than this tooth usually looks.`)
  }
  if (!p.crownOnly && !p.partialRoot && Math.abs(p.length - 1) >= 0.08) {
    out.push(`Looks about ${Math.round(Math.abs(p.length - 1) * 100)}% ${p.length > 1 ? 'longer' : 'shorter'} than usual.`)
  }
  return out
}

function ToothPosition({ fdi, pose, lengthMm }: { fdi: number; pose: ToothPose; lengthMm: number }) {
  const items = describePosition(fdi, pose, lengthMm)
  return (
    <section>
      <p className="label mb-2">Position on this X-ray</p>
      {items.length ? (
        <ul className="flex flex-col gap-1.5 text-[12.5px]">
          {items.map((i) => (
            <li key={i} className="flex gap-2.5">
              <span className="text-accent">●</span>
              <span className="text-ink">{i}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[12.5px] text-muted">Sits like this tooth usually does on an OPG.</p>
      )}
      <p className="mt-2 text-[11.5px] leading-snug text-faint">
        Compared with the same tooth on 634 expert-labelled OPGs. The 3D tooth is moved by the same amount. An OPG
        distorts angles and sizes, so treat these as rough.
      </p>
    </section>
  )
}

function ToothFindings({ r }: { r?: ToothRestorations }) {
  const items = describeRestorations(r)
  if (!r) return null
  return (
    <section>
      <p className="label mb-2">Seen on this X-ray</p>
      {items.length ? (
        <ul className="flex flex-col gap-1.5 text-[12.5px]">
          {items.map((i) => (
            <li key={i} className="flex gap-2.5">
              <span className="text-accent">●</span>
              <span className="text-ink">{i}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[12.5px] text-muted">No crown, filling or root-canal filling flagged.</p>
      )}
      <p className="mt-2 text-[11.5px] leading-snug text-faint">
        Found from how dense the tooth looks and the model’s restoration maps. The 3D shows a generic version; the X-ray can’t tell material, fit or quality.
      </p>
    </section>
  )
}

function tally(result: InferenceResult) {
  let caps = 0, fills = 0, rct = 0, implants = result.implants?.length ?? 0
  const bridges = new Set<number>()
  for (const t of result.teeth) {
    const r = t.restorations
    if (!r) continue
    if (r.crown?.kind === 'cap') caps++
    else if (r.crown) fills++
    if (r.rootCanal) rct++
    if (r.implant) implants++
    if (r.bridgeId) bridges.add(r.bridgeId)
  }
  return { caps, fills, rct, implants, bridges: bridges.size }
}

function Findings({ result }: { result: InferenceResult }) {
  if (!result.teeth.some((t) => t.restorations) && !result.implants) return null
  const t = tally(result)
  const rows: [string, number][] = [
    ['Crowns / caps', t.caps],
    ['Fillings', t.fills],
    ['Root-canal fillings', t.rct],
    ['Implants', t.implants],
    ['Bridges (joined crowns)', t.bridges],
  ]
  return (
    <section className="flex flex-col gap-2">
      <p className="label">Seen on the X-ray</p>
      <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-[12.5px]">
        {rows.map(([k, n]) => (
          <div key={k} className="contents">
            <dt className={n ? 'text-ink' : 'text-faint'}>{k}</dt>
            <dd className={`font-mono tabular ${n ? 'text-ink' : 'text-faint'}`}>{n}</dd>
          </div>
        ))}
      </dl>
      <p className="text-[11.5px] leading-snug text-faint">Drawn on the 3D teeth with their own materials. Detected by the model from the picture, so it can miss or over-call.</p>
    </section>
  )
}

function Status({ label, detail, ok = true }: { label: string; detail: string; ok?: boolean }) {
  return (
    <li className="flex gap-2.5">
      <span className={ok ? 'text-accent' : 'text-faint'}>{ok ? '✓' : '–'}</span>
      <span>
        <span className="text-ink">{label}</span>
        <span className="block font-mono text-[10.5px] leading-snug text-faint">{detail}</span>
      </span>
    </li>
  )
}

function Dim({ k, v }: { k: string; v: number }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-line pb-1">
      <dt className="text-faint">{k}</dt>
      <dd className="font-mono text-ink">{v.toFixed(1)} mm</dd>
    </div>
  )
}

function PatientSummary({ result }: { result: InferenceResult }) {
  const upper = result.teeth.filter((t) => toothRef(t.fdi).upper).length
  const lower = result.teeth.length - upper
  return (
    <>
      <section>
        <p className="label">Your mouth</p>
        <p className="mt-2 text-[26px] font-light leading-tight text-ink">{result.teeth.length} teeth identified</p>
        <p className="mt-1 text-[13px] text-muted">Tap a tooth to explore it.</p>
      </section>
      <section className="flex flex-col gap-3">
        {[
          ['Upper jaw', upper],
          ['Lower jaw', lower],
        ].map(([k, n]) => (
          <div key={String(k)}>
            <div className="mb-1.5 flex justify-between text-[12.5px]">
              <span className="text-ink">{k}</span>
              <span className="font-mono text-muted tabular">{n} teeth</span>
            </div>
            <div className="flex gap-[3px]">
              {Array.from({ length: 16 }, (_, i) => (
                <span key={i} className={`h-4 flex-1 rounded-[2px] ${i < Number(n) ? 'bg-enamel/80' : 'bg-line-strong'}`} />
              ))}
            </div>
          </div>
        ))}
      </section>
      <PatientFindings result={result} />
      <section>
        <Odontogram result={result} />
      </section>
      <p className="text-[12px] leading-relaxed text-faint">
        This is an educational 3D-style picture built from a flat X-ray. It can’t tell you whether anything is wrong. Your
        dentist is the right person for that.
      </p>
    </>
  )
}

function PatientFindings({ result }: { result: InferenceResult }) {
  const t = tally(result)
  const parts = [
    t.caps ? `${t.caps} crown${t.caps > 1 ? 's' : ''} or cap${t.caps > 1 ? 's' : ''}` : '',
    t.fills ? `${t.fills} filling${t.fills > 1 ? 's' : ''}` : '',
    t.rct ? `${t.rct} root-canal-treated tooth${t.rct > 1 ? ' teeth' : ''}`.replace('tooth teeth', 'teeth') : '',
    t.implants ? `${t.implants} implant${t.implants > 1 ? 's' : ''}` : '',
  ].filter(Boolean)
  if (!result.teeth.some((x) => x.restorations) && !result.implants) return null
  return (
    <section>
      <p className="label mb-2">Dental work that shows up</p>
      <p className="text-[13.5px] leading-relaxed text-ink">{parts.length ? parts.join(', ') + '.' : 'None spotted.'}</p>
      <p className="mt-1 text-[12px] leading-relaxed text-faint">In 3D, crowns look metallic, root-canal fillings glow inside the root, and implants look like screws. It is an automatic guess from the picture.</p>
    </section>
  )
}

function PatientTooth({ tooth, r }: { tooth: ToothRef; r?: ToothRestorations }) {
  const select = useMouthTwin((s) => s.select)
  return (
    <>
      <section className="flex items-start justify-between">
        <div>
          <p className="label">Tooth #{tooth.fdi}</p>
          <p className="mt-2 text-[22px] font-light leading-tight text-ink">{tooth.friendlyName}</p>
          <p className="mt-1 text-[13px] text-muted">{tooth.shortName}</p>
        </div>
        <button onClick={() => select(null)} className="rounded px-2 py-1 text-[12px] text-muted hover:bg-raised hover:text-ink">
          Back
        </button>
      </section>
      <p className="text-[14px] leading-relaxed text-ink">{tooth.friendlyRole}</p>
      {describeRestorations(r).length > 0 && (
        <p className="rounded-md border border-line bg-panel-2 px-3 py-2 text-[13px] text-ink">
          On the X-ray this tooth shows: {describeRestorations(r).join(', ').toLowerCase()}.
        </p>
      )}
      <section>
        <p className="label mb-2">Explore</p>
        <ul className="flex flex-col gap-1.5 text-[13px] text-ink">
          <li>✓ Tooth surface</li>
          <li>✓ Root{tooth.roots > 1 ? `s (${tooth.roots})` : ''}</li>
          <li>✓ Surrounding anatomy</li>
        </ul>
      </section>
      <p className="text-[12px] leading-relaxed text-faint">The outline comes from your X-ray. How thick it looks is an estimate, because an X-ray is flat.</p>
    </>
  )
}
