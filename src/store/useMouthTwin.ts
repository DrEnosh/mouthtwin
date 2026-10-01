import { create } from 'zustand'
import type { InferenceResult } from '../types/inference'
import type { PipelineError } from '../features/pipeline/errors'
import type { LoadedImage } from '../features/pipeline/loadImage'
import type { InferenceProvider } from '../features/pipeline/providers'

export type AppStage = 'landing' | 'processing' | 'workspace'
export type ViewMode = 'image' | 'model' | 'split'
export type AudienceMode = 'clinical' | 'anatomy' | 'patient'
export type Lens = 'anatomy' | 'teeth' | 'xray' | 'section' | 'timeline'

export interface Layers {
  /** Reference CBCT teeth fitted to the OPG */
  teeth: boolean
  /** Mandible and maxilla from the reference CBCT */
  bone: boolean
  /** Inferior alveolar canals from the reference CBCT */
  canals: boolean
  /** Maxillary sinuses from the reference CBCT */
  sinus: boolean
  /** The OPG itself pushed into 3D (brightness relief) */
  relief: boolean
  /** Crowns, fillings, root-canal fillings, implants and bridges found on the OPG, drawn on the 3D teeth */
  restorations: boolean
}

const LENS_LAYERS: Record<'anatomy' | 'teeth' | 'xray', Layers> = {
  anatomy: { teeth: true, bone: true, canals: true, sinus: true, relief: false, restorations: true },
  teeth: { teeth: true, bone: false, canals: false, sinus: false, relief: false, restorations: true },
  xray: { teeth: false, bone: false, canals: false, sinus: false, relief: true, restorations: false },
}

interface State {
  stage: AppStage
  image: LoadedImage | null
  result: InferenceResult | null
  provider: InferenceProvider | null
  error: PipelineError | null
  /** File kept only while the user decides whether to override a soft validation failure. */
  pendingFile: File | null

  view: ViewMode
  audience: AudienceMode
  lens: Lens
  layers: Layers
  showNumbers: boolean
  showMapping: boolean
  isolate: boolean
  hoveredFdi: number | null
  selectedFdi: number | null
  /** Bumped to replay the OPG → arch transformation. */
  revealKey: number
  autoRotate: boolean
  cameraResetKey: number

  beginProcessing: (image: LoadedImage, provider: InferenceProvider) => void
  finishProcessing: (result: InferenceResult) => void
  fail: (error: PipelineError, pendingFile?: File | null) => void
  clearError: () => void
  goHome: () => void

  setView: (v: ViewMode) => void
  setAudience: (m: AudienceMode) => void
  setLens: (l: Lens) => void
  toggleLayer: (k: keyof Layers) => void
  setShowNumbers: (v: boolean) => void
  setShowMapping: (v: boolean) => void
  setIsolate: (v: boolean) => void
  hover: (fdi: number | null) => void
  select: (fdi: number | null) => void
  replayReveal: () => void
  setAutoRotate: (v: boolean) => void
  resetCamera: () => void
}

export const useMouthTwin = create<State>((set, get) => ({
  stage: 'landing',
  image: null,
  result: null,
  provider: null,
  error: null,
  pendingFile: null,

  view: 'model',
  audience: 'clinical',
  lens: 'anatomy',
  layers: { teeth: true, bone: true, canals: true, sinus: false, relief: false, restorations: true },
  showNumbers: false,
  showMapping: true,
  isolate: true,
  hoveredFdi: null,
  selectedFdi: null,
  revealKey: 0,
  autoRotate: true,
  cameraResetKey: 0,

  beginProcessing: (image, provider) => {
    const prev = get().image
    if (prev?.revocable && prev.url !== image.url) URL.revokeObjectURL(prev.url)
    set({ stage: 'processing', image, provider, result: null, error: null, pendingFile: null, selectedFdi: null, hoveredFdi: null })
  },
  finishProcessing: (result) =>
    set((s) => ({ stage: 'workspace', result, view: 'model', revealKey: s.revealKey + 1, autoRotate: true })),
  fail: (error, pendingFile = null) => set({ stage: 'landing', error, pendingFile }),
  clearError: () => set({ error: null, pendingFile: null }),
  goHome: () => {
    const prev = get().image
    if (prev?.revocable) URL.revokeObjectURL(prev.url)
    set({ stage: 'landing', image: null, result: null, provider: null, selectedFdi: null, hoveredFdi: null })
  },

  setView: (view) => set({ view }),
  setAudience: (audience) =>
    set(() => {
      if (audience === 'patient') return { audience, showNumbers: false, layers: { ...LENS_LAYERS.teeth }, lens: 'teeth' }
      if (audience === 'anatomy') return { audience, layers: { ...LENS_LAYERS.anatomy }, lens: 'anatomy' }
      return { audience }
    }),
  setLens: (lens) =>
    set(() => {
      if (lens === 'anatomy' || lens === 'teeth' || lens === 'xray') return { lens, layers: { ...LENS_LAYERS[lens] } }
      return { lens }
    }),
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setShowNumbers: (showNumbers) => set({ showNumbers }),
  setShowMapping: (showMapping) => set({ showMapping }),
  setIsolate: (isolate) => set({ isolate }),
  hover: (hoveredFdi) => set({ hoveredFdi }),
  select: (selectedFdi) => set({ selectedFdi, autoRotate: false }),
  replayReveal: () => set((s) => ({ revealKey: s.revealKey + 1, selectedFdi: null, autoRotate: true, view: s.view === 'image' ? 'model' : s.view })),
  setAutoRotate: (autoRotate) => set({ autoRotate }),
  resetCamera: () => set((s) => ({ cameraResetKey: s.cameraResetKey + 1, selectedFdi: null })),
}))

// exposed for automated visual checks
if (typeof window !== 'undefined') (window as unknown as { __mtStore: unknown }).__mtStore = useMouthTwin
