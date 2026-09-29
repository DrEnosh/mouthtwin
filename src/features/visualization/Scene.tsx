import { Environment, Grid, Lightformer } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Suspense, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult } from '../../types/inference'
import { AnatomyModel } from './AnatomyModel'
import { CameraRig } from './CameraRig'
import { OpgWrap } from './OpgWrap'
import { Relief } from './Relief'
import { linkAnchors, reveal, skipReveal, startReveal, tickReveal, toothWorldCenter } from './runtime'

function RevealDriver({ animate }: { animate: boolean }) {
  const revealKey = useMouthTwin((s) => s.revealKey)
  useLayoutEffect(() => {
    if (animate) startReveal(performance.now())
    else skipReveal()
  }, [revealKey, animate])
  useFrame(() => tickReveal(performance.now()))
  return null
}

/** Projects the active tooth to page coordinates for the split-view link line. */
function ProjectionTracker() {
  const { camera, gl } = useThree()
  const v = useMemo(() => new THREE.Vector3(), [])
  useFrame(() => {
    const s = useMouthTwin.getState()
    const fdi = s.hoveredFdi ?? s.selectedFdi
    if (s.view !== 'split' || fdi === null || reveal.running) {
      linkAnchors.model.visible = false
      return
    }
    const c = toothWorldCenter(fdi, v)
    if (!c) return
    c.project(camera)
    const rect = gl.domElement.getBoundingClientRect()
    linkAnchors.model.x = rect.left + ((c.x + 1) / 2) * rect.width
    linkAnchors.model.y = rect.top + ((1 - c.y) / 2) * rect.height
    linkAnchors.model.visible = c.z < 1 && c.x > -1.05 && c.x < 1.05
  })
  return null
}

function Floor() {
  const ref = useRef<THREE.Group>(null)
  useFrame(() => {
    if (ref.current) ref.current.visible = reveal.morph > 0.6
  })
  return (
    <group ref={ref}>
      <Grid
        position={[0, -46, 0]}
        args={[400, 400]}
        cellSize={5}
        cellThickness={0.5}
        cellColor="#1b2429"
        sectionSize={25}
        sectionThickness={0.9}
        sectionColor="#24323a"
        fadeDistance={260}
        fadeStrength={1.6}
        infiniteGrid
      />
    </group>
  )
}

export function MouthTwinScene({
  result,
  imageUrl,
  image,
  animate,
}: {
  result: InferenceResult
  imageUrl: string
  image: HTMLImageElement
  animate: boolean
}) {
  const select = useMouthTwin((s) => s.select)
  const showRelief = useMouthTwin((s) => s.layers.relief)
  const hasFlatStart = animate

  return (
    <Canvas
      className="!absolute inset-0"
      dpr={[1, 2]}
      camera={{ fov: 32, near: 1, far: 2000, position: [0, 0, 210] }}
      gl={{ antialias: true, alpha: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.05 }}
      onPointerMissed={() => select(null)}
    >
      <RevealDriver animate={hasFlatStart} />
      <ambientLight intensity={0.35} />
      <hemisphereLight args={['#dfe9ee', '#1a1410', 0.55]} />
      <directionalLight position={[60, 120, 110]} intensity={2.1} />
      <directionalLight position={[-90, 30, -80]} intensity={0.9} color="#9cc9d9" />
      <directionalLight position={[0, -80, 60]} intensity={0.35} color="#f0d9c0" />
      <Environment resolution={128} frames={1}>
        <Lightformer form="rect" intensity={2.2} position={[0, 60, 40]} scale={[120, 30, 1]} />
        <Lightformer form="rect" intensity={1.1} color="#9cc9d9" position={[-90, 10, -40]} scale={[60, 60, 1]} />
        <Lightformer form="ring" intensity={0.8} position={[80, 20, 60]} scale={30} />
      </Environment>

      <Suspense fallback={null}>
        <AnatomyModel result={result} />
      </Suspense>
      {showRelief && (
        <Suspense fallback={null}>
          <Relief result={result} imageUrl={imageUrl} image={image} interactive={false} />
        </Suspense>
      )}

      {hasFlatStart && (
        <Suspense fallback={null}>
          <OpgWrap url={imageUrl} result={result} />
        </Suspense>
      )}
      <Floor />
      <CameraRig hasFlatStart={hasFlatStart} />
      <ProjectionTracker />
    </Canvas>
  )
}
