import { OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { TEETH } from '../../data/fdi'
import { archPose } from './archLayout'
import { getToothGeometry } from './toothGeometry'

export interface ArchPreviewProps {
  hovered?: number | null
  selected?: number | null
  onHover?: (fdi: number | null, x?: number, y?: number) => void
  onSelect?: (fdi: number | null) => void
  interacting?: (on: boolean) => void
}

const WIRE = new THREE.Color('#86c5d8')
const LIT = new THREE.Color('#d6f1fa')

/** Landing-page hero: the template arch as a slowly turning radiographic wireframe. Hover and click a tooth. */
function Arch({ hovered = null, selected = null, onHover, onSelect }: ArchPreviewProps) {
  const group = useRef<THREE.Group>(null)
  const teeth = useMemo(
    () =>
      TEETH.map((t) => ({
        t,
        pose: archPose(t),
        geo: getToothGeometry(t),
        wire: new THREE.MeshBasicMaterial({ color: WIRE.clone(), wireframe: true, transparent: true, opacity: 0.13, depthWrite: false }),
        body: new THREE.MeshBasicMaterial({ color: WIRE.clone(), transparent: true, opacity: 0.05, depthWrite: false, blending: THREE.AdditiveBlending }),
      })),
    [],
  )
  useEffect(() => () => teeth.forEach((x) => { x.wire.dispose(); x.body.dispose() }), [teeth])

  useFrame((state, delta) => {
    if (group.current) {
      const e = state.clock.elapsedTime
      if (selected === null) group.current.rotation.y += delta * 0.12
      group.current.position.y = Math.sin(e * 0.5) * 1.2
    }
    for (const x of teeth) {
      const on = x.t.fdi === selected ? 1 : x.t.fdi === hovered ? 0.7 : 0
      x.wire.opacity = THREE.MathUtils.damp(x.wire.opacity, 0.13 + 0.5 * on, 10, delta)
      x.body.opacity = THREE.MathUtils.damp(x.body.opacity, 0.05 + 0.2 * on, 10, delta)
      x.wire.color.lerpColors(WIRE, LIT, on)
    }
  })

  const fdiOf = (e: ThreeEvent<PointerEvent | MouseEvent>) => {
    let o: THREE.Object3D | null = e.object
    while (o && o.userData.fdi === undefined) o = o.parent
    return o ? (o.userData.fdi as number) : null
  }

  return (
    <group ref={group} rotation={[0.35, 0, 0]}>
      {teeth.map(({ t, pose, geo, wire, body }) => (
        <group
          key={t.fdi}
          userData={{ fdi: t.fdi }}
          position={pose.position}
          quaternion={pose.quaternion}
          onPointerMove={(e: ThreeEvent<PointerEvent>) => {
            const f = fdiOf(e)
            if (f === null || !onHover) return
            e.stopPropagation()
            onHover(f, e.nativeEvent.clientX, e.nativeEvent.clientY)
            document.body.style.cursor = 'pointer'
          }}
          onPointerOut={() => {
            onHover?.(null)
            document.body.style.cursor = ''
          }}
          onClick={(e: ThreeEvent<MouseEvent>) => {
            const f = fdiOf(e)
            if (f === null || !onSelect) return
            e.stopPropagation()
            onSelect(f === selected ? null : f)
          }}
        >
          <group scale={pose.mirror}>
            <mesh geometry={geo.crown} material={wire} />
            <mesh geometry={geo.roots} material={wire} />
            <mesh geometry={geo.crown} material={body} />
            <mesh geometry={geo.roots} material={body} />
          </group>
        </group>
      ))}
    </group>
  )
}

export function ArchPreview(props: ArchPreviewProps) {
  const coarse = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ fov: 30, position: [0, 38, 150] }}
      gl={{ antialias: true, alpha: true }}
      onPointerMissed={() => props.onSelect?.(null)}
    >
      <group rotation={[0.18, 0, 0]}>
        <Arch {...props} />
      </group>
      {props.onSelect && (
        <OrbitControls
          makeDefault
          enableZoom={false}
          enablePan={false}
          enableRotate={!coarse}
          rotateSpeed={0.5}
          minPolarAngle={0.6}
          maxPolarAngle={2.0}
          onStart={() => props.interacting?.(true)}
          onEnd={() => props.interacting?.(false)}
        />
      )}
    </Canvas>
  )
}
