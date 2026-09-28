import { Html } from '@react-three/drei'
import { useFrame, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { ToothRef } from '../../data/fdi'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult, ToothDetection } from '../../types/inference'
import { archPose, flatPose } from './archLayout'
import { reveal, toothObjects } from './runtime'
import { getToothGeometry } from './toothGeometry'

const ACCENT = new THREE.Color('#86c5d8')
const ENAMEL = new THREE.Color('#efe7d8')
const ROOT = new THREE.Color('#d9c6a2')

interface Props {
  tooth: ToothRef
  detection?: ToothDetection
  result: InferenceResult
}

export function Tooth({ tooth, detection, result }: Props) {
  const outer = useRef<THREE.Group>(null)
  const inner = useRef<THREE.Group>(null)
  const geo = getToothGeometry(tooth)
  const pose = useMemo(() => archPose(tooth), [tooth])
  const flat = useMemo(() => (detection ? flatPose(tooth, detection, result) : null), [tooth, detection, result])

  const crownMat = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: ENAMEL,
        roughness: 0.3,
        metalness: 0,
        clearcoat: 0.8,
        clearcoatRoughness: 0.22,
        sheen: 0.4,
        sheenColor: new THREE.Color('#ffffff'),
        emissive: ACCENT,
        emissiveIntensity: 0,
      }),
    [],
  )
  const rootMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: ROOT, roughness: 0.55, emissive: ACCENT, emissiveIntensity: 0 }),
    [],
  )
  useEffect(() => () => { crownMat.dispose(); rootMat.dispose() }, [crownMat, rootMat])

  useEffect(() => {
    const obj = inner.current
    if (!obj) return
    toothObjects.set(tooth.fdi, { object: obj, localCenter: geo.center })
    return () => { toothObjects.delete(tooth.fdi) }
  }, [tooth.fdi, geo.center])

  const hovered = useMouthTwin((s) => s.hoveredFdi === tooth.fdi)
  const selected = useMouthTwin((s) => s.selectedFdi === tooth.fdi)
  const layers = useMouthTwin((s) => s.layers)
  const showNumbers = useMouthTwin((s) => s.showNumbers)
  const hover = useMouthTwin((s) => s.hover)
  const select = useMouthTwin((s) => s.select)

  const tmpQ = useMemo(() => new THREE.Quaternion(), [])
  const one = useMemo(() => new THREE.Vector3(1, 1, 1), [])

  useFrame((state, delta) => {
    const o = outer.current
    if (!o) return
    const m = reveal.morph
    if (flat && m < 1) {
      o.position.lerpVectors(flat.position, pose.position, m)
      tmpQ.copy(flat.quaternion).slerp(pose.quaternion, m)
      o.quaternion.copy(tmpQ)
      o.scale.lerpVectors(flat.scale, one, m)
    } else {
      o.position.copy(pose.position)
      o.quaternion.copy(pose.quaternion)
      o.scale.copy(one)
    }

    const { selectedFdi, isolate } = useMouthTwin.getState()
    const dimmed = isolate && selectedFdi !== null && selectedFdi !== tooth.fdi
    const target = (dimmed ? 0.1 : 1) * (flat ? reveal.appear : 1)
    const opacity = reveal.running ? target : THREE.MathUtils.damp(crownMat.opacity, target, 10, delta)
    for (const mat of [crownMat, rootMat]) {
      mat.opacity = opacity
      const wantTransparent = opacity < 0.995
      if (mat.transparent !== wantTransparent) {
        mat.transparent = wantTransparent
        mat.depthWrite = !wantTransparent || opacity > 0.6
        mat.needsUpdate = true
      }
    }
    const pulse = 0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 3.2)
    const glow = selected ? 0.22 + 0.16 * pulse : hovered ? 0.2 : 0
    crownMat.emissiveIntensity = THREE.MathUtils.damp(crownMat.emissiveIntensity, glow, 12, delta)
    rootMat.emissiveIntensity = crownMat.emissiveIntensity * 0.8
  })

  const onOver = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    hover(tooth.fdi)
    document.body.style.cursor = 'pointer'
  }
  const onOut = () => {
    if (useMouthTwin.getState().hoveredFdi === tooth.fdi) hover(null)
    document.body.style.cursor = ''
  }
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    if (reveal.running) return
    select(selected ? null : tooth.fdi)
  }

  const labelY = -(tooth.crownMm + tooth.rootMm + 4)
  const labelVisible = layers.template && (showNumbers || hovered || selected)

  return (
    <group ref={outer}>
      <group ref={inner} scale={pose.mirror}>
        <mesh
          geometry={geo.crown}
          material={crownMat}
          visible={layers.template}
          onPointerOver={onOver}
          onPointerOut={onOut}
          onClick={onClick}
          castShadow
        />
        <mesh
          geometry={geo.roots}
          material={rootMat}
          visible={layers.template}
          onPointerOver={onOver}
          onPointerOut={onOut}
          onClick={onClick}
        />
        {labelVisible && (
          <Html position={[0, labelY, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`fdi-chip ${selected ? 'is-selected' : ''} ${hovered ? 'is-hovered' : ''}`}>{tooth.fdi}</div>
          </Html>
        )}
      </group>
    </group>
  )
}
