import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { TEETH } from '../../data/fdi'
import { archPose } from './archLayout'
import { getToothGeometry } from './toothGeometry'

/** Landing-page backdrop: the template arch drawn as a slowly turning radiographic wireframe. */
function Arch() {
  const group = useRef<THREE.Group>(null)
  const mats = useMemo(
    () => ({
      wire: new THREE.MeshBasicMaterial({ color: '#86c5d8', wireframe: true, transparent: true, opacity: 0.13, depthWrite: false }),
      body: new THREE.MeshBasicMaterial({ color: '#86c5d8', transparent: true, opacity: 0.05, depthWrite: false, blending: THREE.AdditiveBlending }),
    }),
    [],
  )
  const teeth = useMemo(() => TEETH.map((t) => ({ t, pose: archPose(t), geo: getToothGeometry(t) })), [])
  useFrame((state) => {
    if (!group.current) return
    const e = state.clock.elapsedTime
    group.current.rotation.y = e * 0.12
    group.current.position.y = Math.sin(e * 0.5) * 1.2
  })
  return (
    <group ref={group} rotation={[0.35, 0, 0]}>
      {teeth.map(({ t, pose, geo }) => (
        <group key={t.fdi} position={pose.position} quaternion={pose.quaternion}>
          <group scale={pose.mirror}>
            <mesh geometry={geo.crown} material={mats.wire} />
            <mesh geometry={geo.roots} material={mats.wire} />
            <mesh geometry={geo.crown} material={mats.body} />
            <mesh geometry={geo.roots} material={mats.body} />
          </group>
        </group>
      ))}
    </group>
  )
}

export function ArchPreview() {
  return (
    <Canvas dpr={[1, 1.75]} camera={{ fov: 30, position: [0, 38, 150] }} gl={{ antialias: true, alpha: true }}>
      <group rotation={[0.18, 0, 0]}>
        <Arch />
      </group>
    </Canvas>
  )
}
