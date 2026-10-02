/**
 * Landing-page hero: the real reference dentition (ToothFairy2 CBCT, CC BY-SA 4.0) as an object you can handle.
 * Drag to turn it, hover a tooth to read its name, click to pin it. Nothing here comes from a patient.
 */
import { Environment, Lightformer, OrbitControls, useGLTF } from '@react-three/drei'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { Suspense, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { crownHeight, toothFrame } from './restorationGeometry'

const URL_GLB = `${import.meta.env.BASE_URL}models/reference-anatomy.glb`
const ENAMEL = new THREE.Color('#f4efe6')
const ROOT = new THREE.Color('#d9c29a')
const HOVER = new THREE.Color('#86c5d8')

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

interface Props {
  hovered: number | null
  selected: number | null
  onHover: (fdi: number | null, x?: number, y?: number) => void
  onSelect: (fdi: number | null) => void
  showBone: boolean
}

function Dentition({ hovered, selected, onHover, onSelect, showBone }: Props) {
  const gltf = useGLTF(URL_GLB)
  const group = useRef<THREE.Group>(null)
  const built = useMemo(() => {
    const root = gltf.scene.clone(true)
    const teeth = new Map<number, THREE.MeshPhysicalMaterial>()
    const bone: THREE.Mesh[] = []
    const boneMat = new THREE.MeshPhysicalMaterial({
      color: '#c9d3d9', roughness: 0.55, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.FrontSide,
    })
    const canalMat = new THREE.MeshStandardMaterial({ color: '#e5c36a', emissive: '#a77d22', emissiveIntensity: 0.4, transparent: true, opacity: 0.55 })
    const box = new THREE.Box3()
    root.updateMatrixWorld(true)
    root.traverse((o) => {
      const m = o as THREE.Mesh
      if (!m.isMesh) return
      if (m.name.startsWith('tooth_')) {
        const fdi = Number(m.name.slice(6))
        const geo = m.geometry.clone()
        m.geometry = geo
        const { sign, origin } = toothFrame(geo, fdi)
        const crown = crownHeight(fdi)
        const pos = geo.attributes.position as THREE.BufferAttribute
        const col = new Float32Array(pos.count * 3)
        const c = new THREE.Color()
        for (let i = 0; i < pos.count; i++) {
          const depth = (pos.getY(i) - origin) * sign
          c.copy(ENAMEL).lerp(ROOT, smooth(crown - 2, crown + 3, depth))
          col.set([c.r, c.g, c.b], i * 3)
        }
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
        const mat = new THREE.MeshPhysicalMaterial({
          vertexColors: true, roughness: 0.28, metalness: 0, clearcoat: 0.85, clearcoatRoughness: 0.18,
          sheen: 0.4, sheenColor: new THREE.Color('#fff6e8'), emissive: HOVER, emissiveIntensity: 0,
        })
        m.material = mat
        teeth.set(fdi, mat)
        geo.computeBoundingBox()
        box.union((geo.boundingBox as THREE.Box3).clone().applyMatrix4(m.matrixWorld))
      } else if (m.name === 'mandible') {
        m.material = boneMat
        m.raycast = () => {}
        m.renderOrder = 2
        bone.push(m)
      } else if (m.name.startsWith('canal')) {
        m.material = canalMat
        m.raycast = () => {}
        bone.push(m)
      } else {
        m.visible = false // maxilla (cut flat by the scan's field of view) and sinuses stay out of the hero
        m.raycast = () => {}
      }
    })
    const center = box.getCenter(new THREE.Vector3())
    return { root, teeth, bone, boneMat, canalMat, center }
  }, [gltf])

  useEffect(() => () => {
    built.teeth.forEach((m) => m.dispose())
    built.boneMat.dispose(); built.canalMat.dispose()
  }, [built])

  useEffect(() => {
    built.bone.forEach((m) => (m.visible = showBone))
  }, [built, showBone])

  useFrame((state, delta) => {
    for (const [fdi, mat] of built.teeth) {
      const target = fdi === selected ? 0.32 : fdi === hovered ? 0.2 : 0
      mat.emissiveIntensity = THREE.MathUtils.damp(mat.emissiveIntensity, target, 12, delta)
    }
    if (group.current) {
      // entrance: rise into place once
      const t = Math.min(1, state.clock.elapsedTime / 1.6)
      const e = 1 - Math.pow(1 - t, 3)
      group.current.position.y = -built.center.y - (1 - e) * 14
      group.current.scale.setScalar(0.94 + 0.06 * e)
    }
  })

  const fdiOf = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.object.name.startsWith('tooth_') ? Number(e.object.name.slice(6)) : null)

  return (
    <group ref={group} position={[-built.center.x, -built.center.y, -built.center.z]}>
      <primitive
        object={built.root}
        onPointerMove={(e: ThreeEvent<PointerEvent>) => {
          const f = fdiOf(e)
          if (f === null) return
          e.stopPropagation()
          onHover(f, e.nativeEvent.clientX, e.nativeEvent.clientY)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          onHover(null)
          document.body.style.cursor = ''
        }}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          const f = fdiOf(e)
          if (f === null) return
          e.stopPropagation()
          onSelect(f === selected ? null : f)
        }}
      />
    </group>
  )
}

export function HeroAnatomy(props: Props & { interacting: (on: boolean) => void }) {
  const { interacting, ...rest } = props
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  // on touch screens a drag must scroll the page, so the jaw only turns by itself and taps select teeth
  const coarse = typeof window !== 'undefined' && window.matchMedia?.('(pointer: coarse)').matches
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ fov: 28, position: [0, 70, 215] }}
      gl={{ antialias: true, alpha: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.08 }}
      onPointerMissed={() => props.onSelect(null)}
    >
      <ambientLight intensity={0.3} />
      <hemisphereLight args={['#e8f0f3', '#20170f', 0.6]} />
      <directionalLight position={[50, 110, 90]} intensity={2.2} />
      <directionalLight position={[-80, 40, -60]} intensity={1.1} color="#9cc9d9" />
      <directionalLight position={[0, -60, 80]} intensity={0.4} color="#f0d9c0" />
      <Environment resolution={128} frames={1}>
        <Lightformer form="rect" intensity={2.4} position={[0, 60, 40]} scale={[120, 30, 1]} />
        <Lightformer form="rect" intensity={1.2} color="#9cc9d9" position={[-90, 10, -40]} scale={[60, 60, 1]} />
        <Lightformer form="ring" intensity={0.8} position={[80, 20, 60]} scale={30} />
      </Environment>
      <Suspense fallback={null}>
        <Dentition {...rest} />
      </Suspense>
      <OrbitControls
        makeDefault
        enableZoom={false}
        enablePan={false}
        enableRotate={!coarse}
        autoRotate={!reduced && props.selected === null}
        autoRotateSpeed={0.55}
        minPolarAngle={0.35}
        maxPolarAngle={2.2}
        rotateSpeed={0.6}
        onStart={() => interacting(true)}
        onEnd={() => interacting(false)}
      />
    </Canvas>
  )
}

useGLTF.preload(URL_GLB)
