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
  /** teeth to light up (e.g. all molars) */
  highlight?: ReadonlySet<number> | null
  /** quiz feedback: the tooth clicked and whether it was right; `key` restarts the flash */
  flash?: { fdi: number; ok: boolean; answer: number; key: number } | null
  onHover?: (fdi: number | null, x?: number, y?: number) => void
  onSelect?: (fdi: number | null) => void
  interacting?: (on: boolean) => void
}

const WIRE = new THREE.Color('#86c5d8')
const LIT = new THREE.Color('#d6f1fa')
const GOOD = new THREE.Color('#8fe0ae')
const BAD = new THREE.Color('#f08a78')
const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

/** Vertical light that sweeps across the arch once, like a panoramic scanner arm. */
function makeSweepMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uAlpha: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
    fragmentShader:
      'varying vec2 vUv; uniform float uAlpha; void main(){ float x = abs(vUv.x - 0.5) * 2.0; float core = exp(-x * x * 40.0); float glow = exp(-x * x * 4.0) * 0.25; float fade = smoothstep(0.0, 0.25, vUv.y) * smoothstep(1.0, 0.75, vUv.y); float a = (core + glow) * fade * uAlpha; gl_FragColor = vec4(vec3(0.62, 0.86, 0.95), a); }',
  })
}

function Arch({ hovered = null, selected = null, highlight = null, flash = null, onHover, onSelect }: ArchPreviewProps) {
  const group = useRef<THREE.Group>(null)
  const sweep = useRef<THREE.Mesh>(null)
  const refs = useRef<(THREE.Group | null)[]>([])
  const flashAt = useRef(-10)
  const reduced = useMemo(() => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches, [])
  const teeth = useMemo(
    () =>
      TEETH.map((t) => ({
        t,
        pose: archPose(t),
        geo: getToothGeometry(t),
        // reveal order: from the midline outwards, upper slightly before lower
        start: 0.2 + 0.11 * (t.position - 1) + (t.upper ? 0 : 0.06),
        wire: new THREE.MeshBasicMaterial({ color: WIRE.clone(), wireframe: true, transparent: true, opacity: 0, depthWrite: false }),
        body: new THREE.MeshBasicMaterial({ color: WIRE.clone(), transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }),
      })),
    [],
  )
  const sweepMat = useMemo(makeSweepMaterial, [])
  useEffect(() => () => { teeth.forEach((x) => { x.wire.dispose(); x.body.dispose() }); sweepMat.dispose() }, [teeth, sweepMat])
  useEffect(() => {
    if (flash) flashAt.current = -1 // set on next frame
  }, [flash])

  const tmp = useMemo(() => new THREE.Vector3(), [])
  useFrame((state, delta) => {
    const e = reduced ? 10 : state.clock.elapsedTime
    if (flashAt.current === -1) flashAt.current = state.clock.elapsedTime
    const sinceFlash = state.clock.elapsedTime - flashAt.current
    if (group.current) {
      if (selected === null && !flash) group.current.rotation.y += delta * 0.12
      group.current.position.y = Math.sin(e * 0.5) * 1.2
      // lean gently toward the pointer
      const tx = 0.35 + state.pointer.y * -0.08
      group.current.rotation.x = THREE.MathUtils.damp(group.current.rotation.x, tx, 3, delta)
      group.current.rotation.z = THREE.MathUtils.damp(group.current.rotation.z, state.pointer.x * 0.05, 3, delta)
    }
    // one scanner sweep after the teeth have arrived
    const sweepT = (e - 1.7) / 1.5
    const sweepX = -78 + 156 * sweepT
    if (sweep.current) {
      sweep.current.visible = sweepT > 0 && sweepT < 1
      sweep.current.position.x = sweepX
      sweepMat.uniforms.uAlpha.value = Math.sin(Math.PI * Math.min(1, Math.max(0, sweepT)))
    }
    teeth.forEach((x, i) => {
      const r = smooth(x.start, x.start + 0.55, e)
      const g = refs.current[i]
      let lit = 0
      if (g) {
        g.scale.setScalar(0.55 + 0.45 * r)
        if (sweep.current?.visible) {
          g.getWorldPosition(tmp)
          lit = Math.exp(-(((tmp.x - sweepX) / 9) ** 2))
        }
      }
      const isFlash = flash && sinceFlash < 1.6 && (x.t.fdi === flash.fdi || x.t.fdi === flash.answer)
      let on = x.t.fdi === selected ? 1 : x.t.fdi === hovered ? 0.7 : highlight?.has(x.t.fdi) ? 0.55 : 0
      on = Math.max(on, lit)
      const target = isFlash ? 1 : on
      x.wire.opacity = THREE.MathUtils.damp(x.wire.opacity, (0.13 + 0.5 * target) * r, 10, delta)
      x.body.opacity = THREE.MathUtils.damp(x.body.opacity, (0.05 + 0.2 * target) * r, 10, delta)
      if (isFlash) {
        const good = x.t.fdi === flash.answer
        x.wire.color.copy(good ? GOOD : BAD)
      } else x.wire.color.lerpColors(WIRE, LIT, on)
    })
  })

  const fdiOf = (e: ThreeEvent<PointerEvent | MouseEvent>) => {
    let o: THREE.Object3D | null = e.object
    while (o && o.userData.fdi === undefined) o = o.parent
    return o ? (o.userData.fdi as number) : null
  }

  return (
    <>
      <mesh ref={sweep} material={sweepMat} position={[0, 0, 30]} visible={false}>
        <planeGeometry args={[22, 120]} />
      </mesh>
      <group ref={group} rotation={[0.35, 0, 0]}>
        {teeth.map(({ t, pose, geo, wire, body }, i) => (
          <group
            key={t.fdi}
            ref={(g) => {
              refs.current[i] = g
            }}
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
              onSelect(f)
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
    </>
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
