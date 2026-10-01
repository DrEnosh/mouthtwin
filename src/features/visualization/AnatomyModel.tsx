/**
 * Reference anatomy from a real CBCT (ToothFairy2, CC BY-SA 4.0), fitted to the patient's OPG.
 *
 * What comes from the OPG (via the tooth model): which teeth are present, each tooth's relative
 * length and its mesiodistal tilt. What comes from the reference CBCT: the 3D shape of every tooth,
 * root and bone, and the arch form. The UI labels it that way.
 */
import { Html, useGLTF } from '@react-three/drei'
import { useFrame, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult } from '../../types/inference'
import { OPG_ORDER_LOWER, OPG_ORDER_UPPER } from '../../data/fdi'
import { reveal, toothAnchors } from './runtime'
import {
  CAP_COLOR, IMPLANT_COLOR, addRestorationAttributes, canalPaths, crownHeight, extractCrown, implantGeometry, makeRestoUniforms,
  patchToothMaterial, taperedTube, toothFrame, type RestoUniforms,
} from './restorationGeometry'

export const ANATOMY_URL = `${import.meta.env.BASE_URL}models/reference-anatomy.glb`

const ENAMEL = new THREE.Color('#f3eee4')
const ROOT = new THREE.Color('#dcc7a1')
const ACCENT = new THREE.Color('#86c5d8')
const UP = new THREE.Vector3(0, 1, 0)

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

interface ToothPart {
  fdi: number
  mesh: THREE.Mesh
  material: THREE.MeshPhysicalMaterial
  upper: boolean
  /** shader uniforms for crown / filling / translucent root */
  resto: RestoUniforms
  /** what the X-ray showed, as target values for the uniforms */
  want: { cap: number; fill: number; rootAlpha: number }
}

/** Extra objects drawn for restorations (canal fillings, implants, bridge bars). */
interface Extra {
  obj: THREE.Object3D
  mats: THREE.Material[]
  /** tooth or slot this belongs to; null when it has no position in the chart */
  fdi: number | null
  /** follows the tooth layer (canal fillings) instead of only the restoration layer */
  withTeeth: boolean
}

/** Long-axis angle from vertical of a 2D point set (image coords, y down). Positive = apex toward +x. */
function imageTilt(points: [number, number][], aspect: number): number | null {
  if (points.length < 6) return null
  let mx = 0, my = 0
  for (const [x, y] of points) { mx += x * aspect; my += y }
  mx /= points.length; my /= points.length
  let sxx = 0, syy = 0, sxy = 0
  for (const [x, y] of points) {
    const dx = x * aspect - mx, dy = y - my
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy) // major axis angle from +x
  const ax = Math.cos(angle), ay = Math.sin(angle)
  const dir = ay >= 0 ? [ax, ay] : [-ax, -ay] // point it downwards (image +y)
  return Math.atan2(dir[0], dir[1])
}

/** Same measure for a mesh, projected on the plane spanned by the arch tangent and vertical. */
function meshTilt(geo: THREE.BufferGeometry, tangent: THREE.Vector3): number {
  const pos = geo.attributes.position as THREE.BufferAttribute
  const pts: [number, number][] = []
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i += 3) {
    v.fromBufferAttribute(pos, i)
    pts.push([v.dot(tangent), -v.y]) // image-like: x along tangent, y down
  }
  return imageTilt(pts, 1) ?? 0
}

export function AnatomyModel({ result }: { result: InferenceResult }) {
  const gltf = useGLTF(ANATOMY_URL)
  const layers = useMouthTwin((s) => s.layers)
  const showNumbers = useMouthTwin((s) => s.showNumbers)
  const hoveredFdi = useMouthTwin((s) => s.hoveredFdi)
  const selectedFdi = useMouthTwin((s) => s.selectedFdi)
  const hover = useMouthTwin((s) => s.hover)
  const select = useMouthTwin((s) => s.select)
  const labelEls = useRef(new Map<number, HTMLDivElement>())
  const toCam = useMemo(() => new THREE.Vector3(), [])

  const detections = useMemo(() => new Map(result.teeth.map((d) => [d.fdi, d])), [result])

  const built = useMemo(() => {
    const root = gltf.scene.clone(true)
    const teeth: ToothPart[] = []
    const bone: THREE.Mesh[] = []
    const canals: THREE.Mesh[] = []
    const sinus: THREE.Mesh[] = []
    const extras: Extra[] = []
    const refGeo = new Map<number, THREE.BufferGeometry>()

    const boneMat = new THREE.MeshPhysicalMaterial({
      color: '#d3dbe0', roughness: 0.5, metalness: 0, transparent: true, opacity: 0.24, depthWrite: false,
      clearcoat: 0.3, side: THREE.FrontSide,
    })
    // the reference CBCT's field of view cuts the upper jaw flat; fade the top instead of showing the cut
    const topY = { value: 30 }
    for (const mat of [boneMat] as THREE.Material[]) {
      mat.onBeforeCompile = (shader) => {
        shader.uniforms.uTopY = topY
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', '#include <common>\nvarying float vObjY;')
          .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObjY = position.y;')
        shader.fragmentShader = shader.fragmentShader
          .replace('#include <common>', '#include <common>\nvarying float vObjY;\nuniform float uTopY;')
          .replace('#include <opaque_fragment>', 'diffuseColor.a *= 1.0 - smoothstep(uTopY - 12.0, uTopY - 1.0, vObjY);\n#include <opaque_fragment>')
      }
      mat.customProgramCacheKey = () => 'bone-fade'
    }
    const canalMat = new THREE.MeshStandardMaterial({ color: '#e7c35f', emissive: '#c99a2e', emissiveIntensity: 0.35, roughness: 0.4, transparent: true })
    const sinusMat = new THREE.MeshPhysicalMaterial({ color: '#7fb6c9', transparent: true, opacity: 0.12, depthWrite: false, roughness: 0.3 })

    // arch centre in the horizontal plane, for outward (buccal) directions
    const centers: THREE.Vector3[] = []
    root.traverse((o) => {
      if ((o as THREE.Mesh).isMesh && o.name.startsWith('tooth_')) {
        const b = new THREE.Box3().setFromObject(o)
        centers.push(b.getCenter(new THREE.Vector3()))
      }
    })
    const archCenter = centers.reduce((a, c) => a.add(c), new THREE.Vector3()).divideScalar(Math.max(1, centers.length))
    archCenter.z -= 12

    // reference lengths per tooth (for relative scaling against the OPG)
    const refLen = new Map<number, number>()
    root.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.isMesh && m.name.startsWith('tooth_')) {
        m.geometry.computeBoundingBox()
        const bb = m.geometry.boundingBox as THREE.Box3
        refLen.set(Number(m.name.slice(6)), bb.max.y - bb.min.y)
      }
    })
    const detected = result.teeth.filter((d) => refLen.has(d.fdi))
    const imgAspect = result.image.width / result.image.height
    const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 1
    const medOpg = median(detected.map((d) => d.bbox[3]))
    const medRef = median(detected.map((d) => refLen.get(d.fdi) as number))

    root.traverse((o) => {
      const m = o as THREE.Mesh
      if (!m.isMesh) return
      const name = m.name
      if (name.startsWith('tooth_')) {
        const fdi = Number(name.slice(6))
        const upper = fdi < 30
        const geo = m.geometry.clone()
        m.geometry = geo
        const depthAttr = addRestorationAttributes(geo, fdi)
        const refPaths = detections.get(fdi)?.restorations?.rootCanal ? canalPaths(geo, fdi) : []
        // enamel-to-root colour: whiter near the occlusal plane, warmer toward the apex
        const pos = geo.attributes.position as THREE.BufferAttribute
        const col = new Float32Array(pos.count * 3)
        const c = new THREE.Color()
        const crownH = crownHeight(fdi)
        for (let i = 0; i < pos.count; i++) {
          c.copy(ENAMEL).lerp(ROOT, smooth(crownH - 2, crownH + 2, depthAttr[i]))
          col.set([c.r, c.g, c.b], i * 3)
        }
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
        const material = new THREE.MeshPhysicalMaterial({
          vertexColors: true, roughness: 0.32, metalness: 0, clearcoat: 0.7, clearcoatRoughness: 0.25,
          sheen: 0.3, emissive: ACCENT, emissiveIntensity: 0, transparent: false,
        })
        const resto = makeRestoUniforms()
        patchToothMaterial(material, resto)
        m.material = material
        const rs = detections.get(fdi)?.restorations
        const want = {
          cap: rs?.crown?.kind === 'cap' || rs?.implant ? 1 : 0,
          fill: rs?.crown?.kind === 'filling' ? 1 : 0,
          rootAlpha: rs?.implant ? 1 : rs?.rootCanal ? 0.6 : 0,
        }

        // fit to the OPG: relative length and mesiodistal tilt, pivoting at the crown
        const det = detections.get(fdi)
        geo.computeBoundingBox()
        const bb = geo.boundingBox as THREE.Box3
        const center = bb.getCenter(new THREE.Vector3())
        const out = center.clone().sub(archCenter).setY(0).normalize()
        const tangent = new THREE.Vector3().crossVectors(UP, out).normalize() // points patient right → left at the front
        if (det) {
          const len = refLen.get(fdi) as number
          const s = THREE.MathUtils.clamp(det.bbox[3] / medOpg / (len / medRef), 0.85, 1.18)
          const pivotY = upper ? bb.min.y : bb.max.y // occlusal end stays in occlusion
          const tImg = imageTilt(det.polygon as [number, number][], imgAspect)
          const tRef = meshTilt(geo, tangent)
          const dTilt = tImg === null ? 0 : THREE.MathUtils.clamp(0.7 * (tImg - tRef), -0.2, 0.2)
          const pivot = new THREE.Vector3(center.x, pivotY, center.z)
          const mtx = new THREE.Matrix4()
            .makeTranslation(pivot.x, pivot.y, pivot.z)
            .multiply(new THREE.Matrix4().makeRotationAxis(out, dTilt))
            .multiply(new THREE.Matrix4().makeScale(1, s, 1))
            .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z))
          geo.applyMatrix4(mtx)
          geo.computeVertexNormals()
          geo.computeBoundingBox()
          for (const path of refPaths) {
            const pts = path.map((p) => p.clone().applyMatrix4(mtx))
            const canalGeo = taperedTube(pts, 0.8, 0.32)
            const mat = new THREE.MeshStandardMaterial({
              color: '#f7f0e4', emissive: '#ffcf8a', emissiveIntensity: 0.55, roughness: 0.35, transparent: true,
            })
            const cm = new THREE.Mesh(canalGeo, mat)
            cm.name = `resto_canal_${fdi}`
            cm.raycast = () => {}
            cm.renderOrder = 1
            extras.push({ obj: cm, mats: [mat], fdi, withTeeth: true })
          }
        }
        const b2 = geo.boundingBox as THREE.Box3
        const c2 = b2.getCenter(new THREE.Vector3())
        const labelY = upper ? b2.max.y + 5 : b2.min.y - 5
        toothAnchors.set(fdi, {
          center: c2,
          normal: out,
          label: new THREE.Vector3(c2.x, labelY, c2.z).addScaledVector(out, 3),
          upper,
        })
        m.visible = Boolean(det)
        refGeo.set(fdi, geo)
        teeth.push({ fdi, mesh: m, material, upper, resto, want })
      } else if (name === 'mandible' || name === 'maxilla') {
        m.raycast = () => {}
        m.material = boneMat
        m.renderOrder = 2
        bone.push(m)
      } else if (name.startsWith('canal')) {
        m.raycast = () => {}
        m.material = canalMat
        canals.push(m)
      } else if (name.startsWith('sinus')) {
        m.raycast = () => {}
        m.material = sinusMat
        m.renderOrder = 1
        sinus.push(m)
      }
    })
    // implants: fixture + abutment (+ crown when the X-ray shows one), at the tooth position they replace
    const capMat = () =>
      new THREE.MeshPhysicalMaterial({ color: CAP_COLOR, metalness: 0.72, roughness: 0.26, clearcoat: 0.5, transparent: true })
    const implantMat = () =>
      new THREE.MeshPhysicalMaterial({ color: IMPLANT_COLOR, metalness: 0.95, roughness: 0.32, transparent: true })
    const placeImplant = (fdi: number, crowned: boolean) => {
      const g = refGeo.get(fdi)
      if (!g) return
      g.computeBoundingBox()
      const bb = g.boundingBox as THREE.Box3
      const c = bb.getCenter(new THREE.Vector3())
      const upper = fdi < 30
      const { origin, crown } = toothFrame(g, fdi)
      const molar = fdi % 10 >= 6, incisor = fdi % 10 <= 2
      const { fixture, abutment } = implantGeometry(molar ? 2.4 : incisor ? 1.7 : 2.0)
      const grp = new THREE.Group()
      const fm = implantMat(), am = implantMat()
      const f = new THREE.Mesh(fixture, fm), a = new THREE.Mesh(abutment, am)
      f.raycast = () => {}; a.raycast = () => {}
      grp.add(f, a)
      const mats: THREE.Material[] = [fm, am]
      if (crowned) {
        const cg = extractCrown(g, fdi) // y = depth from the occlusal end, crest at y = crown
        // implant frame: crest at y = 0, crown above (+y)
        cg.translate(-c.x, -crown, -c.z)
        cg.scale(1, -1, 1)
        const cmat = capMat()
        const cm = new THREE.Mesh(cg, cmat)
        cm.raycast = () => {}
        grp.add(cm)
        mats.push(cmat)
      }
      grp.position.set(c.x, origin + (upper ? crown : -crown), c.z)
      if (upper) grp.rotation.z = Math.PI
      grp.name = `resto_implant_${fdi}`
      extras.push({ obj: grp, mats, fdi, withTeeth: false })
    }
    for (const im of result.implants ?? []) if (im.slot !== null && !detections.has(im.slot)) placeImplant(im.slot, im.crowned)
    for (const d of result.teeth) if (d.restorations?.implant) placeImplant(d.fdi, false)

    // bridges: bars joining neighbouring crowns that share one radiopaque block
    const bridges = new Map<number, number[]>()
    for (const d of result.teeth) {
      const id = d.restorations?.bridgeId
      if (id) bridges.set(id, [...(bridges.get(id) ?? []), d.fdi])
    }
    const crownCentre = (fdi: number) => {
      const g = refGeo.get(fdi)
      if (!g) return null
      g.computeBoundingBox()
      const c = (g.boundingBox as THREE.Box3).getCenter(new THREE.Vector3())
      const { origin, sign } = toothFrame(g, fdi)
      c.y = origin + sign * 3.6
      return c
    }
    for (const members of bridges.values()) {
      for (const [a, b] of [[OPG_ORDER_UPPER, 0], [OPG_ORDER_LOWER, 0]] as [number[], number][]) {
        void b
        const idx = members.map((f) => a.indexOf(f)).filter((i) => i >= 0).sort((x, y) => x - y)
        for (let k = 0; k + 1 < idx.length; k++) {
          if (idx[k + 1] - idx[k] !== 1) continue
          const p = crownCentre(a[idx[k]]), q = crownCentre(a[idx[k + 1]])
          if (!p || !q) continue
          const len = p.distanceTo(q)
          const bar = new THREE.CylinderGeometry(1.25, 1.25, len, 14)
          const mat = capMat()
          const mesh = new THREE.Mesh(bar, mat)
          mesh.position.copy(p).add(q).multiplyScalar(0.5)
          mesh.quaternion.setFromUnitVectors(UP, q.clone().sub(p).normalize())
          mesh.raycast = () => {}
          mesh.name = `resto_bridge_${a[idx[k]]}`
          extras.push({ obj: mesh, mats: [mat], fdi: a[idx[k]], withTeeth: false })
        }
      }
    }
    for (const e of extras) root.add(e.obj)

    const maxilla = bone.find((m) => m.name === 'maxilla')
    if (maxilla) {
      maxilla.geometry.computeBoundingBox()
      topY.value = (maxilla.geometry.boundingBox as THREE.Box3).max.y
    }
    // anchors only for teeth the OPG actually shows
    for (const t of teeth) if (!detections.has(t.fdi)) toothAnchors.delete(t.fdi)
    return { root, teeth, bone, canals, sinus, boneMat, canalMat, sinusMat, extras }
  }, [gltf, detections, result])

  useEffect(
    () => () => {
      toothAnchors.clear()
      built.teeth.forEach((t) => { t.material.dispose(); t.mesh.geometry.dispose() })
      built.boneMat.dispose(); built.canalMat.dispose(); built.sinusMat.dispose()
      built.extras.forEach((e) => { e.mats.forEach((m) => m.dispose()); e.obj.traverse((o) => (o as THREE.Mesh).geometry?.dispose()) })
    },
    [built],
  )

  useEffect(() => {
    built.bone.forEach((m) => (m.visible = layers.bone))
    built.canals.forEach((m) => (m.visible = layers.canals))
    built.sinus.forEach((m) => (m.visible = layers.sinus))
  }, [built, layers])

  useFrame((state, delta) => {
    const s = useMouthTwin.getState()
    const k = smooth(0.35, 1, reveal.morph)
    const isolating = s.isolate && s.selectedFdi !== null
    const pulse = 0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 3.2)
    for (const t of built.teeth) {
      if (!detections.has(t.fdi)) continue
      t.mesh.visible = s.layers.teeth
      const dim = isolating && t.fdi !== s.selectedFdi
      const target = (dim ? 0.1 : 1) * k
      const mat = t.material
      mat.opacity = reveal.running ? target : THREE.MathUtils.damp(mat.opacity, target, 10, delta)
      const on = s.layers.restorations ? 1 : 0
      t.resto.uCap.value = t.want.cap * on
      t.resto.uFill.value = t.want.fill * on
      t.resto.uRootAlpha.value = t.want.rootAlpha * on
      const seeThrough = t.want.rootAlpha * on > 0
      const tr = mat.opacity < 0.995 || seeThrough
      const dw = seeThrough ? false : !tr || mat.opacity > 0.6
      if (mat.transparent !== tr || mat.depthWrite !== dw) { mat.transparent = tr; mat.depthWrite = dw; mat.needsUpdate = true }
      const glow = t.fdi === s.selectedFdi ? 0.18 + 0.14 * pulse : t.fdi === s.hoveredFdi ? 0.18 : 0
      mat.emissiveIntensity = THREE.MathUtils.damp(mat.emissiveIntensity, glow, 12, delta)
    }
    for (const e of built.extras) {
      e.obj.visible = s.layers.restorations && (!e.withTeeth || s.layers.teeth)
      const dim = isolating && (e.fdi === null || e.fdi !== s.selectedFdi)
      const target = (dim ? 0.1 : 1) * k
      for (const m of e.mats) {
        const mm = m as THREE.MeshStandardMaterial
        mm.opacity = reveal.running ? target : THREE.MathUtils.damp(mm.opacity, target, 10, delta)
        mm.depthWrite = mm.opacity > 0.6
      }
    }
    built.boneMat.opacity = (isolating ? 0.08 : 0.24) * k
    built.canalMat.opacity = (isolating ? 0.3 : 1) * k
    built.sinusMat.opacity = (isolating ? 0.05 : 0.12) * k
    // hide labels of teeth on the far side of the arch (HTML labels are not depth-tested)
    for (const [fdi, el] of labelEls.current) {
      const a = toothAnchors.get(fdi)
      if (!a) continue
      toCam.copy(state.camera.position).sub(a.label)
      el.style.opacity = toCam.dot(a.normal) > 0 || fdi === s.selectedFdi ? '1' : '0'
    }
  })

  const fdiOf = (e: ThreeEvent<PointerEvent | MouseEvent>) => {
    const n = e.object.name
    return n.startsWith('tooth_') ? Number(n.slice(6)) : null
  }

  const labels = result.teeth.filter((t) => toothAnchors.has(t.fdi) && (showNumbers || t.fdi === hoveredFdi || t.fdi === selectedFdi))

  return (
    <group>
      <primitive
        object={built.root}
        onPointerOver={(e: ThreeEvent<PointerEvent>) => {
          const f = fdiOf(e)
          if (f === null || reveal.running) return
          e.stopPropagation()
          hover(f)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={(e: ThreeEvent<PointerEvent>) => {
          if (fdiOf(e) !== null) {
            hover(null)
            document.body.style.cursor = ''
          }
        }}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          const f = fdiOf(e)
          if (f === null || reveal.running) return
          e.stopPropagation()
          select(f === useMouthTwin.getState().selectedFdi ? null : f)
        }}
      />
      {!reveal.running &&
        labels.map((t) => {
          const a = toothAnchors.get(t.fdi) as NonNullable<ReturnType<typeof toothAnchors.get>>
          return (
            <Html key={t.fdi} position={a.label} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
              <div
                ref={(el) => {
                  if (el) labelEls.current.set(t.fdi, el)
                  else labelEls.current.delete(t.fdi)
                }}
                className={`fdi-chip ${t.fdi === selectedFdi ? 'is-selected' : ''} ${t.fdi === hoveredFdi ? 'is-hovered' : ''}`}
              >
                {t.fdi}
              </div>
            </Html>
          )
        })}
    </group>
  )
}

