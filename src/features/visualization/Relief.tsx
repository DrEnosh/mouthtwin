/**
 * The 3D model built from the uploaded OPG's own pixels.
 *
 * Two closed shells (teeth, bone) are extruded out of the image both ways along the arch normal,
 * so the outline of every tooth, root, gap and restoration is exactly what is in the picture.
 * The reveal grows the relief out of the flat image, then bends it onto the arch.
 */
import { Html } from '@react-three/drei'
import { useFrame, useLoader, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useMouthTwin } from '../../store/useMouthTwin'
import type { InferenceResult } from '../../types/inference'
import { archFrame, FOCAL_ARCH, PANO_WIDTH, panoHeight, panoToArcMapper } from './archLayout'
import { analyzeOpg, type ReliefData } from './reliefAnalysis'
import { reveal, toothAnchors } from './runtime'

interface Frame {
  /** flat and arch positions of the base surface per column/row */
  flatX: Float32Array
  archX: Float32Array
  archZ: Float32Array
  nX: Float32Array
  nZ: Float32Array
  flatY: (yn: number) => number
  archY: (xn: number, yn: number) => number
}

function makeFrame(data: ReliefData, result: InferenceResult): Frame {
  const { gw } = data
  const ph = panoHeight(result)
  const toArc = panoToArcMapper(result)
  const flatX = new Float32Array(gw)
  const archX = new Float32Array(gw)
  const archZ = new Float32Array(gw)
  const nX = new Float32Array(gw)
  const nZ = new Float32Array(gw)
  for (let x = 0; x < gw; x++) {
    const xn = x / (gw - 1)
    const f = archFrame(FOCAL_ARCH, toArc(xn))
    flatX[x] = (xn - 0.5) * PANO_WIDTH
    archX[x] = f.p.x
    archZ[x] = f.p.z
    nX[x] = f.n.x
    nZ[x] = f.n.z
  }
  return {
    flatX, archX, archZ, nX, nZ,
    flatY: (yn) => (0.5 - yn) * ph,
    archY: (xn, yn) => (data.occ(xn) - yn) * ph, // flattens the panoramic "smile"
  }
}

/** Closed shell: a front surface at +H and a back surface at −H that meet where H = 0. */
function buildShell(
  data: ReliefData,
  H: Float32Array,
  field: Float32Array,
  level: number,
  frame: Frame,
  step: number,
): THREE.BufferGeometry {
  const { gw, gh } = data
  const F = (x: number, y: number) => field[Math.min(gh - 1, Math.max(0, y)) * gw + Math.min(gw - 1, Math.max(0, x))]
  const lerpCol = (arr: Float32Array, fx: number) => {
    const x0 = Math.max(0, Math.min(gw - 1, Math.floor(fx)))
    const x1 = Math.min(gw - 1, x0 + 1)
    const t = fx - x0
    return arr[x0] * (1 - t) + arr[x1] * t
  }
  const cols = Math.floor((gw - 1) / step) + 1
  const rows = Math.floor((gh - 1) / step) + 1
  const hAt = (c: number, r: number) => H[r * step * gw + c * step]
  const used = new Int32Array(cols * rows).fill(-1)
  const cells: number[] = []
  for (let r = 0; r < rows - 1; r++)
    for (let c = 0; c < cols - 1; c++)
      if (hAt(c, r) > 0 || hAt(c + 1, r) > 0 || hAt(c, r + 1) > 0 || hAt(c + 1, r + 1) > 0) cells.push(r * cols + c)

  let n = 0
  for (const cell of cells) {
    const r = Math.floor(cell / cols), c = cell % cols
    for (const k of [r * cols + c, r * cols + c + 1, (r + 1) * cols + c, (r + 1) * cols + c + 1]) if (used[k] < 0) used[k] = n++
  }
  const count = n * 2
  const pos = new Float32Array(count * 3)
  const flat = new Float32Array(count * 3)
  const base = new Float32Array(count * 3)
  const nrm = new Float32Array(count * 3)
  const hh = new Float32Array(count)
  const uv = new Float32Array(count * 2)
  const id = new Float32Array(count)

  for (let k = 0; k < used.length; k++) {
    const vi = used[k]
    if (vi < 0) continue
    const r = Math.floor(k / cols), c = k % cols
    const gx = Math.min(gw - 1, c * step)
    const gy = Math.min(gh - 1, r * step)
    const h = H[gy * gw + gx]
    // outline vertices slide to the sub-pixel iso-line so edges aren't stair-stepped
    let fx = gx
    let fy = gy
    if (h === 0) {
      const f = F(gx, gy) - level
      const dx = (F(gx + step, gy) - F(gx - step, gy)) / (2 * step)
      const dy = (F(gx, gy + step) - F(gx, gy - step)) / (2 * step)
      const g2 = dx * dx + dy * dy
      if (g2 > 1e-8) {
        const k = -f / g2
        fx = gx + Math.max(-step, Math.min(step, k * dx))
        fy = gy + Math.max(-step, Math.min(step, k * dy))
      }
    }
    const xn = fx / (gw - 1)
    const yn = fy / (gh - 1)
    const flatY = frame.flatY(yn)
    const ay = frame.archY(xn, yn)
    const ax = lerpCol(frame.archX, fx)
    const az = lerpCol(frame.archZ, fx)
    const nx = lerpCol(frame.nX, fx)
    const nz = lerpCol(frame.nZ, fx)
    const flX = lerpCol(frame.flatX, fx)
    for (const side of [0, 1]) {
      const i = vi + side * n
      const s = side ? -h : h
      flat.set([flX, flatY, 0], i * 3)
      base.set([ax, ay, az], i * 3)
      nrm.set([nx, 0, nz], i * 3)
      pos.set([ax + nx * s, ay, az + nz * s], i * 3)
      hh[i] = s
      uv.set([xn, 1 - yn], i * 2)
      id[i] = data.ids[gy * gw + gx]
    }
  }

  const index: number[] = []
  for (const cell of cells) {
    const r = Math.floor(cell / cols), c = cell % cols
    const tl = used[r * cols + c], tr = used[r * cols + c + 1]
    const bl = used[(r + 1) * cols + c], br = used[(r + 1) * cols + c + 1]
    index.push(bl, br, tr, bl, tr, tl) // front, facing outward
    index.push(bl + n, tr + n, br + n, bl + n, tl + n, tr + n) // back, reversed
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('aFlat', new THREE.BufferAttribute(flat, 3))
  geo.setAttribute('aBase', new THREE.BufferAttribute(base, 3))
  geo.setAttribute('aN', new THREE.BufferAttribute(nrm, 3))
  geo.setAttribute('aH', new THREE.BufferAttribute(hh, 1))
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  geo.setAttribute('aId', new THREE.BufferAttribute(id, 1))
  geo.setIndex(count > 65535 ? new THREE.Uint32BufferAttribute(index, 1) : new THREE.Uint16BufferAttribute(index, 1))
  geo.computeVertexNormals()
  geo.computeBoundingSphere()
  return geo
}

interface ReliefUniforms {
  uMorph: { value: number }
  uGrow: { value: number }
  uSel: { value: number }
  uHov: { value: number }
  uIso: { value: number }
  uPulse: { value: number }
  uAccent: { value: THREE.Color }
  uBase: { value: THREE.Color }
  uLum: { value: THREE.Vector2 }
}

function makeMaterial(kind: 'tooth' | 'bone', map: THREE.Texture) {
  const uniforms: ReliefUniforms = {
    uMorph: { value: 1 },
    uGrow: { value: 1 },
    uSel: { value: 0 },
    uHov: { value: 0 },
    uIso: { value: 1 },
    uPulse: { value: 0 },
    uAccent: { value: new THREE.Color('#86c5d8') },
    uBase: { value: kind === 'tooth' ? new THREE.Color(1.04, 0.95, 0.8) : new THREE.Color(0.78, 0.82, 0.86) },
    uLum: { value: kind === 'tooth' ? new THREE.Vector2(0.3, 0.85) : new THREE.Vector2(0.12, 1.05) },
  }
  const mat = new THREE.MeshStandardMaterial({
    map,
    roughness: kind === 'tooth' ? 0.36 : 0.75,
    metalness: 0,
    transparent: kind === 'bone',
    opacity: kind === 'bone' ? 0.42 : 1,
    depthWrite: kind === 'tooth',
    side: kind === 'bone' ? THREE.DoubleSide : THREE.FrontSide,
  })
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        attribute vec3 aFlat; attribute vec3 aBase; attribute vec3 aN; attribute float aH; attribute float aId;
        uniform float uMorph; uniform float uGrow;
        flat varying float vId;`,
      )
      .replace(
        '#include <beginnormal_vertex>',
        `vec3 tAxis = cross( vec3( 0.0, 1.0, 0.0 ), aN );
        vec3 flatNormal = vec3( dot( normal, tAxis ), normal.y, dot( normal, aN ) );
        vec3 objectNormal = normalize( mix( flatNormal, normal, uMorph ) );
        #ifdef USE_TANGENT
          vec3 objectTangent = vec3( tangent.xyz );
        #endif`,
      )
      .replace(
        '#include <begin_vertex>',
        `vId = aId;
        vec3 flatP = aFlat + vec3(0.0, 0.0, aH * uGrow);
        vec3 archP = aBase + aN * aH * uGrow;
        vec3 transformed = mix(flatP, archP, uMorph);`,
      )
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uSel; uniform float uHov; uniform float uIso; uniform float uPulse;
        uniform vec3 uAccent; uniform vec3 uBase; uniform vec2 uLum;
        flat varying float vId;`,
      )
      .replace(
        '#include <map_fragment>',
        `vec4 texelColor = texture2D( map, vMapUv );
        float lum = dot( texelColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
        diffuseColor.rgb = uBase * ( uLum.x + uLum.y * lum );
        float isSel = uSel > 0.5 ? 1.0 - step( 0.5, abs( vId - uSel ) ) : 0.0;
        float isHov = uHov > 0.5 ? 1.0 - step( 0.5, abs( vId - uHov ) ) : 0.0;
        float dim = uIso * step( 0.5, uSel ) * ( 1.0 - isSel );
        diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * 0.16, dim );
        diffuseColor.rgb = mix( diffuseColor.rgb, uAccent, isHov * 0.28 + isSel * 0.22 );`,
      )
      .replace(
        '#include <emissivemap_fragment>',
        `#include <emissivemap_fragment>
        totalEmissiveRadiance += uAccent * isSel * ( 0.12 + 0.18 * uPulse );`,
      )
  }
  mat.customProgramCacheKey = () => `relief-${kind}`
  return { mat, uniforms }
}

/** Invisible low-poly surface used for picking (cheap to raycast, follows the morph). */
function usePicker(result: InferenceResult, data: ReliefData, frame: Frame) {
  return useMemo(() => {
    const ph = panoHeight(result)
    const geo = new THREE.PlaneGeometry(PANO_WIDTH, ph, 180, 40)
    const flat = Float32Array.from(geo.attributes.position.array as Float32Array)
    const wrapped = new Float32Array(flat.length)
    for (let i = 0; i < flat.length; i += 3) {
      const xn = flat[i] / PANO_WIDTH + 0.5
      const yn = 0.5 - flat[i + 1] / ph
      const gx = Math.round(Math.min(1, Math.max(0, xn)) * (data.gw - 1))
      wrapped[i] = frame.archX[gx]
      wrapped[i + 1] = frame.archY(xn, yn)
      wrapped[i + 2] = frame.archZ[gx]
    }
    return { geo, flat, wrapped }
  }, [result, data, frame])
}

export function Relief({ result, imageUrl, image }: { result: InferenceResult; imageUrl: string; image: HTMLImageElement }) {
  const texture = useLoader(THREE.TextureLoader, imageUrl)
  const data = useMemo(() => analyzeOpg(image, result), [image, result])
  const frame = useMemo(() => makeFrame(data, result), [data, result])
  const toothGeo = useMemo(() => buildShell(data, data.toothH, data.toothField, data.toothLevel, frame, 1), [data, frame])
  const boneGeo = useMemo(() => buildShell(data, data.boneH, data.boneField, data.boneLevel, frame, 2), [data, frame])
  const tooth = useMemo(() => makeMaterial('tooth', texture), [texture])
  const bone = useMemo(() => makeMaterial('bone', texture), [texture])
  const picker = usePicker(result, data, frame)
  const pickMesh = useRef<THREE.Mesh>(null)
  const labelEls = useRef(new Map<number, HTMLDivElement>())
  const toCam = useMemo(() => new THREE.Vector3(), [])
  const lastMorph = useRef(-1)

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = 8
    texture.needsUpdate = true
  }, [texture])
  useEffect(
    () => () => {
      toothGeo.dispose()
      boneGeo.dispose()
      tooth.mat.dispose()
      bone.mat.dispose()
      picker.geo.dispose()
    },
    [toothGeo, boneGeo, tooth, bone, picker],
  )

  // anchors for camera fly-to, labels and the split-view link line
  useEffect(() => {
    for (const t of result.teeth) {
      const cx = t.bbox[0] + t.bbox[2] / 2
      const cy = t.bbox[1] + t.bbox[3] / 2
      const gx = Math.round(Math.min(1, Math.max(0, cx)) * (data.gw - 1))
      const n = new THREE.Vector3(frame.nX[gx], 0, frame.nZ[gx])
      const center = new THREE.Vector3(frame.archX[gx], frame.archY(cx, cy), frame.archZ[gx]).addScaledVector(n, 1)
      const upper = t.fdi < 30
      const ly = upper ? t.bbox[1] - 0.035 : t.bbox[1] + t.bbox[3] + 0.045
      const label = new THREE.Vector3(frame.archX[gx], frame.archY(cx, ly), frame.archZ[gx]).addScaledVector(n, 3)
      toothAnchors.set(t.fdi, { center, normal: n, label, upper })
    }
    return () => toothAnchors.clear()
  }, [result, data, frame])

  const layers = useMouthTwin((s) => s.layers)
  const showNumbers = useMouthTwin((s) => s.showNumbers)
  const hoveredFdi = useMouthTwin((s) => s.hoveredFdi)
  const selectedFdi = useMouthTwin((s) => s.selectedFdi)
  const hover = useMouthTwin((s) => s.hover)
  const select = useMouthTwin((s) => s.select)

  useFrame((state) => {
    const s = useMouthTwin.getState()
    // hide labels of teeth on the far side of the arch (HTML labels are not depth-tested)
    for (const [fdi, el] of labelEls.current) {
      const a = toothAnchors.get(fdi)
      if (!a) continue
      toCam.copy(state.camera.position).sub(a.label)
      const facing = toCam.dot(a.normal) > 0 || fdi === s.selectedFdi
      el.style.opacity = facing ? '1' : '0'
    }
    const pulse = 0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 3.2)
    for (const u of [tooth.uniforms, bone.uniforms]) {
      u.uMorph.value = reveal.morph
      u.uGrow.value = reveal.appear
      u.uSel.value = s.selectedFdi ?? 0
      u.uHov.value = s.hoveredFdi ?? 0
      u.uIso.value = s.isolate ? 1 : 0
      u.uPulse.value = pulse
    }
    const isolating = s.isolate && s.selectedFdi !== null
    bone.mat.opacity = (isolating ? 0.14 : 0.42) * Math.min(1, reveal.appear * 1.2)

    // keep the picking surface in step with the morph
    const m = reveal.morph
    if (m !== lastMorph.current) {
      lastMorph.current = m
      const arr = picker.geo.attributes.position.array as Float32Array
      for (let i = 0; i < arr.length; i++) arr[i] = picker.flat[i] + (picker.wrapped[i] - picker.flat[i]) * m
      picker.geo.attributes.position.needsUpdate = true
      picker.geo.computeBoundingSphere()
    }
  })

  const idAt = (e: ThreeEvent<PointerEvent | MouseEvent>): number | null => {
    if (!e.uv) return null
    const gx = Math.round(e.uv.x * (data.gw - 1))
    const gy = Math.round((1 - e.uv.y) * (data.gh - 1))
    if (gx < 0 || gy < 0 || gx >= data.gw || gy >= data.gh) return null
    // accept hits on or right next to tooth pixels
    for (let dy = -3; dy <= 3; dy++)
      for (let dx = -3; dx <= 3; dx++) {
        const x = gx + dx, y = gy + dy
        if (x < 0 || y < 0 || x >= data.gw || y >= data.gh) continue
        const i = y * data.gw + x
        if (data.toothMask[i] && data.ids[i]) return data.ids[i]
      }
    return null
  }

  const labels = result.teeth.filter((t) => showNumbers || t.fdi === hoveredFdi || t.fdi === selectedFdi)

  return (
    <group>
      {layers.teeth && <mesh geometry={toothGeo} material={tooth.mat} frustumCulled={false} raycast={() => null} />}
      {layers.bone && <mesh geometry={boneGeo} material={bone.mat} frustumCulled={false} renderOrder={2} raycast={() => null} />}
      <mesh
        ref={pickMesh}
        geometry={picker.geo}
        frustumCulled={false}
        onPointerMove={(e) => {
          e.stopPropagation()
          if (reveal.running) return
          const id = idAt(e)
          if (id !== useMouthTwin.getState().hoveredFdi) hover(id)
          document.body.style.cursor = id ? 'pointer' : ''
        }}
        onPointerOut={() => {
          hover(null)
          document.body.style.cursor = ''
        }}
        onClick={(e) => {
          e.stopPropagation()
          if (reveal.running) return
          const id = idAt(e)
          select(id && id !== useMouthTwin.getState().selectedFdi ? id : null)
        }}
      >
        <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} side={THREE.DoubleSide} />
      </mesh>
      {!reveal.running &&
        labels.map((t) => {
          const a = toothAnchors.get(t.fdi)
          if (!a) return null
          return (
            <Html key={t.fdi} position={a.label} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
              <div
                ref={(el) => {
                  if (el) labelEls.current.set(t.fdi, el)
                  else labelEls.current.delete(t.fdi)
                }}
                className={`fdi-chip ${t.fdi === selectedFdi ? 'is-selected' : ''} ${t.fdi === hoveredFdi ? 'is-hovered' : ''}`}>{t.fdi}</div>
            </Html>
          )
        })}
    </group>
  )
}
