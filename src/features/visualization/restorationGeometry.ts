/**
 * Geometry and materials for what the X-ray shows on a tooth: crown / cap, filling, root-canal filling,
 * implant, bridge. Everything here is generic (the OPG does not tell us the real shape or material), and
 * is only drawn where the model found something.
 *
 * Depth below is measured from each tooth's own occlusal end along its long axis (see toothFrame); the crown is
 * the part shallower than crownHeight(fdi).
 */
import * as THREE from 'three'

/** Crown height (mm) by tooth type, for the crown / root boundary. */
export function crownHeight(fdi: number): number {
  const t = fdi % 10
  return t <= 2 ? 10 : t === 3 ? 10.5 : t <= 5 ? 8.5 : t === 8 ? 7 : 7.5
}

/** The tooth's own frame: y of its occlusal end, and +1 (upper, root toward +y) / -1 (lower). Depth = (y - origin) * sign. */
export function toothFrame(geo: THREE.BufferGeometry, fdi: number) {
  geo.computeBoundingBox()
  const bb = geo.boundingBox as THREE.Box3
  const sign = fdi < 30 ? 1 : -1
  return { sign, origin: sign === 1 ? bb.min.y : bb.max.y, crown: crownHeight(fdi) }
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

/** Per-vertex masks used by the tooth shader: crown, occlusal filling patch, root, plus depth for colouring. Call before fitting. */
export function addRestorationAttributes(geo: THREE.BufferGeometry, fdi: number) {
  const { sign, origin, crown } = toothFrame(geo, fdi)
  const pos = geo.attributes.position as THREE.BufferAttribute
  const n = pos.count
  // occlusal surface centre (x, z) of the vertices closest to the occlusal end
  let cx = 0, cz = 0, cn = 0
  for (let i = 0; i < n; i++) {
    if ((pos.getY(i) - origin) * sign < 1.6) { cx += pos.getX(i); cz += pos.getZ(i); cn++ }
  }
  cx /= Math.max(1, cn); cz /= Math.max(1, cn)
  const cap = new Float32Array(n), fill = new Float32Array(n), root = new Float32Array(n), depth = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const d = (pos.getY(i) - origin) * sign
    depth[i] = d
    cap[i] = 1 - smooth(crown - 0.5, crown + 0.5, d)
    root[i] = smooth(crown - 1.5, crown + 0.5, d)
    const r = Math.hypot(pos.getX(i) - cx, pos.getZ(i) - cz)
    fill[i] = (1 - smooth(1.5, 3.3, r)) * (1 - smooth(0.8, 3.6, d))
  }
  geo.setAttribute('aCap', new THREE.BufferAttribute(cap, 1))
  geo.setAttribute('aFill', new THREE.BufferAttribute(fill, 1))
  geo.setAttribute('aRoot', new THREE.BufferAttribute(root, 1))
  geo.setAttribute('aDepth', new THREE.BufferAttribute(depth, 1))
  return depth
}

export interface RestoUniforms {
  uCap: { value: number }
  uFill: { value: number }
  uRootAlpha: { value: number }
  /** depth (mm from the occlusal end) beyond which the tooth is not drawn: crown-only teeth, e.g. developing wisdom teeth */
  uCut: { value: number }
  uCapColor: { value: THREE.Color }
  uFillColor: { value: THREE.Color }
}

export const CAP_COLOR = '#d9dfe7'
export const FILL_COLOR = '#8f98a3'

export function makeRestoUniforms(): RestoUniforms {
  return {
    uCap: { value: 0 },
    uFill: { value: 0 },
    uRootAlpha: { value: 0 },
    uCut: { value: 1e3 },
    uCapColor: { value: new THREE.Color(CAP_COLOR) },
    uFillColor: { value: new THREE.Color(FILL_COLOR) },
  }
}

/** Crown → metallic cap, occlusal patch → filling, root → translucent (so a canal filling shows through). */
export function patchToothMaterial(mat: THREE.MeshPhysicalMaterial, u: RestoUniforms) {
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u)
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute float aCap;\nattribute float aFill;\nattribute float aRoot;\nattribute float aDepth;\nvarying float vCap;\nvarying float vFill;\nvarying float vRoot;\nvarying float vDepth;',
      )
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvCap = aCap; vFill = aFill; vRoot = aRoot; vDepth = aDepth;')
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        '#include <common>\nvarying float vCap;\nvarying float vFill;\nvarying float vRoot;\nvarying float vDepth;\nuniform float uCut;\nuniform float uCap;\nuniform float uFill;\nuniform float uRootAlpha;\nuniform vec3 uCapColor;\nuniform vec3 uFillColor;',
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        if (vDepth > uCut) discard;
        float capM = smoothstep(0.42, 0.58, vCap) * uCap;
        float fillM = smoothstep(0.35, 0.65, vFill) * uFill;
        float rim = smoothstep(0.12, 0.42, vCap) * (1.0 - smoothstep(0.42, 0.62, vCap)) * uCap;
        diffuseColor.rgb = mix(diffuseColor.rgb, uCapColor, capM);
        diffuseColor.rgb = mix(diffuseColor.rgb, uFillColor, fillM);
        diffuseColor.rgb *= 1.0 - 0.6 * rim;
        diffuseColor.a *= 1.0 - uRootAlpha * vRoot;`,
      )
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.26, max(capM, fillM));')
      .replace(
        '#include <metalnessmap_fragment>',
        '#include <metalnessmap_fragment>\nmetalnessFactor = mix(metalnessFactor, 0.72, capM);\nmetalnessFactor = mix(metalnessFactor, 0.75, fillM);',
      )
  }
  mat.customProgramCacheKey = () => 'tooth-resto'
}

type V3 = THREE.Vector3

/**
 * Root-canal paths of a reference tooth, in the tooth's original coordinates: one polyline per root
 * (pulp chamber → apex). Roots are found as separated clusters of vertices deep below the crown.
 */
export function canalPaths(geo: THREE.BufferGeometry, fdi: number): V3[][] {
  const { sign, origin, crown: CEJ } = toothFrame(geo, fdi)
  const pos = geo.attributes.position as THREE.BufferAttribute
  const depth = (i: number) => (pos.getY(i) - origin) * sign
  let maxD = 0
  for (let i = 0; i < pos.count; i++) maxD = Math.max(maxD, depth(i))
  if (maxD < CEJ + 2) return []
  const slice = (d0: number, d1: number, filter?: (i: number) => boolean) => {
    let x = 0, z = 0, n = 0
    for (let i = 0; i < pos.count; i++) {
      const d = depth(i)
      if (d < d0 || d > d1) continue
      if (filter && !filter(i)) continue
      x += pos.getX(i); z += pos.getZ(i); n++
    }
    return n ? { x: x / n, z: z / n, n } : null
  }
  const at = (d: number, c: { x: number; z: number }): V3 => new THREE.Vector3(c.x, origin + d * sign, c.z)

  // separate roots at 72-90 % of the depth: split the vertices along their main horizontal direction where there is a gap
  const band = slice(maxD * 0.72, maxD * 0.9)
  if (!band) return []
  let sxx = 0, szz = 0, sxz = 0
  for (let i = 0; i < pos.count; i++) {
    const d = depth(i)
    if (d < maxD * 0.72 || d > maxD * 0.9) continue
    const dx = pos.getX(i) - band.x, dz = pos.getZ(i) - band.z
    sxx += dx * dx; szz += dz * dz; sxz += dx * dz
  }
  const ang = 0.5 * Math.atan2(2 * sxz, sxx - szz)
  const ux = Math.cos(ang), uz = Math.sin(ang)
  const deep: { u: number }[] = []
  for (let i = 0; i < pos.count; i++) {
    const d = depth(i)
    if (d < maxD * 0.72 || d > maxD * 0.9) continue
    deep.push({ u: (pos.getX(i) - band.x) * ux + (pos.getZ(i) - band.z) * uz })
  }
  deep.sort((a, b) => a.u - b.u)
  const clusters: { u: number }[][] = [[]]
  for (let k = 0; k < deep.length; k++) {
    if (k && deep[k].u - deep[k - 1].u > 1.1) clusters.push([])
    clusters[clusters.length - 1].push(deep[k])
  }
  const roots = clusters.filter((c) => c.length >= 8)
  const chamber = slice(CEJ - 3.5, CEJ - 1.5)
  if (!chamber) return []
  const paths: V3[][] = []
  for (const c of roots.length ? roots : [[]]) {
    const inCluster = c.length
      ? (i: number) => {
          const u = (pos.getX(i) - band.x) * ux + (pos.getZ(i) - band.z) * uz
          return u >= c[0].u - 0.9 && u <= c[c.length - 1].u + 0.9
        }
      : undefined
    const apex = slice(maxD * 0.93, maxD + 1, inCluster) ?? slice(maxD * 0.85, maxD + 1)
    const mid = slice(CEJ + 0.5, CEJ + (maxD - CEJ) * 0.55, inCluster)
    const upper = slice(CEJ - 0.5, CEJ + 1.5, inCluster)
    if (!apex) continue
    const pts: V3[] = [at(CEJ - 3, chamber)]
    if (upper) pts.push(at(CEJ + 0.5, upper))
    if (mid) pts.push(at(CEJ + (maxD - CEJ) * 0.35, mid))
    pts.push(at(maxD - 0.6, apex))
    paths.push(pts)
  }
  return paths
}

/** A tapered tube along a smooth curve (canal filling). */
export function taperedTube(points: V3[], r0: number, r1: number): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal')
  const tubular = 28, radial = 10
  const g = new THREE.TubeGeometry(curve, tubular, 1, radial, false)
  const pos = g.attributes.position as THREE.BufferAttribute
  const c = new THREE.Vector3(), p = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    const ring = Math.floor(i / (radial + 1))
    const t = ring / tubular
    curve.getPointAt(t, c)
    p.fromBufferAttribute(pos, i).sub(c)
    const r = r0 + (r1 - r0) * t
    p.multiplyScalar(r).add(c)
    pos.setXYZ(i, p.x, p.y, p.z)
  }
  g.computeVertexNormals()
  return g
}

/** Only the crown faces of a tooth, for implant crowns. Returned in the tooth's frame: y = depth from the occlusal end. */
export function extractCrown(geo: THREE.BufferGeometry, fdi: number): THREE.BufferGeometry {
  const { sign, origin, crown } = toothFrame(geo, fdi)
  const src = geo.index ? geo.toNonIndexed() : geo.clone()
  const pos = src.attributes.position as THREE.BufferAttribute
  const dep = (i: number) => (pos.getY(i) - origin) * sign
  const keep: number[] = []
  for (let f = 0; f < pos.count; f += 3) {
    if (dep(f) < crown + 0.3 && dep(f + 1) < crown + 0.3 && dep(f + 2) < crown + 0.3) keep.push(f, f + 1, f + 2)
  }
  const out = new THREE.BufferGeometry()
  const arr = new Float32Array(keep.length * 3)
  keep.forEach((vi, k) => {
    arr[k * 3] = pos.getX(vi); arr[k * 3 + 1] = dep(vi); arr[k * 3 + 2] = pos.getZ(vi)
  })
  out.setAttribute('position', new THREE.BufferAttribute(arr, 3))
  out.computeVertexNormals()
  return out
}

/** Generic threaded implant: fixture (in the bone) + abutment. Local axis = +y, crestal end at y = 0, fixture below. */
export function implantGeometry(radius: number, length = 10.5): { fixture: THREE.BufferGeometry; abutment: THREE.BufferGeometry } {
  const pts: THREE.Vector2[] = []
  const pitch = 0.42
  const n = Math.floor(length / pitch)
  pts.push(new THREE.Vector2(0.001, 0))
  pts.push(new THREE.Vector2(radius * 0.92, 0))
  for (let i = 0; i < n; i++) {
    const t = i / n
    const taper = radius * (1 - 0.22 * smooth(0.55, 1, t))
    const y0 = -i * pitch
    pts.push(new THREE.Vector2(taper * 0.86, y0 - pitch * 0.15))
    pts.push(new THREE.Vector2(taper * 1.0, y0 - pitch * 0.5))
    pts.push(new THREE.Vector2(taper * 0.86, y0 - pitch * 0.85))
  }
  pts.push(new THREE.Vector2(radius * 0.5, -length))
  pts.push(new THREE.Vector2(0.001, -length - 0.2))
  const fixture = new THREE.LatheGeometry(pts, 28)
  fixture.computeVertexNormals()
  const abutment = new THREE.CylinderGeometry(radius * 0.62, radius * 0.78, 3.4, 20)
  abutment.translate(0, 1.7, 0)
  return { fixture, abutment }
}

export const IMPLANT_COLOR = '#aab2bb'
