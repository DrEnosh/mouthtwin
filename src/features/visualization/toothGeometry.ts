/**
 * Procedural template tooth geometry.
 *
 * Local frame (millimetres): occlusal surface at y = 0, crown extends to y = -crown, roots below.
 * +x = distal, +z = buccal/labial. Upper teeth and patient-right teeth are produced by mirroring
 * this frame, so one geometry serves all four quadrants of a given tooth position.
 */
import * as THREE from 'three'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { ToothRef } from '../../data/fdi'

type V2 = [number, number]

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)))
  return t * t * (3 - 2 * t)
}

function crownProfile(kind: ToothRef['kind']): V2[] {
  switch (kind) {
    case 'incisor':
      return [[0, 0], [0.55, -0.005], [0.9, -0.05], [1, -0.28], [0.97, -0.6], [0.86, -0.88], [0.78, -1]]
    case 'canine':
      return [[0, 0.02], [0.22, -0.07], [0.58, -0.2], [0.9, -0.38], [1, -0.55], [0.92, -0.8], [0.8, -1]]
    case 'premolar':
      return [[0, -0.02], [0.35, 0], [0.72, -0.04], [0.94, -0.16], [1, -0.4], [0.95, -0.68], [0.84, -0.9], [0.78, -1]]
    default:
      return [[0, -0.02], [0.4, 0], [0.78, -0.03], [0.96, -0.14], [1, -0.38], [0.96, -0.66], [0.86, -0.9], [0.8, -1]]
  }
}

const ROOT_PROFILE: V2[] = [[0.7, 0.1], [0.74, -0.04], [0.66, -0.38], [0.46, -0.72], [0.22, -0.94], [0, -1]]

function resample(points: V2[], n: number): THREE.Vector2[] {
  const spline = new THREE.SplineCurve(points.map(([x, y]) => new THREE.Vector2(x, y)))
  return spline.getPoints(n)
}

function smoothed(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  geo.deleteAttribute('normal')
  geo.deleteAttribute('uv')
  const merged = mergeVertices(geo, 1e-4)
  merged.computeVertexNormals()
  return merged
}

function gauss(x: number, z: number, cx: number, cz: number, s: number) {
  return Math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / s)
}

function buildCrown(ref: ToothRef): THREE.BufferGeometry {
  const pts = resample(crownProfile(ref.kind), 26)
  pts.push(new THREE.Vector2(0.001, -1)) // close the cervical end so the crown reads as solid without roots
  pts.reverse() // LatheGeometry expects bottom → top for outward-facing normals
  const geo = new THREE.LatheGeometry(pts, 40)
  const pos = geo.attributes.position as THREE.BufferAttribute
  const w = ref.widthMm / 2
  const d = ref.depthMm / 2
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i)
    let y = pos.getY(i)
    let z = pos.getZ(i)
    const depthFromTop = -y

    // incisal edges are thin buccolingually; canines taper to a cusp tip
    if (ref.kind === 'incisor') z *= 0.22 + 0.78 * smoothstep(0, 0.8, depthFromTop)
    if (ref.kind === 'canine') z *= 0.5 + 0.5 * smoothstep(0, 0.65, depthFromTop)
    // buccal surfaces are a little flatter than lingual on anterior teeth
    if (ref.kind === 'incisor' && z > 0) z *= 0.85

    const occ = smoothstep(-0.35, 0, y) // 1 at the occlusal table, 0 below
    if (ref.kind === 'molar') {
      const cusps =
        gauss(x, z, -0.42, 0.42, 0.09) + gauss(x, z, 0.42, 0.42, 0.09) + gauss(x, z, -0.42, -0.42, 0.09) + gauss(x, z, 0.42, -0.42, 0.09)
      const fissure = gauss(x, z, 0, 0, 0.1) * 0.6 + Math.exp(-(z * z) / 0.02) * 0.4
      y += occ * (0.13 * cusps - 0.08 * fissure)
    } else if (ref.kind === 'premolar') {
      const cusps = gauss(x, z, 0, 0.45, 0.12) * 1.2 + gauss(x, z, 0, -0.45, 0.12) * 0.8
      y += occ * (0.14 * cusps - 0.07 * Math.exp(-(z * z) / 0.03))
    }
    x *= w
    z *= d
    y *= ref.crownMm
    pos.setXYZ(i, x, y, z)
  }
  return smoothed(geo)
}

interface RootSpec {
  ox: number // offset across (fraction of half width, +distal)
  oz: number // offset buccolingual (fraction of half depth, +buccal)
  rx: number // radius scale x
  rz: number // radius scale z
  len: number // length scale
  apexX: number // apex drift (fraction of width, +distal)
  apexZ: number // apex drift (fraction of depth, +buccal)
}

function rootSpecs(ref: ToothRef): RootSpec[] {
  if (ref.roots === 1) return [{ ox: 0, oz: 0, rx: 1, rz: 1, len: 1, apexX: 0.08, apexZ: 0 }]
  if (ref.roots === 2 && ref.upper) {
    // upper first premolar: buccal and palatal roots
    return [
      { ox: 0, oz: 0.42, rx: 0.8, rz: 0.55, len: 1, apexX: 0.06, apexZ: 0.12 },
      { ox: 0, oz: -0.42, rx: 0.8, rz: 0.55, len: 0.95, apexX: 0.06, apexZ: -0.12 },
      { ox: 0, oz: 0, rx: 0.95, rz: 0.95, len: 0.3, apexX: 0, apexZ: 0 },
    ]
  }
  if (ref.roots === 2) {
    // lower molars: mesial and distal roots, broad buccolingually
    return [
      { ox: -0.45, oz: 0, rx: 0.42, rz: 0.82, len: 1, apexX: -0.02, apexZ: 0 },
      { ox: 0.45, oz: 0, rx: 0.4, rz: 0.78, len: 0.92, apexX: 0.14, apexZ: 0 },
      { ox: 0, oz: 0, rx: 0.95, rz: 0.95, len: 0.25, apexX: 0, apexZ: 0 },
    ]
  }
  // upper molars: mesiobuccal, distobuccal, palatal
  return [
    { ox: -0.4, oz: 0.38, rx: 0.42, rz: 0.48, len: 0.92, apexX: 0.02, apexZ: 0.1 },
    { ox: 0.42, oz: 0.38, rx: 0.36, rz: 0.4, len: 0.84, apexX: 0.12, apexZ: 0.1 },
    { ox: 0, oz: -0.42, rx: 0.5, rz: 0.52, len: 1, apexX: 0.04, apexZ: -0.3 },
    { ox: 0, oz: 0, rx: 0.95, rz: 0.95, len: 0.25, apexX: 0, apexZ: 0 },
  ]
}

function buildRoots(ref: ToothRef): THREE.BufferGeometry {
  const w = ref.widthMm / 2
  const d = ref.depthMm / 2
  const profile = resample(ROOT_PROFILE, 16).reverse()
  const parts = rootSpecs(ref).map((spec) => {
    const geo = new THREE.LatheGeometry(profile, 24)
    const pos = geo.attributes.position as THREE.BufferAttribute
    const len = ref.rootMm * spec.len
    for (let i = 0; i < pos.count; i++) {
      const t = Math.max(0, -pos.getY(i)) // 0 at the cervical end, 1 at the apex
      const x = pos.getX(i) * w * 0.86 * spec.rx + spec.ox * w + spec.apexX * ref.widthMm * t * t * 1.4
      const z = pos.getZ(i) * d * 0.86 * spec.rz + spec.oz * d + spec.apexZ * ref.depthMm * t
      const y = -ref.crownMm + 1.1 + pos.getY(i) * len
      pos.setXYZ(i, x, y, z)
    }
    return smoothed(geo)
  })
  return mergeGeometries(parts, false) as THREE.BufferGeometry
}

export interface ToothGeometry {
  crown: THREE.BufferGeometry
  roots: THREE.BufferGeometry
  /** Centre of the tooth in local coordinates, used for camera targeting. */
  center: THREE.Vector3
}

const cache = new Map<string, ToothGeometry>()

export function getToothGeometry(ref: ToothRef): ToothGeometry {
  const key = `${ref.upper ? 'u' : 'l'}${ref.position}`
  let g = cache.get(key)
  if (!g) {
    g = {
      crown: buildCrown(ref),
      roots: buildRoots(ref),
      center: new THREE.Vector3(0, -(ref.crownMm + ref.rootMm) * 0.35, 0),
    }
    cache.set(key, g)
  }
  return g
}
