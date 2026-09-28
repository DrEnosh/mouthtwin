import { useFrame, useLoader } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { InferenceResult } from '../../types/inference'
import { archFrame, FOCAL_ARCH, PANO_WIDTH, panoHeight, panoToArcMapper } from './archLayout'
import { reveal } from './runtime'

const SEGMENTS = 200

/**
 * The OPG as a plane that starts flat in front of the camera and bends back onto the arch,
 * reversing the panoramic unrolling. Fades out as the tooth models take over.
 */
export function OpgWrap({ url, result }: { url: string; result: InferenceResult }) {
  const texture = useLoader(THREE.TextureLoader, url)
  const mesh = useRef<THREE.Mesh>(null)
  const ph = panoHeight(result)
  const toArc = useMemo(() => panoToArcMapper(result), [result])
  const occ = useMemo(() => {
    const c = result.occlusalCurve ?? { a: 0, b: 0, c: 0.5 }
    return (xn: number) => c.a * xn * xn + c.b * xn + c.c
  }, [result])

  const geometry = useMemo(() => new THREE.PlaneGeometry(PANO_WIDTH, ph, SEGMENTS, 1), [ph])
  const flat = useMemo(() => Float32Array.from(geometry.attributes.position.array as Float32Array), [geometry])
  const wrapped = useMemo(() => {
    const out = new Float32Array(flat.length)
    for (let i = 0; i < flat.length; i += 3) {
      const x = flat[i]
      const y = flat[i + 1]
      const { p, n } = archFrame(FOCAL_ARCH, toArc(x / PANO_WIDTH + 0.5))
      out[i] = p.x + n.x * 0.05
      const xn = x / PANO_WIDTH + 0.5
      const yn = 0.5 - y / ph
      out[i + 1] = (occ(xn) - yn) * ph
      out[i + 2] = p.z + n.z * 0.05
    }
    return out
  }, [flat, toArc, occ, ph])

  const material = useMemo(
    () => new THREE.MeshBasicMaterial({ transparent: true, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }),
    [],
  )
  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = 4
    material.map = texture
    material.needsUpdate = true
  }, [texture, material])
  useEffect(() => () => { geometry.dispose(); material.dispose() }, [geometry, material])

  const lastMorph = useRef(-1)
  useFrame(() => {
    const m = reveal.morph
    const opacity = 1 - Math.min(1, Math.max(0, (m - 0.3) / 0.55))
    material.opacity = opacity * 0.95
    if (mesh.current) mesh.current.visible = opacity > 0.001
    if (m === lastMorph.current || opacity <= 0.001) return
    lastMorph.current = m
    const pos = geometry.attributes.position as THREE.BufferAttribute
    const arr = pos.array as Float32Array
    for (let i = 0; i < arr.length; i++) arr[i] = flat[i] + (wrapped[i] - flat[i]) * m
    pos.needsUpdate = true
    geometry.computeBoundingSphere()
  })

  return <mesh ref={mesh} geometry={geometry} material={material} renderOrder={-1} raycast={() => null} />
}
