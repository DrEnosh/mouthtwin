import { CameraControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { toothRef } from '../../data/fdi'
import { useMouthTwin } from '../../store/useMouthTwin'
import { archPose, PANO_WIDTH } from './archLayout'
import { reveal, toothAnchors, toothWorldCenter } from './runtime'

const FOV = 32
const FRONT_TARGET = new THREE.Vector3(0, 0, 0)
export const HOME = { pos: new THREE.Vector3(66, 50, 150), target: new THREE.Vector3(0, -4, -10) }

export function CameraRig({ hasFlatStart }: { hasFlatStart: boolean }) {
  const cc = useRef<CameraControls>(null)
  const selectedFdi = useMouthTwin((s) => s.selectedFdi)
  const resetKey = useMouthTwin((s) => s.cameraResetKey)
  const revealKey = useMouthTwin((s) => s.revealKey)
  const size = useThree((s) => s.size)
  const tmpP = useRef(new THREE.Vector3())
  const tmpT = useRef(new THREE.Vector3())

  // narrow viewports need the camera further back
  const aspect = size.width / Math.max(1, size.height)
  const fit = THREE.MathUtils.clamp(1.3 / aspect, 1, 1.9)
  // distance at which the flat panoramic plane fills the width of the viewport
  const frontDist = Math.max(190, (PANO_WIDTH * 1.06) / (2 * Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * aspect))
  const FRONT = { pos: new THREE.Vector3(0, 0, frontDist), target: FRONT_TARGET }

  useEffect(() => {
    const c = cc.current
    if (!c) return
    const onStart = () => useMouthTwin.getState().setAutoRotate(false)
    c.addEventListener('controlstart', onStart)
    return () => c.removeEventListener('controlstart', onStart)
  }, [])

  // start of a reveal: jump to the front view
  useEffect(() => {
    const c = cc.current
    if (!c) return
    if (hasFlatStart) c.setLookAt(0, 0, frontDist, 0, 0, 0, false)
    else c.setLookAt(HOME.pos.x * fit, HOME.pos.y * fit, HOME.pos.z * fit, HOME.target.x, HOME.target.y, HOME.target.z, false)
  }, [revealKey, hasFlatStart, fit, frontDist])

  useEffect(() => {
    const c = cc.current
    if (!c || reveal.running) return
    if (selectedFdi === null) {
      c.setLookAt(HOME.pos.x * fit, HOME.pos.y * fit, HOME.pos.z * fit, HOME.target.x, HOME.target.y, HOME.target.z, true)
      return
    }
    const center = toothWorldCenter(selectedFdi)
    if (!center) return
    const ref = toothRef(selectedFdi)
    const anchor = toothAnchors.get(selectedFdi)
    const buccal = anchor
      ? anchor.normal.clone()
      : new THREE.Vector3(0, 0, 1).applyQuaternion(archPose(ref).quaternion).setY(0).normalize()
    const eye = center.clone().addScaledVector(buccal, 78 * fit).add(new THREE.Vector3(0, ref.upper ? -10 : 18, 0))
    c.setLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, true)
  }, [selectedFdi, resetKey, fit])

  useFrame((_, delta) => {
    const c = cc.current
    if (!c) return
    if (reveal.running && hasFlatStart) {
      const m = reveal.morph
      tmpP.current.lerpVectors(FRONT.pos, tmpT.current.copy(HOME.pos).multiplyScalar(fit), m)
      tmpT.current.lerpVectors(FRONT.target, HOME.target, m)
      c.setLookAt(tmpP.current.x, tmpP.current.y, tmpP.current.z, tmpT.current.x, tmpT.current.y, tmpT.current.z, false)
      return
    }
    const s = useMouthTwin.getState()
    if (s.autoRotate && s.selectedFdi === null) c.azimuthAngle += delta * 0.1
  })

  return (
    <CameraControls
      ref={cc}
      makeDefault
      minDistance={25}
      maxDistance={420}
      smoothTime={0.45}
      draggingSmoothTime={0.12}
    />
  )
}
