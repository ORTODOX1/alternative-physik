'use client'
import { useRef, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Box, Line, Html } from '@react-three/drei'
import * as THREE from 'three'

interface CrystalLatticeProps {
  structure?: 'FCC' | 'BCC' | 'HCP'
  latticeA?: number
  defectConc?: number
  showDeuterium?: boolean
  temperature?: number
}

// Generate FCC lattice positions
function generateFCCPositions(size: number, a: number, defectConc: number) {
  const positions: THREE.Vector3[] = []
  const isDefect: boolean[] = []

  // FCC: corners + face centers
  for (let i = -size; i <= size; i++) {
    for (let j = -size; j <= size; j++) {
      for (let k = -size; k <= size; k++) {
        // Corner atoms
        const corner = new THREE.Vector3(i * a, j * a, k * a)
        positions.push(corner)
        isDefect.push(Math.random() < defectConc)

        // Face centers
        if (i < size && j < size) {
          positions.push(new THREE.Vector3((i + 0.5) * a, (j + 0.5) * a, k * a))
          isDefect.push(Math.random() < defectConc)
        }
        if (i < size && k < size) {
          positions.push(new THREE.Vector3((i + 0.5) * a, j * a, (k + 0.5) * a))
          isDefect.push(Math.random() < defectConc)
        }
        if (j < size && k < size) {
          positions.push(new THREE.Vector3(i * a, (j + 0.5) * a, (k + 0.5) * a))
          isDefect.push(Math.random() < defectConc)
        }
      }
    }
  }
  return { positions, isDefect }
}

// Octahedral interstitial sites in FCC
function generateInterstitialSites(size: number, a: number) {
  const sites: THREE.Vector3[] = []
  for (let i = -size; i < size; i++) {
    for (let j = -size; j < size; j++) {
      for (let k = -size; k < size; k++) {
        // Edge centers
        sites.push(new THREE.Vector3((i + 0.5) * a, j * a, k * a))
        sites.push(new THREE.Vector3(i * a, (j + 0.5) * a, k * a))
        sites.push(new THREE.Vector3(i * a, j * a, (k + 0.5) * a))
      }
    }
  }
  return sites
}

// Deuterium atom that moves through the lattice
function DeuteriumAtom({ sites, speed, delay, onFusion }: {
  sites: THREE.Vector3[], speed: number, delay: number,
  onFusion?: (pos: THREE.Vector3) => void
}) {
  const ref = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.PointLight>(null)
  const [currentSite, setCurrentSite] = useState(Math.floor(Math.random() * sites.length))
  const [nextSite, setNextSite] = useState((currentSite + 1) % sites.length)
  const progress = useRef(delay)

  useFrame((_, delta) => {
    if (!ref.current || sites.length < 2) return
    progress.current += delta * speed

    if (progress.current >= 1) {
      progress.current = 0
      const newCurrent = nextSite
      // Pick random neighbor
      let newNext = Math.floor(Math.random() * sites.length)
      while (newNext === newCurrent) newNext = Math.floor(Math.random() * sites.length)
      setCurrentSite(newCurrent)
      setNextSite(newNext)
    }

    const from = sites[currentSite]
    const to = sites[nextSite]
    if (from && to) {
      ref.current.position.lerpVectors(from, to, progress.current)
      if (glowRef.current) {
        glowRef.current.position.copy(ref.current.position)
      }
    }
  })

  return (
    <group>
      <mesh ref={ref}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial
          color="#00ff88"
          emissive="#00ff44"
          emissiveIntensity={2}
          transparent
          opacity={0.9}
        />
      </mesh>
      <pointLight ref={glowRef} color="#00ff88" intensity={0.5} distance={1.5} />
    </group>
  )
}

// Fusion flash effect
function FusionFlash({ position, active }: { position: THREE.Vector3, active: boolean }) {
  const ref = useRef<THREE.Mesh>(null)
  const [scale, setScale] = useState(0)

  useFrame((_, delta) => {
    if (active && scale < 3) {
      setScale(s => Math.min(s + delta * 8, 3))
    } else if (!active && scale > 0) {
      setScale(s => Math.max(s - delta * 4, 0))
    }
    if (ref.current) {
      ref.current.scale.setScalar(scale)
    }
  })

  if (scale <= 0) return null

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.3, 32, 32]} />
      <meshBasicMaterial
        color="#ffaa00"
        transparent
        opacity={0.6 * (1 - scale / 3)}
      />
    </mesh>
  )
}

export default function CrystalLattice({
  structure = 'FCC',
  latticeA = 1.5,
  defectConc = 0.05,
  showDeuterium = true,
  temperature = 300,
}: CrystalLatticeProps) {
  const groupRef = useRef<THREE.Group>(null)

  const SIZE = 2

  const { positions, isDefect } = useMemo(
    () => generateFCCPositions(SIZE, latticeA, defectConc),
    [latticeA, defectConc]
  )

  const interstitialSites = useMemo(
    () => generateInterstitialSites(SIZE, latticeA),
    [latticeA]
  )

  const dSpeed = useMemo(() => 0.3 + (temperature / 1000) * 2, [temperature])
  const numDeuterium = showDeuterium ? Math.max(3, Math.min(12, Math.floor(6 + defectConc * 20))) : 0

  // Slow rotation
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05
    }
  })

  return (
    <group ref={groupRef}>
      {/* Lattice atoms */}
      {positions.map((pos, i) => (
        isDefect[i] ? (
          // Defect site — vacancy (empty, faint wireframe)
          <mesh key={i} position={pos}>
            <sphereGeometry args={[0.18, 8, 8]} />
            <meshBasicMaterial color="#ff4444" wireframe transparent opacity={0.3} />
          </mesh>
        ) : (
          // Normal atom
          <mesh key={i} position={pos}>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshStandardMaterial
              color="#8899bb"
              metalness={0.8}
              roughness={0.2}
            />
          </mesh>
        )
      ))}

      {/* Deuterium atoms moving through lattice */}
      {Array.from({ length: numDeuterium }).map((_, i) => (
        <DeuteriumAtom
          key={`d-${i}`}
          sites={interstitialSites}
          speed={dSpeed}
          delay={Math.random()}
        />
      ))}

      {/* Ambient glow for defect channels */}
      {defectConc > 0.1 && (
        <pointLight position={[0, 0, 0]} color="#ff6600" intensity={defectConc * 3} distance={SIZE * latticeA * 2} />
      )}
    </group>
  )
}
