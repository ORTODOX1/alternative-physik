'use client'
import { useRef, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Sphere, Text, Line } from '@react-three/drei'
import * as THREE from 'three'

interface ParticleCollisionProps {
  physicsMode: 'maxwell' | 'cherepanov'
  barrierHeight?: number  // 0-1
  defectConc?: number     // 0-0.5
}

// Explosion particles
function ExplosionParticles({ active, position }: { active: boolean, position: [number, number, number] }) {
  const ref = useRef<THREE.Points>(null)
  const [particles] = useState(() => {
    const count = 200
    const positions = new Float32Array(count * 3)
    const velocities = new Float32Array(count * 3)
    const colors = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      positions[i * 3] = 0
      positions[i * 3 + 1] = 0
      positions[i * 3 + 2] = 0
      const theta = Math.random() * Math.PI * 2
      const phi = Math.random() * Math.PI
      const speed = 2 + Math.random() * 6
      velocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed
      velocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed
      velocities[i * 3 + 2] = Math.cos(phi) * speed
      // Gold-orange gradient
      colors[i * 3] = 1
      colors[i * 3 + 1] = 0.5 + Math.random() * 0.5
      colors[i * 3 + 2] = Math.random() * 0.3
    }
    return { positions, velocities, colors }
  })

  const timeRef = useRef(0)
  const [visible, setVisible] = useState(false)

  useFrame((_, delta) => {
    if (!ref.current) return
    const geo = ref.current.geometry
    const pos = geo.attributes.position as THREE.BufferAttribute

    if (active && !visible) {
      setVisible(true)
      timeRef.current = 0
      // Reset positions
      for (let i = 0; i < pos.count; i++) {
        pos.setXYZ(i, 0, 0, 0)
      }
    }

    if (visible) {
      timeRef.current += delta
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i) + particles.velocities[i * 3] * delta
        const y = pos.getY(i) + particles.velocities[i * 3 + 1] * delta
        const z = pos.getZ(i) + particles.velocities[i * 3 + 2] * delta
        pos.setXYZ(i, x, y, z)
      }
      pos.needsUpdate = true

      if (timeRef.current > 1.5) {
        setVisible(false)
      }
    }
  })

  if (!visible) return null

  return (
    <points ref={ref} position={position}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[particles.positions.slice(), 3]}
          count={200}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[particles.colors, 3]}
          count={200}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        vertexColors
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  )
}

// Cherepanov lattice barrier with holes
function CherepanovBarrier({ height, defects }: { height: number, defects: number }) {
  const gridSize = 8
  const cellSize = 0.5
  const holes = useMemo(() => {
    const set = new Set<string>()
    const numHoles = Math.floor(defects * gridSize * gridSize)
    while (set.size < numHoles) {
      const r = Math.floor(Math.random() * gridSize)
      const c = Math.floor(Math.random() * gridSize)
      set.add(`${r}-${c}`)
    }
    return set
  }, [defects])

  return (
    <group position={[0, 0, 0]}>
      {Array.from({ length: gridSize }).map((_, row) =>
        Array.from({ length: gridSize }).map((_, col) => {
          const isHole = holes.has(`${row}-${col}`)
          const y = (row - gridSize / 2 + 0.5) * cellSize
          const z = (col - gridSize / 2 + 0.5) * cellSize
          return isHole ? (
            <mesh key={`${row}-${col}`} position={[0, y, z]}>
              <boxGeometry args={[0.1, cellSize * 0.8, cellSize * 0.8]} />
              <meshBasicMaterial color="#00ff44" transparent opacity={0.3} wireframe />
            </mesh>
          ) : (
            <mesh key={`${row}-${col}`} position={[0, y, z]}>
              <boxGeometry args={[0.15 * height, cellSize * 0.8, cellSize * 0.8]} />
              <meshStandardMaterial color="#6666aa" metalness={0.6} roughness={0.3} transparent opacity={0.6} />
            </mesh>
          )
        })
      )}
    </group>
  )
}

// Maxwell solid wall barrier
function MaxwellBarrier({ height }: { height: number }) {
  const ref = useRef<THREE.Mesh>(null)

  useFrame(() => {
    if (ref.current) {
      const mat = ref.current.material as THREE.MeshStandardMaterial
      mat.opacity = 0.15 + Math.sin(Date.now() * 0.003) * 0.05
    }
  })

  return (
    <mesh ref={ref} position={[0, 0, 0]}>
      <boxGeometry args={[0.3 * height, 4, 4]} />
      <meshStandardMaterial color="#ff4444" emissive="#ff2222" emissiveIntensity={0.5 * height} transparent opacity={0.2} />
    </mesh>
  )
}

// The wall/barrier between particles
function Barrier({ height, defects, mode }: { height: number, defects: number, mode: string }) {
  if (mode === 'cherepanov') {
    return <CherepanovBarrier height={height} defects={defects} />
  }
  return <MaxwellBarrier height={height} />
}

export default function ParticleCollision({
  physicsMode = 'maxwell',
  barrierHeight = 0.8,
  defectConc = 0.05,
}: ParticleCollisionProps) {
  const d1Ref = useRef<THREE.Mesh>(null)
  const d2Ref = useRef<THREE.Mesh>(null)
  const [fusionEvent, setFusionEvent] = useState(false)
  const cycleRef = useRef(0)
  const [bounced, setBounced] = useState(false)

  const canPenetrate = physicsMode === 'cherepanov' ? defectConc > 0.15 : barrierHeight < 0.3

  useFrame((_, delta) => {
    if (!d1Ref.current || !d2Ref.current) return

    cycleRef.current += delta * 0.8

    const cycle = cycleRef.current % 4 // 4-second cycle

    if (cycle < 2) {
      // Approach
      const t = cycle / 2
      d1Ref.current.position.x = -4 + t * 3.8
      d2Ref.current.position.x = 4 - t * 3.8
      setBounced(false)
      setFusionEvent(false)
    } else if (cycle < 2.2) {
      // At barrier
      if (canPenetrate && !fusionEvent) {
        setFusionEvent(true)
      } else if (!canPenetrate) {
        setBounced(true)
      }
    } else {
      // After collision
      if (canPenetrate) {
        // Fusion — particles merge and disappear
        const t = (cycle - 2.2) / 1.8
        d1Ref.current.position.x = -0.2 + t * 0.2
        d2Ref.current.position.x = 0.2 - t * 0.2
        const scale = Math.max(0, 1 - t * 2)
        d1Ref.current.scale.setScalar(scale)
        d2Ref.current.scale.setScalar(scale)
      } else {
        // Bounce back
        const t = (cycle - 2.2) / 1.8
        d1Ref.current.position.x = -0.2 - t * 3.8
        d2Ref.current.position.x = 0.2 + t * 3.8
        d1Ref.current.scale.setScalar(1)
        d2Ref.current.scale.setScalar(1)
      }
    }

    // Reset scale at start of cycle
    if (cycle < 0.1) {
      d1Ref.current.scale.setScalar(1)
      d2Ref.current.scale.setScalar(1)
    }
  })

  return (
    <group>
      {/* Deuteron 1 */}
      <mesh ref={d1Ref} position={[-4, 0, 0]}>
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial color="#00aaff" emissive="#0066ff" emissiveIntensity={1} />
      </mesh>
      <pointLight position={[-4, 0, 0]} color="#0088ff" intensity={2} distance={3} />

      {/* Deuteron 2 */}
      <mesh ref={d2Ref} position={[4, 0, 0]}>
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial color="#00aaff" emissive="#0066ff" emissiveIntensity={1} />
      </mesh>
      <pointLight position={[4, 0, 0]} color="#0088ff" intensity={2} distance={3} />

      {/* Barrier */}
      <Barrier height={barrierHeight} defects={defectConc} mode={physicsMode} />

      {/* Fusion explosion */}
      <ExplosionParticles active={fusionEvent} position={[0, 0, 0]} />

      {/* Labels */}
      <Text position={[-4, 1.2, 0]} fontSize={0.3} color="#88ccff" anchorX="center">
        Deuteron
      </Text>
      <Text position={[4, 1.2, 0]} fontSize={0.3} color="#88ccff" anchorX="center">
        Deuteron
      </Text>
      <Text position={[0, -2.5, 0]} fontSize={0.25} color={canPenetrate ? '#00ff88' : '#ff4444'} anchorX="center">
        {canPenetrate ? 'FUSION!' : bounced ? 'BLOCKED by barrier' : 'Approaching...'}
      </Text>
      <Text position={[0, 2.5, 0]} fontSize={0.2} color="#888888" anchorX="center">
        {physicsMode === 'maxwell' ? 'Maxwell: solid Coulomb wall' : `Cherepanov: lattice with ${Math.round(defectConc * 100)}% defect channels`}
      </Text>
    </group>
  )
}
