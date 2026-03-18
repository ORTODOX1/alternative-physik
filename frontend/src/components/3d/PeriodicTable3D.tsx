'use client'
import { useRef, useState, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Text, Html } from '@react-three/drei'
import * as THREE from 'three'

interface PeriodicTable3DProps {
  hoveredElement?: string | null
  onHover?: (element: string | null) => void
}

// Screening energy data for visualization
const ELEMENTS_DATA: Record<string, {
  symbol: string, Z: number, row: number, col: number,
  Us_eV: number, magnetic: 'ferro' | 'para' | 'dia',
  label?: string
}> = {
  Be: { symbol: 'Be', Z: 4, row: 1, col: 1, Us_eV: 180, magnetic: 'dia' },
  Al: { symbol: 'Al', Z: 13, row: 2, col: 12, Us_eV: 190, magnetic: 'para' },
  Ti: { symbol: 'Ti', Z: 22, row: 3, col: 3, Us_eV: 65, magnetic: 'para' },
  V:  { symbol: 'V', Z: 23, row: 3, col: 4, Us_eV: 140, magnetic: 'para' },
  Mn: { symbol: 'Mn', Z: 25, row: 3, col: 6, Us_eV: 250, magnetic: 'para' },
  Fe: { symbol: 'Fe', Z: 26, row: 3, col: 7, Us_eV: 200, magnetic: 'ferro' },
  Co: { symbol: 'Co', Z: 27, row: 3, col: 8, Us_eV: 350, magnetic: 'ferro' },
  Ni: { symbol: 'Ni', Z: 28, row: 3, col: 9, Us_eV: 420, magnetic: 'ferro' },
  Cu: { symbol: 'Cu', Z: 29, row: 3, col: 10, Us_eV: 120, magnetic: 'dia' },
  Zr: { symbol: 'Zr', Z: 40, row: 4, col: 3, Us_eV: 297, magnetic: 'para' },
  Nb: { symbol: 'Nb', Z: 41, row: 4, col: 4, Us_eV: 160, magnetic: 'para' },
  Pd: { symbol: 'Pd', Z: 46, row: 4, col: 9, Us_eV: 310, magnetic: 'dia', label: 'Pd (poly)' },
  Ag: { symbol: 'Ag', Z: 47, row: 4, col: 10, Us_eV: 95, magnetic: 'dia' },
  Ta: { symbol: 'Ta', Z: 73, row: 5, col: 4, Us_eV: 309, magnetic: 'para' },
  W:  { symbol: 'W', Z: 74, row: 5, col: 5, Us_eV: 150, magnetic: 'para' },
  Pt: { symbol: 'Pt', Z: 78, row: 5, col: 9, Us_eV: 122, magnetic: 'para' },
  Au: { symbol: 'Au', Z: 79, row: 5, col: 10, Us_eV: 70, magnetic: 'dia' },
}

// Special: Pd cold-rolled (the anomaly)
const PD_COLD_ROLLED = {
  symbol: 'Pd*', Z: 46, row: 4, col: 13, Us_eV: 18200,
  magnetic: 'dia' as const, label: 'Pd cold-rolled'
}

const MAGNETIC_COLORS = {
  ferro: new THREE.Color('#ff4444'),
  para: new THREE.Color('#ffaa44'),
  dia: new THREE.Color('#4488ff'),
}

function ElementBar({ data, maxHeight, isSpecial, isHovered, onHover }: {
  data: typeof ELEMENTS_DATA[string],
  maxHeight: number,
  isSpecial?: boolean,
  isHovered: boolean,
  onHover: (symbol: string | null) => void
}) {
  const ref = useRef<THREE.Mesh>(null)
  const [targetScale, setTargetScale] = useState(1)
  const currentScale = useRef(0)

  const height = useMemo(() => {
    // Log scale for height
    return Math.log10(Math.max(data.Us_eV, 10)) / Math.log10(20000) * maxHeight
  }, [data.Us_eV, maxHeight])

  const color = MAGNETIC_COLORS[data.magnetic]

  useFrame((_, delta) => {
    if (!ref.current) return
    // Animate grow
    if (currentScale.current < 1) {
      currentScale.current = Math.min(currentScale.current + delta * 1.5, 1)
      ref.current.scale.y = currentScale.current
    }
    // Hover pulse
    const target = isHovered ? 1.15 : 1
    const s = ref.current.scale.x
    ref.current.scale.x = THREE.MathUtils.lerp(s, target, delta * 8)
    ref.current.scale.z = ref.current.scale.x
  })

  const x = data.col * 1.2
  const z = data.row * 1.2

  return (
    <group position={[x, 0, z]}>
      <mesh
        ref={ref}
        position={[0, height / 2, 0]}
        onPointerEnter={() => onHover(data.symbol)}
        onPointerLeave={() => onHover(null)}
      >
        <boxGeometry args={[0.9, height, 0.9]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHovered ? 0.8 : (isSpecial ? 0.5 : 0.15)}
          metalness={0.4}
          roughness={0.3}
          transparent={!isHovered}
          opacity={isHovered ? 1 : 0.85}
        />
      </mesh>
      {/* Element symbol */}
      <Text
        position={[0, height + 0.3, 0]}
        fontSize={0.35}
        color="white"
        anchorX="center"
        anchorY="bottom"
      >
        {data.symbol}
      </Text>
      {/* Value on hover */}
      {isHovered && (
        <Html position={[0, height + 1.2, 0]} center>
          <div style={{
            background: 'rgba(0,0,0,0.9)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '8px',
            fontSize: '13px',
            whiteSpace: 'nowrap',
            border: `1px solid ${color.getStyle()}`,
          }}>
            <strong>{data.label || data.symbol}</strong> (Z={data.Z})<br />
            Screening: <span style={{ color: '#ffa657', fontWeight: 700 }}>
              {data.Us_eV.toLocaleString()} eV
            </span><br />
            Magnetic: {data.magnetic}
          </div>
        </Html>
      )}
    </group>
  )
}

export default function PeriodicTable3D({ hoveredElement, onHover }: PeriodicTable3DProps) {
  const [localHovered, setLocalHovered] = useState<string | null>(null)
  const hovered = hoveredElement ?? localHovered
  const setHovered = onHover ?? setLocalHovered
  const groupRef = useRef<THREE.Group>(null)

  const maxHeight = 8

  // Slow rotation
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.03
    }
  })

  return (
    <group ref={groupRef} position={[-8, 0, -4]}>
      {/* Regular elements */}
      {Object.values(ELEMENTS_DATA).map(data => (
        <ElementBar
          key={data.symbol}
          data={data}
          maxHeight={maxHeight}
          isHovered={hovered === data.symbol}
          onHover={setHovered}
        />
      ))}

      {/* Pd cold-rolled — THE ANOMALY */}
      <ElementBar
        data={PD_COLD_ROLLED}
        maxHeight={maxHeight}
        isSpecial
        isHovered={hovered === PD_COLD_ROLLED.symbol}
        onHover={setHovered}
      />

      {/* Floor grid */}
      <gridHelper args={[20, 20, '#222244', '#111133']} position={[8, 0, 4]} />

      {/* Legend */}
      <Text position={[17, 0.2, 1]} fontSize={0.3} color="#ff4444" anchorX="left">
        Ferromagnetic
      </Text>
      <Text position={[17, 0.2, 2]} fontSize={0.3} color="#ffaa44" anchorX="left">
        Paramagnetic
      </Text>
      <Text position={[17, 0.2, 3]} fontSize={0.3} color="#4488ff" anchorX="left">
        Diamagnetic
      </Text>
      <Text position={[17, 0.2, 5]} fontSize={0.25} color="#888888" anchorX="left">
        Height = log(screening energy)
      </Text>
    </group>
  )
}
