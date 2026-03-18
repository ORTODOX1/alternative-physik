'use client'
import { useState, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, Environment, PerspectiveCamera } from '@react-three/drei'
import dynamic from 'next/dynamic'

// Dynamic imports for 3D components (avoid SSR)
const CrystalLattice = dynamic(() => import('@/components/3d/CrystalLattice'), { ssr: false })
const ParticleCollision = dynamic(() => import('@/components/3d/ParticleCollision'), { ssr: false })
const PeriodicTable3D = dynamic(() => import('@/components/3d/PeriodicTable3D'), { ssr: false })

type Scene = 'crystal' | 'collision' | 'periodic'

const SCENES: { id: Scene; label: string; icon: string; desc: string }[] = [
  { id: 'crystal', label: 'Crystal Lattice', icon: '🔮', desc: 'FCC Pd lattice with deuterium diffusion. See how defects create fusion channels.' },
  { id: 'collision', label: 'Particle Collision', icon: '💥', desc: 'D-D collision: Maxwell (solid wall) vs Cherepanov (lattice with holes).' },
  { id: 'periodic', label: '3D Periodic Table', icon: '⚛️', desc: 'Screening energy as 3D bars. Notice Pd cold-rolled — the anomaly.' },
]

function SceneCanvas({ scene, controls }: { scene: Scene; controls: Record<string, number | string> }) {
  return (
    <Canvas
      style={{ width: '100%', height: '100%', background: '#050510' }}
      gl={{ antialias: true, alpha: false }}
      dpr={[1, 2]}
    >
      <PerspectiveCamera makeDefault position={[6, 4, 6]} fov={50} />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={30}
        autoRotate={scene === 'periodic'}
        autoRotateSpeed={0.5}
      />

      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <pointLight position={[-5, 5, -5]} intensity={0.5} color="#6666ff" />

      {/* Stars background */}
      <Stars radius={50} depth={50} count={3000} factor={4} fade speed={1} />

      <Suspense fallback={null}>
        {scene === 'crystal' && (
          <CrystalLattice
            structure="FCC"
            latticeA={Number(controls.latticeA) || 1.5}
            defectConc={Number(controls.defectConc) || 0.05}
            showDeuterium={true}
            temperature={Number(controls.temperature) || 300}
          />
        )}
        {scene === 'collision' && (
          <ParticleCollision
            physicsMode={(controls.physicsMode as 'maxwell' | 'cherepanov') || 'maxwell'}
            barrierHeight={Number(controls.barrierHeight) || 0.8}
            defectConc={Number(controls.defectConc) || 0.05}
          />
        )}
        {scene === 'periodic' && (
          <PeriodicTable3D />
        )}
      </Suspense>
    </Canvas>
  )
}

export default function ThreeDPage() {
  const [activeScene, setActiveScene] = useState<Scene>('crystal')

  // Controls state per scene
  const [crystalControls, setCrystalControls] = useState({
    defectConc: 0.05,
    temperature: 300,
    latticeA: 1.5,
  })

  const [collisionControls, setCollisionControls] = useState({
    physicsMode: 'maxwell' as string,
    barrierHeight: 0.8,
    defectConc: 0.05,
  })

  const currentControls = activeScene === 'crystal'
    ? crystalControls
    : activeScene === 'collision'
    ? collisionControls
    : {}

  return (
    <div className="min-h-screen bg-[#050510] text-white">
      {/* Header */}
      <div className="px-6 pt-20 pb-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          3D Nuclear Simulation Lab
        </h1>
        <p className="text-gray-500 mt-1">
          Interactive 3D visualizations of nuclear screening and lattice dynamics
        </p>
      </div>

      {/* Scene tabs */}
      <div className="flex gap-2 px-6 mb-4">
        {SCENES.map(s => (
          <button
            key={s.id}
            onClick={() => setActiveScene(s.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeScene === s.id
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
            }`}
          >
            <span className="mr-1">{s.icon}</span>
            {s.label}
          </button>
        ))}
      </div>

      {/* Description */}
      <div className="px-6 mb-4">
        <p className="text-gray-400 text-sm">
          {SCENES.find(s => s.id === activeScene)?.desc}
          <span className="text-gray-600 ml-2">Drag to rotate, scroll to zoom.</span>
        </p>
      </div>

      {/* Main 3D viewport */}
      <div className="mx-6 rounded-xl overflow-hidden border border-gray-800" style={{ height: '60vh' }}>
        <SceneCanvas scene={activeScene} controls={currentControls} />
      </div>

      {/* Controls panel */}
      <div className="px-6 py-4">
        {activeScene === 'crystal' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ControlSlider
              label="Defect Concentration"
              value={crystalControls.defectConc}
              min={0} max={0.5} step={0.01}
              onChange={v => setCrystalControls(c => ({ ...c, defectConc: v }))}
              format={v => `${(v * 100).toFixed(0)}%`}
              description="Cold-rolled = 50%, Annealed = 0.5%"
            />
            <ControlSlider
              label="Temperature"
              value={crystalControls.temperature}
              min={100} max={1000} step={10}
              onChange={v => setCrystalControls(c => ({ ...c, temperature: v }))}
              format={v => `${v.toFixed(0)} K`}
              description="Affects deuterium diffusion speed"
            />
            <ControlSlider
              label="Lattice Spacing"
              value={crystalControls.latticeA}
              min={1.0} max={2.5} step={0.1}
              onChange={v => setCrystalControls(c => ({ ...c, latticeA: v }))}
              format={v => `${v.toFixed(1)} (a.u.)`}
              description="Pd = 3.89 A, Ni = 3.52 A"
            />
          </div>
        )}

        {activeScene === 'collision' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Physics Mode</label>
              <div className="flex gap-2">
                {(['maxwell', 'cherepanov'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setCollisionControls(c => ({ ...c, physicsMode: mode }))}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                      collisionControls.physicsMode === mode
                        ? mode === 'maxwell'
                          ? 'bg-red-600/30 text-red-400 border border-red-500'
                          : 'bg-purple-600/30 text-purple-400 border border-purple-500'
                        : 'bg-gray-800 text-gray-500'
                    }`}
                  >
                    {mode === 'maxwell' ? 'Maxwell (solid wall)' : 'Cherepanov (lattice)'}
                  </button>
                ))}
              </div>
            </div>
            <ControlSlider
              label="Barrier Height"
              value={collisionControls.barrierHeight}
              min={0} max={1} step={0.05}
              onChange={v => setCollisionControls(c => ({ ...c, barrierHeight: v }))}
              format={v => `${(v * 400).toFixed(0)} keV`}
              description="Maxwell barrier strength"
            />
            <ControlSlider
              label="Defect Channels"
              value={collisionControls.defectConc}
              min={0} max={0.5} step={0.01}
              onChange={v => setCollisionControls(c => ({ ...c, defectConc: v }))}
              format={v => `${(v * 100).toFixed(0)}%`}
              description="Cherepanov: holes in lattice wall"
            />
          </div>
        )}

        {activeScene === 'periodic' && (
          <div className="flex gap-6 items-center text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-red-500" /> Ferromagnetic (Ni, Fe, Co)
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-orange-400" /> Paramagnetic (Ti, Pt, Ta)
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-blue-500" /> Diamagnetic (Au, Cu, Ag)
            </div>
            <div className="text-yellow-500 font-medium">
              Pd cold-rolled = 18,200 eV (the anomaly!)
            </div>
          </div>
        )}
      </div>

      {/* Key insight box */}
      <div className="mx-6 mb-8 p-4 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl border border-purple-500/30">
        <p className="text-sm text-gray-300">
          <span className="text-purple-400 font-bold">Key insight:</span>{' '}
          {activeScene === 'crystal' && (
            <>Increase defect concentration to 50% (cold-rolled). Watch how more deuterium atoms
            can navigate through the lattice — defects create channels that reduce the effective barrier by 60×.</>
          )}
          {activeScene === 'collision' && (
            <>Switch between Maxwell and Cherepanov modes. In Maxwell, the barrier is a solid wall — particles bounce.
            In Cherepanov, the barrier is a lattice — defects create holes that particles pass through.</>
          )}
          {activeScene === 'periodic' && (
            <>Notice the towering Pd* bar — that is cold-rolled Pd with 18,200 eV screening.
            Standard physics predicts ALL bars should be roughly the same height (~30 eV). They are not.</>
          )}
        </p>
      </div>
    </div>
  )
}

// Reusable slider control
function ControlSlider({ label, value, min, max, step, onChange, format, description }: {
  label: string, value: number, min: number, max: number, step: number,
  onChange: (v: number) => void, format: (v: number) => string, description?: string
}) {
  return (
    <div className="bg-gray-900/50 rounded-lg p-3 border border-gray-800">
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm text-gray-300">{label}</label>
        <span className="text-sm font-mono text-blue-400">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
      />
      {description && (
        <p className="text-xs text-gray-600 mt-1">{description}</p>
      )}
    </div>
  )
}
