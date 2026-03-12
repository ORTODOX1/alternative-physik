'use client';

import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Card from '@/components/ui/Card';
import ParameterSlider from '@/components/controls/ParameterSlider';
import PhysicsModeSelector from '@/components/controls/PhysicsModeSelector';
import { PHYSICS_MODES, SCREENING_EXPERIMENTAL, type PhysicsMode } from '@/lib/constants';
import { generateCrossSectionData, generateBarrierVsLoading, generateEnergySweep } from '@/lib/physics-engine';

const MATERIALS = Object.keys(SCREENING_EXPERIMENTAL);

export default function PhysicsPage() {
  const [modes, setModes] = useState<PhysicsMode[]>(['maxwell', 'coulomb_original', 'cherepanov']);
  const [material, setMaterial] = useState('Pd');
  const [energy, setEnergy] = useState(5);
  const [temperature, setTemperature] = useState(340);
  const [loading, setLoading] = useState(0.9);

  const Us = SCREENING_EXPERIMENTAL[material]?.Us_eV ?? 50;

  const crossSectionData = useMemo(
    () => generateCrossSectionData(0.5, 50, 100, Us),
    [Us],
  );

  const barrierData = useMemo(
    () => generateBarrierVsLoading(material, energy, temperature),
    [material, energy, temperature],
  );

  const energySweepData = useMemo(
    () => generateEnergySweep(material, temperature, loading, 0.5, 50, 80),
    [material, temperature, loading],
  );

  const modeColors: Record<string, string> = { maxwell: '#3b82f6', coulomb: '#10b981', cherepanov: '#f59e0b' };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <h1 className="text-2xl font-bold text-white mb-2">Сравнение физических моделей</h1>
      <p className="text-gray-400 mb-6">Три интерпретации электромагнитных взаимодействий и их влияние на ядерные реакции</p>

      {/* Controls */}
      <Card className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <label className="text-sm text-gray-300 mb-2 block">Материал</label>
            <select
              value={material}
              onChange={(e) => setMaterial(e.target.value)}
              className="w-full bg-[#12121a] border border-[#2a2a40] rounded-lg px-3 py-2 text-white text-sm"
            >
              {MATERIALS.map((m) => (
                <option key={m} value={m}>{m} ({SCREENING_EXPERIMENTAL[m].Us_eV} eV)</option>
              ))}
            </select>
          </div>
          <ParameterSlider label="Энергия (keV)" value={energy} min={0.5} max={50} step={0.5} unit="keV" onChange={setEnergy} />
          <ParameterSlider label="Температура" value={temperature} min={200} max={800} step={10} unit="K" onChange={setTemperature} />
          <ParameterSlider label="Загрузка D/Pd" value={loading} min={0} max={1} step={0.01} onChange={setLoading} formatValue={(v) => v.toFixed(2)} />
        </div>
        <div className="mt-4">
          <label className="text-sm text-gray-300 mb-2 block">Режимы физики</label>
          <PhysicsModeSelector selected={modes} onChange={setModes} />
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cross-section chart */}
        <Card title="D-D сечение: голое vs экранированное" subtitle={`${material}, Us = ${Us} eV`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={crossSectionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="E_keV" label={{ value: 'E (keV)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(σ)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  labelStyle={{ color: '#9ca3af' }}
                  formatter={(v: any) => v.toFixed(2)}
                />
                <Legend />
                <Line type="monotone" dataKey="log_sigma_bare" name="Голое σ" stroke="#ef4444" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="log_sigma_screened" name={`Экранированное (${Us} eV)`} stroke="#3b82f6" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Enhancement chart */}
        <Card title="Enhancement factor" subtitle={`Усиление от экранирования в ${material}`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={crossSectionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="E_keV" label={{ value: 'E (keV)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'Enhancement', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => `${v.toFixed(1)}×`}
                />
                <Line type="monotone" dataKey="enhancement" name="Enhancement" stroke="#10b981" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Barrier vs Loading */}
        <Card title="Скорость реакции vs загрузка D/Pd" subtitle="Сравнение 3 моделей">
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={barrierData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="DPd" label={{ value: 'D/Pd', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(rate)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => v.toFixed(1)}
                />
                <Legend />
                {modes.includes('maxwell') && <Line type="monotone" dataKey="maxwell" name="Maxwell" stroke="#3b82f6" dot={false} strokeWidth={2} />}
                {modes.includes('coulomb_original') && <Line type="monotone" dataKey="coulomb" name="Кулон (ориг.)" stroke="#10b981" dot={false} strokeWidth={2} />}
                {modes.includes('cherepanov') && <Line type="monotone" dataKey="cherepanov" name="Черепанов" stroke="#f59e0b" dot={false} strokeWidth={2} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Energy sweep */}
        <Card title="Вероятность проникновения vs энергия" subtitle={`D/Pd = ${loading.toFixed(2)}, T = ${temperature} K`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={energySweepData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="E_keV" label={{ value: 'E (keV)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(P)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => v.toFixed(1)}
                />
                <Legend />
                {modes.includes('maxwell') && <Line type="monotone" dataKey="maxwell" name="Maxwell" stroke="#3b82f6" dot={false} strokeWidth={2} />}
                {modes.includes('coulomb_original') && <Line type="monotone" dataKey="coulomb" name="Кулон (ориг.)" stroke="#10b981" dot={false} strokeWidth={2} />}
                {modes.includes('cherepanov') && <Line type="monotone" dataKey="cherepanov" name="Черепанов" stroke="#f59e0b" dot={false} strokeWidth={2} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Physics modes detail */}
      <h2 className="text-xl font-bold text-white mt-10 mb-4">Детали моделей</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {PHYSICS_MODES.map((m) => (
          <Card key={m.id} accent={m.id === 'maxwell' ? 'blue' : m.id === 'coulomb_original' ? 'green' : 'amber'}>
            <h3 className="text-white font-semibold mb-3">{m.label}</h3>
            <p className="text-sm text-gray-400 mb-3">{m.description}</p>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between"><span className="text-gray-500">Единица заряда</span><span className="text-gray-300 font-mono">{m.charge_unit}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Закон силы</span><span className="text-gray-300 font-mono">{m.force_law}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Тип барьера</span><span className="text-gray-300">{m.barrier_type}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Эл. поле</span><span className={m.field_exists ? 'text-green-400' : 'text-red-400'}>{m.field_exists ? 'Существует' : 'Не существует'}</span></div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
