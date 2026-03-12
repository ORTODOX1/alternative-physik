'use client';

import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Card from '@/components/ui/Card';
import ParameterSlider from '@/components/controls/ParameterSlider';
import PhysicsModeSelector from '@/components/controls/PhysicsModeSelector';
import { SCREENING_EXPERIMENTAL, LOADING, type PhysicsMode } from '@/lib/constants';
import { calculateBarrier, generateCrossSectionData, generateBarrierVsLoading, generateEnergySweep, diffusionCoefficient, enhancementFactor, mckubreExcessPower } from '@/lib/physics-engine';

const MATERIALS = Object.keys(SCREENING_EXPERIMENTAL);

export default function SimulatorPage() {
  const [modes, setModes] = useState<PhysicsMode[]>(['maxwell', 'coulomb_original', 'cherepanov']);
  const [material, setMaterial] = useState('Pd');
  const [energy, setEnergy] = useState(2.5);
  const [temperature, setTemperature] = useState(340);
  const [loadingRatio, setLoadingRatio] = useState(0.90);
  const [current, setCurrent] = useState(0.5);

  const Us = SCREENING_EXPERIMENTAL[material]?.Us_eV ?? 50;

  // Barrier calculations for each mode
  const barriers = useMemo(
    () => modes.map((m) => calculateBarrier(m, material, energy, temperature, loadingRatio)),
    [modes, material, energy, temperature, loadingRatio],
  );

  // Enhancement
  const enhancement = useMemo(() => enhancementFactor(energy, Us), [energy, Us]);

  // Diffusion
  const diff = useMemo(() => {
    try { return diffusionCoefficient(material.replace(/_.*/, ''), temperature); } catch { return null; }
  }, [material, temperature]);

  // McKubre excess power
  const excessPower = useMemo(
    () => mckubreExcessPower(current, loadingRatio, 0.001),
    [current, loadingRatio],
  );

  // Charts
  const energySweep = useMemo(
    () => generateEnergySweep(material, temperature, loadingRatio, 0.5, 50, 100),
    [material, temperature, loadingRatio],
  );

  const loadingSweep = useMemo(
    () => generateBarrierVsLoading(material, energy, temperature),
    [material, energy, temperature],
  );

  // Temperature sweep
  const tempSweep = useMemo(() => {
    const data = [];
    for (let T = 200; T <= 800; T += 10) {
      const point: Record<string, number> = { T_K: T };
      for (const m of modes) {
        const r = calculateBarrier(m, material, energy, T, loadingRatio);
        point[m] = Math.log10(Math.max(r.reaction_rate_relative, 1e-300));
      }
      data.push(point);
    }
    return data;
  }, [modes, material, energy, loadingRatio]);

  const modeColors: Record<string, string> = { maxwell: '#3b82f6', coulomb_original: '#10b981', cherepanov: '#f59e0b' };
  const modeLabels: Record<string, string> = { maxwell: 'Maxwell', coulomb_original: 'Кулон (ориг.)', cherepanov: 'Черепанов' };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <h1 className="text-2xl font-bold text-white mb-2">Интерактивный симулятор</h1>
      <p className="text-gray-400 mb-6">Крутите параметры — графики обновляются в реальном времени</p>

      {/* Controls panel */}
      <Card className="mb-6" accent="blue">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <label className="text-sm text-gray-300 mb-2 block">Материал мишени</label>
            <select value={material} onChange={(e) => setMaterial(e.target.value)} className="w-full bg-[#12121a] border border-[#2a2a40] rounded-lg px-3 py-2 text-white text-sm">
              {MATERIALS.map((m) => <option key={m} value={m}>{m} (Us={SCREENING_EXPERIMENTAL[m].Us_eV} eV)</option>)}
            </select>
          </div>
          <ParameterSlider label="Энергия пучка" value={energy} min={0.1} max={50} step={0.1} unit="keV" onChange={setEnergy} formatValue={(v) => v.toFixed(1)} />
          <ParameterSlider label="Температура" value={temperature} min={200} max={800} step={5} unit="K" onChange={setTemperature} />
          <ParameterSlider label="Загрузка D/Pd" value={loadingRatio} min={0} max={1} step={0.01} onChange={setLoadingRatio} formatValue={(v) => v.toFixed(2)} />
          <ParameterSlider label="Ток электролиза" value={current} min={0} max={2} step={0.01} unit="A/cm²" onChange={setCurrent} formatValue={(v) => v.toFixed(2)} />
          <div>
            <label className="text-sm text-gray-300 mb-2 block">Режимы физики</label>
            <PhysicsModeSelector selected={modes} onChange={setModes} />
          </div>
        </div>
      </Card>

      {/* Results summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="rounded-xl border border-blue-500/30 bg-blue-500/5 p-4">
          <div className="text-xs text-gray-400">Экранирование</div>
          <div className="text-lg font-bold font-mono text-blue-400">{Us} eV</div>
        </div>
        <div className="rounded-xl border border-green-500/30 bg-green-500/5 p-4">
          <div className="text-xs text-gray-400">Enhancement ({energy} keV)</div>
          <div className="text-lg font-bold font-mono text-green-400">{enhancement > 1e6 ? enhancement.toExponential(1) : enhancement.toFixed(1)}×</div>
        </div>
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <div className="text-xs text-gray-400">Диффузия ({temperature}K)</div>
          <div className="text-lg font-bold font-mono text-amber-400">{diff ? diff.toExponential(1) : '—'} cm²/s</div>
        </div>
        <div className="rounded-xl border border-purple-500/30 bg-purple-500/5 p-4">
          <div className="text-xs text-gray-400">McKubre P_ex</div>
          <div className="text-lg font-bold font-mono text-purple-400">{excessPower > 0 ? excessPower.toFixed(1) : '0'} W</div>
        </div>
      </div>

      {/* Barrier results per mode */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {barriers.map((b) => (
          <Card key={b.mode}>
            <div className="flex items-center gap-2 mb-3">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: modeColors[b.mode] }} />
              <h3 className="text-white font-semibold">{modeLabels[b.mode]}</h3>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between"><span className="text-gray-500">Барьер</span><span className="font-mono text-gray-300">{b.barrier_keV.toFixed(0)} keV</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Эффективный</span><span className="font-mono text-gray-300">{b.effective_barrier_keV.toFixed(1)} keV</span></div>
              <div className="flex justify-between"><span className="text-gray-500">P(проник.)</span><span className="font-mono text-blue-400">{b.penetration_probability.toExponential(2)}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Rate (отн.)</span><span className="font-mono text-amber-400">{b.reaction_rate_relative.toExponential(2)}</span></div>
            </div>
            <p className="text-xs text-gray-500 mt-3">{b.notes}</p>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Проникновение vs энергия" subtitle={`D/Pd=${loadingRatio}, T=${temperature}K`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={energySweep}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="E_keV" label={{ value: 'E (keV)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(P)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }} formatter={(v: any) => v.toFixed(1)} />
                <Legend />
                {modes.map((m) => <Line key={m} type="monotone" dataKey={m} name={modeLabels[m]} stroke={modeColors[m]} dot={false} strokeWidth={2} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Скорость реакции vs загрузка" subtitle={`E=${energy} keV, T=${temperature}K`}>
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={loadingSweep}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="DPd" label={{ value: 'D/Pd', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(rate)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }} formatter={(v: any) => v.toFixed(1)} />
                <Legend />
                {modes.includes('maxwell') && <Line type="monotone" dataKey="maxwell" name="Maxwell" stroke="#3b82f6" dot={false} strokeWidth={2} />}
                {modes.includes('coulomb_original') && <Line type="monotone" dataKey="coulomb" name="Кулон (ориг.)" stroke="#10b981" dot={false} strokeWidth={2} />}
                {modes.includes('cherepanov') && <Line type="monotone" dataKey="cherepanov" name="Черепанов" stroke="#f59e0b" dot={false} strokeWidth={2} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Скорость реакции vs температура" subtitle={`E=${energy} keV, D/Pd=${loadingRatio}`} className="lg:col-span-2">
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={tempSweep}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="T_K" label={{ value: 'T (K)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis label={{ value: 'log₁₀(rate)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }} formatter={(v: any) => v.toFixed(1)} />
                <Legend />
                {modes.map((m) => <Line key={m} type="monotone" dataKey={m} name={modeLabels[m]} stroke={modeColors[m]} dot={false} strokeWidth={2} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Loading thresholds */}
      <Card title="Пороги загрузки D/Pd (справка)" className="mt-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          {([
            ['Max (1 atm, RT)', LOADING.Pd_max_1atm_RT],
            ['Max (электролиз)', LOADING.Pd_max_electrolysis],
            ['LENR порог (McKubre)', LOADING.LENR_threshold_McKubre],
            ['LENR порог (Storms)', LOADING.LENR_threshold_Storms],
          ] as const).map(([label, val]) => (
            <div key={label}>
              <span className="text-gray-500 text-xs block">{label}</span>
              <span className={`font-mono text-lg ${loadingRatio >= val ? 'text-green-400' : 'text-gray-400'}`}>{val}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
