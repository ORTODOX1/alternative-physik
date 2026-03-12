'use client';

import { useState, useMemo } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import Card from '@/components/ui/Card';
import ParameterSlider from '@/components/controls/ParameterSlider';
import { LATTICE, SCREENING_EXPERIMENTAL, DIFFUSION } from '@/lib/constants';
import { generateDiffusionData } from '@/lib/physics-engine';

const screeningBarData = Object.entries(SCREENING_EXPERIMENTAL)
  .map(([name, d]) => ({ name, Us_eV: d.Us_eV, error: d.error_eV, source: d.source }))
  .sort((a, b) => b.Us_eV - a.Us_eV);

const latticeBarData = Object.entries(LATTICE).map(([name, d]) => ({
  name, a_A: d.a_A, debye_K: d.debye_K, e_density: d.e_density_A3, structure: d.structure, color: d.color,
}));

const diffusionMetals = Object.keys(DIFFUSION);

export default function MaterialsPage() {
  const [tempRange, setTempRange] = useState(600);

  const diffusionData = useMemo(
    () => generateDiffusionData(diffusionMetals, 200, tempRange, 80),
    [tempRange],
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <h1 className="text-2xl font-bold text-white mb-2">Свойства материалов</h1>
      <p className="text-gray-400 mb-6">Экранирование, решётки, диффузия дейтерия</p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Screening energies */}
        <Card title="Энергии экранирования Us (eV)" subtitle="Экспериментальные данные: Kasagi, Raiola, Huke">
          <div className="h-96">
            <ResponsiveContainer>
              <BarChart data={screeningBarData} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis type="number" label={{ value: 'Us (eV)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis type="category" dataKey="name" width={60} tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any, _name: any, props: any) => [`${v} eV (${props?.payload?.source ?? ''})`, 'Us']}
                />
                <Bar dataKey="Us_eV" radius={[0, 4, 4, 0]}>
                  {screeningBarData.map((_, i) => (
                    <Cell key={i} fill={`hsl(${200 + i * 10}, 70%, ${55 - i}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Lattice parameters */}
        <Card title="Параметры решётки" subtitle="Постоянная решётки a (A) и температура Дебая">
          <div className="h-96">
            <ResponsiveContainer>
              <BarChart data={latticeBarData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="name" />
                <YAxis yAxisId="left" label={{ value: 'a (A)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'θ_D (K)', angle: 90, position: 'insideRight', fill: '#9ca3af' }} />
                <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }} />
                <Legend />
                <Bar yAxisId="left" dataKey="a_A" name="a (A)" radius={[4, 4, 0, 0]}>
                  {latticeBarData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
                <Bar yAxisId="right" dataKey="debye_K" name="θ_D (K)" fill="#6366f180" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Diffusion */}
      <Card title="Диффузия дейтерия D(T)" subtitle="D = D₀·exp(-E_a / k_B·T)" className="mb-6">
        <div className="mb-4 max-w-xs">
          <ParameterSlider label="Макс. температура" value={tempRange} min={300} max={1200} step={50} unit="K" onChange={setTempRange} />
        </div>
        <div className="h-80">
          <ResponsiveContainer>
            <LineChart data={diffusionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis dataKey="T_K" label={{ value: 'T (K)', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
              <YAxis label={{ value: 'log₁₀(D) [cm²/s]', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                formatter={(v: any) => [`10^${v.toFixed(1)} cm²/s`]}
                labelFormatter={(v) => `T = ${v} K`}
              />
              <Legend />
              {diffusionMetals.map((metal, i) => (
                <Line key={metal} type="monotone" dataKey={metal} name={metal} stroke={LATTICE[metal]?.color ?? `hsl(${i * 90}, 70%, 55%)`} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Material table */}
      <Card title="Сводная таблица">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a40] text-left text-gray-400">
                <th className="pb-3 pr-4">Металл</th>
                <th className="pb-3 pr-4">Структура</th>
                <th className="pb-3 pr-4">a (A)</th>
                <th className="pb-3 pr-4">θ_D (K)</th>
                <th className="pb-3 pr-4">ρ_e (A⁻³)</th>
                <th className="pb-3 pr-4">Us (eV)</th>
                <th className="pb-3">D₃₀₀ (cm²/s)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(LATTICE).map(([name, l]) => {
                const s = SCREENING_EXPERIMENTAL[name];
                const d = DIFFUSION[name];
                return (
                  <tr key={name} className="border-b border-[#2a2a40]/50 hover:bg-[#22223a] transition-colors">
                    <td className="py-2.5 pr-4 font-semibold" style={{ color: l.color }}>{name}</td>
                    <td className="py-2.5 pr-4 text-gray-300">{l.structure}</td>
                    <td className="py-2.5 pr-4 font-mono text-gray-300">{l.a_A}</td>
                    <td className="py-2.5 pr-4 font-mono text-gray-300">{l.debye_K}</td>
                    <td className="py-2.5 pr-4 font-mono text-gray-300">{l.e_density_A3}</td>
                    <td className="py-2.5 pr-4 font-mono text-blue-400">{s?.Us_eV ?? '—'}</td>
                    <td className="py-2.5 font-mono text-gray-300">{d ? d.D_300K.toExponential(1) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
