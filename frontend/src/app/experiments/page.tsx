'use client';

import { useState } from 'react';
import { BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, ZAxis } from 'recharts';
import Card from '@/components/ui/Card';
import { EXCESS_HEAT_DATA, TRANSMUTATION_DATA } from '@/lib/constants';

const METHODS = ['all', ...new Set(EXCESS_HEAT_DATA.map((d) => d.method))];
const MATERIALS_LIST = ['all', ...new Set(EXCESS_HEAT_DATA.map((d) => d.material))];

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316', '#06b6d4', '#ec4899', '#84cc16', '#64748b'];

export default function ExperimentsPage() {
  const [method, setMethod] = useState('all');
  const [materialFilter, setMaterialFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'excess_W' | 'COP' | 'duration_days' | 'reproducibility'>('excess_W');

  const filtered = EXCESS_HEAT_DATA
    .filter((d) => method === 'all' || d.method === method)
    .filter((d) => materialFilter === 'all' || d.material === materialFilter)
    .sort((a, b) => b[sortBy] - a[sortBy]);

  const scatterData = EXCESS_HEAT_DATA.map((d) => ({
    ...d,
    DPd_display: d.DPd ?? 0,
    size: d.COP * 50,
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <h1 className="text-2xl font-bold text-white mb-2">Экспериментальные данные LENR</h1>
      <p className="text-gray-400 mb-6">Excess heat, COP, трансмутации из разных лабораторий мира</p>

      {/* Filters */}
      <Card className="mb-6">
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Метод</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)} className="bg-[#12121a] border border-[#2a2a40] rounded-lg px-3 py-1.5 text-white text-sm">
              {METHODS.map((m) => <option key={m} value={m}>{m === 'all' ? 'Все методы' : m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Материал</label>
            <select value={materialFilter} onChange={(e) => setMaterialFilter(e.target.value)} className="bg-[#12121a] border border-[#2a2a40] rounded-lg px-3 py-1.5 text-white text-sm">
              {MATERIALS_LIST.map((m) => <option key={m} value={m}>{m === 'all' ? 'Все материалы' : m}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Сортировка</label>
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)} className="bg-[#12121a] border border-[#2a2a40] rounded-lg px-3 py-1.5 text-white text-sm">
              <option value="excess_W">Excess heat (W)</option>
              <option value="COP">COP</option>
              <option value="duration_days">Длительность</option>
              <option value="reproducibility">Воспроизводимость</option>
            </select>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Excess heat bar chart */}
        <Card title="Excess Heat по лабораториям" subtitle="Мощность (W)">
          <div className="h-96">
            <ResponsiveContainer>
              <BarChart data={filtered} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis type="number" label={{ value: 'W', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis type="category" dataKey="lab" width={120} tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  formatter={(v: any) => [`${v} W`, 'Excess heat']}
                />
                <Bar dataKey="excess_W" radius={[0, 4, 4, 0]}>
                  {filtered.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* COP bar chart */}
        <Card title="COP (коэффициент мощности)" subtitle="Отношение выхода к входу">
          <div className="h-96">
            <ResponsiveContainer>
              <BarChart data={filtered} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis type="number" label={{ value: 'COP', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis type="category" dataKey="lab" width={120} tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => [`${v}×`, 'COP']}
                />
                <Bar dataKey="COP" radius={[0, 4, 4, 0]}>
                  {filtered.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Scatter: excess heat vs D/Pd */}
        <Card title="Excess Heat vs загрузка D/Pd" subtitle="Размер точки = COP">
          <div className="h-80">
            <ResponsiveContainer>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="DPd_display" name="D/Pd" label={{ value: 'D/Pd', position: 'insideBottom', offset: -5, fill: '#9ca3af' }} />
                <YAxis dataKey="excess_W" name="Excess W" label={{ value: 'W', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <ZAxis dataKey="size" range={[40, 400]} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any, name: any) => [name === 'D/Pd' ? Number(v).toFixed(2) : `${v} W`, name]}
                  labelFormatter={() => ''}
                />
                <Scatter data={scatterData} fill="#3b82f6">
                  {scatterData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Reproducibility */}
        <Card title="Воспроизводимость" subtitle="Доля успешных репликаций">
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={filtered} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="lab" width={120} tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => [`${(v * 100).toFixed(0)}%`, 'Воспроизводимость']}
                />
                <Bar dataKey="reproducibility" radius={[0, 4, 4, 0]}>
                  {filtered.map((d, i) => <Cell key={i} fill={d.reproducibility >= 0.8 ? '#10b981' : d.reproducibility >= 0.5 ? '#f59e0b' : '#ef4444'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* Data table */}
      <Card title="Сводная таблица экспериментов" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a40] text-left text-gray-400 text-xs">
                <th className="pb-2 pr-3">Лаборатория</th>
                <th className="pb-2 pr-3">Материал</th>
                <th className="pb-2 pr-3">Метод</th>
                <th className="pb-2 pr-3">W</th>
                <th className="pb-2 pr-3">COP</th>
                <th className="pb-2 pr-3">D/Pd</th>
                <th className="pb-2 pr-3">T (K)</th>
                <th className="pb-2">Воспр.</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d, i) => (
                <tr key={i} className="border-b border-[#2a2a40]/50 hover:bg-[#22223a] transition-colors">
                  <td className="py-2 pr-3 text-white font-medium">{d.lab}</td>
                  <td className="py-2 pr-3 text-gray-300">{d.material}</td>
                  <td className="py-2 pr-3 text-gray-400">{d.method}</td>
                  <td className="py-2 pr-3 font-mono text-blue-400">{d.excess_W}</td>
                  <td className="py-2 pr-3 font-mono text-amber-400">{d.COP}×</td>
                  <td className="py-2 pr-3 font-mono text-gray-300">{d.DPd ?? '—'}</td>
                  <td className="py-2 pr-3 font-mono text-gray-300">{d.temperature_K}</td>
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      d.reproducibility >= 0.8 ? 'bg-green-500/20 text-green-400' :
                      d.reproducibility >= 0.5 ? 'bg-amber-500/20 text-amber-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {(d.reproducibility * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Transmutations */}
      <Card title="Трансмутации (Iwamura / Mitsubishi)" subtitle="Подтверждено Toyota, SPring-8">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a40] text-left text-gray-400">
                <th className="pb-2 pr-4">Реакция</th>
                <th className="pb-2 pr-4">Δ массы</th>
                <th className="pb-2 pr-4">Детекция</th>
                <th className="pb-2">Источник</th>
              </tr>
            </thead>
            <tbody>
              {TRANSMUTATION_DATA.map((t, i) => (
                <tr key={i} className="border-b border-[#2a2a40]/50 hover:bg-[#22223a]">
                  <td className="py-2.5 pr-4 font-mono text-purple-400">{t.reaction}</td>
                  <td className="py-2.5 pr-4 font-mono text-amber-400">{t.delta_mass}</td>
                  <td className="py-2.5 pr-4 text-gray-300">{t.detection}</td>
                  <td className="py-2.5 text-gray-400">{t.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
