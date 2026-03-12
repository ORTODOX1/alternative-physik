'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import Card from '@/components/ui/Card';
import { EQPET_SCREENING, BARRIER_FACTORS, TSC_PARAMS } from '@/lib/constants';

const eqpetChartData = EQPET_SCREENING.map((e) => ({
  ...e,
  log_Us: Math.log10(e.Us_eV),
  log_b0: e.b0_pm > 0 ? Math.log10(e.b0_pm) : 0,
}));

const barrierChartData = BARRIER_FACTORS.map((b) => ({
  label: b.label,
  log_bf_2D: Math.log10(Math.max(b.bf_2D, 1e-300)),
  log_bf_4D: Math.log10(Math.max(b.bf_4D, 1e-300)),
  log_rate: Math.log10(Math.max(b.rate_4D, 1e-300)),
}));

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316'];

export default function TSCPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <h1 className="text-2xl font-bold text-white mb-2">TSC теория Takahashi</h1>
      <p className="text-gray-400 mb-6">
        Tetrahedral Symmetric Condensation: 4 дейтрона конденсируются в тетраэдр → ⁸Be* → 2α
      </p>

      {/* TSC params */}
      <Card title="Параметры TSC конденсации" className="mb-6">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
          {([
            ['Начальная дистанция D-D', `${TSC_PARAMS.initial_dd_distance_pm} pm`],
            ['Радиус TSC (старт)', `${TSC_PARAMS.tsc_radius_start_pm} pm`],
            ['Мин. радиус TSC', `${TSC_PARAMS.tsc_radius_min_fm} fm`],
            ['Время конденсации 4D', `${TSC_PARAMS.condensation_time_4D_fs} фс`],
            ['Время конденсации 4H', `${TSC_PARAMS.condensation_time_4H_fs} фс`],
            ['Макс. плотность TSC', `${TSC_PARAMS.max_tsc_density_per_cm3.toExponential(0)} /см³`],
            ['Макс. fusion rate', `${TSC_PARAMS.max_fusion_rate_MW_per_cm3} МВт/см³`],
            ['n/⁴He ratio', `${TSC_PARAMS.neutron_to_He4_ratio.toExponential(0)}`],
          ] as const).map(([label, value]) => (
            <div key={label}>
              <span className="text-gray-500 text-xs block">{label}</span>
              <span className="font-mono text-white">{value}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* EQPET screening energies */}
        <Card title="EQPET экранирование" subtitle="Квазичастицы e*(m,Z) и их энергии экранирования">
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={eqpetChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#9ca3af' }} angle={-15} />
                <YAxis label={{ value: 'log₁₀(Us eV)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(_v: any, _name: any, props: any) => {
                    const p = props?.payload;
                    return p ? [`Us=${p.Us_eV} eV, b₀=${p.b0_pm} pm`, 'EQPET'] : [];
                  }}
                />
                <Bar dataKey="log_Us" name="log₁₀(Us)" radius={[4, 4, 0, 0]}>
                  {eqpetChartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Barrier factors */}
        <Card title="Барьерные факторы (E_d = 0.22 eV)" subtitle="Вероятность проникновения для 2D и 4D">
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={barrierChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="label" />
                <YAxis label={{ value: 'log₁₀(factor)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: 8 }}
                  formatter={(v: any) => [v.toFixed(0), 'log₁₀']}
                />
                <Legend />
                <Bar dataKey="log_bf_2D" name="2D barrier" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="log_bf_4D" name="4D barrier" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="log_rate" name="4D rate" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* EQPET table */}
      <Card title="EQPET квазичастицы — полная таблица" className="mb-6">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a40] text-left text-gray-400 text-xs">
                <th className="pb-2 pr-4">e*(m,Z)</th>
                <th className="pb-2 pr-4">Us (eV)</th>
                <th className="pb-2 pr-4">b₀ (pm)</th>
                <th className="pb-2 pr-4">R_dd (pm)</th>
                <th className="pb-2">Глубина ловушки (eV)</th>
              </tr>
            </thead>
            <tbody>
              {EQPET_SCREENING.map((e, i) => (
                <tr key={i} className="border-b border-[#2a2a40]/50 hover:bg-[#22223a]">
                  <td className="py-2.5 pr-4 font-mono" style={{ color: COLORS[i] }}>{e.label}</td>
                  <td className="py-2.5 pr-4 font-mono text-blue-400">{e.Us_eV.toLocaleString()}</td>
                  <td className="py-2.5 pr-4 font-mono text-gray-300">{e.b0_pm}</td>
                  <td className="py-2.5 pr-4 font-mono text-gray-300">{e.Rdd_pm ?? '—'}</td>
                  <td className="py-2.5 font-mono text-gray-300">{e.trap_eV ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Barrier factors table */}
      <Card title="Барьерные факторы при E_d = 0.22 eV">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a40] text-left text-gray-400 text-xs">
                <th className="pb-2 pr-4">e*(m,Z)</th>
                <th className="pb-2 pr-4">BF (2D)</th>
                <th className="pb-2 pr-4">BF (4D)</th>
                <th className="pb-2">Rate 4D (f/s/cluster)</th>
              </tr>
            </thead>
            <tbody>
              {BARRIER_FACTORS.map((b, i) => (
                <tr key={i} className="border-b border-[#2a2a40]/50 hover:bg-[#22223a]">
                  <td className="py-2.5 pr-4 font-mono text-white">{b.label}</td>
                  <td className="py-2.5 pr-4 font-mono text-blue-400">{b.bf_2D.toExponential(0)}</td>
                  <td className="py-2.5 pr-4 font-mono text-green-400">{b.bf_4D.toExponential(0)}</td>
                  <td className="py-2.5 font-mono text-amber-400">{b.rate_4D.toExponential(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Reaction chain */}
      <Card title="Реакционная цепочка TSC" className="mt-6">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          {['4D', '→', 'TSC (1.4 фс)', '→', '⁸Be* (47.6 МэВ)', '→', '2⁴He + γ', '→', '23.8 МэВ'].map((step, i) => (
            <span key={i} className={i % 2 === 1 ? 'text-gray-500 text-lg' : 'px-3 py-1.5 rounded-lg bg-[#12121a] border border-[#2a2a40] font-mono text-white'}>
              {step}
            </span>
          ))}
        </div>
        <p className="text-sm text-gray-400 mt-4">
          Ключевое: нейтрон/⁴He ≈ 10⁻¹² — практически безнейтронная реакция. Вся энергия в альфа-частицы и гамма.
          Максимальная мощность: {TSC_PARAMS.max_fusion_rate_MW_per_cm3} МВт/см³ Pd.
        </p>
      </Card>
    </div>
  );
}

type EqpetChartEntry = (typeof eqpetChartData)[number];
