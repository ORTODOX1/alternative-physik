'use client';

import Link from 'next/link';
import Card from '@/components/ui/Card';
import { NUCLEAR, EXCESS_HEAT_DATA, SCREENING_EXPERIMENTAL, PHYSICS_MODES, TRANSMUTATION_DATA } from '@/lib/constants';

const STATS = [
  { label: 'Экспериментов', value: EXCESS_HEAT_DATA.length, color: '#3b82f6' },
  { label: 'Материалов', value: Object.keys(SCREENING_EXPERIMENTAL).length, color: '#10b981' },
  { label: 'Режимов физики', value: PHYSICS_MODES.length, color: '#f59e0b' },
  { label: 'Трансмутаций', value: TRANSMUTATION_DATA.length, color: '#8b5cf6' },
];

const SECTIONS = [
  { href: '/physics', title: 'Сравнение физических моделей', desc: '3 режима: Maxwell, оригинальный Кулон, Черепанов. Барьеры, сечения, предсказания', accent: 'blue' as const, icon: '⚛' },
  { href: '/materials', title: 'Свойства материалов', desc: 'Решётки, диффузия дейтерия, экранирование. Pd, Ni, Ti, Fe, Au, Pt, W', accent: 'green' as const, icon: '◆' },
  { href: '/experiments', title: 'Экспериментальные данные', desc: 'Excess heat, COP, трансмутации. Fleischmann-Pons, McKubre, Iwamura', accent: 'amber' as const, icon: '◉' },
  { href: '/simulator', title: 'Интерактивный симулятор', desc: 'Крутите параметры в реальном времени. Расчёт 3 моделей параллельно', accent: 'purple' as const, icon: '▶' },
  { href: '/tsc', title: 'TSC теория Takahashi', desc: 'EQPET экранирование, барьерные факторы, конденсация 4D → ⁸Be*', accent: 'red' as const, icon: '✦' },
];

export default function DashboardPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 animate-fade-in">
      <div className="mb-10">
        <h1 className="text-3xl sm:text-4xl font-bold text-white mb-3">
          LENR Alternative Physics <span className="text-blue-400">Simulator</span>
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl">
          ML-платформа для симуляции ядерных процессов. Три режима физики: стандартный Maxwell,
          оригинальный Кулон (масса электричества) и Черепанов (фотонная масса).
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-10">
        {STATS.map((s) => (
          <div key={s.label} className="rounded-xl border border-[#2a2a40] bg-[#1a1a2e] p-4">
            <div className="text-2xl font-bold font-mono" style={{ color: s.color }}>{s.value}</div>
            <div className="text-sm text-gray-400 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      <Card title="Ядерные константы D-D" className="mb-8">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
          {([
            ['Энергия Гамова', `${NUCLEAR.gamow_energy_DD_keV} keV`],
            ['Кулоновский барьер', `${NUCLEAR.coulomb_barrier_vacuum_keV} keV`],
            ['D+D → T+p', `${NUCLEAR.Q_DpT_MeV} MeV`],
            ['D+D → ³He+n', `${NUCLEAR.Q_Dn3He_MeV} MeV`],
            ['D+D → ⁴He+γ', `${NUCLEAR.Q_D4He_gamma_MeV} MeV`],
            ['4D → ⁸Be*', `${NUCLEAR.Q_4D_8Be_MeV} MeV`],
            ['⁴He связь', `${NUCLEAR.He4_binding_MeV} MeV`],
            ['α (тонкая стр.)', `1/${(1 / NUCLEAR.fine_structure_alpha).toFixed(3)}`],
          ] as const).map(([label, value]) => (
            <div key={label} className="flex flex-col">
              <span className="text-gray-500 text-xs">{label}</span>
              <span className="font-mono text-white">{value}</span>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
        {PHYSICS_MODES.map((m) => (
          <div key={m.id} className="rounded-xl border bg-[#1a1a2e] p-5 transition-all hover:bg-[#22223a]" style={{ borderColor: `${m.color}30` }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: m.color }} />
              <h3 className="text-white font-semibold">{m.label}</h3>
            </div>
            <div className="space-y-2 text-sm">
              <div><span className="text-gray-500">Заряд: </span><span className="text-gray-300">{m.charge_unit}</span></div>
              <div><span className="text-gray-500">Сила: </span><span className="text-gray-300 font-mono text-xs">{m.force_law}</span></div>
              <div><span className="text-gray-500">Барьер: </span><span className="text-gray-300">{m.barrier_type}</span></div>
              <div><span className="text-gray-500">Эл. поле: </span><span className={m.field_exists ? 'text-green-400' : 'text-red-400'}>{m.field_exists ? 'Да' : 'Нет'}</span></div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="text-xl font-bold text-white mb-4">Разделы</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {SECTIONS.map((s) => (
          <Link key={s.href} href={s.href}>
            <Card accent={s.accent} hover className="h-full">
              <div className="flex items-start gap-3">
                <span className="text-2xl">{s.icon}</span>
                <div>
                  <h3 className="text-white font-semibold mb-1">{s.title}</h3>
                  <p className="text-sm text-gray-400">{s.desc}</p>
                </div>
              </div>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
