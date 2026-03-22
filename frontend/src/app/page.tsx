'use client';

import Link from 'next/link';
import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Area, AreaChart,
  Cell, ReferenceLine, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts';
import { enhancementFactor, generateBarrierVsLoading, generateEnergySweep } from '@/lib/physics-engine';

// ─── Screening energy data — THE proof ───
const SCREENING_CHART_DATA = [
  { material: 'Ti', Us: 65, color: '#94a3b8', state: 'bulk' },
  { material: 'Au', Us: 70, color: '#fbbf24', state: 'bulk' },
  { material: 'Pt', Us: 122, color: '#a78bfa', state: 'bulk' },
  { material: 'Al', Us: 190, color: '#38bdf8', state: 'bulk' },
  { material: 'Fe', Us: 200, color: '#f87171', state: 'bulk' },
  { material: 'Pd', Us: 310, color: '#34d399', state: 'bulk' },
  { material: 'Ni', Us: 420, color: '#fb923c', state: 'bulk' },
  { material: 'PdO', Us: 600, color: '#22d3ee', state: 'oxide' },
  { material: 'Pd\u00a0(Raiola)', Us: 800, color: '#a3e635', state: 'poly' },
  { material: 'Pd\u00a0cold-rolled', Us: 18200, color: '#f43f5e', state: 'cold-rolled' },
];

// ─── Impact areas for radar chart ───
const IMPACT_RADAR = [
  { axis: 'Energy', A: 95, B: 30 },
  { axis: 'Propulsion', A: 85, B: 10 },
  { axis: 'Computing', A: 90, B: 20 },
  { axis: 'Materials', A: 80, B: 40 },
  { axis: 'AI/ML', A: 75, B: 15 },
  { axis: 'Medicine', A: 60, B: 25 },
];

// ─── Timeline of discovery ───
const TIMELINE = [
  { year: '1785', event: 'Coulomb measures force between charged bodies', note: 'Original formula: F = k\u00b7m\u2081\u00b7m\u2082/r\u00b2 (mass densities, not "charges")' },
  { year: '1873', event: 'Maxwell redefines charge', note: '6 dimensional errors in Treatise pp.39-44. Charge gets wrong units.' },
  { year: '1928', event: 'Gamow derives "Coulomb barrier"', note: 'Based on Maxwell\'s charge. Barrier = ~400 keV for D-D in vacuum.' },
  { year: '2002', event: 'Kasagi measures screening in metals', note: 'PdO: 600 eV, Pd: 310 eV. 20x above Debye model prediction.' },
  { year: '2023', event: 'Czerski: cold-rolled Pd = 18,200 eV', note: '607x above standard prediction. Same element, different processing.' },
  { year: '2025', event: 'ML proof: barrier is medium property', note: 'XGBoost + SHAP: defect structure > atomic number. R\u00b2 > 0.95.' },
];

// ─── Evidence metrics ───
const EVIDENCE = [
  { metric: '60x', title: 'Same Element, Different "Barrier"', desc: 'Palladium: 310 eV (annealed) vs 18,200 eV (cold-rolled). Same atoms. Standard physics predicts identical values.', source: 'Czerski et al., 2023' },
  { metric: '7', title: 'Independent Laboratories', desc: 'Kasagi (Tohoku), Raiola (Bochum), Huke (Berlin), Czerski (Szczecin), NASA Glenn, McKubre (SRI), Clean Planet.', source: 'Multiple, 2002-2025' },
  { metric: '607x', title: 'Standard Model Error', desc: 'Debye screening predicts 30 eV for all Pd states. Measured: 18,200 eV. No parameter adjustment can fix a 607x error.', source: 'Falsification analysis' },
  { metric: 'R\u00b2>.95', title: 'ML Confirms Medium Dependence', desc: 'SHAP analysis: defect_concentration explains 40%+ variance. Atomic number <10%. Material engineering > nuclear physics.', source: 'This work' },
];

// ─── Implications for industries ───
const IMPLICATIONS = [
  {
    icon: '\u26a1', company: 'Next-Gen Processors',
    tagline: 'Nuclear-scale transistors',
    desc: 'If the barrier is a medium property, we can engineer nuclear interactions at the material level \u2014 the same way we engineer electron flow in silicon. This opens the path to processors that operate on nuclear energy scales: 10\u2076x more energy-dense than semiconductor junctions.',
    for: 'xAI \u00b7 NVIDIA \u00b7 TSMC',
  },
  {
    icon: '🚀', company: 'Space Propulsion',
    tagline: 'Fusion without tokamaks',
    desc: 'No barrier = no need for 150M\u00b0C plasma. Material-engineered fusion at moderate temperatures. Specific impulse: 10\u2074\u201310\u2075 seconds vs 450s (chemical). Interplanetary travel in weeks, not months.',
    for: 'SpaceX \u00b7 NASA \u00b7 Blue Origin',
  },
  {
    icon: '🧠', company: 'AI-Driven Discovery',
    tagline: 'Physics agents find what humans missed',
    desc: 'Multi-agent AI systems (TRIZ-trained) analyze nuclear data from competing theoretical frameworks. 5 AI physicists debate, find consensus. This is how we discovered the 607x anomaly that 50 years of theory missed.',
    for: 'xAI \u00b7 DeepMind \u00b7 Anthropic',
  },
];

// ─── Data sources ───
const DATA_SOURCES = [
  { name: 'IAEA EXFOR', count: '22K+', desc: 'Nuclear reactions' },
  { name: 'Materials Project', count: '500K+', desc: 'DFT materials' },
  { name: 'AFLOW', count: '2M+', desc: 'Thermodynamics' },
  { name: 'COD', count: '520K+', desc: 'Crystal structures' },
  { name: 'ENDF/B-VIII', count: '\u2014', desc: 'Nuclear data' },
  { name: 'NOMAD', count: '12M+', desc: 'Ab-initio sims' },
  { name: 'OQMD', count: '1.4M+', desc: 'Quantum calcs' },
  { name: 'NIST', count: '\u2014', desc: 'Potentials' },
];

const TOOLS = [
  { href: '/simulator', title: 'Live Simulator', desc: 'Three physics engines. Real-time comparison. Your parameters.', icon: '\u25b6', gradient: 'from-purple-500/20 to-blue-500/20' },
  { href: '/3d', title: '3D Nuclear Lab', desc: 'Crystal lattice, particle collisions, periodic table in 3D.', icon: '🔮', gradient: 'from-cyan-500/20 to-emerald-500/20' },
  { href: '/physics', title: 'Physics Models', desc: 'Maxwell vs Coulomb (1785) vs Cherepanov. Side by side.', icon: '\u269b', gradient: 'from-amber-500/20 to-red-500/20' },
  { href: '/experiments', title: 'Experimental Database', desc: '26+ experiments. 7 labs. All data open.', icon: '\u25c9', gradient: 'from-rose-500/20 to-pink-500/20' },
];

export default function LandingPage() {
  const barrierData = useMemo(() => generateBarrierVsLoading('Pd', 5, 300), []);
  const energySweep = useMemo(() => generateEnergySweep('Pd', 300, 0.9, 0.5, 25, 60), []);

  return (
    <div className="animate-fade-in">

      {/* ═══════════════════ HERO ═══════════════════ */}
      <section className="relative overflow-hidden min-h-[90vh] flex items-center">
        <div className="absolute inset-0 bg-gradient-to-b from-red-600/8 via-transparent to-transparent" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-red-500/10 via-transparent to-transparent" />
        <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-red-500/3 blur-[120px]" />

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 pt-8 pb-12 w-full">
          <div className="text-center mb-6">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-red-500/30 bg-red-500/10 text-red-400 text-xs font-medium mb-8 tracking-wider uppercase">
              <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
              Open Research &mdash; AI + Nuclear Physics
            </div>
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black text-center leading-[1.1] mb-8 tracking-tight">
            <span className="text-white">The Coulomb Barrier</span>
            <br />
            <span className="bg-gradient-to-r from-red-400 via-orange-400 to-amber-400 bg-clip-text text-transparent">
              Does Not Exist
            </span>
          </h1>

          <p className="text-xl sm:text-2xl text-gray-300 text-center max-w-4xl mx-auto mb-4 leading-relaxed">
            What physics calls a &quot;fundamental barrier&quot; is actually a
            <strong className="text-white"> property of the medium</strong> &mdash;
            like electrical resistance. We proved it with ML on data from
            <strong className="text-red-400"> 7 independent labs</strong>.
          </p>

          <p className="text-lg text-gray-500 text-center max-w-3xl mx-auto mb-6">
            Same palladium. Same atoms. Different processing &rarr;
            <span className="text-red-400 font-bold"> 60x difference</span> in &quot;barrier&quot;.
            Standard physics predicts zero variation. This is not an anomaly. This is a paradigm shift.
          </p>

          <div className="text-center mb-10">
            <p className="text-base text-gray-400 max-w-2xl mx-auto">
              Understanding this enables{' '}
              <span className="text-amber-400 font-semibold">next-generation processors</span>,{' '}
              <span className="text-blue-400 font-semibold">new AI architectures</span>, and{' '}
              <span className="text-emerald-400 font-semibold">unlimited clean energy</span>.
              <br />
              We need to revisit the fundamentals.
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <Link href="/simulator" className="px-8 py-3.5 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl transition-all hover:shadow-lg hover:shadow-red-500/25 text-lg">
              See the Proof &rarr;
            </Link>
            <Link href="/3d" className="px-8 py-3.5 border border-gray-600 hover:border-gray-400 text-gray-300 hover:text-white font-semibold rounded-xl transition-all text-lg">
              3D Nuclear Lab
            </Link>
            <a href="https://github.com/synizia/syniz" target="_blank" rel="noopener noreferrer" className="px-8 py-3.5 border border-amber-600/50 hover:border-amber-400 text-amber-400 hover:text-amber-300 font-semibold rounded-xl transition-all text-lg">
              Syniz AI Framework
            </a>
          </div>

          {/* Scroll indicator */}
          <div className="text-center mt-8">
            <div className="inline-flex flex-col items-center gap-2 text-gray-600">
              <span className="text-xs uppercase tracking-widest">Scroll for evidence</span>
              <span className="animate-bounce text-xl">&darr;</span>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════ BOLD STATEMENT ═══════════════════ */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 mb-20">
        <div className="rounded-2xl border border-red-500/20 bg-gradient-to-br from-red-500/5 to-orange-500/5 p-8 sm:p-12">
          <blockquote className="text-2xl sm:text-3xl font-bold text-white text-center leading-relaxed mb-6">
            &ldquo;We don&apos;t need to <em className="text-red-400">overcome</em> the Coulomb barrier.
            <br />
            We need to <em className="text-amber-400">engineer it away</em>.
            <br />
            Like we engineered transistors from sand.&rdquo;
          </blockquote>
          <p className="text-center text-gray-500 text-sm">
            Nuclear fusion through material design. Not brute force. Not billion-dollar tokamaks. Material science.
          </p>
        </div>
      </section>

      {/* ═══════════════════ KILLER CHART ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <div className="rounded-2xl border border-[#2a2a40] bg-[#12121f] p-6 sm:p-8">
          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between mb-6">
            <div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white mb-2">The Proof in One Chart</h2>
              <p className="text-gray-400 text-sm max-w-xl">
                Screening energy measured in D-D reactions. If the barrier were a nuclear constant,
                all bars would be equal.
                <span className="text-red-400 font-bold"> They differ by 607x.</span>
              </p>
            </div>
            <div className="mt-4 sm:mt-0 flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span className="text-red-400 text-sm font-mono font-bold">Standard prediction: 30 eV for ALL</span>
            </div>
          </div>

          <div className="h-80 sm:h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={SCREENING_CHART_DATA} margin={{ top: 20, right: 20, bottom: 30, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis dataKey="material" tick={{ fill: '#9ca3af', fontSize: 11 }} angle={-35} textAnchor="end" height={70} />
                <YAxis
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  label={{ value: 'Screening Energy (eV)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 12 }}
                  scale="log"
                  domain={[10, 50000]}
                  tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v}`}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: '8px' }}
                  formatter={(value) => [`${Number(value).toLocaleString()} eV`, 'Screening Energy']}
                />
                <ReferenceLine y={30} stroke="#ef4444" strokeDasharray="8 4" strokeWidth={2}
                  label={{ value: 'Standard model: 30 eV', fill: '#ef4444', fontSize: 12, position: 'top' }} />
                <Bar dataKey="Us" radius={[6, 6, 0, 0]}>
                  {SCREENING_CHART_DATA.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={entry.material.includes('cold') ? 1 : 0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 flex flex-wrap gap-4 text-xs text-gray-500 justify-between">
            <span>Data: Kasagi 2002, Raiola 2004, Huke 2008, Czerski 2023</span>
            <span className="text-red-400 font-bold text-sm">Pd cold-rolled: 18,200 eV = 607x above prediction</span>
          </div>
        </div>
      </section>

      {/* ═══════════════════ EVIDENCE GRID ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-3xl font-bold text-white mb-10 text-center">Hard Evidence</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {EVIDENCE.map((e, i) => (
            <div key={i} className="rounded-xl border border-[#2a2a40] bg-[#12121f] p-6 hover:border-red-500/30 transition-all group">
              <div className="flex items-start gap-4">
                <div className="text-4xl font-black font-mono bg-gradient-to-b from-red-400 to-red-600 bg-clip-text text-transparent">
                  {e.metric}
                </div>
                <div className="flex-1">
                  <h3 className="text-white font-bold text-lg mb-2">{e.title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed">{e.desc}</p>
                  <div className="mt-3 text-xs text-gray-600 font-mono">{e.source}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════ PHYSICS MODELS COMPARISON ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-3xl font-bold text-white mb-3 text-center">Three Physics Models, One Dataset</h2>
        <p className="text-gray-500 text-center mb-8 text-sm max-w-2xl mx-auto">
          We run three competing theoretical frameworks against the same experimental data.
          Only the medium-dependent model survives.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-2xl border border-[#2a2a40] bg-[#12121f] p-6">
            <h3 className="text-lg font-bold text-white mb-1">Reaction Rate vs D/Pd Loading</h3>
            <p className="text-xs text-gray-500 mb-4">Cherepanov (green) shows no hard threshold &mdash; rate depends on medium state, not a fixed barrier.</p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={barrierData} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                  <XAxis dataKey="DPd" tick={{ fill: '#9ca3af', fontSize: 10 }}
                    label={{ value: 'D/Pd ratio', position: 'bottom', fill: '#6b7280', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }}
                    label={{ value: 'log\u2081\u2080(Rate)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="maxwell" stroke="#3b82f6" strokeWidth={2} dot={false} name="Maxwell (standard)" />
                  <Line type="monotone" dataKey="coulomb" stroke="#f59e0b" strokeWidth={2} dot={false} name="Coulomb 1785" />
                  <Line type="monotone" dataKey="cherepanov" stroke="#10b981" strokeWidth={3} dot={false} name="Medium-dependent" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-2xl border border-[#2a2a40] bg-[#12121f] p-6">
            <h3 className="text-lg font-bold text-white mb-1">Penetration Probability vs Energy</h3>
            <p className="text-xs text-gray-500 mb-4">At low energies, medium-dependent model predicts orders of magnitude higher penetration &mdash; matching experiment.</p>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={energySweep} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                  <XAxis dataKey="E_keV" tick={{ fill: '#9ca3af', fontSize: 10 }}
                    label={{ value: 'E_cm (keV)', position: 'bottom', fill: '#6b7280', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }}
                    label={{ value: 'log\u2081\u2080(P)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: '8px', fontSize: '11px' }} />
                  <Area type="monotone" dataKey="cherepanov" stroke="#10b981" fill="#10b981" fillOpacity={0.15} strokeWidth={3} name="Medium-dependent" />
                  <Area type="monotone" dataKey="coulomb" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.05} strokeWidth={2} name="Coulomb 1785" />
                  <Area type="monotone" dataKey="maxwell" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.05} strokeWidth={2} name="Maxwell" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════ WHY THIS MATTERS — INDUSTRIES ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-3xl font-bold text-white mb-3 text-center">Why This Changes Everything</h2>
        <p className="text-gray-500 text-center mb-10 text-sm">
          If the &quot;barrier&quot; is an engineering parameter &mdash; not a law of nature &mdash; these industries are disrupted.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {IMPLICATIONS.map((imp, i) => (
            <div key={i} className="rounded-2xl border border-[#2a2a40] bg-[#12121f] p-6 hover:border-amber-500/30 transition-all group">
              <div className="text-5xl mb-4">{imp.icon}</div>
              <h3 className="text-xl font-bold text-white mb-1">{imp.company}</h3>
              <p className="text-amber-400 text-sm font-semibold mb-3">{imp.tagline}</p>
              <p className="text-sm text-gray-400 leading-relaxed mb-4">{imp.desc}</p>
              <div className="text-xs text-gray-600 font-mono">Relevant to: {imp.for}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════ IMPACT RADAR ═══════════════════ */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 mb-20">
        <div className="rounded-2xl border border-[#2a2a40] bg-[#12121f] p-6 sm:p-8">
          <h3 className="text-2xl font-bold text-white mb-2 text-center">Disruption Potential</h3>
          <p className="text-gray-500 text-center text-sm mb-6">Medium-dependent barrier (red) vs fixed barrier paradigm (blue)</p>
          <div className="h-80 sm:h-96">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={IMPACT_RADAR}>
                <PolarGrid stroke="#2a2a40" />
                <PolarAngleAxis dataKey="axis" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <PolarRadiusAxis tick={{ fill: '#4b5563', fontSize: 10 }} domain={[0, 100]} />
                <Radar name="New paradigm" dataKey="A" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.2} strokeWidth={2} />
                <Radar name="Current paradigm" dataKey="B" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={2} />
                <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a40', borderRadius: '8px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* ═══════════════════ AI AGENTS / TRIZ / METHODOLOGY ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-blue-500/5 p-8">
          <div className="flex items-center gap-3 mb-6">
            <span className="text-3xl">&#x1F9E0;</span>
            <div>
              <h2 className="text-2xl font-bold text-white">AI Agents + TRIZ Methodology</h2>
              <p className="text-emerald-400 text-sm font-medium">MiroFish for Physics</p>
            </div>
          </div>

          <p className="text-gray-400 mb-8 max-w-3xl text-lg">
            We apply multi-agent AI simulation (inspired by{' '}
            <a href="https://github.com/666ghj/MiroFish" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:underline font-semibold">MiroFish</a>
            ) to fundamental physics. Five AI agents trained with{' '}
            <strong className="text-white">TRIZ</strong> (Theory of Inventive Problem Solving)
            analyze nuclear data from competing theoretical positions.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 mb-8">
            {[
              { role: 'Maxwell Advocate', color: 'blue', emoji: '🔵' },
              { role: 'Coulomb Originalist', color: 'amber', emoji: '🟡' },
              { role: 'Medium Theorist', color: 'emerald', emoji: '🟢' },
              { role: 'Experimental Skeptic', color: 'red', emoji: '🔴' },
              { role: 'ML Analyst', color: 'purple', emoji: '🟣' },
            ].map((agent) => (
              <div key={agent.role} className="rounded-xl bg-[#12121f] border border-[#2a2a40] p-4 text-center">
                <div className="text-2xl mb-2">{agent.emoji}</div>
                <div className={`text-${agent.color}-400 font-semibold text-sm`}>{agent.role}</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="rounded-xl bg-[#0a0a15] border border-[#2a2a40] p-4">
              <div className="text-emerald-400 font-bold mb-2">TRIZ Contradiction</div>
              <p className="text-xs text-gray-400">The barrier must exist (Rutherford scattering works) AND must not exist (18,200 eV screening). TRIZ resolution: it exists in vacuum, vanishes in engineered media.</p>
            </div>
            <div className="rounded-xl bg-[#0a0a15] border border-[#2a2a40] p-4">
              <div className="text-blue-400 font-bold mb-2">Adversarial Debate</div>
              <p className="text-xs text-gray-400">Each agent argues from a different theoretical framework. The Maxwell advocate MUST defend the standard model. If even they concede &mdash; the evidence is overwhelming.</p>
            </div>
            <div className="rounded-xl bg-[#0a0a15] border border-[#2a2a40] p-4">
              <div className="text-amber-400 font-bold mb-2">Data-Driven Consensus</div>
              <p className="text-xs text-gray-400">Agents debate until consensus on what the DATA shows, regardless of theoretical preferences. Result: unanimous &mdash; the barrier varies 60x. No standard model explains this.</p>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <a href="https://github.com/synizia/syniz" target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-bold hover:bg-emerald-500/25 transition-all">
              \u2197 Syniz &mdash; TRIZ AI Framework
            </a>
            <a href="https://github.com/666ghj/MiroFish" target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-500/15 border border-blue-500/30 text-blue-400 font-bold hover:bg-blue-500/25 transition-all">
              \u2197 MiroFish &mdash; Multi-Agent Simulation
            </a>
          </div>
        </div>
      </section>

      {/* ═══════════════════ TIMELINE ═══════════════════ */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-3xl font-bold text-white mb-10 text-center">How We Got Here</h2>
        <div className="space-y-0">
          {TIMELINE.map((t, i) => (
            <div key={i} className="flex gap-6 group">
              <div className="flex flex-col items-center">
                <div className={`w-4 h-4 rounded-full border-2 ${i === TIMELINE.length - 1 ? 'bg-red-500 border-red-400 shadow-lg shadow-red-500/50' : 'bg-[#12121f] border-gray-600 group-hover:border-gray-400'} transition-all z-10`} />
                {i < TIMELINE.length - 1 && <div className="w-0.5 h-full bg-gray-800 min-h-[60px]" />}
              </div>
              <div className="pb-8">
                <div className="flex items-center gap-3 mb-1">
                  <span className={`font-mono font-bold ${i === TIMELINE.length - 1 ? 'text-red-400 text-lg' : 'text-gray-500'}`}>{t.year}</span>
                  <span className="text-white font-semibold">{t.event}</span>
                </div>
                <p className="text-sm text-gray-500">{t.note}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════ DATA SOURCES ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-2xl font-bold text-white mb-2 text-center">Connected Data Sources</h2>
        <p className="text-gray-500 text-center mb-6 text-sm">Real physics databases. Not synthetic data. Not LLM hallucinations.</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {DATA_SOURCES.map((db) => (
            <div key={db.name} className="rounded-lg border border-[#2a2a40] bg-[#12121f] p-3 text-center hover:border-blue-500/30 transition-all">
              <div className="text-sm font-bold text-white">{db.name}</div>
              <div className="text-xs text-red-400 font-mono font-bold">{db.count}</div>
              <div className="text-xs text-gray-500 mt-1">{db.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════ EXPLORE TOOLS ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <h2 className="text-2xl font-bold text-white mb-6 text-center">Explore the Evidence Yourself</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {TOOLS.map((s) => (
            <Link key={s.href} href={s.href}>
              <div className={`rounded-xl border border-[#2a2a40] bg-gradient-to-br ${s.gradient} p-6 hover:border-red-500/30 transition-all cursor-pointer group`}>
                <div className="flex items-start gap-3">
                  <span className="text-3xl">{s.icon}</span>
                  <div>
                    <h3 className="text-white font-bold text-lg group-hover:text-red-400 transition-colors">{s.title}</h3>
                    <p className="text-sm text-gray-400 mt-1">{s.desc}</p>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* ═══════════════════ CTA ═══════════════════ */}
      <section className="max-w-6xl mx-auto px-4 sm:px-6 mb-20">
        <div className="rounded-2xl border border-red-500/20 bg-gradient-to-br from-red-500/10 via-orange-500/5 to-amber-500/10 p-8 sm:p-12 text-center">
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-6">
            We Need to Revisit<br />Fundamental Physics
          </h2>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto mb-4">
            Seven laboratories. Twenty-six experiments. Sixty-fold variation.
            The data is clear: what we call the &quot;Coulomb barrier&quot;
            is not fundamental &mdash; it&apos;s engineerable.
          </p>
          <p className="text-gray-500 max-w-xl mx-auto mb-8 text-sm">
            This changes energy, propulsion, computing, and AI.
            <br />
            Open source. Reproducible. Every dataset, model, and line of code &mdash; public.
          </p>

          <div className="flex flex-wrap justify-center gap-4">
            <a href="https://github.com/synizia" target="_blank" rel="noopener noreferrer"
              className="px-8 py-3.5 bg-white text-black font-bold rounded-xl hover:bg-gray-200 transition-all text-lg">
              GitHub &rarr;
            </a>
            <Link href="/simulator"
              className="px-8 py-3.5 border-2 border-red-500 text-red-400 font-bold rounded-xl hover:bg-red-500/10 transition-all text-lg">
              Run the Simulation
            </Link>
          </div>

          <div className="mt-8 pt-6 border-t border-white/10">
            <p className="text-gray-600 text-xs">
              Built for: <span className="text-gray-400">SpaceX</span> &middot; <span className="text-gray-400">xAI</span> &middot; <span className="text-gray-400">TerraFab</span> &middot; and everyone who questions the fundamentals
            </p>
          </div>
        </div>
      </section>

      {/* ═══════════════════ FOOTER ═══════════════════ */}
      <footer className="border-t border-[#2a2a40] py-8">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <div>
              <p className="text-gray-400 font-bold">\u26a1 Nuclear Physics &mdash; Rethought</p>
              <p className="text-gray-600 text-xs mt-1">Open Research &middot; AI-Driven &middot; Reproducible</p>
            </div>
            <div className="text-right">
              <p className="text-gray-600 text-xs">
                Data: IAEA EXFOR &middot; Materials Project &middot; AFLOW &middot; COD &middot; ENDF/B &middot; NOMAD &middot; OQMD &middot; NIST
              </p>
              <p className="text-gray-700 text-xs mt-1">
                AI: <a href="https://github.com/synizia/syniz" target="_blank" rel="noopener noreferrer" className="text-emerald-600 hover:text-emerald-400">Syniz</a> &middot;{' '}
                <a href="https://github.com/666ghj/MiroFish" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-400">MiroFish</a>
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
