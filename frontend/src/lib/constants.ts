/**
 * LENR Alternative Physics — Physical Constants & Experimental Data
 * Ported from lenr_constants.py
 */

// =============================================================================
// NUCLEAR REFERENCE CONSTANTS
// =============================================================================
export const NUCLEAR = {
  gamow_energy_DD_keV: 986.0,
  coulomb_barrier_vacuum_keV: 400.0,
  Q_DpT_MeV: 4.033,
  Q_Dn3He_MeV: 3.269,
  Q_D4He_gamma_MeV: 23.847,
  Q_4D_8Be_MeV: 47.6,
  He4_binding_MeV: 28.296,
  Be8_above_2alpha_keV: 91.84,
  Be8_width_eV: 6.0,
  Be8_halflife_s: 8.19e-17,
  S0_DpT_keVb: 55.0,
  S0_Dn3He_keVb: 52.0,
  fine_structure_alpha: 1.0 / 137.036,
  deuteron_mass_MeV: 1875.613,
  reduced_mass_DD_MeV: 937.807,
  branch_ratio_DpT: 0.5,
  branch_ratio_Dn3He: 0.5,
  branch_ratio_D4He_gamma: 1e-7,
} as const;

// =============================================================================
// SCREENING ENERGIES (eV) — EXPERIMENTAL
// =============================================================================
export interface ScreeningEntry {
  Us_eV: number;
  error_eV: number;
  enhancement_2_5keV?: number;
  source: string;
}

export const SCREENING_EXPERIMENTAL: Record<string, ScreeningEntry> = {
  PdO:       { Us_eV: 600, error_eV: 60,  enhancement_2_5keV: 50.0, source: 'Kasagi (Tohoku)' },
  Pd:        { Us_eV: 310, error_eV: 30,  enhancement_2_5keV: 10.0, source: 'Kasagi (Tohoku)' },
  Pd_Raiola: { Us_eV: 800, error_eV: 90,  source: 'Raiola (Bochum)' },
  Pd_Huke:   { Us_eV: 313, error_eV: 2,   source: 'Huke (Berlin)' },
  Fe:        { Us_eV: 200, error_eV: 20,  enhancement_2_5keV: 5.0,  source: 'Kasagi (Tohoku)' },
  Au:        { Us_eV: 70,  error_eV: 10,  enhancement_2_5keV: 1.5,  source: 'Kasagi (Tohoku)' },
  Ti:        { Us_eV: 65,  error_eV: 10,  enhancement_2_5keV: 1.2,  source: 'Kasagi (Tohoku)' },
  Ta:        { Us_eV: 309, error_eV: 12,  source: 'Raiola (Bochum)' },
  Zr:        { Us_eV: 297, error_eV: 8,   source: 'Huke (Berlin)' },
  Ni:        { Us_eV: 420, error_eV: 50,  source: 'Raiola (Bochum)' },
  Al:        { Us_eV: 190, error_eV: 15,  source: 'Huke (Berlin)' },
  BeO:       { Us_eV: 180, error_eV: 40,  source: 'NASA TP-2020' },
  Pt:        { Us_eV: 122, error_eV: 20,  source: 'Raiola (Bochum)' },
  C:         { Us_eV: 25,  error_eV: 5,   source: 'Huke (Berlin)' },
};

// =============================================================================
// TAKAHASHI TSC / EQPET THEORY
// =============================================================================
export interface EQPETEntry {
  label: string;
  Us_eV: number;
  b0_pm: number;
  Rdd_pm: number | null;
  trap_eV: number | null;
}

export const EQPET_SCREENING: EQPETEntry[] = [
  { label: '(1,1) electron',     Us_eV: 36,    b0_pm: 40,    Rdd_pm: 101,  trap_eV: -15.4 },
  { label: '(1,1)×2 D₂',        Us_eV: 72,    b0_pm: 20,    Rdd_pm: 73,   trap_eV: -37.8 },
  { label: '(2,2) Cooper pair',  Us_eV: 360,   b0_pm: 4,     Rdd_pm: 33.8, trap_eV: -259 },
  { label: '(4,4) quadruplet',   Us_eV: 4000,  b0_pm: 0.36,  Rdd_pm: 15.1, trap_eV: -2460 },
  { label: '(6,6)',              Us_eV: 9600,  b0_pm: 0.15,  Rdd_pm: null, trap_eV: null },
  { label: '(8,8) octal',       Us_eV: 22154, b0_pm: 0.065, Rdd_pm: null, trap_eV: null },
];

export const TSC_PARAMS = {
  initial_dd_distance_pm: 74,
  tsc_radius_start_pm: 45.8,
  tsc_radius_min_fm: 20,
  condensation_time_4D_fs: 1.4,
  condensation_time_4H_fs: 1.0,
  max_tsc_density_per_cm3: 1e22,
  max_fusion_rate_MW_per_cm3: 46,
  neutron_to_He4_ratio: 1e-12,
} as const;

export interface BarrierFactorEntry {
  label: string;
  bf_2D: number;
  bf_4D: number;
  rate_4D: number;
}

export const BARRIER_FACTORS: BarrierFactorEntry[] = [
  { label: '(1,1)', bf_2D: 1e-125, bf_4D: 1e-250, rate_4D: 1e-252 },
  { label: '(2,2)', bf_2D: 1e-7,   bf_4D: 1e-15,  rate_4D: 1e-17 },
  { label: '(4,4)', bf_2D: 3e-4,   bf_4D: 1e-7,   rate_4D: 1e-9 },
  { label: '(8,8)', bf_2D: 4e-1,   bf_4D: 1e-1,   rate_4D: 1e-3 },
];

// =============================================================================
// MATERIAL PROPERTIES
// =============================================================================
export interface LatticeEntry {
  structure: string;
  a_A: number;
  c_A?: number;
  debye_K: number;
  e_density_A3: number;
  color: string;
}

export const LATTICE: Record<string, LatticeEntry> = {
  Pd: { structure: 'FCC', a_A: 3.8907, debye_K: 274, e_density_A3: 0.34,  color: '#3b82f6' },
  Ni: { structure: 'FCC', a_A: 3.5240, debye_K: 450, e_density_A3: 0.16,  color: '#10b981' },
  Ti: { structure: 'HCP', a_A: 2.9508, c_A: 4.6855, debye_K: 420, e_density_A3: 0.051, color: '#8b5cf6' },
  Fe: { structure: 'BCC', a_A: 2.8665, debye_K: 470, e_density_A3: 0.170, color: '#ef4444' },
  Au: { structure: 'FCC', a_A: 4.0782, debye_K: 165, e_density_A3: 0.059, color: '#f59e0b' },
  Pt: { structure: 'FCC', a_A: 3.9242, debye_K: 240, e_density_A3: 0.066, color: '#6366f1' },
  W:  { structure: 'BCC', a_A: 3.1652, debye_K: 400, e_density_A3: 0.063, color: '#64748b' },
  Cu: { structure: 'FCC', a_A: 3.6149, debye_K: 343, e_density_A3: 0.085, color: '#f97316' },
  Ag: { structure: 'FCC', a_A: 4.0853, debye_K: 225, e_density_A3: 0.059, color: '#94a3b8' },
};

export interface DiffusionEntry {
  D0_cm2s: number;
  Ea_eV: number;
  D_300K: number;
}

export const DIFFUSION: Record<string, DiffusionEntry> = {
  Pd: { D0_cm2s: 2.0e-3, Ea_eV: 0.230, D_300K: 1e-7 },
  Ni: { D0_cm2s: 2.4e-2, Ea_eV: 0.457, D_300K: 5e-10 },
  Fe: { D0_cm2s: 7.4e-4, Ea_eV: 0.041, D_300K: 1.5e-5 },
  Ti: { D0_cm2s: 2e-3,   Ea_eV: 0.34,  D_300K: 3e-9 },
};

export const LOADING = {
  Pd_max_1atm_RT: 0.70,
  Pd_max_electrolysis: 0.92,
  Pd_max_highP_77K: 0.97,
  LENR_threshold_McKubre: 0.84,
  LENR_threshold_Storms: 0.90,
  McKubre_M: 2.33e5,
  McKubre_i0: 0.4,
  McKubre_x0: 0.832,
  Pd_alpha_beta_low: 0.017,
  Pd_alpha_beta_high: 0.58,
  Pd_critical_T_C: 276,
} as const;

// =============================================================================
// EXCESS HEAT DATA
// =============================================================================
export interface ExcessHeatEntry {
  lab: string;
  material: string;
  method: string;
  excess_W: number;
  COP: number;
  duration_days: number;
  DPd: number | null;
  temperature_K: number;
  pressure_Pa: number;
  reproducibility: number;
}

export const EXCESS_HEAT_DATA: ExcessHeatEntry[] = [
  { lab: 'Fleischmann-Pons', material: 'Pd', method: 'electrolysis', excess_W: 150, COP: 40, duration_days: 14, DPd: 0.95, temperature_K: 340, pressure_Pa: 1e5, reproducibility: 0.30 },
  { lab: 'McKubre / SRI', material: 'Pd', method: 'electrolysis', excess_W: 2.1, COP: 1.38, duration_days: 60, DPd: 0.90, temperature_K: 340, pressure_Pa: 1e5, reproducibility: 0.17 },
  { lab: 'Kitamura / Kobe', material: 'PdNi/ZrO₂', method: 'gas loading', excess_W: 24, COP: 2.0, duration_days: 42, DPd: 0.80, temperature_K: 500, pressure_Pa: 1e5, reproducibility: 1.0 },
  { lab: 'Li Xing Zhong', material: 'Pd', method: 'gas loading', excess_W: 87, COP: 1.1, duration_days: 40, DPd: 0.12, temperature_K: 350, pressure_Pa: 9e4, reproducibility: 1.0 },
  { lab: 'Iwamura / Clean Planet', material: 'NiCu', method: 'gas permeation', excess_W: 5, COP: 1.5, duration_days: 589, DPd: 0.80, temperature_K: 343, pressure_Pa: 5e4, reproducibility: 1.0 },
  { lab: 'Storms', material: 'Pd', method: 'electrolysis', excess_W: 7.5, COP: 1.2, duration_days: 0.5, DPd: 0.82, temperature_K: 340, pressure_Pa: 1e5, reproducibility: 0.30 },
  { lab: 'Arata / Zhang', material: 'nano-Pd', method: 'gas loading', excess_W: 24, COP: 1.5, duration_days: 21, DPd: 3.0, temperature_K: 300, pressure_Pa: 5e5, reproducibility: 0.80 },
  { lab: 'Piantelli', material: 'Ni', method: 'gas loading (H)', excess_W: 38.9, COP: 1.38, duration_days: 278, DPd: 0.50, temperature_K: 500, pressure_Pa: 1e5, reproducibility: 0.60 },
  { lab: 'Brillouin', material: 'Ni', method: 'gas loading (H)', excess_W: 60, COP: 2.25, duration_days: 30, DPd: 0.50, temperature_K: 500, pressure_Pa: 1e5, reproducibility: 0.80 },
  { lab: 'Constantan 2025', material: 'Constantan', method: 'gas loading', excess_W: 209, COP: 3.91, duration_days: 0.001, DPd: null, temperature_K: 400, pressure_Pa: 1e5, reproducibility: 0.50 },
];

// =============================================================================
// TRANSMUTATION DATA
// =============================================================================
export interface TransmutationEntry {
  reaction: string;
  delta_mass: string;
  detection: string;
  source: string;
}

export const TRANSMUTATION_DATA: TransmutationEntry[] = [
  { reaction: '¹³³Cs → ¹⁴¹Pr', delta_mass: '+4D (+8 mass)', detection: 'XPS, SIMS, ICP-MS, SPring-8', source: 'Iwamura / Mitsubishi' },
  { reaction: '⁸⁸Sr → ⁹⁶Mo',   delta_mass: '+4D',           detection: 'XPS, SIMS, ICP-MS, SPring-8', source: 'Iwamura / Mitsubishi' },
  { reaction: '¹³⁷Ba → ¹⁴⁹Sm',  delta_mass: '+6D',           detection: 'ICCF-10',                     source: 'Iwamura' },
  { reaction: 'Ca → Ti',         delta_mass: '+4D',           detection: 'Nikkei 2014',                  source: 'Iwamura / Clean Planet' },
];

// =============================================================================
// PHYSICS MODES
// =============================================================================
export type PhysicsMode = 'maxwell' | 'coulomb_original' | 'cherepanov';

export interface PhysicsModeConfig {
  id: PhysicsMode;
  label: string;
  description: string;
  charge_unit: string;
  force_law: string;
  barrier_type: string;
  field_exists: boolean;
  color: string;
}

export const PHYSICS_MODES: PhysicsModeConfig[] = [
  {
    id: 'maxwell',
    label: 'Maxwell (стандарт)',
    description: 'Стандартная электромагнитная теория',
    charge_unit: '[L³/² · T⁻¹ · M¹/²]',
    force_law: 'F = k·q₁·q₂/r²',
    barrier_type: 'Кулоновский электростатический',
    field_exists: true,
    color: '#3b82f6',
  },
  {
    id: 'coulomb_original',
    label: 'Кулон (оригинал 1785)',
    description: 'Заряд = масса электричества',
    charge_unit: '[M · L⁻²] (плотность масс)',
    force_law: 'F = k·ρ₁·ρ₂/r²',
    barrier_type: 'Взаимодействие плотностей масс',
    field_exists: false,
    color: '#10b981',
  },
  {
    id: 'cherepanov',
    label: 'Черепанов',
    description: 'Фотонная масса, нет заряда',
    charge_unit: 'Нет (заряда не существует)',
    force_law: 'Магнитный поток B[кг/сек]',
    barrier_type: 'Сопротивление среды',
    field_exists: false,
    color: '#f59e0b',
  },
];

// DD Cross-section reference data (Bosch-Hale)
export const DD_CROSSSECTION_REFERENCE = [
  { E_keV: 1,  P_gamow: 3.6e-43, sigma_mb: 1e-40 },
  { E_keV: 5,  P_gamow: 3.5e-20, sigma_mb: 1e-18 },
  { E_keV: 10, P_gamow: 5.4e-14, sigma_mb: 1e-12 },
  { E_keV: 25, P_gamow: 2.6e-9,  sigma_mb: 1e-7 },
  { E_keV: 50, P_gamow: 8.5e-7,  sigma_mb: 1e-4 },
];
