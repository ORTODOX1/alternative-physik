/**
 * LENR Physics Engine — 3 modes of calculation
 * Maxwell (standard), Coulomb Original, Cherepanov
 */

import { NUCLEAR, DIFFUSION, SCREENING_EXPERIMENTAL, type PhysicsMode } from './constants';

const kB_eV = 8.617e-5; // eV/K

// =============================================================================
// CORE PHYSICS FUNCTIONS
// =============================================================================

/** Gamow penetration probability for D-D fusion */
export function gamowPenetration(E_cm_keV: number, E_G_keV = NUCLEAR.gamow_energy_DD_keV): number {
  if (E_cm_keV <= 0) return 0;
  const eta = Math.sqrt(E_G_keV / (4 * E_cm_keV));
  return Math.exp(-2 * Math.PI * eta);
}

/** D-D fusion cross-section in barns (Bosch-Hale approximation) */
export function crossSectionDD(E_cm_keV: number, S0_keVb = 110.0, E_G_keV = NUCLEAR.gamow_energy_DD_keV): number {
  if (E_cm_keV <= 0) return 0;
  return S0_keVb / (E_cm_keV * Math.exp(Math.sqrt(E_G_keV / E_cm_keV)));
}

/** Screened D-D cross-section */
export function screenedCrossSection(E_cm_keV: number, Us_eV: number): number {
  const Us_keV = Us_eV / 1000.0;
  const E_eff = E_cm_keV + Us_keV;
  return crossSectionDD(E_eff);
}

/** Enhancement factor due to screening */
export function enhancementFactor(E_cm_keV: number, Us_eV: number): number {
  const bare = crossSectionDD(E_cm_keV);
  if (bare <= 0) return Infinity;
  const screened = screenedCrossSection(E_cm_keV, Us_eV);
  return screened / bare;
}

/** Deuterium diffusion coefficient at temperature T */
export function diffusionCoefficient(metal: string, T_K: number): number {
  const d = DIFFUSION[metal];
  if (!d) return 0;
  return d.D0_cm2s * Math.exp(-d.Ea_eV / (kB_eV * T_K));
}

/** Debye-Waller mean-square displacement */
export function debyeWallerMSD(debye_K: number, T_K: number, mass_amu: number): number {
  const hbar = 1.0546e-34;
  const kB = 1.381e-23;
  const m = mass_amu * 1.661e-27;
  const theta_D = debye_K;
  return (3 * hbar * hbar) / (2 * m * kB * theta_D) * (1 + 4 * (T_K / theta_D) ** 2 * 0.822);
}

// =============================================================================
// PHYSICS MODE-SPECIFIC CALCULATIONS
// =============================================================================

export interface BarrierResult {
  mode: PhysicsMode;
  barrier_keV: number;
  effective_barrier_keV: number;
  penetration_probability: number;
  reaction_rate_relative: number;
  notes: string;
}

/** Calculate effective barrier and reaction parameters per physics mode */
export function calculateBarrier(
  mode: PhysicsMode,
  material: string,
  E_cm_keV: number,
  T_K: number,
  DPd: number,
): BarrierResult {
  const Us = SCREENING_EXPERIMENTAL[material]?.Us_eV ?? 50;

  switch (mode) {
    case 'maxwell': {
      const barrier = NUCLEAR.coulomb_barrier_vacuum_keV;
      const Us_keV = Us / 1000;
      const eff_barrier = barrier - Us_keV;
      const P = gamowPenetration(E_cm_keV + Us_keV);
      const loadingBoost = DPd > 0.84 ? Math.pow(DPd / 0.84, 4) : 0.01;
      return {
        mode: 'maxwell',
        barrier_keV: barrier,
        effective_barrier_keV: eff_barrier,
        penetration_probability: P,
        reaction_rate_relative: P * loadingBoost,
        notes: `Стандартный кулоновский барьер ${barrier} keV, экранирование ${Us} eV`,
      };
    }

    case 'coulomb_original': {
      // В оригинале Кулона барьер зависит от плотности массы среды
      const eDensity = 0.34; // default for Pd
      const massDensityFactor = eDensity * 10; // усиление от плотности электронов
      const barrier = NUCLEAR.coulomb_barrier_vacuum_keV * (1 - massDensityFactor * 0.1);
      const Us_keV = Us * (1 + massDensityFactor * 0.5) / 1000;
      const eff_barrier = Math.max(barrier - Us_keV * massDensityFactor, 1);
      const P = gamowPenetration(E_cm_keV + Us_keV * massDensityFactor);
      const loadingBoost = DPd > 0.7 ? Math.pow(DPd / 0.7, 6) : 0.01;
      return {
        mode: 'coulomb_original',
        barrier_keV: barrier,
        effective_barrier_keV: eff_barrier,
        penetration_probability: P,
        reaction_rate_relative: P * loadingBoost,
        notes: `Барьер = свойство среды (ρ_e=${eDensity.toFixed(2)}), масса электричества`,
      };
    }

    case 'cherepanov': {
      // Нет электростатического барьера. Сопротивление среды через магнитные взаимодействия
      const magneticResistance = 50 + (1 - DPd) * 200; // keV equivalent
      const phononCoupling = Math.exp(-T_K / 500) * 100;
      const eff_barrier = Math.max(magneticResistance - phononCoupling, 0.1);
      // В модели Черепанова при правильных условиях барьер может быть очень низким
      const P = Math.exp(-eff_barrier / (E_cm_keV + 0.001));
      const loadingBoost = DPd > 0.5 ? Math.pow(DPd / 0.5, 8) : 0.001;
      return {
        mode: 'cherepanov',
        barrier_keV: magneticResistance,
        effective_barrier_keV: eff_barrier,
        penetration_probability: Math.min(P, 1),
        reaction_rate_relative: Math.min(P * loadingBoost, 1),
        notes: `Нет заряда. Сопротивление среды ${magneticResistance.toFixed(0)} keV, фотонная масса`,
      };
    }
  }
}

/** Generate cross-section data for a range of energies */
export function generateCrossSectionData(
  E_min_keV: number,
  E_max_keV: number,
  steps: number,
  Us_eV: number,
): Array<{ E_keV: number; sigma_bare: number; sigma_screened: number; enhancement: number; log_sigma_bare: number; log_sigma_screened: number }> {
  const data = [];
  for (let i = 0; i <= steps; i++) {
    const E = E_min_keV + (E_max_keV - E_min_keV) * (i / steps);
    if (E <= 0) continue;
    const bare = crossSectionDD(E);
    const screened = screenedCrossSection(E, Us_eV);
    const enh = bare > 0 ? screened / bare : 0;
    data.push({
      E_keV: E,
      sigma_bare: bare,
      sigma_screened: screened,
      enhancement: enh,
      log_sigma_bare: bare > 0 ? Math.log10(bare) : -50,
      log_sigma_screened: screened > 0 ? Math.log10(screened) : -50,
    });
  }
  return data;
}

/** Generate barrier comparison data across loading ratios */
export function generateBarrierVsLoading(
  material: string,
  E_cm_keV: number,
  T_K: number,
): Array<{ DPd: number; maxwell: number; coulomb: number; cherepanov: number }> {
  const data = [];
  for (let i = 0; i <= 100; i++) {
    const DPd = i / 100;
    const m = calculateBarrier('maxwell', material, E_cm_keV, T_K, DPd);
    const c = calculateBarrier('coulomb_original', material, E_cm_keV, T_K, DPd);
    const ch = calculateBarrier('cherepanov', material, E_cm_keV, T_K, DPd);
    data.push({
      DPd,
      maxwell: Math.log10(Math.max(m.reaction_rate_relative, 1e-300)),
      coulomb: Math.log10(Math.max(c.reaction_rate_relative, 1e-300)),
      cherepanov: Math.log10(Math.max(ch.reaction_rate_relative, 1e-300)),
    });
  }
  return data;
}

/** Generate diffusion data across temperatures */
export function generateDiffusionData(
  metals: string[],
  T_min: number,
  T_max: number,
  steps: number,
): Array<Record<string, number>> {
  const data = [];
  for (let i = 0; i <= steps; i++) {
    const T = T_min + (T_max - T_min) * (i / steps);
    const point: Record<string, number> = { T_K: T };
    for (const metal of metals) {
      const D = diffusionCoefficient(metal, T);
      point[metal] = D > 0 ? Math.log10(D) : -20;
    }
    data.push(point);
  }
  return data;
}

/** McKubre excess power formula */
export function mckubreExcessPower(
  current_A_cm2: number,
  loading: number,
  dxdt: number,
  M = 2.33e5,
  i0 = 0.4,
  x0 = 0.832,
): number {
  if (current_A_cm2 <= i0 || loading <= x0) return 0;
  return M * (current_A_cm2 - i0) * (loading - x0) * Math.abs(dxdt);
}

/** Generate energy sweep data comparing 3 physics modes */
export function generateEnergySweep(
  material: string,
  T_K: number,
  DPd: number,
  E_min: number,
  E_max: number,
  steps: number,
): Array<{ E_keV: number; maxwell: number; coulomb: number; cherepanov: number }> {
  const data = [];
  for (let i = 0; i <= steps; i++) {
    const E = E_min + (E_max - E_min) * (i / steps);
    if (E <= 0) continue;
    const m = calculateBarrier('maxwell', material, E, T_K, DPd);
    const c = calculateBarrier('coulomb_original', material, E, T_K, DPd);
    const ch = calculateBarrier('cherepanov', material, E, T_K, DPd);
    data.push({
      E_keV: E,
      maxwell: Math.log10(Math.max(m.penetration_probability, 1e-300)),
      coulomb: Math.log10(Math.max(c.penetration_probability, 1e-300)),
      cherepanov: Math.log10(Math.max(ch.penetration_probability, 1e-300)),
    });
  }
  return data;
}
