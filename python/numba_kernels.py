"""
Numba JIT-accelerated kernels for LENR physics calculations.
Hot paths compiled to native code for 10-100x speedup over pure Python.

Usage:
    from numba_kernels import (
        gamow_penetration_jit, cross_section_DD_jit,
        enhancement_factor_jit, batch_barrier_maxwell,
        batch_barrier_cherepanov, batch_reaction_probability,
    )

Falls back to NumPy if Numba is not available.
"""

import numpy as np

try:
    from numba import njit, prange, float64, int32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ===========================================================================
# Numba-accelerated scalar kernels
# ===========================================================================

if HAS_NUMBA:

    @njit(float64(float64), cache=True)
    def gamow_penetration_jit(E_cm_keV):
        """Gamow penetration factor P = exp(-2*pi*eta)."""
        if E_cm_keV <= 0:
            return 0.0
        E_G = 986.0  # Gamow energy for D-D (keV)
        eta = np.sqrt(E_G / E_cm_keV)
        exponent = -2.0 * np.pi * eta
        if exponent < -700.0:
            return 0.0
        return np.exp(exponent)

    @njit(float64(float64), cache=True)
    def cross_section_DD_jit(E_cm_keV):
        """D-D fusion cross section (Bosch-Hale) in barns."""
        if E_cm_keV <= 0:
            return 0.0
        E_G = 986.0
        S_keVb = 55.0
        sigma = (S_keVb / E_cm_keV) * np.exp(-np.sqrt(E_G / E_cm_keV) * np.pi)
        return sigma

    @njit(float64(float64, float64), cache=True)
    def enhancement_factor_jit(E_cm_keV, Us_eV):
        """Screening enhancement factor."""
        if E_cm_keV <= 0 or Us_eV <= 0:
            return 1.0
        Us_keV = Us_eV / 1000.0
        E_G = 986.0
        exponent = np.pi * np.sqrt(E_G) * Us_keV / (2.0 * E_cm_keV ** 1.5)
        if exponent > 500.0:
            return np.exp(500.0)
        return np.exp(exponent)

    @njit(float64(float64, float64), cache=True)
    def _diffusion_arrhenius(D0, Ea_over_kB_T):
        """Arrhenius diffusion: D = D0 * exp(-Ea / kB*T)."""
        if Ea_over_kB_T > 500.0:
            return 0.0
        return D0 * np.exp(-Ea_over_kB_T)

    # -----------------------------------------------------------------------
    # Batch vectorized kernels — process entire arrays at once
    # -----------------------------------------------------------------------

    @njit(cache=True, parallel=True)
    def batch_gamow_penetration(E_cm_array):
        """Vectorized Gamow penetration for array of energies."""
        n = len(E_cm_array)
        result = np.empty(n)
        for i in prange(n):
            result[i] = gamow_penetration_jit(E_cm_array[i])
        return result

    @njit(cache=True, parallel=True)
    def batch_cross_section_DD(E_cm_array):
        """Vectorized D-D cross section."""
        n = len(E_cm_array)
        result = np.empty(n)
        for i in prange(n):
            result[i] = cross_section_DD_jit(E_cm_array[i])
        return result

    @njit(cache=True, parallel=True)
    def batch_enhancement(E_cm_array, Us_eV_array):
        """Vectorized enhancement factor."""
        n = len(E_cm_array)
        result = np.empty(n)
        for i in prange(n):
            result[i] = enhancement_factor_jit(E_cm_array[i], Us_eV_array[i])
        return result

    @njit(cache=True, parallel=True)
    def batch_barrier_maxwell(
        E_cm_array, Us_eV_array, D_loading_array,
        coulomb_barrier_keV=400.0,
    ):
        """Vectorized Maxwell barrier calculation.

        Returns: (effective_barrier, penetration, rate) — each array of shape (n,)
        """
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)

        for i in prange(n):
            Us_keV = Us_eV_array[i] / 1000.0
            eff_barrier[i] = coulomb_barrier_keV - Us_keV

            P = gamow_penetration_jit(E_cm_array[i] + Us_keV)
            penetration[i] = P

            loading = D_loading_array[i]
            if loading > 0.84:
                boost = ((loading - 0.84) / 0.16) ** 4
            else:
                boost = 0.01
            rate[i] = P * boost

        return eff_barrier, penetration, rate

    @njit(cache=True, parallel=True)
    def batch_barrier_coulomb_original(
        E_cm_array, Us_eV_array, D_loading_array, e_density_array,
        coulomb_barrier_keV=400.0,
    ):
        """Vectorized Coulomb Original barrier calculation."""
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)

        for i in prange(n):
            mdf = e_density_array[i] * 10.0
            barrier = coulomb_barrier_keV * (1.0 - mdf * 0.1)
            Us_keV = Us_eV_array[i] * (1.0 + mdf * 0.5) / 1000.0
            eb = barrier - Us_keV * mdf
            if eb < 1.0:
                eb = 1.0
            eff_barrier[i] = eb

            P = gamow_penetration_jit(E_cm_array[i] + Us_keV * mdf)
            penetration[i] = P

            loading = D_loading_array[i]
            if loading > 0.7:
                boost = ((loading - 0.7) / 0.3) ** 6
            else:
                boost = 0.01
            rate[i] = P * boost

        return eff_barrier, penetration, rate

    @njit(cache=True, parallel=True)
    def batch_barrier_cherepanov(
        E_cm_array, T_K_array, D_loading_array,
    ):
        """Vectorized Cherepanov barrier calculation."""
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)

        for i in prange(n):
            loading = D_loading_array[i]
            T = T_K_array[i]

            mag_res = 50.0 + (1.0 - loading) * 200.0
            phonon = np.exp(-T / 500.0) * 100.0
            eb = mag_res - phonon
            if eb < 0.1:
                eb = 0.1
            eff_barrier[i] = eb

            E = E_cm_array[i]
            exp_arg = -eb / (E + 0.001)
            if exp_arg < -700.0:
                P = 0.0
            else:
                P = np.exp(exp_arg)
            if P > 1.0:
                P = 1.0
            penetration[i] = P

            if loading > 0.5:
                boost = ((loading - 0.5) / 0.5) ** 8
            else:
                boost = 0.001
            r = P * boost
            if r > 1.0:
                r = 1.0
            rate[i] = r

        return eff_barrier, penetration, rate

    @njit(cache=True, parallel=True)
    def batch_reaction_probability(
        D_loading_array, Us_eV_array, E_cm_array,
        material_is_Pd_array, material_is_PdO_array, material_is_Ni_array,
        T_K_array, pressure_Pa_array,
    ):
        """Vectorized reaction probability estimation."""
        n = len(D_loading_array)
        result = np.empty(n)

        for i in prange(n):
            prob = 0.0
            loading = D_loading_array[i]

            if loading > 0.84:
                prob += 0.3 * ((loading - 0.84) / 0.16) ** 2
            if loading > 0.90:
                prob += 0.2

            Us = Us_eV_array[i]
            add_us = Us / 2000.0
            if add_us > 0.3:
                add_us = 0.3
            prob += add_us

            E = E_cm_array[i]
            if E > 1.0:
                add_e = E / 100.0
                if add_e > 0.15:
                    add_e = 0.15
                prob += add_e

            if material_is_Pd_array[i]:
                prob += 0.1
            if material_is_PdO_array[i]:
                prob += 0.15
            if material_is_Ni_array[i]:
                prob += 0.05

            T = T_K_array[i]
            if 300.0 < T < 600.0:
                prob += 0.05

            if pressure_Pa_array[i] > 1e5:
                prob += 0.05

            if prob < 0.001:
                prob = 0.001
            if prob > 0.95:
                prob = 0.95
            result[i] = prob

        return result

else:
    # -----------------------------------------------------------------------
    # Pure NumPy fallbacks (no Numba)
    # -----------------------------------------------------------------------

    def gamow_penetration_jit(E_cm_keV):
        if E_cm_keV <= 0:
            return 0.0
        E_G = 986.0
        eta = np.sqrt(E_G / E_cm_keV)
        exponent = -2.0 * np.pi * eta
        if exponent < -700:
            return 0.0
        return float(np.exp(exponent))

    def cross_section_DD_jit(E_cm_keV):
        if E_cm_keV <= 0:
            return 0.0
        E_G = 986.0
        S_keVb = 55.0
        return float((S_keVb / E_cm_keV) * np.exp(-np.sqrt(E_G / E_cm_keV) * np.pi))

    def enhancement_factor_jit(E_cm_keV, Us_eV):
        if E_cm_keV <= 0 or Us_eV <= 0:
            return 1.0
        Us_keV = Us_eV / 1000.0
        E_G = 986.0
        exponent = np.pi * np.sqrt(E_G) * Us_keV / (2.0 * E_cm_keV ** 1.5)
        return float(np.exp(min(exponent, 500.0)))

    def batch_gamow_penetration(E_cm_array):
        return np.array([gamow_penetration_jit(e) for e in E_cm_array])

    def batch_cross_section_DD(E_cm_array):
        return np.array([cross_section_DD_jit(e) for e in E_cm_array])

    def batch_enhancement(E_cm_array, Us_eV_array):
        return np.array([enhancement_factor_jit(e, u) for e, u in zip(E_cm_array, Us_eV_array)])

    def batch_barrier_maxwell(E_cm_array, Us_eV_array, D_loading_array, coulomb_barrier_keV=400.0):
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)
        for i in range(n):
            Us_keV = Us_eV_array[i] / 1000.0
            eff_barrier[i] = coulomb_barrier_keV - Us_keV
            P = gamow_penetration_jit(E_cm_array[i] + Us_keV)
            penetration[i] = P
            loading = D_loading_array[i]
            boost = ((loading - 0.84) / 0.16) ** 4 if loading > 0.84 else 0.01
            rate[i] = P * boost
        return eff_barrier, penetration, rate

    def batch_barrier_coulomb_original(E_cm_array, Us_eV_array, D_loading_array, e_density_array, coulomb_barrier_keV=400.0):
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)
        for i in range(n):
            mdf = e_density_array[i] * 10.0
            barrier = coulomb_barrier_keV * (1.0 - mdf * 0.1)
            Us_keV = Us_eV_array[i] * (1.0 + mdf * 0.5) / 1000.0
            eff_barrier[i] = max(barrier - Us_keV * mdf, 1.0)
            P = gamow_penetration_jit(E_cm_array[i] + Us_keV * mdf)
            penetration[i] = P
            loading = D_loading_array[i]
            boost = ((loading - 0.7) / 0.3) ** 6 if loading > 0.7 else 0.01
            rate[i] = P * boost
        return eff_barrier, penetration, rate

    def batch_barrier_cherepanov(E_cm_array, T_K_array, D_loading_array):
        n = len(E_cm_array)
        eff_barrier = np.empty(n)
        penetration = np.empty(n)
        rate = np.empty(n)
        for i in range(n):
            loading = D_loading_array[i]
            mag_res = 50.0 + (1.0 - loading) * 200.0
            phonon = np.exp(-T_K_array[i] / 500.0) * 100.0
            eb = max(mag_res - phonon, 0.1)
            eff_barrier[i] = eb
            exp_arg = -eb / (E_cm_array[i] + 0.001)
            P = min(np.exp(max(exp_arg, -700.0)), 1.0)
            penetration[i] = P
            boost = ((loading - 0.5) / 0.5) ** 8 if loading > 0.5 else 0.001
            rate[i] = min(P * boost, 1.0)
        return eff_barrier, penetration, rate

    def batch_reaction_probability(D_loading_array, Us_eV_array, E_cm_array,
                                    material_is_Pd_array, material_is_PdO_array, material_is_Ni_array,
                                    T_K_array, pressure_Pa_array):
        n = len(D_loading_array)
        result = np.empty(n)
        for i in range(n):
            prob = 0.0
            loading = D_loading_array[i]
            if loading > 0.84:
                prob += 0.3 * ((loading - 0.84) / 0.16) ** 2
            if loading > 0.90:
                prob += 0.2
            prob += min(Us_eV_array[i] / 2000.0, 0.3)
            if E_cm_array[i] > 1.0:
                prob += min(E_cm_array[i] / 100.0, 0.15)
            if material_is_Pd_array[i]:
                prob += 0.1
            if material_is_PdO_array[i]:
                prob += 0.15
            if material_is_Ni_array[i]:
                prob += 0.05
            if 300.0 < T_K_array[i] < 600.0:
                prob += 0.05
            if pressure_Pa_array[i] > 1e5:
                prob += 0.05
            result[i] = np.clip(prob, 0.001, 0.95)
        return result
