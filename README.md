# LENR Alternative Physics Simulator + Cold Star

> From critique of Maxwell's errors to infinite energy source and next-generation AI processor.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ORTODOX1/alternative-physik/blob/main/python/notebooks/training.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Vision: Cold Star

**Cold Star** -- next-generation computing architecture powered by controlled nuclear reactions in crystal lattices. Self-sustaining energy (COP > 1), nuclear-scale memory density (22 ZB/cm3), and resonance-based AI inference.

```
Axiom 1: Maxwell's 6 errors         ->  Charge is not fundamental
Axiom 2: Ether & vortex magnetism   ->  Continuous medium, no particles
Axiom 3: Electroculture experiments  ->  Macro proof of ether flows
Axiom 4: LENR in Japanese labs       ->  Micro proof: controlled compression
Axiom 5: Cold Star                   ->  Compression = Energy + Logic + AI
```

> **Full logic chain & roadmap:** [docs/roadmap.md](docs/roadmap.md)

---

## Project Structure

```
alternative-physik/
|
|-- axioms/                              # Foundational postulates
|   |-- 01_criticism_of_standard_model.md   # Maxwell's errors, Coulomb vs Maxwell charge
|   |-- 02_ether_and_vortex_magnetism.md    # Ether, photon mass, vortex model
|   +-- 03_from_ether_to_cold_star.md       # Synthesis: ether -> energy -> computing
|
|-- experiments/                         # Key experimental evidence
|   |-- plant_grounding.md                  # Electroculture: macro proof of ether flows
|   |-- electroculture_overview.md          # Methods and history of electroculture
|   +-- deuterium_lenr_japan.md             # Takahashi, Iwamura, Kasagi, Mizuno, Kozima
|
|-- models/                              # Mathematical and geometric models
|   |-- ether_barrier_and_screening.md      # Coulomb barrier as ether viscosity
|   +-- vortex/
|       +-- vortex_dynamics.md              # Toroidal vortex model of particles
|
|-- computing/                           # Application to computing
|   +-- cold_star/
|       |-- 01_physics_foundation.md        # Experimental basis for Cold Star
|       |-- 02_computing_architecture.md    # Nuclear bits, isotopic memory, processor
|       +-- 03_cold_star_architecture.md    # Full architecture + LLM + roadmap
|
|-- docs/
|   +-- roadmap.md                          # Complete logic chain of the project
|
|-- lenr_constants.py                    # Physical constants & experimental data
|-- python/
|   |-- physics_engine.py                # 3-mode physics engine
|   |-- data_generator.py               # Synthetic + real data generator
|   |-- numba_kernels.py                # JIT-accelerated compute kernels
|   |-- requirements.txt                # Python dependencies
|   |-- models/
|   |   |-- xgboost_classifier.py       # Reaction prediction + SHAP
|   |   |-- dnn_regressor.py            # Excess heat prediction
|   |   +-- anomaly_detector.py         # Outlier detection
|   +-- notebooks/
|       +-- training.ipynb              # Complete training pipeline (Colab)
|
+-- frontend/                            # Next.js interactive visualization
    +-- src/
        |-- app/                         # Pages: dashboard, physics, materials
        |-- components/                  # Reusable UI components
        +-- lib/                         # TS physics engine & constants
```

---

## Research Objective

This project investigates whether **cold fusion / LENR reactions** can be predicted using machine learning models trained on physics simulations and experimental data. We compare three fundamentally different physical frameworks to determine which best explains observed anomalous heat and transmutation phenomena.

**Ultimate goal:** use ML predictions to engineer optimal conditions for a **Cold Star** prototype -- a self-powered nuclear-scale computer capable of running LLMs without external energy.

### Key Research Questions

1. **Can ML models predict LENR reaction occurrence** given material properties, loading ratios, temperature, and energy conditions?
2. **Which physical framework** (Maxwell, Coulomb Original, or Cherepanov) best matches experimental data?
3. **What are the dominant factors** driving excess heat production in metal-deuteride systems?
4. **Is the McKubre loading threshold** (D/Pd > 0.84) a genuine phase transition or a gradual effect?
5. **Can we engineer optimal lattice structures** for maximum barrier reduction (toward Cold Star)?

---

## Three Physics Modes

The simulator implements three competing interpretations of electromagnetic interactions:

### 1. Maxwell (Standard)
- Classical electromagnetic theory with Coulomb barrier (~400 keV for D-D)
- Electron screening reduces the effective barrier (Assenbaum et al., 1987)
- Gamow penetration: P = exp(-2*pi*eta), where eta = sqrt(E_G/E)
- Well-established framework; struggles to explain LENR-scale reaction rates

### 2. Coulomb Original (1785)
- Based on Coulomb's original memoir: charge = mass density of electricity
- Force law: F = k*(rho_1*rho_2)/r^2 -- interaction of mass densities, not point charges
- Barrier depends on electron mass density of the host lattice
- Predicts stronger screening in high-density metals

### 3. Cherepanov Framework
- No electric charge; photon mass replaces electromagnetic field concept
- Magnetic flux B [kg/s] as the fundamental quantity
- "Barrier" = medium resistance (magnetic lattice interactions)
- Predicts barrier engineering through material structure
- **Key for Cold Star:** barrier is not fundamental, it can be engineered

> **Deep dive:** [axioms/01_criticism_of_standard_model.md](axioms/01_criticism_of_standard_model.md)

---

## Experimental Data Sources

All experimental data used in this project comes from published, peer-reviewed research:

| Laboratory | Key Result | Reference |
|-----------|-----------|-----------|
| **Kasagi / Tohoku University** | Screening energies: PdO (600 eV), Pd (310 eV), Fe (200 eV) | Kasagi et al., J. Phys. Soc. Japan, 2002 |
| **Raiola / Bochum** | Enhanced screening: Pd (800+/-90 eV), Ni (420+/-50 eV) | Raiola et al., Eur. Phys. J. A, 2004 |
| **Huke / Berlin** | Vacancy-dependent screening: Zr (100-600 eV) | Huke et al., Phys. Rev. C, 2008 |
| **McKubre / SRI** | Excess heat 2.1W, COP 1.38, D/Pd threshold >= 0.84 | McKubre, ICCF proceedings |
| **Fleischmann-Pons** | 20-240 W/cm3 excess heat, COP >40 | Fleischmann & Pons, J. Electroanal. Chem., 1989 |
| **Iwamura / Mitsubishi** | Transmutation: Cs-133 -> Pr-141 (+4D), confirmed by Toyota | Iwamura et al., Jpn. J. Appl. Phys., 2002 |
| **Kitamura / Kobe** | 3-24W excess, 110W burst, Pd*Ni/ZrO2 nano | Kitamura et al., ICCF-17 |
| **Takahashi TSC theory** | 4D -> Be-8* -> 2*alpha, barrier factor ~0.1 for e*(8,8) | Takahashi, JCMNS, 2009 |
| **Mizuno / Hokkaido** | Heat-after-death: 114 MJ from 100g Pd; Neutrons from SUS304+H2 | Mizuno, Eur. J. Appl. Phys., 2025 |
| **Mizuno (transmutation)** | Host Pd -> Cu, Zn, Fe, Cr, Ca; anomalous Cr isotopes | Mizuno et al., Int. J. SMER, 1998 |

### Additional Data
- **Li Xing Zhong (China)**: 41W sustained, 87W peak, 79.58 MJ total
- **Storms**: 7.5W (20% input), D/Pd = 0.82, 740 min duration
- **Constantan-D2 (2025)**: 183-209W, COP 3.76-3.91
- **Mizuno R19**: 55 experiments, Ni-mesh + Pd coating, COP 1.2-1.5
- **Mizuno SUS304**: 10 neutron experiments with H2 (NOT deuterium), 0.7 MeV peak

> **Deep dive:** [experiments/deuterium_lenr_japan.md](experiments/deuterium_lenr_japan.md)

---

## ML Pipeline

### Data Generation
- **Synthetic**: 10,000+ samples via physics engine sweep across 8 materials (Pd, Ni, Fe, Ti, Au, Pt, PdO, SUS304)
- **Experimental**: 65+ real data points from published LENR experiments
- **Mizuno R19**: 55 real measurements (Ni+D2, COP 1.2-1.5)
- **Mizuno SUS304**: 10 neutron experiments (stainless steel + H2, 2020-2025)

### Features (19 dimensions)
```
lattice_constant_A, debye_temperature_K, electron_density_A3,
screening_energy_eV, beam_energy_keV, temperature_K,
deuterium_loading, pressure_Pa, diffusion_coefficient,
enhancement_factor, log_cross_section,
barrier_reduction_{maxwell,coulomb,cherepanov},
log_rate_{maxwell,coulomb,cherepanov},
above_loading_threshold, above_storms_threshold
```

### Models

| Model | Task | Target | Method |
|-------|------|--------|--------|
| **XGBoost Classifier** | Will a reaction occur? | Binary (0/1) | Gradient boosting + SHAP |
| **DNN Regressor** | How much excess heat? | Continuous (W) | Physics-informed loss |
| **Anomaly Detector** | Data quality filter | Outlier scores | Isolation Forest |

### Physics-Informed Loss
The DNN regressor uses a custom loss function that penalizes physically impossible predictions:
- Excess heat must be ~0 when D/Pd loading < 0.5
- Excess heat should not occur when barrier reduction is negligible

### Performance Acceleration
Critical numerical kernels are JIT-compiled via **Numba** for 10-100x speedup:
- Gamow penetration factor
- D-D cross-section (Bosch-Hale)
- Batch barrier calculations for all 3 physics modes

---

## Quick Start

### Google Colab (recommended, free GPU)

Click the badge above or open:
```
https://colab.research.google.com/github/ORTODOX1/alternative-physik/blob/main/python/notebooks/training.ipynb
```
Select **Runtime -> Change runtime type -> T4 GPU**, then **Run All**.

### Local Setup

```bash
# Python ML pipeline
cd python
pip install -r requirements.txt
jupyter notebook notebooks/training.ipynb

# Frontend (optional)
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Cold Star Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **0: ML Simulation** | Predict optimal LENR conditions | **In progress** (this repo) |
| **1: Energy Prototype** | Reproduce Mizuno R19, achieve COP > 2 | Planning |
| **2: Controlled Transmutation** | Reproduce Iwamura Cs->Pr, addressable | Concept |
| **3: Nuclear Memory** | Write/read 1 bit in isotopic state | Concept |
| **4: Logic Operations** | AND/OR via resonance conditions | Concept |
| **5: Cold Star Alpha** | Integrated energy + memory + logic | Concept |
| **6: LLM on Cold Star** | AI inference via resonance waves | Vision |

> **Full architecture:** [computing/cold_star/03_cold_star_architecture.md](computing/cold_star/03_cold_star_architecture.md)

---

## Key Physical Constants

| Quantity | Value |
|----------|-------|
| D-D Gamow energy E_G | 986 keV |
| Coulomb barrier (vacuum) | ~400 keV |
| D+D -> T+p Q-value | 4.033 MeV |
| D+D -> 3He+n Q-value | 3.269 MeV |
| D+D -> 4He+gamma Q-value | 23.847 MeV |
| 4D -> 8Be* (TSC) | 47.6 MeV |
| McKubre threshold | D/Pd > 0.84 |
| S-factor D(d,p)T | ~55 keV*b |
| Heat-after-death (Mizuno) | 114 MJ / 100g Pd |
| Best COP (Constantan-D2) | 3.76-3.91 |

---

## Expected Model Predictions

Based on the physics and experimental data, the trained models should show:

1. **Screening energy** as the #1 feature (>40% SHAP importance)
2. **D/Pd loading threshold** visible as a sharp transition near 0.84
3. **Temperature** contributing <10% -- consistent with Kasagi observations
4. **Three physics modes** yielding different atomic-level predictions but converging at macroscopic scale
5. **Cherepanov mode** predicting lower effective barriers for high-density lattices

---

## References

1. Kasagi, J. et al. "Energetic Protons and alpha Particles." *J. Phys. Soc. Japan* 71, 2881 (2002)
2. Raiola, F. et al. "Enhanced electron screening in d(d,p)t." *Eur. Phys. J. A* 19, 283 (2004)
3. Huke, A. et al. "Enhancement of deuteron-fusion reactions in metals." *Phys. Rev. C* 78, 015803 (2008)
4. McKubre, M.C.H. "The Fleischmann-Pons Effect." *ICCF-15* (2009)
5. Fleischmann, M. & Pons, S. "Electrochemically Induced Nuclear Fusion of Deuterium." *J. Electroanal. Chem.* 261, 301 (1989)
6. Iwamura, Y. et al. "Elemental Analysis of Pd Complexes." *Jpn. J. Appl. Phys.* 41, 4642 (2002)
7. Takahashi, A. "TSC-Induced Nuclear Reactions." *JCMNS* 1, 86 (2009)
8. Bosch, H.S. & Hale, G.M. "Improved Formulas for Fusion Cross-Sections." *Nuclear Fusion* 32, 611 (1992)
9. Storms, E. "The Science of Low Energy Nuclear Reaction." World Scientific (2007)
10. Assenbaum, H.J. et al. "Effects of electron screening." *Z. Phys. A* 327, 461 (1987)
11. Mizuno, T. "Nuclear Transmutation: The Reality of Cold Fusion." Infinite Energy Press (1998)
12. Mizuno, T. et al. "Confirmation of Isotopic Distribution Changes." *Int. J. SMER* 6, 45 (1998)
13. Mizuno, T. "Neutrons Produced by Heating Processed Metals." *Eur. J. Appl. Phys.* (2025)
14. Kozima, H. "The Science of the Cold Fusion Phenomenon." Elsevier (2006)
15. Cherepanov, A.I. "Analysis of Maxwell's errors." (2024)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Experimental data compiled from published work by Kasagi (Tohoku), Raiola (Bochum), Huke (Berlin), McKubre (SRI International), Fleischmann-Pons, Iwamura (Mitsubishi/Clean Planet), Kitamura (Kobe), Takahashi (Osaka), Mizuno (Hokkaido), Kozima, and Li Xing Zhong. Alternative physics framework based on analysis by A.I. Cherepanov.
