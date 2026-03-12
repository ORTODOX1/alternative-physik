# LENR Alternative Physics ML Simulation — Project Context

## Цель проекта
ML-платформа для симуляции ядерных процессов с альтернативными физическими допущениями. Три режима физики: стандартный (Maxwell), оригинальный Кулон (масса электричества), и Черепанов (фотонная масса, нет заряда). Предсказание: будет ли реакция синтеза, сколько энергии выделится, при каких условиях.

---

## Альтернативная физическая рамка (Черепанов А.И.)

### Ключевые тезисы
1. **Ошибки Максвелла (пп.39-44 трактата "Электричество и Магнетизм")**: 6 арифметических ошибок в размерностях. Правая и левая части формул не совпадают по dimensions. F[F] ≠ e[Q]·e'[Q]·f[F]·r⁻²[L⁻²]
2. **Заряд Кулона ≠ Заряд Максвелла**: У Кулона заряд = масса электричества D[M·L⁻²]. У Максвелла заряд e[L³/²·T⁻¹·M¹/²] — получен из ошибочных формул.
3. **Электрического поля не существует** — это артефакт ошибочной математики Максвелла.
4. **Электронов, протонов, нейтронов не существует** как заряженных частиц. Томсон, Резерфорд, Чедвик ошибались, опираясь на ошибочную теорию Максвелла.
5. **Катодные лучи** — не поток электронов, а поток фотонной массы.
6. **Магнитный поток B[кг/сек]** — это и есть электрический ток, свет, все виды излучения.
7. **Кулоновский барьер** — это "максвелловский барьер", свойство среды (электронного экранирования + решётки), а НЕ фундаментальная константа. Можно инженерить материалом.
8. **Формула Кулона (оригинал 1785)**: F = k·(mp·me)/r², где mp и me — плотности масс электричества на шарах.
9. **Милликен ошибался**: между электродами было магнитное поле, не электрическое. Величина 1.602×10⁻¹⁹ Кл реальна, но интерпретация неверна.
10. **18 моделей атома водорода** (Гареев) — все описывают одни данные → эксперимент не выбирает одну теорию.

### Источники (Черепанов)
- Analysis of Maxwell's errors for Bob Greenier, 17.07.2024
- Открытие электрона в 1897 году под вопросом
- Недопонимание разницы между математикой и физикой, 29.07.2023
- Фатальные ошибки Милликена, 07.08.2021
- 3-я Памятка Кулона — TROISIEME MEMOIRE, 1785
- Двенадцать уравнений Максвелла, 10.10.2024
- Карл Шребер "Размерности электрических величин", 1899

---

## Экспериментальные данные LENR

### Screening Energies (Kasagi, Tohoku University, 2002)
| Материал | Us (eV) | Enhancement @ 2.5 keV | Ошибка (eV) |
|----------|---------|----------------------|-------------|
| PdO | 600 | 50× | ±60 |
| Pd | 310 | 10× | ±30 |
| Fe | 200 | 5× | ±20 |
| Au | 70 | 1.5× | ±10 |
| Ti | 65 | 1.2× | ±10 |

### Расширенные данные (Raiola, Huke, Czerski)
| Metal | U_s (eV) | Group |
|-------|----------|-------|
| Pd | 800 ± 90 | Raiola (Bochum) |
| Pd | 313 ± 2 | Huke (Berlin, lower limit) |
| Ta | 309 ± 12 | Raiola |
| Zr | 297 ± 8 | Huke (100-600 eV depending on vacancies) |
| Ni | 420 ± 50 | Raiola |
| Al | 190 ± 15 | Huke |
| Be(BeO) | 180 ± 40 | NASA |
| Pt | ~122 | Raiola |

### Excess Heat Data (Multiple Labs)
| Группа | Мощность | COP | Длительность | Энергия | D/M |
|--------|----------|-----|-------------|---------|-----|
| Fleischmann-Pons | 20-240 W/cm³ | >40× | Дни-недели | 150 MJ/cm³ | D/Pd>0.9 |
| McKubre/SRI | 2.1 W | 1.38× | Недели-месяцы | — | D/Pd≥0.9 |
| Kitamura/Kobe | 3-24 W; 110W burst | — | Недели | 100 MJ/mol | Pd·Ni/ZrO₂ |
| Iwamura/Clean Planet | ~5 W | >1× | 589 дней | 1.1 MJ | NiCu |
| Li Xing Zhong (China) | 41 W; max 87 W | ~1.1 | 83 часа | 79.58 MJ | D/Pd=0.12 |
| Storms | 7.5 W (20% input) | 1.2× | 740 мин | — | D/Pd=0.82 |
| Constantan-D₂ (2025) | 183-209 W | 3.76-3.91× | ~30 сек | — | — |

### Трансмутации (Iwamura/Mitsubishi, replicated by Toyota)
| Реакция | Δ массы | Детекция |
|---------|---------|----------|
| ¹³³Cs → ¹⁴¹Pr | +4D (+8 mass) | XPS, SIMS, ICP-MS, SPring-8 |
| ⁸⁸Sr → ⁹⁶Mo | +4D | Same |
| ¹³⁷Ba → ¹⁴⁹Sm | +6D | ICCF-10 |
| Ca → Ti | +4D | Nikkei 2014 |

---

## Takahashi TSC Theory Parameters

### EQPET Screening Energies
| Quasi-particle e*(m,Z) | U_s (eV) | b₀ (pm) | R_dd (pm) |
|------------------------|----------|---------|-----------|
| (1,1) normal electron | 36 | 40 | 101 |
| (1,1)×2 D₂ molecule | 72 | 20 | 73 |
| (2,2) Cooper pair | 360 | 4 | 33.8 |
| (4,4) quadruplet | 4,000 | 0.36 | 15.1 |
| (6,6) | 9,600 | 0.15 | — |
| (8,8) octal | 22,154 | 0.065 | — |

### TSC Condensation
- Initial inter-deuteron distance: 74 pm (D₂ ground state)
- TSC radius start: 45.8 pm
- Minimum radius: ~20 fm
- Condensation time: 1.4 fs (4D), 1.0 fs (4H)
- Reaction: 4D → ⁸Be*(47.6 MeV) → 2⁴He
- Max fusion rate: 46 MW/cm³-Pd
- Neutron/⁴He ratio: ~10⁻¹²

### Barrier Factors (at E_d = 0.22 eV)
| e*(m,Z) | Barrier factor (2D) | Barrier factor (4D) | Rate 4D (f/s/cluster) |
|---------|--------------------|--------------------|----------------------|
| (1,1) | 10⁻¹²⁵ | 10⁻²⁵⁰ | 10⁻²⁵² |
| (2,2) | 10⁻⁷ | 10⁻¹⁵ | 10⁻¹⁷ |
| (4,4) | 3×10⁻⁴ | 10⁻⁷ | 10⁻⁹ |
| (8,8) | 4×10⁻¹ | 1×10⁻¹ | 10⁻³ |

---

## Nuclear Reference Constants

| Quantity | Value |
|----------|-------|
| D-D Gamow energy E_G | 986 keV |
| Coulomb barrier (vacuum) | ~350-400 keV |
| D+D → T+p Q-value | 4.033 MeV |
| D+D → ³He+n Q-value | 3.269 MeV |
| D+D → ⁴He+γ Q-value | 23.847 MeV |
| 4D → ⁸Be* excitation | 47.6 MeV |
| ⁴He binding energy | 28.296 MeV (7.074 MeV/nucleon) |
| ⁸Be above 2α threshold | 91.84 keV |
| ⁸Be width Γ | 6 eV |
| ⁸Be half-life | 8.19 × 10⁻¹⁷ s |
| S(0) D(d,p)T | ~55 keV·b |
| S(0) D(d,n)³He | ~52 keV·b |
| Fine structure α | 1/137.036 |

### D-D Cross-sections (Bosch-Hale parameterization)
σ(E) = S(E) / [E · exp(√(E_G/E))]

| E_CM (keV) | P = exp(−2πη) | Approx σ (mb) |
|------------|---------------|---------------|
| 1 | ~3.6 × 10⁻⁴³ | ~10⁻⁴⁰ |
| 5 | ~3.5 × 10⁻²⁰ | ~10⁻¹⁸ |
| 10 | ~5.4 × 10⁻¹⁴ | ~10⁻¹² |
| 25 | ~2.6 × 10⁻⁹ | ~10⁻⁷ |
| 50 | ~8.5 × 10⁻⁷ | ~10⁻⁴ |

---

## Material Properties for Simulation

### Lattice Parameters
| Metal | Structure | a (Å) | θ_D (K) |
|-------|-----------|-------|---------|
| Pd | FCC | 3.8907 | 274 |
| Ni | FCC | 3.5240 | 450 |
| Ti (α) | HCP | 2.9508 | 420 |
| Fe (α) | BCC | 2.8665 | 470 |
| Au | FCC | 4.0782 | 165 |
| Pt | FCC | 3.9242 | 240 |
| W | BCC | 3.1652 | 400 |

### Deuterium Diffusion (D = D₀·exp(−E_a/k_BT))
| Metal | D₀ (cm²/s) | E_a (eV) | D(300K) (cm²/s) |
|-------|-----------|---------|-----------------|
| Pd | 2.0×10⁻³ | 0.230 | ~1×10⁻⁷ |
| Ni | 2.4×10⁻² | 0.457 | ~5×10⁻¹⁰ |
| Fe | 7.4×10⁻⁴ | 0.041 | ~1.5×10⁻⁵ |
| Ti | ~2×10⁻³ | 0.34 | ~3×10⁻⁹ |

### Loading Ratios
- Pd: max D/Pd ≈ 0.7 (1 atm RT), 0.82-0.92 (electrolysis), 0.97 (high P/77K)
- LENR threshold: D/Pd > 0.84-0.90 (McKubre/SRI)
- McKubre formula: P_ex = M × (i − i₀) × (x − x₀) × |δx/δt|
  - M = 2.33 × 10⁵ V/cm, i₀ = 0.4 A/cm², x₀ = 0.832
- Pd α→β phase transition: 0.017 < D/Pd < 0.58 at RT
- Critical T: 276°C for Pd-D

---

## Физический движок — 3 режима

### Режим "maxwell" (стандарт)
- Заряд: e [L³/²·T⁻¹·M¹/²] (единица СИ: Кл)
- Сила: F = k·q₁·q₂/r²
- Кулоновский барьер: ~400 keV (D-D vacuum)
- Электрическое поле: E = F/q
- Модель экранирования: V_eff(r) = V_Coulomb(r) · exp(−r/λ_D) − U_s

### Режим "coulomb_original"
- Заряд = масса электричества: D [M·L⁻²] (плотность)
- Сила: F = k·(ρ₁·ρ₂)/r² — взаимодействие плотностей масс
- Барьер: зависит от плотности массы среды
- Нет электрического поля в максвелловском смысле

### Режим "cherepanov"
- Нет заряда, нет электрического поля
- Есть фотонная масса (эфирная масса)
- Магнитный поток B [кг/сек] = ток = свет = все излучения
- "Барьер" = сопротивление среды (магнитные взаимодействия)
- Радиоактивный элемент = аккумулятор фотонной массы
- Interaction через магнитные свойства решётки

---

## Архитектура ML-моделей

### Фаза 1: Данные
- Реальных точек: ~50-80 (Kasagi 5 + excess heat 30+ + transmutation data)
- Синтетические: 5000-10000 (через физический движок с шумом)
- GAN для расширения: physics-constrained generator

### Фаза 2: Feature Engineering
Key features:
- material_type, screening_energy_eV, beam_energy_keV
- temperature_K, deuterium_loading_DPd, pressure_Pa
- lattice_constant_A, debye_temperature_K
- electron_density_interstitial, diffusion_coefficient
- tetrahedral_bond_order (для TSC)
- barrier_reduction_ratio = V_eff / V_vacuum
- physics_mode: "maxwell" / "coulomb_original" / "cherepanov"

### Фаза 3: Модели
1. **XGBoost Classifier**: "Будет ли TSC/реакция?" → binary, + SHAP
2. **DNN Regressor**: "Сколько excess heat (W)?" → с physics loss constraint
3. **GNN**: Геометрия 4D кластера (PyTorch Geometric)
4. **RL Agent**: Оптимизация условий для max reaction rate
5. **Anomaly Detection** (Isolation Forest): фильтрация данных

### Фаза 4: Валидация
- Leave-One-Lab-Out cross-validation
- SHAP analysis: какие параметры важны
- Сравнение предсказаний 3 режимов физики

---

## Доступные базы данных
- Freire 2021 LENR Database (Excel): lenr-canr.org/Collections/FreireLOpreliminar.xlsx
- EXFOR via NucML (Python): github.com/pedrojrv/nucml
- ENDF/B-VIII.1: NNDC BNL (D-D cross-sections)
- LENR-CANR.org: 2500+ papers
- JCMNS: 39+ volumes open-access
- Bosch-Hale (1992): analytical D-D cross-section formulas

---

## Стек технологий
- Python 3.12 (NumPy, Pandas, Scikit-learn, XGBoost, PyTorch, SHAP)
- Frontend: Next.js + React (для интерфейса симуляции)
- Backend: Node.js / Python API
- Deployment: Ubuntu 24.04 remote server (84.201.138.131)
- GPU: RTX 5080 (локально), A100 (при необходимости аренда)

---

## Ключевые предсказания модели (что проверяем)
1. SHAP покажет screening_energy как #1 фактор (>40% importance)
2. Temperature почти не влияет (<10%) — подтверждение Kasagi
3. D/Pd loading threshold ~0.84 будет виден как ступенька
4. 3 режима физики дадут разные предсказания на атомном уровне но одинаковые на макро
5. Модель воспроизведёт формулу Takahashi: E_barrier_eff ≈ U_s при TSC
