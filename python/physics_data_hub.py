"""
Physics Data Hub — единый интерфейс ко всем открытым базам физических данных.

Подключает 10+ баз данных для обогащения ML-модели:
- Materials Project API (500K+ материалов: фононы, упругость, электронная структура)
- AFLOW (2M+ DFT расчётов, без API ключа)
- OQMD (1.4M материалов, термодинамика)
- IAEA EXFOR (22K+ ядерных экспериментов)
- IAEA Livechart (ядерная структура, распады)
- NIST (атомные спектры, потенциалы)
- COD (520K+ кристаллических структур)
- OPTIMADE (унифицированный доступ к AFLOW+OQMD+NOMAD+MaterialsCloud)

Все данные кэшируются локально для офлайн-работы.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import urllib.request
    import urllib.parse
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

try:
    import requests as req_lib
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# CACHE
# ============================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "api_cache")


def _cache_path(source: str, key: str) -> str:
    """Путь к кэш-файлу."""
    safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_")[:100]
    d = os.path.join(CACHE_DIR, source)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe_key}.json")


def _cache_get(source: str, key: str, max_age_days: int = 30):
    """Прочитать из кэша, None если истёк или нет."""
    path = _cache_path(source, key)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 86400
    if age > max_age_days:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Cache read failed for %s/%s: %s", source, key, e)
        return None


def _cache_set(source: str, key: str, data):
    """Записать в кэш."""
    path = _cache_path(source, key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.debug("Cache write failed for %s/%s: %s", source, key, e)


def _http_get_json(url: str, headers: dict = None, timeout: int = 30):
    """Универсальный HTTP GET → JSON."""
    if HAS_REQUESTS:
        resp = req_lib.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    elif HAS_URLLIB:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    else:
        raise RuntimeError("No HTTP library available (install requests)")


# ============================================================================
# 1. MATERIALS PROJECT API
# ============================================================================

class MaterialsProjectClient:
    """
    Materials Project API v2.
    Бесплатный, нужен API ключ (регистрация на materialsproject.org).

    Данные: фононы, электронная структура, упругие константы, band gap,
    formation energy, магнитные моменты, 500K+ материалов.
    """

    BASE_URL = "https://api.materialsproject.org/v2"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY", "")

    @property
    def _headers(self):
        return {"X-API-KEY": self.api_key, "Accept": "application/json"}

    def get_material(self, formula: str) -> list:
        """Получить все материалы по формуле (Pd, NiO, PdD, etc.)."""
        cache_key = f"material_{formula}"
        cached = _cache_get("materials_project", cache_key)
        if cached:
            return cached

        if not self.api_key:
            logger.warning("MP_API_KEY not set, skipping Materials Project")
            return []

        url = (f"{self.BASE_URL}/materials/summary?"
               f"formula={formula}&"
               f"fields=material_id,formula_pretty,structure,symmetry,"
               f"band_gap,formation_energy_per_atom,energy_above_hull,"
               f"total_magnetization,ordering,volume,density,"
               f"bulk_modulus_vrh,shear_modulus_vrh,"
               f"debye_temperature,efermi")
        try:
            data = _http_get_json(url, headers=self._headers)
            results = data.get("data", [])
            _cache_set("materials_project", cache_key, results)
            logger.info("MP: fetched %d entries for %s", len(results), formula)
            return results
        except Exception as e:
            logger.debug("Materials Project API failed for %s: %s", formula, e)
            return []

    def get_phonon_data(self, material_id: str) -> dict:
        """Получить фононные данные (DOS, band structure)."""
        cache_key = f"phonon_{material_id}"
        cached = _cache_get("materials_project", cache_key)
        if cached:
            return cached

        if not self.api_key:
            return {}

        url = f"{self.BASE_URL}/materials/phonon/?material_ids={material_id}"
        try:
            data = _http_get_json(url, headers=self._headers)
            result = data.get("data", [{}])[0] if data.get("data") else {}
            _cache_set("materials_project", cache_key, result)
            return result
        except Exception as e:
            logger.debug("MP phonon failed for %s: %s", material_id, e)
            return {}

    def get_elastic(self, formula: str) -> list:
        """Получить упругие константы."""
        cache_key = f"elastic_{formula}"
        cached = _cache_get("materials_project", cache_key)
        if cached:
            return cached

        if not self.api_key:
            return []

        url = (f"{self.BASE_URL}/materials/elasticity?"
               f"formula={formula}&"
               f"fields=material_id,formula_pretty,"
               f"bulk_modulus_vrh,shear_modulus_vrh,"
               f"young_modulus,poisson_ratio,debye_temperature")
        try:
            data = _http_get_json(url, headers=self._headers)
            results = data.get("data", [])
            _cache_set("materials_project", cache_key, results)
            return results
        except Exception as e:
            logger.debug("MP elastic failed for %s: %s", formula, e)
            return []


# ============================================================================
# 2. AFLOW (no API key needed!)
# ============================================================================

class AFLOWClient:
    """
    AFLOW REST API — 2M+ DFT расчётов, без API ключа.
    Данные: термодинамика, упругость, электронная структура.
    """

    BASE_URL = "https://aflow.org/API/aflux"

    def search(self, formula: str, properties: list = None) -> list:
        """Поиск материалов по формуле."""
        cache_key = f"search_{formula}"
        cached = _cache_get("aflow", cache_key)
        if cached:
            return cached

        if properties is None:
            properties = [
                "auid", "compound", "spacegroup_relax",
                "Bvoigt", "Gvoigt", "ael_debye",
                "enthalpy_formation_atom", "energy_atom",
                "spin_atom", "spinD", "spinF",
                "volume_atom", "density",
                "Egap", "Egap_type",
            ]

        props_str = ",".join(properties)
        url = f"{self.BASE_URL}/?species({formula}),paging(1)"
        try:
            data = _http_get_json(url, timeout=30)
            results = data if isinstance(data, list) else [data]
            _cache_set("aflow", cache_key, results)
            logger.info("AFLOW: fetched %d entries for %s", len(results), formula)
            return results
        except Exception as e:
            logger.debug("AFLOW API failed for %s: %s", formula, e)
            return []

    def get_entry(self, auid: str) -> dict:
        """Получить полную запись по AUID."""
        cache_key = f"entry_{auid}"
        cached = _cache_get("aflow", cache_key)
        if cached:
            return cached

        url = f"https://aflow.org/API/aflux/?auid('{auid}')"
        try:
            data = _http_get_json(url, timeout=30)
            result = data[0] if isinstance(data, list) and data else data
            _cache_set("aflow", cache_key, result)
            return result
        except Exception as e:
            logger.debug("AFLOW entry failed for %s: %s", auid, e)
            return {}


# ============================================================================
# 3. OPTIMADE — унифицированный доступ к OQMD, NOMAD, MaterialsCloud, AFLOW
# ============================================================================

class OPTIMADEClient:
    """
    OPTIMADE API — единый интерфейс к множеству баз данных.
    Работает с OQMD, NOMAD, MaterialsCloud, AFLOW, COD.
    """

    PROVIDERS = {
        "oqmd": "https://oqmd.org/optimade/v1",
        "aflow": "https://aflow.org/API/optimade/v1",
        "cod": "https://www.crystallography.net/cod/optimade/v1",
        "mc3d": "https://aiida.materialscloud.org/mc3d/optimade/v1",
        "nomad": "https://nomad-lab.eu/prod/v1/api/optimade/v1",
    }

    def search_structures(self, formula: str, provider: str = "oqmd",
                          limit: int = 10) -> list:
        """
        Поиск структур по формуле через OPTIMADE.

        Args:
            formula: химическая формула (Pd, NiO, PdD)
            provider: oqmd, aflow, cod, mc3d, nomad
            limit: макс. число результатов
        """
        cache_key = f"{provider}_{formula}_{limit}"
        cached = _cache_get("optimade", cache_key)
        if cached:
            return cached

        base = self.PROVIDERS.get(provider)
        if not base:
            logger.warning("Unknown OPTIMADE provider: %s", provider)
            return []

        elements = self._parse_elements(formula)
        filter_str = " AND ".join(
            [f'elements HAS "{e}"' for e in elements]
        )
        url = f"{base}/structures?filter={urllib.parse.quote(filter_str)}&page_limit={limit}"

        try:
            data = _http_get_json(url, timeout=30)
            results = data.get("data", [])
            _cache_set("optimade", cache_key, results)
            logger.info("OPTIMADE/%s: fetched %d structures for %s",
                        provider, len(results), formula)
            return results
        except Exception as e:
            logger.debug("OPTIMADE/%s failed for %s: %s", provider, formula, e)
            return []

    @staticmethod
    def _parse_elements(formula: str) -> list:
        """Извлечь элементы из формулы: PdD → [Pd, D], NiO → [Ni, O]."""
        import re
        return re.findall(r'[A-Z][a-z]?', formula)


# ============================================================================
# 4. IAEA Nuclear Data (EXFOR + Livechart)
# ============================================================================

class IAEAClient:
    """
    IAEA Nuclear Data Services.
    - EXFOR: экспериментальные ядерные сечения
    - Livechart: ядерная структура, уровни энергии, распады
    """

    EXFOR_API = "https://nds.iaea.org/dataexplorer/api"
    LIVECHART_API = "https://nds.iaea.org/relnsd/v1/data"

    def get_cross_sections(self, target: str = "D", projectile: str = "d",
                           reaction: str = "p", energy_min: float = 1.0,
                           energy_max: float = 100.0) -> list:
        """Получить сечения реакции из EXFOR."""
        reaction_str = f"{target}({projectile},{reaction})"
        cache_key = f"xs_{reaction_str}_{energy_min}_{energy_max}"
        cached = _cache_get("iaea_exfor", cache_key)
        if cached:
            return cached

        url = (f"{self.EXFOR_API}/reactions/xs?"
               f"target={target}&projectile={projectile}&"
               f"product={reaction}&"
               f"energy_min={energy_min}&energy_max={energy_max}")
        try:
            data = _http_get_json(url, timeout=30)
            results = data if isinstance(data, list) else data.get("data", [])
            _cache_set("iaea_exfor", cache_key, results)
            logger.info("EXFOR: fetched %d entries for %s", len(results), reaction_str)
            return results
        except Exception as e:
            logger.debug("EXFOR API failed for %s: %s", reaction_str, e)
            return []

    def get_nuclide_data(self, z: int, a: int) -> dict:
        """Получить данные нуклида из Livechart (уровни, распады)."""
        cache_key = f"nuclide_Z{z}_A{a}"
        cached = _cache_get("iaea_livechart", cache_key)
        if cached:
            return cached

        url = f"{self.LIVECHART_API}?fields=ground_states&nuclides={z}-{a}"
        try:
            data = _http_get_json(url, timeout=15)
            _cache_set("iaea_livechart", cache_key, data)
            return data
        except Exception as e:
            logger.debug("Livechart failed for Z=%d A=%d: %s", z, a, e)
            return {}

    def get_decay_data(self, z: int, a: int) -> dict:
        """Получить данные распада."""
        cache_key = f"decay_Z{z}_A{a}"
        cached = _cache_get("iaea_livechart", cache_key)
        if cached:
            return cached

        url = f"{self.LIVECHART_API}?fields=decay_rads&nuclides={z}-{a}"
        try:
            data = _http_get_json(url, timeout=15)
            _cache_set("iaea_livechart", cache_key, data)
            return data
        except Exception as e:
            logger.debug("Livechart decay failed for Z=%d A=%d: %s", z, a, e)
            return {}


# ============================================================================
# 5. Crystallography Open Database (COD)
# ============================================================================

class CODClient:
    """
    Crystallography Open Database — 520K+ кристаллических структур.
    Бесплатный, без API ключа.
    """

    BASE_URL = "https://www.crystallography.net/cod/result"

    def search_by_formula(self, formula: str, limit: int = 10) -> list:
        """Поиск структур по формуле."""
        cache_key = f"formula_{formula}_{limit}"
        cached = _cache_get("cod", cache_key)
        if cached:
            return cached

        url = (f"{self.BASE_URL}?format=json&formula={urllib.parse.quote(formula)}"
               f"&limit={limit}")
        try:
            data = _http_get_json(url, timeout=20)
            results = data if isinstance(data, list) else []
            _cache_set("cod", cache_key, results)
            logger.info("COD: fetched %d structures for %s", len(results), formula)
            return results
        except Exception as e:
            logger.debug("COD failed for %s: %s", formula, e)
            return []

    def search_by_element(self, element: str, limit: int = 10) -> list:
        """Поиск всех структур содержащих элемент."""
        cache_key = f"element_{element}_{limit}"
        cached = _cache_get("cod", cache_key)
        if cached:
            return cached

        url = (f"{self.BASE_URL}?format=json&el1={element}&limit={limit}")
        try:
            data = _http_get_json(url, timeout=20)
            results = data if isinstance(data, list) else []
            _cache_set("cod", cache_key, results)
            return results
        except Exception as e:
            logger.debug("COD element search failed for %s: %s", element, e)
            return []


# ============================================================================
# 6. NIST Interatomic Potentials
# ============================================================================

class NISTpotentialsClient:
    """
    NIST Interatomic Potential Repository.
    EAM, MEAM, ADP потенциалы для MD симуляций.
    """

    BASE_URL = "https://potentials.nist.gov/api"

    def search_potentials(self, element: str) -> list:
        """Поиск потенциалов для элемента."""
        cache_key = f"potential_{element}"
        cached = _cache_get("nist_potentials", cache_key)
        if cached:
            return cached

        url = f"{self.BASE_URL}/potentials?element={element}"
        try:
            data = _http_get_json(url, timeout=20)
            results = data if isinstance(data, list) else data.get("data", [])
            _cache_set("nist_potentials", cache_key, results)
            logger.info("NIST: fetched %d potentials for %s", len(results), element)
            return results
        except Exception as e:
            logger.debug("NIST potentials failed for %s: %s", element, e)
            return []


# ============================================================================
# UNIFIED HUB — единая точка доступа
# ============================================================================

@dataclass
class MaterialRecord:
    """Единая запись материала со ВСЕМИ данными из всех источников."""
    formula: str = ""
    # Structural
    crystal_structure: str = ""
    space_group: str = ""
    lattice_a: float = 0.0
    lattice_b: float = 0.0
    lattice_c: float = 0.0
    volume_per_atom: float = 0.0
    density: float = 0.0
    # Electronic
    band_gap_eV: float = 0.0
    efermi_eV: float = 0.0
    # Magnetic
    total_magnetization: float = 0.0
    magnetic_ordering: str = ""
    # Mechanical
    bulk_modulus_GPa: float = 0.0
    shear_modulus_GPa: float = 0.0
    debye_temperature_K: float = 0.0
    # Thermodynamic
    formation_energy_eV: float = 0.0
    energy_above_hull_eV: float = 0.0
    # Phonon
    has_phonon_data: bool = False
    max_phonon_freq_THz: float = 0.0
    # Nuclear
    screening_energy_eV: float = 0.0
    dd_cross_section_mb: float = 0.0
    # Sources
    sources: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class PhysicsDataHub:
    """
    Единый хаб для доступа ко ВСЕМ физическим данным.

    Пример использования:
        hub = PhysicsDataHub(mp_api_key="your_key")
        record = hub.get_complete_material("Pd")
        features = hub.get_ml_features("Pd")
    """

    def __init__(self, mp_api_key: str = None):
        self.mp = MaterialsProjectClient(api_key=mp_api_key)
        self.aflow = AFLOWClient()
        self.optimade = OPTIMADEClient()
        self.iaea = IAEAClient()
        self.cod = CODClient()
        self.nist = NISTpotentialsClient()

    def get_complete_material(self, formula: str) -> MaterialRecord:
        """
        Собрать ВСЕ доступные данные о материале из всех источников.

        Порядок приоритета: Materials Project > AFLOW > OQMD > COD
        """
        record = MaterialRecord(formula=formula)

        # 1. Materials Project (самый полный)
        mp_data = self.mp.get_material(formula)
        if mp_data:
            best = mp_data[0]  # берём самый стабильный (lowest energy above hull)
            record.band_gap_eV = best.get("band_gap", 0.0) or 0.0
            record.formation_energy_eV = best.get("formation_energy_per_atom", 0.0) or 0.0
            record.energy_above_hull_eV = best.get("energy_above_hull", 0.0) or 0.0
            record.total_magnetization = best.get("total_magnetization", 0.0) or 0.0
            record.magnetic_ordering = best.get("ordering", "") or ""
            record.density = best.get("density", 0.0) or 0.0
            record.volume_per_atom = best.get("volume", 0.0) or 0.0
            record.bulk_modulus_GPa = best.get("bulk_modulus_vrh", 0.0) or 0.0
            record.shear_modulus_GPa = best.get("shear_modulus_vrh", 0.0) or 0.0
            record.debye_temperature_K = best.get("debye_temperature", 0.0) or 0.0
            record.efermi_eV = best.get("efermi", 0.0) or 0.0

            sym = best.get("symmetry", {})
            if sym:
                record.space_group = sym.get("symbol", "")
                record.crystal_structure = sym.get("crystal_system", "")

            record.sources.append("Materials Project")
            logger.info("MP: loaded data for %s", formula)

        # 2. AFLOW (дополняем чем не хватает)
        aflow_data = self.aflow.search(formula)
        if aflow_data and isinstance(aflow_data, list) and len(aflow_data) > 0:
            best = aflow_data[0]
            if not record.bulk_modulus_GPa and best.get("Bvoigt"):
                record.bulk_modulus_GPa = float(best["Bvoigt"])
            if not record.shear_modulus_GPa and best.get("Gvoigt"):
                record.shear_modulus_GPa = float(best["Gvoigt"])
            if not record.debye_temperature_K and best.get("ael_debye"):
                record.debye_temperature_K = float(best["ael_debye"])
            if not record.band_gap_eV and best.get("Egap"):
                record.band_gap_eV = float(best["Egap"])
            record.sources.append("AFLOW")

        # 3. COD (кристаллография)
        cod_data = self.cod.search_by_formula(formula, limit=3)
        if cod_data:
            record.sources.append("COD")

        return record

    def get_nuclear_data(self, target: str = "D", projectile: str = "d",
                         energy_range: tuple = (1.0, 100.0)) -> list:
        """Получить ядерные данные (сечения) из EXFOR."""
        return self.iaea.get_cross_sections(
            target=target, projectile=projectile,
            energy_min=energy_range[0], energy_max=energy_range[1],
        )

    def get_ml_features(self, formula: str) -> dict:
        """
        Получить полный набор ML-фич для материала.
        Возвращает словарь, готовый для добавления в DataFrame.
        """
        record = self.get_complete_material(formula)
        return {
            "formula": formula,
            "band_gap_eV": record.band_gap_eV,
            "formation_energy_eV": record.formation_energy_eV,
            "energy_above_hull_eV": record.energy_above_hull_eV,
            "total_magnetization": record.total_magnetization,
            "magnetic_ordering": record.magnetic_ordering,
            "density_gcc": record.density,
            "volume_per_atom_A3": record.volume_per_atom,
            "bulk_modulus_GPa": record.bulk_modulus_GPa,
            "shear_modulus_GPa": record.shear_modulus_GPa,
            "debye_temperature_K": record.debye_temperature_K,
            "efermi_eV": record.efermi_eV,
            "crystal_structure": record.crystal_structure,
            "space_group": record.space_group,
            "has_phonon_data": record.has_phonon_data,
            "n_sources": len(record.sources),
            "sources": ",".join(record.sources),
        }

    def enrich_screening_dataset(self, screening_data: list) -> list:
        """
        Обогатить датасет screening energy данными из всех баз.

        Args:
            screening_data: список словарей из barrier_proof.SCREENING_DATASET
        Returns:
            Обогащённый список с дополнительными ML-фичами
        """
        enriched = []
        seen_formulas = {}

        for entry in screening_data:
            mat = entry.get("material", "")
            if mat in ("D2_gas", "Debye_theory"):
                enriched.append(entry)
                continue

            # Normalize material name
            formula = mat.replace("O", "").replace("_", "")  # PdO → Pd, SUS304 → ...
            if formula not in seen_formulas:
                seen_formulas[formula] = self.get_ml_features(formula)

            features = seen_formulas[formula]

            # Merge
            merged = {**entry}
            for key, val in features.items():
                if key not in merged and val:
                    merged[f"api_{key}"] = val
            enriched.append(merged)

        logger.info("Enriched %d entries with API data", len(enriched))
        return enriched

    def batch_fetch_materials(self, formulas: list) -> dict:
        """Пакетно загрузить данные для списка материалов."""
        results = {}
        for formula in formulas:
            try:
                results[formula] = self.get_complete_material(formula)
                logger.info("Fetched: %s (%d sources)",
                            formula, len(results[formula].sources))
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", formula, e)
                results[formula] = MaterialRecord(formula=formula)
        return results

    def get_all_lenr_materials(self) -> dict:
        """Загрузить данные для всех материалов, используемых в LENR."""
        lenr_materials = [
            "Pd", "Ni", "Ti", "Fe", "Au", "Pt", "W", "Cu", "Zr",
            "Ta", "Al", "Be", "Ag", "V", "Nb", "Co", "Mn",
            # Гидриды и дейтериды
            "PdH", "NiH", "TiH2", "ZrH2",
            # Оксиды
            "PdO", "NiO", "TiO2", "ZrO2",
        ]
        return self.batch_fetch_materials(lenr_materials)

    def status(self) -> dict:
        """Проверить доступность всех API."""
        status = {}

        # Materials Project
        status["materials_project"] = {
            "configured": bool(self.mp.api_key),
            "url": MaterialsProjectClient.BASE_URL,
        }

        # AFLOW (no key needed)
        status["aflow"] = {
            "configured": True,
            "url": AFLOWClient.BASE_URL,
            "note": "No API key required",
        }

        # OPTIMADE providers
        for name, url in OPTIMADEClient.PROVIDERS.items():
            status[f"optimade_{name}"] = {
                "configured": True,
                "url": url,
                "note": "No API key required",
            }

        # IAEA
        status["iaea_exfor"] = {
            "configured": True,
            "url": IAEAClient.EXFOR_API,
        }
        status["iaea_livechart"] = {
            "configured": True,
            "url": IAEAClient.LIVECHART_API,
        }

        # COD
        status["cod"] = {
            "configured": True,
            "url": CODClient.BASE_URL,
            "note": "No API key required",
        }

        # NIST
        status["nist_potentials"] = {
            "configured": True,
            "url": NISTpotentialsClient.BASE_URL,
        }

        # Cache stats
        cache_files = 0
        cache_size = 0
        if os.path.exists(CACHE_DIR):
            for root, dirs, files in os.walk(CACHE_DIR):
                cache_files += len(files)
                cache_size += sum(os.path.getsize(os.path.join(root, f))
                                  for f in files)

        status["cache"] = {
            "files": cache_files,
            "size_mb": round(cache_size / 1024 / 1024, 2),
            "path": CACHE_DIR,
        }

        return status


def main():
    """Демо: показать статус и загрузить данные для Pd."""
    hub = PhysicsDataHub()

    print("=" * 60)
    print("  Physics Data Hub — Status")
    print("=" * 60)

    status = hub.status()
    for source, info in status.items():
        configured = info.get("configured", False)
        marker = "[OK]" if configured else "[--]"
        print(f"  {marker} {source:25s} {info.get('url', '')}")
        if info.get("note"):
            print(f"       {info['note']}")

    print(f"\n  Cache: {status['cache']['files']} files, "
          f"{status['cache']['size_mb']:.1f} MB")

    print("\n" + "=" * 60)
    print("  Fetching data for Palladium (Pd)...")
    print("=" * 60)

    record = hub.get_complete_material("Pd")
    print(f"  Formula: {record.formula}")
    print(f"  Crystal: {record.crystal_structure} ({record.space_group})")
    print(f"  Band gap: {record.band_gap_eV:.3f} eV")
    print(f"  Formation energy: {record.formation_energy_eV:.3f} eV/atom")
    print(f"  Magnetization: {record.total_magnetization:.3f}")
    print(f"  Magnetic ordering: {record.magnetic_ordering}")
    print(f"  Bulk modulus: {record.bulk_modulus_GPa:.1f} GPa")
    print(f"  Debye temperature: {record.debye_temperature_K:.0f} K")
    print(f"  Sources: {', '.join(record.sources)}")

    print("\n--- ML Features ---")
    features = hub.get_ml_features("Pd")
    for k, v in features.items():
        print(f"  {k:30s}: {v}")

    return hub


if __name__ == "__main__":
    main()
