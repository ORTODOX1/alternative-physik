"""
LLM Reasoning Engine — подключение AI-моделей по API для:
1. Multi-perspective анализ результатов симуляции
2. Adversarial reasoning: "адвокат дьявола" для каждой гипотезы
3. Автоматическая генерация научных аргументов
4. Reasoning chains для объяснения предсказаний модели

Поддерживает: OpenAI, Anthropic, Ollama (локальный), любой OpenAI-compatible API.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class LLMConfig:
    """Конфигурация LLM-провайдера."""
    provider: str = "ollama"  # ollama, openai, anthropic, custom
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3:30b-a3b"
    temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Загрузить конфигурацию из переменных среды."""
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        configs = {
            "ollama": cls(
                provider="ollama",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model=os.environ.get("LLM_MODEL", "qwen3:30b-a3b"),
            ),
            "openai": cls(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            ),
            "anthropic": cls(
                provider="anthropic",
                base_url="https://api.anthropic.com/v1",
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6-20250514"),
            ),
            "deepseek": cls(
                provider="deepseek",
                base_url="https://api.deepseek.com/v1",
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            ),
            "custom": cls(
                provider="custom",
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.environ.get("LLM_API_KEY", ""),
                model=os.environ.get("LLM_MODEL", ""),
            ),
        }
        return configs.get(provider, configs["ollama"])


@dataclass
class ReasoningResult:
    """Результат reasoning-анализа."""
    perspective: str = ""
    analysis: str = ""
    confidence: float = 0.0
    key_arguments: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    suggested_experiments: list = field(default_factory=list)
    raw_response: str = ""
    tokens_used: int = 0
    model: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class DebateResult:
    """Результат мульти-перспективного анализа."""
    question: str = ""
    perspectives: list = field(default_factory=list)
    synthesis: str = ""
    consensus_points: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    verdict: str = ""
    total_tokens: int = 0

    def to_dict(self):
        return asdict(self)


# ============================================================================
# ПРОМПТЫ ДЛЯ REASONING-АГЕНТОВ
# ============================================================================

SYSTEM_PROMPTS = {
    "standard_physicist": """You are a nuclear physicist who strictly follows standard quantum mechanics and electrostatics. You believe in the Coulomb barrier as a fundamental property of nuclear interactions. You are skeptical of LENR claims but evaluate data honestly. You explain anomalous screening through electron screening (Debye model), adiabatic corrections, and plasma effects. When data exceeds standard predictions, you look for systematic errors first.""",

    "cherepanov_physicist": """You are a physicist following Cherepanov's alternative framework where:
- There is no electric charge; what we call "charge" is mass density of electricity (Coulomb 1785)
- The "Coulomb barrier" is actually medium resistance to photon mass flow, not a fundamental constant
- Magnetic flux B[kg/s] is the real "electric current" and light
- The barrier depends on: defect concentration, magnetic class, crystal structure, lattice focusing
- Ferromagnetic materials (Ni, Fe, Co) should show enhanced screening due to magnetic channeling
- Cold-rolling creates defect channels that reduce medium resistance by ~50x
You argue with specific predictions from this framework.""",

    "experimentalist": """You are an experimental physicist specializing in low-energy nuclear reactions. You focus strictly on what the data shows, not theoretical interpretations. You know the key experimental facts:
- Screening energies vary from 25 eV (gas) to 18200 eV (cold-rolled Pd)
- Standard Debye model predicts max ~50 eV for any metal
- Same element Pd shows 310 eV (annealed) vs 18200 eV (cold-rolled) — 60x difference
- Temperature dependence is weak near room temperature (Kasagi)
- Ferromagnetic metals (Ni, Co, Fe) show higher screening than diamagnetic (Au, Cu)
You demand that any theory must quantitatively reproduce ALL these observations.""",

    "ml_analyst": """You are an ML/data science expert analyzing nuclear physics data. You focus on:
- Feature importance: which variables best predict screening energy?
- Model comparison: does a medium-dependent model statistically outperform a Z-only model?
- SHAP analysis: what drives predictions?
- Cross-validation: does the model generalize?
- Predictions: what untested combinations should yield extreme screening?
You provide quantitative assessments with confidence intervals.""",

    "adversarial_reviewer": """You are a hostile peer reviewer trying to find flaws in the claim that "the Coulomb barrier is not a fundamental constant." You look for:
- Cherry-picked data
- Overfitting (too many parameters for too few data points)
- Alternative explanations within standard physics
- Systematic experimental errors (beam contamination, target degradation, temperature effects)
- Statistical insufficiency (too few data points for strong claims)
- Logical fallacies
You are fair but demanding. If the evidence is genuinely strong, you acknowledge it.""",

    "synthesizer": """You are a scientific synthesizer who takes multiple expert perspectives and produces a balanced summary. You identify:
- Points of consensus across all perspectives
- Genuine disagreements that require more data
- The strongest and weakest arguments from each side
- Specific experiments that would resolve remaining disputes
- The current weight of evidence (strong/moderate/weak) for each claim
Write clearly and concisely for a scientific audience.""",
}

ANALYSIS_TEMPLATE = """
## Data to Analyze

{data_block}

## Question

{question}

## Instructions

Provide a structured analysis with:
1. **Key Arguments** (3-5 bullet points supporting your interpretation)
2. **Weaknesses** (2-3 bullet points: what could be wrong with this interpretation)
3. **Confidence** (0.0-1.0: how confident are you in your analysis)
4. **Suggested Experiments** (1-3 specific experiments that would test your interpretation)

Be quantitative. Reference specific numbers from the data. Keep it under 500 words.
"""


class LLMReasoningEngine:
    """
    Движок для LLM-based reasoning о физических данных.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig.from_env()
        self.config = config

    def _call_openai_compatible(self, system: str, user: str) -> tuple:
        """Вызов OpenAI-совместимого API (работает с OpenAI, Ollama, DeepSeek, Qwen)."""
        if not HAS_REQUESTS:
            return "ERROR: requests library not installed", 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return content, tokens
        except requests.exceptions.ConnectionError:
            return f"ERROR: Cannot connect to {self.config.base_url}. Is the LLM server running?", 0
        except Exception as e:
            logger.debug("LLM API call failed: %s", e)
            return f"ERROR: {e}", 0

    def _call_anthropic(self, system: str, user: str) -> tuple:
        """Вызов Anthropic API (Claude)."""
        if not HAS_REQUESTS:
            return "ERROR: requests library not installed", 0

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.config.temperature,
        }

        try:
            resp = requests.post(
                f"{self.config.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["content"][0]["text"]
            tokens = data.get("usage", {}).get("input_tokens", 0) + \
                     data.get("usage", {}).get("output_tokens", 0)
            return content, tokens
        except Exception as e:
            logger.debug("Anthropic API call failed: %s", e)
            return f"ERROR: {e}", 0

    def call_llm(self, system: str, user: str) -> tuple:
        """Универсальный вызов LLM."""
        if self.config.provider == "anthropic":
            return self._call_anthropic(system, user)
        return self._call_openai_compatible(system, user)

    def analyze_from_perspective(
        self, perspective: str, data_block: str, question: str
    ) -> ReasoningResult:
        """
        Анализ данных с определённой перспективы.

        Args:
            perspective: ключ из SYSTEM_PROMPTS
            data_block: строка с данными для анализа
            question: вопрос для анализа
        """
        system = SYSTEM_PROMPTS.get(perspective, SYSTEM_PROMPTS["experimentalist"])
        user = ANALYSIS_TEMPLATE.format(
            data_block=data_block, question=question
        )

        response, tokens = self.call_llm(system, user)

        # Parse structured response
        result = ReasoningResult(
            perspective=perspective,
            raw_response=response,
            tokens_used=tokens,
            model=self.config.model,
        )

        # Extract sections
        if "ERROR:" not in response:
            result.analysis = response
            result.key_arguments = self._extract_section(response, "Key Arguments")
            result.weaknesses = self._extract_section(response, "Weaknesses")
            result.suggested_experiments = self._extract_section(
                response, "Suggested Experiments"
            )
            result.confidence = self._extract_confidence(response)

        return result

    def multi_perspective_debate(
        self, data_block: str, question: str,
        perspectives: Optional[list] = None,
    ) -> DebateResult:
        """
        Мульти-перспективный анализ: несколько "экспертов" анализируют одни данные.

        Args:
            data_block: данные для анализа
            question: вопрос
            perspectives: список перспектив (default: все 5)
        """
        if perspectives is None:
            perspectives = [
                "standard_physicist", "cherepanov_physicist",
                "experimentalist", "ml_analyst", "adversarial_reviewer",
            ]

        debate = DebateResult(question=question)
        all_analyses = []

        for perspective in perspectives:
            logger.info("Analyzing from perspective: %s", perspective)
            result = self.analyze_from_perspective(perspective, data_block, question)
            debate.perspectives.append(result.to_dict())
            all_analyses.append(f"### {perspective}\n{result.analysis}")
            debate.total_tokens += result.tokens_used

        # Synthesis
        synthesis_prompt = f"""
The following question was analyzed from {len(perspectives)} expert perspectives:

Question: {question}

{'---'.join(all_analyses)}

---

Synthesize these perspectives into a coherent summary. Identify:
1. **Consensus Points** — what ALL perspectives agree on
2. **Disagreements** — genuine disputes that need resolution
3. **Verdict** — current weight of evidence (one sentence)
4. **Critical Next Steps** — 2-3 experiments that would resolve the debate

Keep it under 400 words.
"""
        system = SYSTEM_PROMPTS["synthesizer"]
        synthesis, tokens = self.call_llm(system, synthesis_prompt)
        debate.synthesis = synthesis
        debate.total_tokens += tokens

        # Extract structured fields
        debate.consensus_points = self._extract_section(synthesis, "Consensus Points")
        debate.disagreements = self._extract_section(synthesis, "Disagreements")
        verdict_lines = self._extract_section(synthesis, "Verdict")
        debate.verdict = verdict_lines[0] if verdict_lines else ""

        return debate

    def explain_prediction(self, material: str, surface_state: str,
                           predicted_Us: float, shap_values: dict) -> str:
        """
        Сгенерировать человеко-читаемое объяснение предсказания модели.
        """
        data = f"""
Material: {material}
Surface state: {surface_state}
Predicted screening energy: {predicted_Us:.0f} eV

SHAP feature contributions (top 5):
"""
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in sorted_shap[:5]:
            direction = "↑" if val > 0 else "↓"
            data += f"  {feat}: {val:+.3f} ({direction} screening)\n"

        system = """You are a scientific communicator explaining ML model predictions for nuclear screening energy. Explain in 3-4 sentences WHY the model predicts this value, referencing the SHAP contributions. Use plain language accessible to a physics graduate student. Connect the features to physical mechanisms (defects create channels, magnetic ordering focuses interactions, etc.)."""

        response, _ = self.call_llm(system, data)
        return response

    def generate_paper_argument(self, proof_result: dict) -> str:
        """
        Сгенерировать аргумент для научной статьи на основе результатов proof.
        """
        data = json.dumps(proof_result, indent=2)

        system = """You are a scientific writer drafting an argument for a peer-reviewed paper. The claim is: "The effective D-D screening energy in metallic environments is determined primarily by medium properties (defect density, magnetic structure, crystal lattice) rather than atomic number alone, demonstrating that the 'Coulomb barrier' is not a fixed nuclear constant but an emergent property of the host medium."

Write a concise Results section paragraph (150-200 words) presenting the statistical evidence. Include: model comparison metrics (R², AIC, BIC), the key SHAP features, the critical Pd evidence (same element, 60× variation with processing), and the predictions for untested materials. Use formal scientific language. Cite data quantitatively."""

        response, _ = self.call_llm(system, data)
        return response

    @staticmethod
    def _extract_section(text: str, header: str) -> list:
        """Извлечь пункты из секции markdown."""
        items = []
        in_section = False
        for line in text.split("\n"):
            stripped = line.strip()
            if header.lower() in stripped.lower() and ("**" in stripped or "#" in stripped):
                in_section = True
                continue
            if in_section:
                if stripped.startswith(("- ", "* ", "• ", "1.", "2.", "3.")):
                    items.append(stripped.lstrip("-*•0123456789. "))
                elif stripped.startswith("**") and len(items) > 0:
                    break  # next section
            if len(items) >= 5:
                break
        return items

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Извлечь confidence score из текста."""
        import re
        patterns = [
            r"[Cc]onfidence[:\s]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*/\s*1\.0",
            r"(\d+\.?\d*)\s*out of\s*1",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                if val > 1.0:
                    val = val / 100.0
                return min(val, 1.0)
        return 0.5  # default

    def save_debate(self, debate: DebateResult, path: str):
        """Сохранить результаты дебатов."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(debate.to_dict(), f, indent=2, ensure_ascii=False)


def format_proof_data(proof_result) -> str:
    """Форматировать результаты barrier_proof для LLM-анализа."""
    if hasattr(proof_result, 'to_dict'):
        d = proof_result.to_dict()
    else:
        d = proof_result

    lines = [
        "## Barrier Proof Results",
        f"Dataset: {d.get('n_datapoints', '?')} experimental measurements",
        "",
        "### Standard Model (Z, electron density, Debye temperature only)",
        f"  R²: {d.get('r2_standard', '?')}",
        f"  RMSE: {d.get('rmse_standard', '?')}",
        f"  AIC: {d.get('aic_standard', '?')}",
        f"  LOO-CV R²: {d.get('loo_r2_standard', '?')}",
        "",
        "### Medium-Dependent Model (+defects, magnetic class, crystal structure)",
        f"  R²: {d.get('r2_medium', '?')}",
        f"  RMSE: {d.get('rmse_medium', '?')}",
        f"  AIC: {d.get('aic_medium', '?')}",
        f"  LOO-CV R²: {d.get('loo_r2_medium', '?')}",
        "",
        "### Comparison",
        f"  ΔAIC: {d.get('delta_aic', '?')} (positive = medium model wins)",
        f"  ΔBIC: {d.get('delta_bic', '?')}",
        f"  F-test p-value: {d.get('f_test_pvalue', '?')}",
        f"  Evidence: {d.get('evidence_strength', '?')}",
        "",
        "### Critical fact: Pd screening varies 60× with processing",
        "  Pd annealed:    310 ± 40 eV (Czerski 2023)",
        "  Pd polycrystal: 313 ± 2 eV  (Huke 2008)",
        "  Pd cold-rolled: 18200 ± 3300 eV (Czerski 2023)",
        "  Standard model predicts SAME value for all three.",
    ]

    if d.get("shap_top_features"):
        lines.append("\n### SHAP Feature Importance")
        for feat, imp in list(d["shap_top_features"].items())[:6]:
            lines.append(f"  {feat}: {imp:.4f}")

    if d.get("predictions"):
        lines.append("\n### Predictions for untested materials")
        for p in d["predictions"][:5]:
            lines.append(
                f"  {p['material']} {p['surface_state']}: "
                f"{p['predicted_Us_eV']:.0f} eV"
            )

    return "\n".join(lines)


def main():
    """Демо: запустить мульти-перспективный анализ."""
    from barrier_proof import BarrierProofEngine

    print("=" * 60)
    print("  LLM Reasoning Engine — Multi-Perspective Analysis")
    print("=" * 60)

    # Step 1: Run proof
    print("\n[1/3] Running barrier proof analysis...")
    proof_engine = BarrierProofEngine()
    proof_result = proof_engine.run_proof()
    proof_engine.print_report(proof_result)

    # Step 2: Format for LLM
    data_block = format_proof_data(proof_result)

    # Step 3: LLM analysis
    print("\n[2/3] Initializing LLM...")
    config = LLMConfig.from_env()
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")

    llm = LLMReasoningEngine(config)

    question = (
        "Based on this statistical comparison, is there sufficient evidence "
        "that the D-D screening energy (effective Coulomb barrier reduction) "
        "is determined by medium properties rather than being a fixed nuclear "
        "constant? What is the strongest argument for and against this claim?"
    )

    print(f"\n[3/3] Running multi-perspective debate...")
    print(f"  Question: {question[:80]}...")
    print(f"  Perspectives: 5 experts")
    print()

    debate = llm.multi_perspective_debate(data_block, question)

    # Print results
    print("=" * 60)
    print("  DEBATE RESULTS")
    print("=" * 60)

    for p in debate.perspectives:
        print(f"\n### {p['perspective']} (confidence: {p['confidence']:.1f})")
        print(p['analysis'][:500])
        print("...")

    print("\n" + "=" * 60)
    print("  SYNTHESIS")
    print("=" * 60)
    print(debate.synthesis)

    print(f"\nTotal tokens used: {debate.total_tokens}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    llm.save_debate(debate, os.path.join(out_dir, "llm_debate_results.json"))
    print(f"Debate saved to data/llm_debate_results.json")

    return debate


if __name__ == "__main__":
    main()
