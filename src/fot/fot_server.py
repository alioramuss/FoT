"""FoT global aggregation (server) pipeline.

This module is the OpenClaw port of the current Federation over Text server
(``server_text.py`` / ``TextBasedInsightAggregationServer`` in the research
code). Each round the server:

1. Collects every uploaded reasoning trace (``problem_*.json`` / ``paper_*.json``)
   into one indexed store without deduplication.
2. Builds a relationship profile over the traces (Prompt 4): clusters of highly
   similar traces plus prerequisite / composition / alternative / complementary /
   derivation / similarity relationships within and across clusters.
3. Updates the insight library (Prompt 5) by merging the previous library with
   the new traces, guided by the complete cluster and relationship profile. One
   trace may contribute to several insights, which enables cross-domain merges.

Prompt numbers follow Appendix E.1 of "Federation over Text: Insight Sharing for
Multi-Agent Reasoning". The profiling result is checkpointed (sha256 over the
collected traces) so an interrupted aggregation can resume at Prompt 5.

There are no fallbacks: unreadable trace files, an OpenClaw failure, or a model
response that is not exactly the required JSON shape raises ``ValueError``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

from fotclaw.openclaw_adapter import render_insight_markdown, run_openclaw_prompt

METADATA_KEYS = frozenset(
    {
        "paper_name",
        "problem",
        "problem_id",
        "iteration",
        "is_correct",
        "number_output_tokens",
        "loop_count",
    }
)
RELATIONSHIP_TYPES = (
    "prerequisite",
    "complementary",
    "alternative",
    "similar",
    "derived_from",
    "composes_with",
)
PROFILING_CHECKPOINT_VERSION = 1
PROMPT5_EXAMPLES = """
**Example 1 - Transformer Architecture:**

Input reasoning traces:
- reasoning_trace_VisionTransformerImageClassification: "I need to classify medical X-ray images into disease categories. CNNs aren't working well - they can't capture relationships between distant regions like how fluid in the lower right lung might relate to heart enlargement. Let me try Vision Transformer (ViT). I'll divide each X-ray into 16x16 patches - a 224x224 image gives 196 patches. Each patch becomes like a token in NLP. I flatten each to a 256-dim vector. Since transformers don't know spatial positions, I add position embeddings so the model knows patch [0,0] is top-left. I prepend a [CLS] token to gather global info. Feeding through 12 transformer encoder layers - the self-attention lets every patch attend to every other patch directly, so patch [2,5] can look at patch [10,12] even though they're far apart spatially. This is exactly what I need! After 12 layers, I extract the [CLS] token and feed to MLP classifier. Training on 50k chest X-rays: 94.2% accuracy, beating ResNet-50's 89.3%. The attention maps show it's correctly attending to both lungs simultaneously for pneumonia detection, linking heart size to lung fluid - this cross-region reasoning is what CNNs miss. The key: treating image patches as tokens with self-attention enables global spatial reasoning."

- reasoning_trace_TransformerNextWordPrediction: "I'm building autocomplete for a text editor. The challenge: predict next word given arbitrary-length context. RNNs struggle with long sequences - the hidden state forgets earlier context. Let me use a transformer decoder. I tokenize 'The cat sat on the' using WordPiece → tokens [254, 8901, 4523, 651, 278]. Convert each to 512-dim embedding and add positional encodings. Critical part: causal masking so the model can't cheat by seeing future words. When predicting token at position 4, it should only see 0-3. I implement lower-triangular attention mask. Processing through 6 decoder layers with masked self-attention. At position 4 ('the'), attention computes similarity between its query and keys of previous words. It attends strongly to 'sat' (0.8) and 'on' (0.7), weakly to 'cat' (0.2). Using 12 heads helps - different heads capture different patterns: head-2 learns syntax (preposition+article), head-5 learns semantics (actions+objects), head-8 learns long-range dependencies. After final layer, project last hidden state through 50k-dim softmax. For 'the': top predictions are 'mat' (0.73), 'floor' (0.12), 'rug' (0.08). Deployed in production - users accept top-3 suggestion 85% of the time, reducing typing by 40%. Transformer's self-attention captures context way better than RNN's sequential processing."

Output aggregated insight:
{
  "insight_transformerArchitecture": "This fundamental neural network architecture applies across natural language processing, computer vision, time series analysis, graph neural networks, and multi-modal learning. This insight is essential for modern AI applications including language models, image processing, code generation, and scientific computing. When you need to capture relationships between all elements simultaneously (self-attention), you're working with sequences of variable length, you need parallel processing of sequences, or when the problem involves understanding context and relationships. Details: 1) Design input representation - convert your data into embeddings (token embeddings for text, patch embeddings for images, node embeddings for graphs), add positional encodings to preserve sequence information, and prepare input for transformer blocks 2) Create models with transformer blocks 3) Apply task-specific architecture - use encoder-only for understanding (BERT, ViT), decoder-only for generation (GPT), or encoder-decoder for translation"
}

**Example 2 - Surface-Enhanced Raman Spectroscopy:**

Input reasoning traces:
- reasoning_trace_SERSMedicalDetectionR6G: "I need to detect cancer biomarkers in blood at incredibly low concentrations - 10^-12 M, like finding molecules in a swimming pool. ELISA only goes to 10^-9 M, not sensitive enough for early diagnosis. Let me try SERS - Surface-Enhanced Raman Spectroscopy. Metal nanoparticles create huge EM field enhancements. I synthesize 60nm gold nanoparticles via citrate reduction. At 785nm laser, these have plasmon resonance amplifying local field by ~10^6. But I need selectivity too - can't detect everything. So I functionalize the gold with anti-PSA antibodies for prostate cancer. When I add patient serum, only PSA proteins bind. Here's the clever part: I add R6G (Rhodamine 6G) reporter molecules. R6G has enormous Raman cross-section and when it sits in nanogaps between gold particles, field enhancement shoots to 10^8 or 10^10. Incubate 30 min for PSA binding, add R6G which sticks near bound PSA. Hit with 785nm laser at 5mW - I see characteristic R6G peaks at 1650, 1510, 1310 cm^-1. Peak intensity directly proportional to PSA amount. Integrate 60 sec for good SNR. Comparing to calibration: detecting PSA at 0.1 ng/mL - that's 10,000x more sensitive than ELISA! On clinical samples, detected prostate cancer 3-6 months earlier than conventional tests. The breakthrough: combining selective antibody recognition with SERS amplification gives single-molecule sensitivity while maintaining specificity."

- reasoning_trace_SERSPollutantDetection: "Monitoring river water for pesticides. EPA limit for malathion is 0.1 ppb but standard chromatography needs 1 ppb minimum. I need 10x better for early warning. SERS might work. Instead of spherical particles, I'll fabricate silver nanorod arrays - sharp tips create hotter hotspots than spheres. Using oblique angle deposition: 80nm nanorods with 4:1 aspect ratio on silicon. Gaps between rods only 5-10nm - perfect for trapping molecules. I calculate enhancement should hit 10^10 at 532nm. Collect river water, filter through 0.2μm to remove debris and bacteria. Drop 50μL onto nanorod substrate. During 5-min adsorption, pesticide molecules diffuse into nanogaps. Small gap means molecules guaranteed in enhancement zone (<10nm from metal). Rinse gently - removes interfering organics/salts but leaves adsorbed pesticides. Excite with 532nm at 2mW, matching silver plasmon peak. Even at 0.01 ppb malathion, clear peaks at 1440 cm^-1 (P=S stretch), 1080 cm^-1 (P-O-C), 640 cm^-1 (C-S). Measuring 1440 peak height vs calibration standards for quantification. Tested 50 river sites, cross-validated against LC-MS: R^2=0.97 correlation. Best part: do this in field with portable Raman - no lab needed. Real-time monitoring at 10x below regulatory limits. The nanorod geometry is critical - those sharp tips and tight gaps push enhancement to 10^10."

Output aggregated insight:
{
  "insight_surfaceEnhancedRamanSpectroscopy": "This powerful technique applies across analytical chemistry, materials science, biosensing, pharmaceutical analysis, environmental monitoring, and forensics. The technique achieves single-molecule sensitivity (10^6-10^11 enhancement) while providing molecular structural information through vibrational fingerprints. When to use: When you need ultra-sensitive detection below conventional analytical limits, when you want label-free molecular identification, when analyzing trace contaminants or biomarkers, or when field-portable real-time analysis is required. Common steps: 1) Prepare SERS-active substrate - synthesize plasmonic nanostructures (gold/silver nanoparticles, nanorods, nanostars) optimizing particle size (20-100 nm), shape, and inter-particle spacing (1-10 nm gaps) to maximize electromagnetic field enhancement at laser wavelength 2) Functionalize substrate if needed - modify metallic surface with antibodies, aptamers, or molecular recognition elements for selective analyte binding and improved specificity 3) Prepare and apply sample - process sample (filter, dilute, concentrate as needed), deposit onto SERS substrate via drop-casting or flow-through, allow adsorption time for molecules to enter hot spots (<10 nm from metal surface) 4) Select laser parameters - choose wavelength matching plasmon resonance (532, 633, or 785 nm), optimize power (0.1-10 mW) to avoid sample damage while maximizing signal 5) Acquire SERS spectrum - collect Raman scattered light with appropriate integration time, record vibrational spectrum showing characteristic molecular peaks 6) Analyze spectral fingerprint - identify molecules by comparing peak positions to reference spectra, quantify concentration from peak intensities using calibration curves, assess molecular orientation from peak ratios 7) Validate and control quality - average multiple spots for reproducibility, use internal standards, verify with orthogonal methods, consider substrate heterogeneity and enhancement factor variations"
}
""".strip()


class GlobalReasoningServer(ABC):
    """Abstract FoT global aggregation pipeline with overridable steps 1, 2, and 3."""

    def __init__(
        self,
        *,
        agent_name: str,
        workspace: str | Path,
        openclaw_path: str | None = None,
        input_dirs: list[str] | None = None,
        num_insights: int | None = None,
        custom_prompt_section: str = "",
        timeout_seconds: float = 3600.0,
        max_files: int | None = None,
        seed: int | None = None,
    ):
        self.agent_name = agent_name
        self.workspace = Path(workspace)
        self.openclaw_path = openclaw_path
        if isinstance(input_dirs, str):
            input_dirs = [input_dirs]
        self.input_dirs = input_dirs or ["output"]
        self.num_insights = num_insights
        self.custom_prompt_section = (custom_prompt_section or "").strip()
        self.timeout_seconds = timeout_seconds
        self.max_files = max_files
        self.seed = seed
        self.output_dir: Path | None = None
        self.insight_store: dict[str, str] = {}
        self.insight_relationships: dict[str, Any] = {}
        self.aggregation_steps: list[dict[str, Any]] = []
        self.encyclopedia: str = ""
        self.encyclopedia_dict: dict[str, str] = {}

    def reset_state(self) -> None:
        self.insight_store = {}
        self.insight_relationships = {}
        self.aggregation_steps = []
        self.encyclopedia = ""
        self.encyclopedia_dict = {}

    @abstractmethod
    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        """Run global aggregation step 1 (trace collection) and return a step payload."""

    @abstractmethod
    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        """Run global aggregation step 2 (relationship profiling) and return a step payload."""

    @abstractmethod
    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run global aggregation step 3 (insight-library update) and return a step payload."""

    def _record_step(
        self,
        result: dict[str, Any],
        *,
        step_number: int,
        default_name: str,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"FoT global step {step_number} must return a dict.")
        step = dict(result)
        step["step"] = step_number
        step.setdefault("name", default_name)
        step.setdefault("timestamp", time.time())
        self.aggregation_steps.append(step)
        return step

    def _normalize_str_dict(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a dict of strings, got {type(raw).__name__}.")
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(value, str):
                raise ValueError(f"Entry '{key}' must be a string, got {type(value).__name__}.")
            normalized[str(key)] = re.sub(r"\s+", " ", value).strip()
        return normalized

    def _sync_collection_state(self, result: dict[str, Any]) -> None:
        self.insight_store = self._normalize_str_dict(result.get("insight_store"))

    def _sync_profiling_state(self, result: dict[str, Any]) -> None:
        profiling = result.get("profiling")
        if isinstance(profiling, dict):
            self.insight_relationships = profiling

    def _sync_extraction_state(self, result: dict[str, Any]) -> None:
        encyclopedia_dict = result.get("encyclopedia_dict")
        if encyclopedia_dict is None and isinstance(result.get("encyclopedia"), dict):
            encyclopedia_dict = result.get("encyclopedia")
        normalized = self._normalize_str_dict(encyclopedia_dict)
        self.encyclopedia_dict = normalized
        self.encyclopedia = json.dumps(normalized, indent=2, ensure_ascii=False)

    def collect_insight_books(self, json_files: list[str] | None = None) -> dict[str, Any]:
        collection = self._record_step(
            self.global_step_1(json_files),
            step_number=1,
            default_name="Global Step 1",
        )
        self._sync_collection_state(collection)
        collection["insight_store"] = dict(self.insight_store)
        return collection

    def aggregate_and_build_encyclopedia(
        self,
        json_files: list[str] | None = None,
        output_dir: str = "output",
        existing_encyclopedia_path: str | None = None,
    ) -> dict[str, Any]:
        """Run the three FoT server stages and return every intermediate result."""

        self.reset_state()
        self.output_dir = Path(output_dir)
        collection = self.collect_insight_books(json_files)
        if not self.insight_store:
            raise ValueError("Reasoning traces are empty: nothing to aggregate.")

        profiling = self._record_step(
            self.global_step_2(collection),
            step_number=2,
            default_name="Global Step 2",
        )
        self._sync_profiling_state(profiling)
        profiling["profiling"] = self.insight_relationships

        existing = _load_existing_encyclopedia(
            Path(existing_encyclopedia_path) if existing_encyclopedia_path else Path(output_dir) / "insight.json"
        )

        extraction = self._record_step(
            self.global_step_3(collection, profiling, existing),
            step_number=3,
            default_name="Global Step 3",
        )
        self._sync_extraction_state(extraction)
        if not self.encyclopedia_dict:
            raise ValueError("Insight library is empty after aggregation.")
        extraction["encyclopedia"] = self.encyclopedia
        extraction["encyclopedia_dict"] = dict(self.encyclopedia_dict)

        return {
            "collection": collection,
            "profiling": profiling,
            "extraction": extraction,
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "encyclopedia_dict": dict(self.encyclopedia_dict),
            "insight_store": dict(self.insight_store),
            "insight_relationships": self.insight_relationships,
            "insight_datastore_tokens": len(
                re.findall(r"\S+", json.dumps(self.insight_store, ensure_ascii=False))
            ),
            "total_output_tokens": _sum_output_tokens(profiling, extraction),
        }

    def save_results(self, result: dict[str, Any], output_dir: str = "output") -> tuple[str, str]:
        """Write ``insight.json``, ``insight.md``, and (when present) ``profiling.json``."""

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "insight.json"
        markdown_path = directory / "insight.md"
        _write_text_atomic(json_path, json.dumps(self.encyclopedia_dict, indent=2, ensure_ascii=False))
        _write_text_atomic(markdown_path, render_insight_markdown(self.encyclopedia_dict))
        profiling = result.get("insight_relationships") or self.insight_relationships
        if isinstance(profiling, dict) and profiling:
            _write_text_atomic(directory / "profiling.json", json.dumps(profiling, indent=2, ensure_ascii=False))
        return str(json_path), str(markdown_path)


class OpenClawFoTServer(GlobalReasoningServer):
    """Default OpenClaw-backed implementation of the global FoT pipeline."""

    def _call_model(self, prompt: str, step_name: str) -> tuple[str, dict[str, Any]]:
        session_id = f"{step_name}_{int(time.time() * 1000)}"
        result = run_openclaw_prompt(
            agent_name=self.agent_name,
            prompt=prompt,
            workspace=self.workspace,
            session_id=session_id,
            timeout_seconds=self.timeout_seconds,
            openclaw_path=self.openclaw_path,
        )
        if result["status"] != "success":
            raise RuntimeError(f"{step_name} failed: {result['stderr'] or result['status']}")
        return result["response_text"], result["usage"]

    def _prepend_custom_prompt(self, prompt: str) -> str:
        if not self.custom_prompt_section:
            return prompt
        return f"{self.custom_prompt_section}\n\n{prompt}"

    # ------------------------------------------------------------ step 1

    def _discover_trace_files(self) -> list[Path]:
        files: set[Path] = set()
        for root in self.input_dirs:
            base = Path(root)
            if not base.exists():
                continue
            files.update(base.rglob("problem_*.json"))
            files.update(base.rglob("paper_*.json"))
        ordered = sorted(files, key=lambda path: (_numeric_suffix(path), str(path)))
        if self.max_files is not None and self.max_files < len(ordered):
            rng = random.Random(self.seed)
            rng.shuffle(ordered)
            ordered = ordered[: self.max_files]
        return ordered

    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        """Collect ALL traces with globally indexed keys (no deduplication)."""

        files = [Path(item) for item in json_files] if json_files is not None else self._discover_trace_files()
        if not files:
            raise ValueError("Reasoning traces are empty: no problem*.json or paper*.json files were found.")

        all_traces: dict[str, str] = {}
        trace_counter = 0

        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Cannot read reasoning-trace file {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Reasoning-trace file {path} must contain a JSON object.")
            if "insight_book" in payload:
                trace_book = payload["insight_book"]
            elif "behavior_book" in payload:
                trace_book = payload["behavior_book"]
            else:
                trace_book = payload
            if not isinstance(trace_book, dict):
                raise ValueError(f"Reasoning traces in {path} must be a JSON object.")
            file_added = 0
            for key, value in trace_book.items():
                if key in METADATA_KEYS:
                    continue
                if not _nonempty_str(value):
                    raise ValueError(f"Reasoning trace '{key}' in {path} is empty or not a string.")
                trace_counter += 1
                file_added += 1
                all_traces[f"{key}_{trace_counter:06d}"] = re.sub(r"\s+", " ", value).strip()
            if not file_added:
                raise ValueError(f"Reasoning-trace file {path} contains no traces.")

        return {
            "name": "Collect Insights",
            "files_processed": len(files),
            "total_insights_collected": trace_counter,
            "insight_store": all_traces,
        }

    # ------------------------------------------------------------ step 2

    def _get_text_profiling_prompt(self, insight_store: dict[str, str]) -> str:
        """Prompt 4: constructing relationships between reasoning traces."""

        insights_text = "\n".join(f"- {name}: {desc}" for name, desc in insight_store.items())
        return f"""
You are analyzing a collection of reasoning traces generated for problem-solving. Understand their relationships and structure and build a profile of their relationships.

**Collected {len(insight_store)} traces in total:**
Note: This collection includes ALL traces from all problems without deduplication. Similar names with different indices (e.g., _001, _002) represent different occurrences that may have variations in their descriptions.

{insights_text}

**Your Task:**
Analyze these traces and build a profile of their relationships:

1. **Identify Clusters**:
   Group related traces that share:
   - Resolve the same or similar problem
   - Similar approaches or techniques
   - Nearly identical traces (e.g., same trace with minor variations in description or parameters)
   - Traces in the same cluster should be highly similar.

2. **Build Trace Relationships**:
Record all important relationships - traces don't exist in isolation and build a relationship graph that records:
   - **Prerequisite relationships**: Traces that must be learned/used before others
   - **Composition relationships**: Traces that can be chained/composed together
   - **Alternative relationships**: Different approaches to the same problem
   - **Complementary relationships**: Traces that work better together than individually used
   - **Derivation relationships**: Traces derived from or based on others
   - **Similar relationships**: Traces that are similar but not identical
Map relationships between traces within clusters and across clusters.

# Output Format:
{{
  "clusters": [
    {{
      "cluster_id": 0,
      "cluster_name": "Domain/Theme Name",
      "traces": ["name1", "name2", "name3"],
      "theme": "What is the high-level technical idea of the traces in this cluster?"
    }}
  ],
  "relationships": [
    {{
      "trace_a": "trace_name1",
      "trace_b": "trace_name2",
      "relationship_type": "prerequisite/complementary/alternative/similar/derived_from/composes_with",
      "description": "How these traces relate to each other and Why"
    }}
  ]
}}

**Output your analysis as JSON only:**
""".strip()

    def _profiling_fingerprint(self, insight_store: dict[str, str]) -> str:
        payload = {
            "insight_store": insight_store,
            "agent_name": self.agent_name,
            "custom_prompt_section": self.custom_prompt_section,
        }
        return _sha256_json(payload)

    def _load_profiling_checkpoint(self, fingerprint: str) -> dict[str, Any] | None:
        if self.output_dir is None:
            return None
        profiling_path = self.output_dir / "profiling.json"
        checkpoint_path = self.output_dir / "profiling_checkpoint.json"
        if not (profiling_path.exists() and checkpoint_path.exists()):
            return None
        try:
            profiling = json.loads(profiling_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Corrupt profiling checkpoint in {self.output_dir}: {exc}") from exc
        # A checkpoint for a different trace batch is simply not reused.
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("version") == PROFILING_CHECKPOINT_VERSION
            and checkpoint.get("insight_store_sha256") == fingerprint
            and checkpoint.get("profiling_sha256") == _sha256_json(profiling)
        ):
            validate_profiling(profiling)
            return profiling
        return None

    def _save_profiling_checkpoint(self, profiling: dict[str, Any], fingerprint: str, response: str) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(self.output_dir / "profiling_raw_response.txt", response)
        _write_text_atomic(self.output_dir / "profiling.json", json.dumps(profiling, indent=2, ensure_ascii=False))
        checkpoint = {
            "version": PROFILING_CHECKPOINT_VERSION,
            "insight_store_sha256": fingerprint,
            "profiling_sha256": _sha256_json(profiling),
        }
        _write_text_atomic(
            self.output_dir / "profiling_checkpoint.json",
            json.dumps(checkpoint, indent=2, ensure_ascii=False),
        )

    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        """Prompt 4 (with checkpoint reuse); raises on any malformed response."""

        insight_store = self._normalize_str_dict(collection_result.get("insight_store"))
        if not insight_store:
            raise ValueError("Reasoning traces are empty.")

        fingerprint = self._profiling_fingerprint(insight_store)
        saved = self._load_profiling_checkpoint(fingerprint)
        if saved is not None:
            return {
                "name": "Text-Based Profiling",
                "profiling": saved,
                "resumed": True,
                "usage": {},
            }

        prompt = self._prepend_custom_prompt(self._get_text_profiling_prompt(insight_store))
        response, usage = self._call_model(prompt, "profiling")
        if not _nonempty_str(response):
            raise ValueError("Cluster and relationship model response is empty.")
        profiling = _extract_json_object(response)
        if profiling is None:
            raise ValueError("Cluster and relationship response is not a valid JSON object.")
        validate_profiling(profiling)
        self._save_profiling_checkpoint(profiling, fingerprint, response)
        return {
            "name": "Text-Based Profiling",
            "prompt": prompt,
            "response": response,
            "profiling": profiling,
            "usage": usage,
        }

    # ------------------------------------------------------------ step 3

    def _get_knowledge_extraction_prompt(
        self,
        insight_store: dict[str, str],
        profiling: dict[str, Any],
        existing_encyclopedia: dict[str, str] | str | None = None,
    ) -> str:
        """Prompt 5: building (updating) the insight library."""

        profiling = profiling if isinstance(profiling, dict) else {}
        # Preserve the complete profiling output; do not flatten clusters,
        # which could silently drop trace membership or themes.
        clusters_text = json.dumps(profiling.get("clusters", []), indent=2, ensure_ascii=False)
        relationships_text = json.dumps(profiling.get("relationships", []), indent=2, ensure_ascii=False)
        clusters_section = (
            "- Use the complete cluster analysis below, including every "
            "cluster's trace membership, theme, and any other returned "
            f"fields, to help organize the insight library:\n{clusters_text}"
        )
        relationships_section = (
            "- Use the complete trace relationships below, including every "
            "relationship's endpoints, type, description, and any other "
            f"returned fields, to help organize the insight library:\n{relationships_text}"
        )

        all_insights_text = ";".join(f"{name}: {desc}" for name, desc in insight_store.items())
        if isinstance(existing_encyclopedia, dict):
            existing_text = json.dumps(existing_encyclopedia, indent=2, ensure_ascii=False) if existing_encyclopedia else ""
        else:
            existing_text = (existing_encyclopedia or "").strip()

        proper_number = f"exactly {self.num_insights}" if self.num_insights is not None else "a reasonable number"
        exact_count_requirement = (
            f"8. The JSON object must contain EXACTLY {self.num_insights} "
            "top-level insight entries. Count them before responding."
            if self.num_insights is not None
            else ""
        )
        uncertainty_rule = (
            f"- Even if uncertain, return exactly {self.num_insights} valid insights; "
            "never return a partial library or free text."
            if self.num_insights is not None
            else "- If uncertain, still output a valid JSON object (possibly with fewer insights), never free text."
        )

        return f"""
**Your Task:**
You are extracting fundamental insights from a collection of problem-solving traces.

**Output Requirements (STRICT):**
1. Return EXACTLY one valid JSON object and nothing else.
2. Do NOT output markdown code fences.
3. Do NOT output explanations, notes, reasoning, prefixes, suffixes, or `<think>` content.
4. Do NOT output list/array at top-level.
5. Every key must start with "insight_".
6. Every value must be a single string.
7. No nested objects, no nested arrays.
{exact_count_requirement}

**Required JSON shape:**
{{
    "insight_name1": "description string",
    "insight_name2": "description string"
}}

**Formatting Rules:**
- Use valid JSON syntax only.
- Keep top-level as key-value pairs only.
- Escape quotes in descriptions with backslash: \\"
{uncertainty_rule}

Your goal is to extract a comprehensive set of fundamental, cross-domain insights that can be derived and applied beyond their original domain and meet the following requirements:
- Combine previous insights (if any): {existing_text if existing_text else "None"} with new insights.
- Extract your insights based on all client reasoning traces: {all_insights_text}. These traces are derived from solving specific problems (bottom-up approach)
{clusters_section}
{relationships_section}
- Your task is to extract multi-disciplinary, fundamental knowledge (top-down approach) which can be generalized to multi-domain problem-solving.
- The extracted insights should be able to DERIVE and GUIDE the use of the collected insights
- The extracted insights cannot be too general. They are not supposed to be knowledge which can be applied to any problem. They should be fundamental knowledge to particular several domains but specific.
- You must extract {proper_number} insights. Do not over-simplify or make descriptions needlessly long.
- DO NOT over-merge insights.

Insights should have following properties:
1. **Extract Reusable Primitives**:
   - For EACH cluster, extract multiple fundamental insights capturing core essence and variations (DO NOT over-merge)
   - Identify cross-domain patterns that apply to multiple fields
   - Create reusable, composable primitives specific enough to be actionable

2. **Knowledge to include**:
   - **Fundamental Level**: Core principles underlying multiple domains
   - **General Level**: Broad techniques for related problem types
   - **Cross-Domain**: Insights transferable beyond origin field

3. **Preserve While Generalizing**:
   - Create fundamental versions that can guide/derive original insights
   - Maintain important variations rather than collapsing into single insight

4. **Description Format:**
   Each description is a single string containing:
   - What the insight is and how it solves problems
   - When to use: problem types, conditions, triggers (be comprehensive and specific)

{PROMPT5_EXAMPLES}
""".strip()

    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Prompt 5; raises unless the response is a valid insight library."""

        insight_store = self._normalize_str_dict(collection_result.get("insight_store"))
        if not insight_store:
            raise ValueError("Reasoning traces are empty.")
        profiling = profiling_result.get("profiling")
        validate_profiling(profiling)
        prompt = self._prepend_custom_prompt(
            self._get_knowledge_extraction_prompt(insight_store, profiling, existing_encyclopedia)
        )
        response, usage = self._call_model(prompt, "aggregate")
        if not _nonempty_str(response):
            raise ValueError("Knowledge-extraction response is empty.")
        library = _extract_json_object(response)
        if library is None:
            raise ValueError("Knowledge-extraction response is not a valid JSON object.")
        validate_insight_library(library, num_insights=self.num_insights)
        library = {key: re.sub(r"\s+", " ", value).strip() for key, value in library.items()}
        return {
            "name": "Knowledge Extraction",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json.dumps(library, indent=2, ensure_ascii=False),
            "encyclopedia_dict": library,
            "usage": usage,
        }


TextBasedInsightAggregationServer = OpenClawFoTServer


# ------------------------------------------------------------------ validation


def validate_profiling(profiling: Any) -> None:
    """Raise ``ValueError`` unless ``profiling`` is a complete, well-formed Prompt-4 result."""

    if not isinstance(profiling, dict):
        raise ValueError("Cluster and relationship information must be a JSON object.")
    clusters = profiling.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Cluster information is empty.")
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            raise ValueError(f"Cluster {index} is not an object.")
        if "cluster_id" not in cluster:
            raise ValueError(f"Cluster {index} is missing 'cluster_id'.")
        if not _nonempty_str(cluster.get("cluster_name")):
            raise ValueError(f"Cluster {index} has an empty 'cluster_name'.")
        traces = cluster.get("traces")
        if not isinstance(traces, list) or not traces or not all(_nonempty_str(trace) for trace in traces):
            raise ValueError(f"Cluster {index} has no valid trace members.")
        if not _nonempty_str(cluster.get("theme")):
            raise ValueError(f"Cluster {index} has an empty 'theme'.")
    relationships = profiling.get("relationships")
    if not isinstance(relationships, list) or not relationships:
        raise ValueError("Relationship information is empty.")
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            raise ValueError(f"Relationship {index} is not an object.")
        for field in ("trace_a", "trace_b", "relationship_type", "description"):
            if not _nonempty_str(relationship.get(field)):
                raise ValueError(f"Relationship {index} has an empty or missing '{field}'.")


def validate_insight_library(insight_library: Any, *, num_insights: int | None = None) -> None:
    """Raise ``ValueError`` unless every entry is ``insight_*`` -> non-empty string."""

    if not isinstance(insight_library, dict) or not insight_library:
        raise ValueError("Insight library is empty.")
    for name, content in insight_library.items():
        if not str(name).startswith("insight_"):
            raise ValueError(f"Insight name '{name}' must start with 'insight_'.")
        if not _nonempty_str(content):
            raise ValueError(f"Insight '{name}' content is empty or not a string.")
    if num_insights is not None and len(insight_library) != num_insights:
        raise ValueError(f"Insight library has {len(insight_library)} entries; exactly {num_insights} were requested.")


# --------------------------------------------------------------------- helpers


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _numeric_suffix(path: Path) -> float:
    match = re.search(r"_(\d+)$", path.stem)
    return int(match.group(1)) if match else float("inf")


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sum_output_tokens(*steps: dict[str, Any]) -> int:
    total = 0
    for step in steps:
        usage = step.get("usage") if isinstance(step, dict) else None
        if isinstance(usage, dict):
            for key in ("output_tokens", "output", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    total += int(value)
                    break
    return total


def _find_json_object(text: str) -> str | None:
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a model response, preserving nested values."""

    if not isinstance(text, str):
        return None
    candidate = _find_json_object(text.strip())
    if candidate is None:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_existing_encyclopedia(path: Path) -> dict[str, str] | None:
    """Load the previous round's insight library; ``None`` only if it does not exist yet."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read existing insight library {path}: {exc}") from exc
    validate_insight_library(payload)
    return dict(payload)


def choose_most_common_model(models: list[str]) -> str | None:
    filtered = [model for model in models if model]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenClaw-backed FoT aggregation server (Prompts 4-5).")
    parser.add_argument("--agent", required=True, help="OpenClaw agent name to use")
    parser.add_argument("--workspace", required=True, help="Workspace path for the OpenClaw agent")
    parser.add_argument("--input-dir", nargs="+", required=True, help="Input directories containing reasoning trace JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for the aggregated insight library")
    parser.add_argument("--openclaw-path", default=None, help="Path to the openclaw binary")
    parser.add_argument("--num-insights", type=int, default=None, help="Optional exact number of insights")
    parser.add_argument("--max-files", type=int, default=None, help="Randomly sample at most this many trace files")
    parser.add_argument("--seed", type=int, default=None, help="Seed for --max-files sampling")
    parser.add_argument("--existing-library", default=None, help="Previous insight.json to merge (default: <output-dir>/insight.json)")
    args = parser.parse_args()

    server = OpenClawFoTServer(
        agent_name=args.agent,
        workspace=args.workspace,
        openclaw_path=args.openclaw_path,
        input_dirs=args.input_dir,
        num_insights=args.num_insights,
        max_files=args.max_files,
        seed=args.seed,
    )
    result = server.aggregate_and_build_encyclopedia(
        output_dir=args.output_dir,
        existing_encyclopedia_path=args.existing_library,
    )
    json_path, markdown_path = server.save_results(result, output_dir=args.output_dir)
    print(json.dumps({"insight_json": json_path, "insight_markdown": markdown_path}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
