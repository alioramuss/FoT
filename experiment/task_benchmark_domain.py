"""
Multi-dataset Benchmark Problem Solving Pipeline (task_benchmark_domain)
- Supports running STEP 1 (insight extraction) across multiple datasets.
- Allows choosing which insight sets to aggregate in STEP 2.
- Evaluates multiple target datasets sequentially in STEP 3 using the shared encyclopedia.

Usage is similar to math_pipeline.py but adds list-style arguments.

## Supported Datasets

### Hugging Face Datasets (via 🤗 datasets library):
- gsm8k, gsm8k_train: Grade School Math (8K examples)
- aime24, aime25, aime26: American Invitational Mathematics Examination
- math500, math1000: Competition math problems
- gpqa, gpqa_diamond: Graduate-level science problems (GPQA benchmark)
- hle: Full text and image questions from Humanity's Last Exam (gated on Hugging Face)
- livecodebench_v6: Cumulative LiveCodeBench release v6

### Local Datasets:
- CSV or JSON files in math_datasets/ directory

### IMOBench (International Mathematical Olympiad Benchmark)
From: https://github.com/google-deepmind/superhuman/tree/main/imobench
See: https://imobench.github.io

IMOBench consists of three specialized benchmarks:

1. **IMO-AnswerBench** (400 problems)
   - Short-answer problems with verifiable final answers
   - Categories: Algebra, Combinatorics, Geometry, Number Theory
   - Difficulty: pre-IMO, IMO-Easy, IMO-Medium, IMO-Hard
   - CSV columns: problem/question, answer/solution, id, difficulty
   - Evaluation: Symbolic comparison with algebraic normalization

2. **IMO-ProofBench** (60 problems)
   - Proof-writing evaluation (not just final answers)
   - Requires human expert grading (0-7 scale)
   - Can use ProofAutoGrader with Gemini 2.5 Pro for automatic evaluation
   - Correlation with human grading: 0.96 (basic), 0.93 (advanced)
   - Not automatically evaluated in this script - use external graders

3. **IMO-GradingBench** (1000 examples)
   - Dataset for evaluating grading capability
   - Problem + proposed solution + human grade (0-7)
   - Classification labels: Correct (7), Almost (6), Partial (1), Incorrect (0)
   - CSV columns: problem, solution, grade, grade_label

### Answer Verification for IMOBench:
- Numeric: Direct numeric comparison with tolerance 1e-6
- Symbolic: Algebraic equivalence checking (normalized forms)
- String: Case-insensitive exact matching
- Partial: Substring matching for multi-answer problems
- Unit handling: Removes common units (degrees, radians, cm, m, etc.)

For proof-based evaluation on IMO-ProofBench, consider:
- Using Gemini 2.5 Pro's ProofAutoGrader (available in superhuman repo)
- Implementing LLM-based grading with reference solutions
- Human expert evaluation for rigorous assessment
"""

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import parallel_utils
from utils import (
    OpenRouterInFlightBudgetError,
    is_openrouter_budget_error,
    normalize_api_model,
)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from client import ChainOfThoughtReader
from client_metacognitive import MetacognitiveClient
from client_trt import TRTClient
from client_hyperagents import HyperAgentsClient
from client_evolveprompt import EvolvePromptClient
from client_ace import ACEClient
from math_datasets.imo_benchmark import imo_evaluator
from math_datasets.livemathbench import livemathbench_evaluator
from math_datasets.utils import extract_numbers
from server_text import TextBasedInsightAggregationServer

try:
    from server import InsightAggregationServer as _InsightAggregationServer
    _GRAPH_SERVER_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:
    _InsightAggregationServer = None
    _GRAPH_SERVER_IMPORT_ERROR = exc

# Dataset registry: (source, path_or_hf_name, data_dir_or_col_map, split_or_none)
# - source="hf": use Hugging Face with (hf_name, data_dir, split)
# - source="json": load JSON from path
# - source="csv": load CSV from path with optional column mapping dict
#
# IMOBench Benchmarks (https://imobench.github.io/):
# - IMO-AnswerBench: 400 short-answer problems (CSV)
#   Columns: problem/question, answer/solution, id, difficulty
#   Evaluation: Symbolic comparison with unit normalization
#
# - IMO-ProofBench: 60 proof-based problems (not directly scored here)
#   Requires: ProofAutoGrader (Gemini 2.5 Pro) or human evaluation
#   Grading: 0-7 scale, ~high correlation with human experts (0.96)
#
# - IMO-GradingBench: 1000 grading examples (CSV)
#   Columns: problem, solution, grade (0-7), grade_label (Correct/Almost/Partial/Incorrect)
#   Use for training automatic graders
# LiveMathBench (Live Mathematical Reasoning Benchmark) — From OpenCompass
#   Configs: v202412_AMC_en, v202412_CCEE_en, v202412_CNMO_en, v202412_WLPMC_en, v202412_hard_en, v202505_hard_en
#   Reference: https://huggingface.co/datasets/opencompass/LiveMathBench
#   Columns: question, answer, question_type
# GPQA (Graduate-Level Google-Proof Q&A Benchmark) — Graduate-level science problems
#   Reference: https://huggingface.co/datasets/Idavidrein/gpqa
#   Diamond variant: https://huggingface.co/datasets/fingertap/GPQA-Diamond
#   Contains graduate-level questions in physics, chemistry, and biology
DATASET_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    "gsm8k": ("hf", "openai/gsm8k", "main", "test"),
    "gsm8k_train": ("hf", "openai/gsm8k", "main", "train"),
    "aime25": ("hf", "math-ai/aime25", None, "test"),
    "aime24": ("hf", "math-ai/aime24", None, "test"),
    "aime26": ("hf", "math-ai/aime26", None, "test"),
    "math500": ("hf", "HuggingFaceH4/MATH-500", None, "test"),
    "math1000": ("hf", "hendrycks/competition_math", None, "test"),
    # GPQA datasets (Graduate-level science problems)
    "gpqa": ("hf", "Idavidrein/gpqa", "gpqa_main", "train"),
    "gpqa_diamond": ("hf", "Idavidrein/gpqa", "gpqa_diamond", "train"),
    # HLE is multimodal; image fields are attached through the OpenRouter
    # solution-generation path while the text fields remain intact.
    "hle": ("hf", "cais/hle", None, "test"),
    # LiveCodeBench datasets (Code generation)
    # Using bzantium/livecodebench (Parquet-based clone, works with datasets 3.0+)
    "livecodebench": ("hf", "bzantium/livecodebench", "release_v6", "test"),
    "livecodebench_v6": (
        "hf",
        "bzantium/livecodebench",
        "release_v6",
        "test",
    ),
    "livecodebench_lite": ("hf", "bzantium/livecodebench", "v6", "test"),  # Latest version increment
    # LiveMathBench datasets (OpenCompass)
    "livemathbench_amc": ("hf", "opencompass/LiveMathBench", "v202412_AMC_en", "test"),
    "livemathbench_ccee": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_CCEE_en",
        "test",
    ),
    "livemathbench_cnmo": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_CNMO_en",
        "test",
    ),
    "livemathbench_wlpmc": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_WLPMC_en",
        "test",
    ),
    "livemathbench_hard_2024": (
        "hf",
        "opencompass/LiveMathBench",
        "v202412_hard_en",
        "test",
    ),
    "livemathbench_hard_2025": (
        "hf",
        "opencompass/LiveMathBench",
        "v202505_hard_en",
        "test",
    ),
    # IMO benchmark (IMOBench) — CSV files from https://github.com/google-deepmind/superhuman/tree/main/imobench
    # Download from: https://github.com/google-deepmind/superhuman/tree/main/imobench
    "imo_answerbench": ("csv", "math_datasets/answerbench.csv", None, None),
    "imo_answerbench_algebra": ("csv", "math_datasets/imo_algebra.csv", None, None),
    "imo_answerbench_geometry": ("csv", "math_datasets/imo_geometry.csv", None, None),
    "imo_answerbench_number_theory": (
        "csv",
        "math_datasets/imo_number_theory.csv",
        None,
        None,
    ),
    # IMO-ProofBench: Requires external evaluation (ProofAutoGrader or human experts)
    "imo_proofbench": ("csv", "math_datasets/proofbench.csv", None, None),
    # IMO-GradingBench: For training/evaluating automatic graders
    "imo_gradingbench": ("csv", "math_datasets/gradingbench.csv", None, None),
}


_CLIENT_CHOICES = ["default", "metacognitive", "trt", "hyperagents", "evolveprompt", "ace"]


def _build_client(
    client_type: str,
    model_name: str,
    device: Optional[str],
    use_api: bool,
    api_key: Optional[str],
    api_provider: str,
    output_dir: str,
    load_in_8bit: bool,
    api_model: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
):
    """Factory: return the requested client instance."""
    common = dict(
        model_name=model_name,
        device=device,
        use_api=use_api,
        api_key=api_key,
        api_provider=api_provider,
        output_dir=output_dir,
        load_in_8bit=load_in_8bit,
    )
    if client_type == "metacognitive":
        client = MetacognitiveClient(**common)
    elif client_type == "trt":
        client = TRTClient(
            **common, **({"api_model": api_model} if api_model else {})
        )
    elif client_type == "hyperagents":
        client = HyperAgentsClient(**common)
    elif client_type == "evolveprompt":
        client = EvolvePromptClient(**common)
    elif client_type == "ace":
        client = ACEClient(
            **common, **({"api_model": api_model} if api_model else {})
        )
    else:  # "default"
        client = ChainOfThoughtReader(
            model_name=model_name,
            device=device,
            use_api=use_api,
            api_key=api_key,
            api_provider=api_provider,
            api_model=api_model,
            reasoning_enabled=reasoning_enabled,
            load_in_8bit=load_in_8bit,
        )
    if use_api:
        client.api_model_name = normalize_api_model(api_provider, api_model)
    return client


_BENCHMARK_PROCESS_PIPELINE = None
_BENCHMARK_PROCESS_CLIENT = None
_BENCHMARK_PROCESS_CONFIG: Optional[Dict[str, Any]] = None


def _initialize_benchmark_process(config: Dict[str, Any]) -> None:
    """Create one pipeline helper and one API client in each worker process."""
    global _BENCHMARK_PROCESS_PIPELINE
    global _BENCHMARK_PROCESS_CLIENT
    global _BENCHMARK_PROCESS_CONFIG

    _BENCHMARK_PROCESS_CONFIG = config
    _BENCHMARK_PROCESS_PIPELINE = BenchmarkDomainPipeline(
        model_name=config["model_name"],
        device=config["device"],
        output_dir=config["output_dir"],
        use_api=config["use_api"],
        api_key=config["api_key"],
        api_provider=config["api_provider"],
        mode=config["mode"],
        num_iterations=1,
        load_in_8bit=config["load_in_8bit"],
        client_type=config["client_type"],
        api_model=config["api_model"],
        reasoning_enabled=config["reasoning_enabled"],
        num_workers=1,
    )
    _BENCHMARK_PROCESS_CLIENT = _build_client(
        client_type=config["client_type"],
        model_name=config["model_name"],
        device=config["device"],
        use_api=config["use_api"],
        api_key=config["api_key"],
        api_provider=config["api_provider"],
        output_dir=config["output_dir"],
        load_in_8bit=config["load_in_8bit"],
        api_model=config["api_model"],
        reasoning_enabled=config["reasoning_enabled"],
    )
    print(
        f"[API worker pid={os.getpid()}] initialized "
        f"{config['client_type']} client for {config['dataset_name']}",
        flush=True,
    )


def _solve_benchmark_problem_process(item):
    """Solve and extract insights inside one process-local API agent."""
    if (
        _BENCHMARK_PROCESS_PIPELINE is None
        or _BENCHMARK_PROCESS_CLIENT is None
        or _BENCHMARK_PROCESS_CONFIG is None
    ):
        raise RuntimeError("Benchmark process worker was not initialized")

    idx, problem_data = item
    pipeline = _BENCHMARK_PROCESS_PIPELINE
    task_client = _BENCHMARK_PROCESS_CLIENT
    config = _BENCHMARK_PROCESS_CONFIG
    dataset_name = config["dataset_name"]
    problem_text, test_cases = pipeline._format_problem(problem_data, dataset_name)
    if not problem_text:
        return problem_text, test_cases, None, None, "missing", os.getpid()

    image = None
    image_source = "none"
    if dataset_name == "hle":
        from hle_datasets.hle import extract_hle_image

        image, image_source = extract_hle_image(problem_data)
    try:
        if image is not None and not isinstance(task_client, ChainOfThoughtReader):
            raise ValueError(
                "Multimodal HLE insight generation currently requires --client default."
            )
        result = task_client.solve_problem(
            task=problem_text,
            insights_section=config["insights_section"],
            **({"image": image} if image is not None else {}),
        )
        return problem_text, test_cases, result, None, image_source, os.getpid()
    except OpenRouterInFlightBudgetError:
        raise
    except Exception as exc:  # noqa: BLE001
        if is_openrouter_budget_error(exc):
            raise OpenRouterInFlightBudgetError(
                "OpenRouter rejected the benchmark request because the account "
                "budget cannot admit it. Stopping this run immediately; completed "
                "checkpoints remain resumable."
            ) from exc
        error = RuntimeError(f"{type(exc).__name__}: {exc}")
        return problem_text, test_cases, None, error, image_source, os.getpid()
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


def _evaluate_benchmark_problem_process(item):
    """Evaluate one problem and atomically checkpoint it in a worker process."""
    if (
        _BENCHMARK_PROCESS_PIPELINE is None
        or _BENCHMARK_PROCESS_CLIENT is None
        or _BENCHMARK_PROCESS_CONFIG is None
    ):
        raise RuntimeError("Benchmark process worker was not initialized")

    idx, problem_data = item
    pipeline = _BENCHMARK_PROCESS_PIPELINE
    task_client = _BENCHMARK_PROCESS_CLIENT
    config = _BENCHMARK_PROCESS_CONFIG
    dataset_name = config["dataset_name"]
    problem_text, test_cases_for_eval = pipeline._format_problem(
        problem_data, dataset_name
    )
    if not problem_text:
        return {"index": idx, "skip": True, "worker_pid": os.getpid()}

    image = None
    image_source = "none"
    if dataset_name == "hle":
        from hle_datasets.hle import extract_hle_image

        image, image_source = extract_hle_image(problem_data)
    try:
        if image is not None and not isinstance(task_client, ChainOfThoughtReader):
            raise ValueError(
                "Multimodal HLE evaluation currently requires --client default."
            )
        prompt = task_client._get_solution_prompt(
            problem_text, insights_section=config["insights_section"]
        )
        if image is not None:
            prompt = (
                "An image is attached to this problem. Use both the image "
                "and the question text; do not ignore visual evidence.\n\n" + prompt
            )
        response, token_info = task_client._call_model(
            prompt,
            None,
            max_new_tokens=config["eval_max_output_tokens"],
            **({"image": image} if image is not None else {}),
        )
        solution = response
        number_output_tokens = token_info.get("output_tokens", 0)
        number_reasoning_tokens = token_info.get("reasoning_tokens", 0)
        loop_count = pipeline._count_consecutive_sentence_loops(solution)
        predicted_answer = pipeline._extract_answer_from_solution(
            solution, dataset_name, problem_data
        )
        if test_cases_for_eval:
            is_correct = pipeline._check_answer_match(
                solution, test_cases_for_eval, dataset_name, problem_text
            )
            ground_truth = None
        else:
            ground_truth = pipeline._get_ground_truth(problem_data, dataset_name)
            is_correct = False
            if predicted_answer:
                is_correct = pipeline._check_answer_match(
                    predicted_answer, ground_truth, dataset_name, problem_text
                )
        output_data = {
            "problem": problem_text,
            "problem_id": problem_data.get("id", idx),
            "solution": solution,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "number_output_tokens": number_output_tokens,
            "number_reasoning_tokens": number_reasoning_tokens,
            "loop_count": loop_count,
            "multimodal_image_source": image_source,
            "worker_pid": os.getpid(),
        }
        output_path = os.path.join(config["eval_dir"], f"problem_{idx:04d}.json")
        temporary_path = f"{output_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_path, output_path)
        return {
            "index": idx,
            "output_data": output_data,
            "ground_truth": ground_truth,
            "test_cases": bool(test_cases_for_eval),
            "is_correct": is_correct,
            "number_output_tokens": number_output_tokens,
            "loop_count": loop_count,
            "worker_pid": os.getpid(),
        }
    except OpenRouterInFlightBudgetError:
        raise
    except Exception as exc:  # noqa: BLE001
        if is_openrouter_budget_error(exc):
            raise OpenRouterInFlightBudgetError(
                "OpenRouter rejected the benchmark request because the account "
                "budget cannot admit it. Stopping this run immediately; completed "
                "checkpoints remain resumable."
            ) from exc
        return {
            "index": idx,
            "error": f"{type(exc).__name__}: {exc}",
            "worker_pid": os.getpid(),
        }
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass


class BenchmarkDomainPipeline:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "math_output",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        mode: str = "text",
        num_iterations: int = 3,
        load_in_8bit: bool = False,
        client_type: str = "default",
        api_model: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        num_workers: int = parallel_utils.DEFAULT_NUM_WORKERS,
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.api_model = (
            normalize_api_model(api_provider, api_model) if api_model else None
        )
        self.reasoning_enabled = reasoning_enabled
        self.mode = mode  # "normal" uses server.py, "text" uses server_text.py
        self.iterative = True  # Always True
        self.num_iterations = num_iterations
        self.load_in_8bit = load_in_8bit
        self.client_type = client_type
        self.num_workers = parallel_utils.worker_count(num_workers)

        os.makedirs(output_dir, exist_ok=True)

        self.client = None
        self.server: Optional[Any] = None
        self.server_text: Optional[TextBasedInsightAggregationServer] = None
        self.encyclopedia_loaded = False

    def _process_config(
        self,
        dataset_name: str,
        insights_section: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Return only serializable state needed by spawned API workers."""
        config = {
            "client_type": self.client_type,
            "model_name": self.model_name,
            "device": self.device,
            "use_api": self.use_api,
            "api_key": self.api_key,
            "api_provider": self.api_provider,
            "api_model": self.api_model,
            "reasoning_enabled": self.reasoning_enabled,
            "output_dir": self.output_dir,
            "load_in_8bit": self.load_in_8bit,
            "mode": self.mode,
            "dataset_name": dataset_name,
            "insights_section": insights_section,
        }
        config.update(extra)
        return config

    def _create_graph_server(self):
        """Construct the optional GraphRAG server only when normal mode needs it."""
        if _InsightAggregationServer is None:
            raise ImportError(
                "--mode normal requires optional GraphRAG dependencies such as "
                "sentence_transformers. Use --mode text or install those dependencies."
            ) from _GRAPH_SERVER_IMPORT_ERROR
        return _InsightAggregationServer(
            model_name=self.model_name,
            device=self.device,
            input_dir=self.output_dir,
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
            api_model=self.api_model,
        )

    def _count_consecutive_sentence_loops(self, text: str) -> int:
        """Count repeated consecutive sentences in the generated text.

        A loop is counted when a sentence is identical to the immediately
        preceding sentence. Multiple consecutive repeats are counted
        individually (e.g., A A A B → loops=2).
        """
        if not text:
            return 0

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        prev = None
        loops = 0
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            if prev is not None and cleaned == prev:
                loops += 1
            prev = cleaned
        return loops

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------
    def _load_local_json(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a local JSON file."""
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.json"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Provide {candidate_path} or update DATASET_REGISTRY."
            )
        with open(candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_csv_file(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a CSV file.

        Tries to infer column names from common patterns:
        - Problem: problem, question, problem_text, task, statement
        - Answer: answer, solution, final_answer, answer_text
        - ID: id, problem_id, num, number
        """
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.csv"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"CSV file for dataset '{dataset_name}' not found at {candidate_path}"
            )

        print(f"Loading CSV file from {candidate_path}...")
        with open(candidate_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {candidate_path} is empty or has no header")

            fieldnames_lower = [fn.lower() for fn in reader.fieldnames]
            print(f"CSV columns: {reader.fieldnames}")

            # Map CSV columns to standard schema
            problem_cols = [
                "problem",
                "question",
                "problem_text",
                "task",
                "statement",
                "text",
            ]
            # Note: For gradingbench, "response" is the student answer being graded
            # For answerbench/proofbench, "short answer" or "solution" is the correct answer
            answer_cols = [
                "answer",
                "solution",
                "final_answer",
                "answer_text",
                "short answer",
                "short_answer",
                "response",  # For gradingbench
            ]
            id_cols = [
                "id",
                "problem_id",
                "problem id",
                "grading_id",
                "grading id",
                "num",
                "number",
                "idx",
            ]

            problem_col = None
            answer_col = None
            id_col = None  # Reserved for future ID column mapping

            for col in problem_cols:
                if col in fieldnames_lower:
                    problem_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            for col in answer_cols:
                if col in fieldnames_lower:
                    answer_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            for col in id_cols:
                if col in fieldnames_lower:
                    id_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            if not problem_col:
                raise ValueError(
                    f"Could not find problem column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {problem_cols}"
                )
            if not answer_col:
                raise ValueError(
                    f"Could not find answer column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {answer_cols}"
                )

            data = []
            for row_idx, row in enumerate(reader, 1):
                data.append(row)

            print(f"Loaded {len(data)} rows from CSV")
            return data

    def _normalize_problems(
        self, raw_problems: List[Dict], dataset_name: str
    ) -> List[Dict]:
        """Ensure a consistent schema across sources."""

        # Helper to find field from problem dict with multiple possible names
        def get_field(obj: Dict, candidates: List[str], default: str = "") -> str:
            for candidate in candidates:
                for key in obj.keys():
                    if key.lower() == candidate.lower():
                        val = obj[key]
                        return str(val) if val is not None else default
            return default

        normalized = []
        for idx, problem in enumerate(raw_problems):
            # Try various column name combinations (case-insensitive)
            problem_text = get_field(
                problem,
                ["problem", "question", "problem_text", "task", "statement", "text"],
            )
            answer_text = get_field(
                problem, ["answer", "solution", "final_answer", "answer_text"]
            )
            id_val = get_field(
                problem, ["id", "problem_id", "num", "number", "idx"], str(idx + 1)
            )

            normalized_problem = {
                "id": int(id_val) if id_val.isdigit() else id_val,
                "problem": problem_text,
                "question": problem_text,  # Keep both for compatibility
                "solution": get_field(problem, ["solution", "step_by_step"]),
                "answer": answer_text,
            }

            # GSM8K stores answer as "solution #### answer"; split when present.
            if dataset_name.startswith("gsm8k"):
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    normalized_problem["solution"] = parts[0].strip()
                    normalized_problem["answer"] = parts[-1].strip()

            # Preserve all other fields from original
            for key, value in problem.items():
                if key not in normalized_problem:
                    normalized_problem[key] = value
            normalized.append(normalized_problem)
        return normalized

    def load_math_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a dataset from Hugging Face, CSV, or JSON file."""
        # Registry lookup
        entry = DATASET_REGISTRY.get(dataset_name)
        if not entry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {', '.join(DATASET_REGISTRY.keys())}"
            )

        source_type, path_or_hf_name, data_dir, split = entry

        # Attempt Hugging Face
        if source_type == "hf":
            if load_dataset is None:
                raise ImportError(
                    "datasets library is required. Install with: pip install datasets"
                )
            print(
                f"Loading dataset '{dataset_name}' from Hugging Face ({path_or_hf_name}, split={split})..."
            )
            if data_dir and split:
                ds = load_dataset(path_or_hf_name, name=data_dir, split=split)
            else:
                ds = load_dataset(path_or_hf_name, split=split)

            raw = []
            for i, item in enumerate(ds):
                if dataset_name == "math1000" and i >= 1000:
                    break

                # Preserve every HLE row, including decoded and data-URI image
                # fields. Images are attached to the solution-generation call.
                if dataset_name == "hle":
                    raw_item = dict(item)
                    raw_item["id"] = item.get("id", i + 1)
                    raw_item["problem"] = item.get("question", "")
                    raw_item["question"] = item.get("question", "")
                    raw.append(raw_item)
                # For GPQA datasets, preserve all original fields
                elif dataset_name and dataset_name.startswith("gpqa"):
                    # Keep all original fields for GPQA (Question, Correct Answer, Incorrect Answer 1/2/3)
                    raw_item = dict(item)
                    raw_item["id"] = item.get("id", i + 1)
                    raw.append(raw_item)
                # For LiveCodeBench datasets, preserve all original fields
                elif dataset_name and dataset_name.startswith("livecodebench"):
                    # Keep all original fields for LiveCodeBench (question_content, test_cases, etc.)
                    raw_item = dict(item)
                    raw_item["id"] = item.get("question_id", item.get("id", i + 1))
                    raw.append(raw_item)
                else:
                    # Extract fields with fallbacks for different dataset formats
                    # Most datasets (AIME, MATH500, GSM8K): use "problem" field
                    # LiveMathBench: uses "question" field
                    problem_text = item.get("problem") or item.get("question", "")
                    solution = item.get("solution", "")
                    answer = item.get("answer", "")

                    # Special handling for GSM8K format (solution contains "#### answer")
                    if dataset_name == "math1000" and "####" in solution:
                        answer = solution.split("####")[-1].strip()

                    raw.append(
                        {
                            "id": item.get("id", i + 1),
                            "problem": problem_text,
                            "question": problem_text,
                            "solution": solution or answer or "",
                            "answer": answer,
                        }
                    )
            print(f"Loaded {len(raw)} problems from Hugging Face")
            # Skip normalization for GPQA and LiveCodeBench datasets to preserve original field structure
            if dataset_name and (
                dataset_name == "hle"
                or dataset_name.startswith("gpqa")
                or dataset_name.startswith("livecodebench")
            ):
                return raw
            return self._normalize_problems(raw, dataset_name)

        # Attempt CSV file
        if source_type == "csv":
            raw = self._load_csv_file(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from CSV for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        # Attempt JSON file
        if source_type == "json":
            raw = self._load_local_json(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from JSON for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        raise ValueError(f"Unknown source type: {source_type}")

    # ------------------------------------------------------------------
    # STEP 1: Insight extraction across multiple datasets
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self.client is None:
            self.client = _build_client(
                client_type=self.client_type,
                model_name=self.model_name,
                device=self.device,
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                output_dir=self.output_dir,
                load_in_8bit=self.load_in_8bit,
                api_model=self.api_model,
                reasoning_enabled=self.reasoning_enabled,
            )
            print(f"[Pipeline] Using client: {self.client_type} ({type(self.client).__name__})")

    def _clear_client_encyclopedia(self) -> None:
        """Clear loaded encyclopedia state on the reusable client."""
        self._ensure_client()
        if hasattr(self.client, "encyclopedia"):
            self.client.encyclopedia = ""
        if hasattr(self.client, "encyclopedia_dict"):
            self.client.encyclopedia_dict = {}
        if hasattr(self.client, "encyclopedia_loaded"):
            self.client.encyclopedia_loaded = False

    def _extract_insights_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> Tuple[str, List[Dict]]:
        """Extract insights from dataset, optionally solving with encyclopedia first.

        Args:
            dataset_name: Name of dataset
            problems: List of problems
            max_problems: Max problems to process
            encyclopedia_path: If provided, solve with encyclopedia before extracting insights
            iteration: Current iteration number (for logging)

        Returns:
            Tuple of (insights_dir, results_list)
        """
        self._ensure_client()
        insights_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(insights_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(
            f"\nIteration {iteration}: Extracting insights for {dataset_name} ({len(worklist)} problems)..."
        )

        # Load encyclopedias once at the start of this dataset iteration
        insights_section = ""
        if encyclopedia_paths:
            valid_eps = [ep for ep in encyclopedia_paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"  Loading {len(valid_eps)} encyclopedias for guidance...")
                self.client.load_encyclopedias(valid_eps, mode=self.mode)

                # Generate insights section once to reuse for all problems
                if self.client.encyclopedia_loaded:
                    if self.client.encyclopedia_dict:
                        # Text mode: Format from dictionary
                        insights_list = []
                        for (
                            insight_name,
                            insight_desc,
                        ) in self.client.encyclopedia_dict.items():
                            insights_list.append(f"**{insight_name}**:\n{insight_desc}")
                        insights_text = "\n\n".join(insights_list)
                    else:
                        # Normal mode: Use raw encyclopedia text
                        insights_text = self.client.encyclopedia

                    insights_section = f"""Available Insights to Guide Your Solution:

{insights_text}

---
INSTRUCTIONS: Review the insights above and actively apply the relevant techniques from insights to solve this problem. Consider which insights can help you approach the problem more effectively.

"""
            else:
                print("  No valid encyclopedias found; proceeding without guidance")
                self._clear_client_encyclopedia()
        else:
            self._clear_client_encyclopedia()

        results = []
        number_output_tokens_list = []
        loop_count_list = []
        indexed_problems = list(enumerate(worklist, 1))
        parallel_clients = self.use_api and self.num_workers > 1

        def solve_one(item):
            idx, problem_data = item
            problem_text, test_cases = self._format_problem(problem_data, dataset_name)
            if not problem_text:
                return problem_text, test_cases, None, None, "missing", os.getpid()
            image = None
            image_source = "none"
            if dataset_name == "hle":
                from hle_datasets.hle import extract_hle_image

                image, image_source = extract_hle_image(problem_data)
            task_client = self.client
            try:
                if image is not None and not isinstance(task_client, ChainOfThoughtReader):
                    raise ValueError(
                        "Multimodal HLE insight generation currently requires "
                        "--client default."
                    )
                result = task_client.solve_problem(
                    task=problem_text,
                    insights_section=insights_section,
                    **({"image": image} if image is not None else {}),
                )
                return problem_text, test_cases, result, None, image_source, os.getpid()
            except OpenRouterInFlightBudgetError:
                raise
            except Exception as exc:  # noqa: BLE001
                if is_openrouter_budget_error(exc):
                    raise OpenRouterInFlightBudgetError(
                        "OpenRouter rejected the benchmark request because the "
                        "account budget cannot admit it. Stopping this run "
                        "immediately; completed checkpoints remain resumable."
                    ) from exc
                return problem_text, test_cases, None, exc, image_source, os.getpid()
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass

        workers = self.num_workers if parallel_clients else 1
        if parallel_clients:
            print(
                f"  Launching {min(workers, len(indexed_problems))} independent "
                "API agent process(es) for insight extraction"
            )
            solve_outcomes = parallel_utils.process_map_ordered(
                _solve_benchmark_problem_process,
                indexed_problems,
                num_workers=workers,
                initializer=_initialize_benchmark_process,
                initargs=(
                    self._process_config(dataset_name, insights_section),
                ),
            )
        else:
            solve_outcomes = parallel_utils.parallel_map_ordered(
                solve_one, indexed_problems, num_workers=1
            )

        for (idx, problem_data), outcome in zip(indexed_problems, solve_outcomes):
            (
                problem_text,
                test_cases_for_eval,
                solved_result,
                solve_error,
                image_source,
                worker_pid,
            ) = outcome

            if not problem_text:
                print(f"  [skip] Problem {idx} missing text")
                continue

            print(
                f"  [{idx}/{len(worklist)}] [pid={worker_pid}] "
                f"{problem_text[:80]}..."
            )

            # Extract solution, reflection, and insights in one call
            predicted_answer = None
            is_correct = False
            try:
                if solve_error is not None:
                    raise solve_error
                result = solved_result or {}

                # Extract solution first
                solution = result.get("solution", "")

                # Extract output tokens from Step 1 (Solution Generation)
                number_output_tokens = 0
                token_info = result.get("token_info", {})
                number_output_tokens = token_info.get("output_tokens", 0)
                number_output_tokens_list.append(number_output_tokens)

                # Loop detection: count repeated consecutive sentences in Step 1 solution
                loop_count = self._count_consecutive_sentence_loops(solution)
                loop_count_list.append(loop_count)

                # Extract answer from solution using dataset-specific extractors
                predicted_answer = self._extract_answer_from_solution(
                    solution, dataset_name, problem_data
                )

                # Get extracted insights
                insight_book = result.get("insight_book", {})
                if not insight_book:
                    print("    No insights extracted")
                    continue

                # Filter out fallback insights
                insight_book = {
                    k: v
                    for k, v in insight_book.items()
                    if not k.startswith("insight_fallback")
                }
                if not insight_book:
                    print("    No insights extracted")
                    continue

                # Check answer correctness (pass problem_text for Gemini grading)
                # For code generation datasets, use test_cases; for others, use ground_truth
                if test_cases_for_eval:
                    # Code generation dataset (e.g., LiveCodeBench)
                    is_correct = self._check_answer_match(
                        solution, test_cases_for_eval, dataset_name, problem_text
                    )
                    status = "✓" if is_correct else "✗"
                    print(f"    {status} Code execution test results")
                else:
                    # Standard dataset
                    # For GPQA datasets, get ground truth from formatter
                    if dataset_name and dataset_name.startswith("gpqa"):
                        from science_datasets.gpqa import gpqa_formatter
                        _, ground_truth = gpqa_formatter(problem_data)
                    else:
                        ground_truth = problem_data.get("answer") or problem_data.get(
                            "solution", ""
                        )

                    if predicted_answer:
                        is_correct = self._check_answer_match(
                            predicted_answer, ground_truth, dataset_name, problem_text
                        )

                    status = "✓" if is_correct else "✗"
                    print(
                        f"    {status} Predicted: {predicted_answer if predicted_answer else 'N/A'} | GT: {ground_truth if ground_truth else 'N/A'}"
                    )

                # Save insights only
                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "insight_book": insight_book,
                    "iteration": iteration,
                    "is_correct": is_correct,
                    "number_output_tokens": number_output_tokens,
                    "loop_count": loop_count,
                    "multimodal_image_source": image_source,
                }

                output_path = os.path.join(insights_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                # Track for accuracy calculation
                results.append(
                    {
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                        "multimodal_image_source": image_source,
                    }
                )
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"    Error processing problem {idx}: {exc}")

        # Calculate and log average output tokens
        if number_output_tokens_list:
            avg_number_output_tokens = sum(number_output_tokens_list) / len(
                number_output_tokens_list
            )
            print(
                f"\n  Dataset '{dataset_name}' - Average Output Tokens: {avg_number_output_tokens:.1f}"
            )

        if loop_count_list:
            total_loop_count = sum(loop_count_list)
            print(f"  Dataset '{dataset_name}' - Total Loop Count: {total_loop_count}")

        return insights_dir, results

    def learn_insights_from_datasets(
        self,
        dataset_names: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        iteration: int = 0,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Learn insights from datasets, optionally solving with encyclopedia first.

        Returns:
            Tuple of (insights_map, accuracy_map) where accuracy_map has dataset -> accuracy
        """
        if not dataset_names:
            raise ValueError("Provide at least one dataset for STEP 1.")

        insights_map: Dict[str, str] = {}
        accuracy_map: Dict[str, float] = {}
        token_map: Dict[str, float] = {}
        loop_map: Dict[str, float] = {}

        # Helper to append per-dataset entry to iterative_summary.json immediately
        summary_file = os.path.join(self.output_dir, "iterative_summary.json")

        for name in dataset_names:
            problems = self.load_math_dataset(name)
            insights_dir, results = self._extract_insights_for_dataset(
                name, problems, max_problems, encyclopedia_paths, iteration
            )
            insights_map[name] = insights_dir

            # Calculate accuracy and average output tokens for this dataset
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                accuracy_map[name] = accuracy

                # Calculate average output tokens
                number_output_tokens_list = [
                    r.get("number_output_tokens", 0) for r in results
                ]
                if number_output_tokens_list:
                    avg_tokens = sum(number_output_tokens_list) / len(
                        number_output_tokens_list
                    )
                    token_map[name] = avg_tokens
                else:
                    token_map[name] = 0.0

                # Calculate total loop count
                loop_counts = [r.get("loop_count", 0) for r in results]
                loop_map[name] = sum(loop_counts) if loop_counts else 0.0
            else:
                accuracy_map[name] = 0.0
                token_map[name] = 0.0
                loop_map[name] = 0.0

            # Build per-question correctness list
            question_correctness = [1 if r["is_correct"] else 0 for r in results] if results else []

            # Append per-dataset summary entry immediately
            entry = {
                "iteration": iteration,
                "dataset": name,
                "accuracy": accuracy_map[name],
                "model": (
                    self.api_model
                    if self.use_api and self.api_model
                    else self.model_name
                ),
                "encyclopedia_used": [
                    ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)
                ],
                "average_output_tokens": token_map.get(name, 0.0),
                "total_loop_count": loop_map.get(name, 0.0),
                "question_correctness": question_correctness,
                "max_problems": max_problems,
            }
            try:
                if os.path.exists(summary_file):
                    with open(summary_file, "r", encoding="utf-8") as f:
                        current = json.load(f)
                else:
                    current = []
                if not isinstance(current, list):
                    current = []
                current.append(entry)
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(current, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Warning: failed to append iterative summary: {e}")

        print("\nFinished STEP 1 across datasets:")
        for name, path in insights_map.items():
            print(
                f"  - {name}: {path} (Accuracy: {accuracy_map[name]:.2%}, Avg Output Tokens: {token_map[name]:.1f})"
            )

        return insights_map, accuracy_map

    # ------------------------------------------------------------------
    # STEP 2: Aggregate chosen insights into one encyclopedia
    # ------------------------------------------------------------------
    def aggregate_insights(
        self,
        insight_sets: List[str],
        r1: float,
        r2: float,
        iteration: Optional[int] = None,
    ) -> Dict[str, str]:
        if not insight_sets:
            raise ValueError("Provide at least one dataset to aggregate in STEP 2.")
        # Build an encyclopedia per dataset folder
        print("\nAggregating insights per dataset:")
        per_dataset_encyclopedias: Dict[str, str] = {}
        for name in insight_sets:
            insights_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(insights_dir):
                raise FileNotFoundError(
                    f"Insights directory not found for {name}: {insights_dir}"
                )
            dataset_files = [
                os.path.join(insights_dir, f)
                for f in os.listdir(insights_dir)
                if f.endswith(".json") and f.startswith("problem_")
            ]
            dataset_files = sorted(dataset_files)
            if not dataset_files:
                print(f"  - {name}: no insight JSON files found; skipping")
                continue

            print(f"  - {name} ({len(dataset_files)} files)")
            if self.mode == "text":
                self.server_text = TextBasedInsightAggregationServer(
                    model_name=self.model_name,
                    device=self.device,
                    input_dirs=[self.output_dir],
                    use_api=self.use_api,
                    api_key=self.api_key,
                    api_provider=self.api_provider,
                    api_model=self.api_model,
                    reasoning_enabled=self.reasoning_enabled,
                )
                result = self.server_text.aggregate_and_build_encyclopedia(
                    json_files=dataset_files, output_dir=insights_dir
                )
                self.server_text.save_results(result, output_dir=insights_dir)
                if iteration is not None:
                    self.server_text.save_profiling(
                        result,
                        output_dir=insights_dir,
                        filename=f"profiling_iter_{iteration:02d}.json",
                    )
                encyclopedia_path = os.path.join(insights_dir, "encyclopedia.json")
            else:
                self.server = self._create_graph_server()
                result = self.server.aggregate_and_build_encyclopedia(
                    json_files=dataset_files, r1=r1, r2=r2, output_dir=insights_dir
                )
                self.server.save_results(result, output_dir=insights_dir)
                encyclopedia_path = os.path.join(insights_dir, "encyclopedia.txt")

            print(f"    Encyclopedia saved to {encyclopedia_path}")
            per_dataset_encyclopedias[name] = os.path.abspath(encyclopedia_path)

        return per_dataset_encyclopedias

    def _dataset_encyclopedia_path(self, dataset_name: str) -> str:
        if self.mode == "text":
            filename = "encyclopedia.json"
        else:
            filename = "encyclopedia.txt"
        return os.path.join(self.output_dir, dataset_name, filename)

    def _iteration_encyclopedia_path(self, iteration: int) -> str:
        extension = "json" if self.mode == "text" else "txt"
        return os.path.join(
            self.output_dir,
            f"encyclopedia_all_iter_{iteration:02d}.{extension}",
        )

    def _resumable_extraction_accuracy(
        self, iteration: int, dataset_list: List[str], max_problems: Optional[int]
    ) -> Optional[Dict[str, float]]:
        """Recover a completed Step 1 when aggregation failed afterward."""
        summary_path = Path(self.output_dir, "iterative_summary.json")
        if not summary_path.is_file():
            return None
        try:
            entries = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(entries, list):
            return None

        accuracy_map: Dict[str, float] = {}
        for dataset_name in dataset_list:
            matches = [
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("iteration") == iteration
                and entry.get("dataset") == dataset_name
                and entry.get("model")
                == (
                    self.api_model
                    if self.use_api and self.api_model
                    else self.model_name
                )
                and entry.get("max_problems", max_problems) == max_problems
            ]
            if not matches:
                return None

            insights_dir = Path(self.output_dir, dataset_name)
            problem_files = sorted(insights_dir.glob("problem_*.json"))
            has_current_iteration = False
            for problem_file in problem_files:
                try:
                    problem_result = json.loads(
                        problem_file.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError, TypeError):
                    continue
                if problem_result.get("iteration") == iteration:
                    has_current_iteration = True
                    break
            if not has_current_iteration:
                return None
            accuracy_map[dataset_name] = float(matches[-1].get("accuracy", 0.0))

        return accuracy_map

    def _find_existing_individual_encyclopedias(
        self, dataset_list: List[str]
    ) -> Dict[str, List[str]]:
        encyclopedia_map: Dict[str, List[str]] = {}
        for dataset_name in dataset_list:
            dataset_ency = self._dataset_encyclopedia_path(dataset_name)
            if os.path.exists(dataset_ency):
                encyclopedia_map[dataset_name] = [os.path.abspath(dataset_ency)]
        return encyclopedia_map

    def generate_combined_encyclopedia(
        self,
        dataset_list: List[str],
        r1: float = 0.95,
        r2: float = 0.4,
        iteration: Optional[int] = None,
    ) -> Optional[str]:
        """Generate a combined encyclopedia from all problem_*.json files across all datasets.

        This method collects all skills from all datasets' problem_*.json files and
        generates a single encyclopedia_all.json saved under output_dir.

        Args:
            dataset_list: List of dataset names to collect skills from
            r1: First threshold for insight aggregation (default: 0.95)
            r2: Second threshold for insight aggregation (default: 0.4)

        Returns:
            Path to the combined encyclopedia (encyclopedia_all.json) or None if failed
        """
        print("\n" + "=" * 80)
        print("Generating Combined Encyclopedia from All Datasets")
        print("=" * 80)

        # Collect all problem_*.json files from all datasets
        all_json_files = []
        for name in dataset_list:
            insights_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(insights_dir):
                print(f"  Warning: Directory not found for {name}: {insights_dir}")
                continue
            dataset_files = [
                os.path.join(insights_dir, f)
                for f in os.listdir(insights_dir)
                if f.endswith(".json") and f.startswith("problem_")
            ]
            dataset_files = sorted(dataset_files)
            if dataset_files:
                all_json_files.extend(dataset_files)
                print(f"  - {name}: {len(dataset_files)} problem files")

        if not all_json_files:
            print("  No problem files found across all datasets!")
            return None

        print(f"\nTotal problem files to aggregate: {len(all_json_files)}")

        # Generate combined encyclopedia using server_text.py
        if self.mode == "text":
            self.server_text = TextBasedInsightAggregationServer(
                model_name=self.model_name,
                device=self.device,
                input_dirs=[self.output_dir],
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                api_model=self.api_model,
                reasoning_enabled=self.reasoning_enabled,
            )
            result = self.server_text.aggregate_and_build_encyclopedia(
                json_files=all_json_files, output_dir=self.output_dir
            )
            # Save as encyclopedia_all.json under output_dir
            encyclopedia_all_path = os.path.join(self.output_dir, "encyclopedia_all.json")
            encyclopedia_dict = self.server_text._try_parse_json(self.server_text.encyclopedia)
            if encyclopedia_dict is None:
                json_content = self.server_text._extract_json_only(self.server_text.encyclopedia)
                encyclopedia_dict = self.server_text._try_parse_json(json_content)
            if encyclopedia_dict is None:
                error_msg = f"ERROR: Could not parse combined encyclopedia as JSON. Encyclopedia content: {self.server_text.encyclopedia[:500]}"
                print(error_msg)
                raise ValueError(error_msg)
            with open(encyclopedia_all_path, "w", encoding="utf-8") as f:
                json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
            if iteration is not None:
                iteration_encyclopedia_path = self._iteration_encyclopedia_path(
                    iteration
                )
                with open(
                    iteration_encyclopedia_path, "w", encoding="utf-8"
                ) as f:
                    json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
                print(
                    "Iteration encyclopedia snapshot saved to: "
                    f"{iteration_encyclopedia_path}"
                )
            self.server_text.save_profiling(
                result,
                output_dir=self.output_dir,
                filename="profiling_all.json",
            )
            if iteration is not None:
                self.server_text.save_profiling(
                    result,
                    output_dir=self.output_dir,
                    filename=f"profiling_all_iter_{iteration:02d}.json",
                )
        else:
            self.server = self._create_graph_server()
            result = self.server.aggregate_and_build_encyclopedia(
                json_files=all_json_files, r1=r1, r2=r2, output_dir=self.output_dir
            )
            # Save as encyclopedia_all.txt under output_dir
            encyclopedia_all_path = os.path.join(self.output_dir, "encyclopedia_all.txt")
            with open(encyclopedia_all_path, "w", encoding="utf-8") as f:
                f.write(self.server.encyclopedia)
            if iteration is not None:
                iteration_encyclopedia_path = self._iteration_encyclopedia_path(
                    iteration
                )
                with open(
                    iteration_encyclopedia_path, "w", encoding="utf-8"
                ) as f:
                    f.write(self.server.encyclopedia)
                print(
                    "Iteration encyclopedia snapshot saved to: "
                    f"{iteration_encyclopedia_path}"
                )

        print(f"\nCombined encyclopedia saved to: {encyclopedia_all_path}")
        return encyclopedia_all_path

    # ------------------------------------------------------------------
    # Eval-only mode: solve + check accuracy, no trace extraction or aggregation
    # ------------------------------------------------------------------
    def _format_problem(self, problem_data: Dict, dataset_name: str):
        """Format a problem for the given dataset. Returns (problem_text, test_cases_for_eval)."""
        problem_text = None
        test_cases_for_eval = None
        if dataset_name and dataset_name.startswith("aime"):
            from math_datasets.aime25 import aime25_formatter
            problem_text, _ = aime25_formatter(problem_data)
        elif dataset_name and "livemathbench" in dataset_name:
            from math_datasets.livemathbench import livemathbench_formatter
            problem_text, _ = livemathbench_formatter(problem_data, dataset_name)
        elif dataset_name and dataset_name.startswith("imo"):
            from math_datasets.imo_benchmark import imo_formatter
            problem_text, _ = imo_formatter(problem_data, dataset_name)
        elif dataset_name == "math500":
            from math_datasets.math500 import math500_formatter
            problem_text, _ = math500_formatter(problem_data)
        elif dataset_name == "gsm8k":
            from math_datasets.gsm8k import gsm8k_formatter
            problem_text, _ = gsm8k_formatter(problem_data)
        elif dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_formatter
            problem_text, _ = gpqa_formatter(problem_data)
        elif dataset_name and "livecodebench" in dataset_name:
            from code_datasets.livecodebench import livecodebench_formatter
            problem_text, test_cases_for_eval = livecodebench_formatter(problem_data)
        else:
            problem_text = problem_data.get("problem") or problem_data.get("question", "")
        return problem_text, test_cases_for_eval

    def _get_ground_truth(self, problem_data: Dict, dataset_name: str) -> str:
        """Get ground truth answer for a problem."""
        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_formatter
            _, ground_truth = gpqa_formatter(problem_data)
            return ground_truth
        if dataset_name == "hle":
            from hle_datasets.hle import get_hle_answer

            return get_hle_answer(problem_data)
        return problem_data.get("answer") or problem_data.get("solution", "")

    def run_eval_only(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        encyclopedia_paths: Optional[List[str]] = None,
        encyclopedia_map: Optional[Dict[str, List[str]]] = None,
        problem_overrides: Optional[Dict[str, List[Dict]]] = None,
        output_subdir: str = "eval_only",
        summary_name: str = "eval_only_summary.json",
        summary_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Eval-only mode: solve problems and check accuracy.

        Uses existing encyclopedia (if provided) to guide solutions,
        but does NOT extract traces or aggregate insights.

        Args:
            dataset_list: List of datasets to evaluate
            max_problems: Max problems per dataset
            encyclopedia_paths: Paths to encyclopedia files to use for guidance

        Returns:
            Summary dict with accuracy per dataset
        """
        self._ensure_client()

        def build_insights_section(paths: Optional[List[str]]) -> str:
            insights_section = ""
            if not paths:
                self._clear_client_encyclopedia()
                return insights_section
            valid_eps = [ep for ep in paths if ep and os.path.exists(ep)]
            if valid_eps:
                print(f"Loading {len(valid_eps)} encyclopedias for guidance...")
                self.client.load_encyclopedias(valid_eps, mode=self.mode)

                if self.client.encyclopedia_loaded:
                    if self.client.encyclopedia_dict:
                        insights_list = []
                        for insight_name, insight_desc in self.client.encyclopedia_dict.items():
                            insights_list.append(f"**{insight_name}**:\n{insight_desc}")
                        insights_text = "\n\n".join(insights_list)
                    else:
                        insights_text = self.client.encyclopedia

                    insights_section = f"""Available Insights to Guide Your Solution:

{insights_text}

---
INSTRUCTIONS: Review the insights above and actively apply the relevant techniques from insights to solve this problem. Consider which insights can help you approach the problem more effectively.

"""
            else:
                self._clear_client_encyclopedia()
            return insights_section

        # Build shared insights_section once unless an individual per-dataset map is provided.
        shared_insights_section = ""
        if encyclopedia_map:
            print("Using individual per-dataset encyclopedias for guidance")
        elif encyclopedia_paths:
            shared_insights_section = build_insights_section(encyclopedia_paths)
        else:
            self._clear_client_encyclopedia()
            print("No valid encyclopedias found; proceeding without guidance")

        print(f"\n{'='*80}")
        print("EVAL-ONLY MODE: Solve + Accuracy Check (no trace extraction / aggregation)")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(
            "Encyclopedia: "
            f"{'individual' if encyclopedia_map else ('yes' if shared_insights_section else 'none')}"
        )
        print(f"{'='*80}\n")

        accuracy_map = {}
        token_map = {}
        loop_map = {}

        for dataset_name in dataset_list:
            if encyclopedia_map:
                dataset_paths = encyclopedia_map.get(dataset_name, [])
                insights_section = build_insights_section(dataset_paths)
                if not insights_section:
                    print(f"No valid encyclopedia found for {dataset_name}; evaluating without guidance")
            else:
                insights_section = shared_insights_section

            if problem_overrides and dataset_name in problem_overrides:
                worklist = problem_overrides[dataset_name]
            else:
                problems = self.load_math_dataset(dataset_name)
                worklist = problems[:max_problems] if max_problems else problems
            print(f"\nEvaluating {dataset_name} ({len(worklist)} problems)...")

            eval_dir = os.path.join(self.output_dir, dataset_name, output_subdir)
            os.makedirs(eval_dir, exist_ok=True)

            indexed_problems = list(enumerate(worklist, 1))
            parallel_clients = self.use_api and self.num_workers > 1
            # OpenRouter performs admission control against the requested
            # completion ceiling, not the tokens ultimately consumed. A 32K
            # ceiling made short-answer GPQA/AIME requests fail with HTTP 402
            # even in a one-worker run. Their full reasoning plus final answer
            # fits comfortably in 4K; retain 16K only for code generation.
            eval_max_output_tokens = (
                4096 if dataset_name in {"aime26", "gpqa_diamond"} else 16384
            )
            execution_backend = (
                f"spawn multiprocessing ({min(self.num_workers, len(indexed_problems))} "
                "processes)"
                if parallel_clients and indexed_problems
                else "single process"
            )
            print(
                f"  Execution backend: {execution_backend}; "
                f"evaluation max_tokens={eval_max_output_tokens}"
            )

            def completed_outcome(item):
                """Return a valid per-problem checkpoint, or None."""
                idx, problem_data = item
                output_path = os.path.join(eval_dir, f"problem_{idx:04d}.json")
                if not os.path.isfile(output_path):
                    return None
                try:
                    with open(output_path, "r", encoding="utf-8") as handle:
                        output_data = json.load(handle)
                except (OSError, ValueError, TypeError):
                    return None
                expected_id = problem_data.get("id", idx)
                if not isinstance(output_data, dict):
                    return None
                if str(output_data.get("problem_id")) != str(expected_id):
                    return None
                required = {
                    "solution",
                    "is_correct",
                    "number_output_tokens",
                    "loop_count",
                }
                if not required.issubset(output_data):
                    return None
                return {
                    "index": idx,
                    "output_data": output_data,
                    "is_correct": bool(output_data["is_correct"]),
                    "number_output_tokens": int(
                        output_data.get("number_output_tokens", 0) or 0
                    ),
                    "loop_count": int(output_data.get("loop_count", 0) or 0),
                    "cached": True,
                }

            def evaluate_one(item):
                idx, problem_data = item
                problem_text, test_cases_for_eval = self._format_problem(problem_data, dataset_name)
                if not problem_text:
                    return {"index": idx, "skip": True}
                image = None
                image_source = "none"
                if dataset_name == "hle":
                    from hle_datasets.hle import extract_hle_image

                    image, image_source = extract_hle_image(problem_data)
                task_client = self.client
                try:
                    if image is not None and not isinstance(task_client, ChainOfThoughtReader):
                        raise ValueError(
                            "Multimodal HLE evaluation currently requires --client default."
                        )
                    prompt = task_client._get_solution_prompt(
                        problem_text, insights_section=insights_section
                    )
                    if image is not None:
                        prompt = (
                            "An image is attached to this problem. Use both the image "
                            "and the question text; do not ignore visual evidence.\n\n"
                            + prompt
                        )
                    response, token_info = task_client._call_model(
                        prompt,
                        None,
                        max_new_tokens=eval_max_output_tokens,
                        **({"image": image} if image is not None else {}),
                    )
                    solution = response
                    number_output_tokens = token_info.get("output_tokens", 0)
                    number_reasoning_tokens = token_info.get("reasoning_tokens", 0)
                    loop_count = self._count_consecutive_sentence_loops(solution)
                    predicted_answer = self._extract_answer_from_solution(
                        solution, dataset_name, problem_data
                    )
                    if test_cases_for_eval:
                        is_correct = self._check_answer_match(
                            solution, test_cases_for_eval, dataset_name, problem_text
                        )
                        ground_truth = None
                    else:
                        ground_truth = self._get_ground_truth(problem_data, dataset_name)
                        is_correct = False
                        if predicted_answer:
                            is_correct = self._check_answer_match(
                                predicted_answer, ground_truth, dataset_name, problem_text
                            )
                    output_data = {
                        "problem": problem_text,
                        "problem_id": problem_data.get("id", idx),
                        "solution": solution,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "number_reasoning_tokens": number_reasoning_tokens,
                        "loop_count": loop_count,
                        "multimodal_image_source": image_source,
                        "worker_pid": os.getpid(),
                    }
                    output_path = os.path.join(
                        eval_dir, f"problem_{idx:04d}.json"
                    )
                    temporary_path = f"{output_path}.tmp.{os.getpid()}"
                    with open(temporary_path, "w", encoding="utf-8") as handle:
                        json.dump(output_data, handle, indent=2, ensure_ascii=False)
                    os.replace(temporary_path, output_path)
                    return {
                        "index": idx,
                        "output_data": output_data,
                        "ground_truth": ground_truth,
                        "test_cases": bool(test_cases_for_eval),
                        "is_correct": is_correct,
                        "number_output_tokens": number_output_tokens,
                        "loop_count": loop_count,
                        "worker_pid": os.getpid(),
                    }
                except OpenRouterInFlightBudgetError:
                    # This is a shared account admission failure, not a bad
                    # answer for one problem. Let the parallel executor cancel
                    # queued work instead of issuing the same doomed request
                    # for every remaining benchmark item.
                    raise
                except Exception as exc:
                    if is_openrouter_budget_error(exc):
                        raise OpenRouterInFlightBudgetError(
                            "OpenRouter rejected the benchmark request because "
                            "the account budget cannot admit it. Stopping this "
                            "run immediately; completed checkpoints remain "
                            "resumable."
                        ) from exc
                    return {
                        "index": idx,
                        "error": str(exc),
                        "worker_pid": os.getpid(),
                    }
                finally:
                    if image is not None:
                        try:
                            image.close()
                        except Exception:
                            pass

            workers = self.num_workers if parallel_clients else 1
            cached_outcomes = {}
            pending_problems = []
            for item in indexed_problems:
                checkpoint = completed_outcome(item)
                if checkpoint is None:
                    pending_problems.append(item)
                else:
                    cached_outcomes[item[0]] = checkpoint
            if cached_outcomes:
                print(
                    f"  Resuming {len(cached_outcomes)} completed problem(s); "
                    f"calling the API for {len(pending_problems)} remaining problem(s)"
                )
            if parallel_clients and pending_problems:
                print(
                    f"  Launching {min(workers, len(pending_problems))} independent "
                    "API agent process(es)"
                )
                fresh_outcomes = parallel_utils.process_map_ordered(
                    _evaluate_benchmark_problem_process,
                    pending_problems,
                    num_workers=workers,
                    initializer=_initialize_benchmark_process,
                    initargs=(
                        self._process_config(
                            dataset_name,
                            insights_section,
                            eval_dir=eval_dir,
                            eval_max_output_tokens=eval_max_output_tokens,
                        ),
                    ),
                )
            else:
                fresh_outcomes = parallel_utils.parallel_map_ordered(
                    evaluate_one, pending_problems, num_workers=1
                )
            fresh_by_index = {
                outcome["index"]: outcome for outcome in fresh_outcomes
            }
            outcomes = [
                cached_outcomes.get(idx) or fresh_by_index[idx]
                for idx, _ in indexed_problems
            ]
            results = []
            evaluation_errors = []
            for outcome in outcomes:
                idx = outcome["index"]
                if outcome.get("skip"):
                    print(f"  [skip] Problem {idx} missing text")
                    continue
                if outcome.get("error"):
                    print(f"    Error processing problem {idx}: {outcome['error']}")
                    evaluation_errors.append((idx, outcome["error"]))
                    continue
                output_data = outcome["output_data"]
                if outcome.get("cached"):
                    print(
                        f"  [{idx}/{len(worklist)}] [resume] "
                        f"{output_data.get('problem', '')[:80]}..."
                    )
                else:
                    print(
                        f"  [{idx}/{len(worklist)}] "
                        f"[pid={outcome.get('worker_pid', os.getpid())}] "
                        f"{output_data['problem'][:80]}..."
                    )
                    status = "+" if outcome["is_correct"] else "x"
                    if outcome["test_cases"]:
                        print(f"    {status} Code execution test results")
                    else:
                        print(
                            f"    {status} Predicted: {output_data['predicted_answer'] or 'N/A'} "
                            f"| GT: {outcome['ground_truth'] or 'N/A'}"
                        )
                results.append({
                    "is_correct": outcome["is_correct"],
                    "number_output_tokens": outcome["number_output_tokens"],
                    "loop_count": outcome["loop_count"],
                })

            if evaluation_errors:
                failed_indices = ", ".join(
                    str(idx) for idx, _ in evaluation_errors[:20]
                )
                raise RuntimeError(
                    f"{len(evaluation_errors)} problem(s) failed in {dataset_name} "
                    f"(indices: {failed_indices}). Successful problem files were "
                    "saved. Rerun the same command to evaluate only missing/failed "
                    "problems; no completion summary was written."
                )

            # Summarize
            if results:
                num_correct = sum(1 for r in results if r["is_correct"])
                accuracy = num_correct / len(results)
                avg_tokens = sum(r["number_output_tokens"] for r in results) / len(results)
                total_loops = sum(r["loop_count"] for r in results)
            else:
                accuracy = 0.0
                avg_tokens = 0.0
                total_loops = 0

            accuracy_map[dataset_name] = accuracy
            token_map[dataset_name] = avg_tokens
            loop_map[dataset_name] = total_loops

            print(f"\n  {dataset_name}: Accuracy={accuracy:.2%}, Avg Tokens={avg_tokens:.1f}, Loops={total_loops}")

        # Save summary
        summary = {
            "mode": "eval_only",
            "datasets": dataset_list,
            "accuracy_per_dataset": accuracy_map,
            "avg_tokens_per_dataset": token_map,
            "loop_count_per_dataset": loop_map,
            "encyclopedia_used": [ep for ep in (encyclopedia_paths or []) if ep and os.path.exists(ep)],
            "individual_encyclopedia_used": {
                ds: [ep for ep in paths if ep and os.path.exists(ep)]
                for ds, paths in (encyclopedia_map or {}).items()
            },
        }
        if summary_metadata:
            summary.update(summary_metadata)

        summary_path = os.path.join(self.output_dir, summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("EVAL-ONLY COMPLETE")
        print(f"{'='*80}")
        for dataset, acc in accuracy_map.items():
            print(f"  {dataset}: {acc:.2%}")
        print(f"\nSummary saved: {summary_path}")

        return summary

    def run_cross_dataset_pipeline(
        self,
        train_datasets: List[str],
        eval_datasets: List[str],
        train_max_problems: Optional[int],
        eval_max_problems: Optional[int] = None,
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step: int = 1,
    ) -> Dict:
        """Run an unguided iteration 0, train insights, then evaluate held-out data."""
        if not train_datasets:
            raise ValueError("Provide at least one training dataset")
        if not eval_datasets:
            raise ValueError("Provide at least one held-out evaluation dataset")
        if self.num_iterations < 1:
            raise ValueError(
                "Cross-dataset training requires --num-iterations of at least 1; "
                "iteration 0 is added automatically as the unguided baseline."
            )
        overlap = sorted(set(train_datasets).intersection(eval_datasets))
        if overlap:
            raise ValueError(
                "Training and held-out datasets must be disjoint; overlap: "
                + ", ".join(overlap)
            )

        start_time = time.time()
        baseline_datasets = list(dict.fromkeys(train_datasets + eval_datasets))
        baseline_summary_path = Path(
            self.output_dir, "iteration_0_baseline_summary.json"
        )
        baseline_summary = None
        if baseline_summary_path.is_file():
            try:
                candidate = json.loads(
                    baseline_summary_path.read_text(encoding="utf-8")
                )
                if (
                    isinstance(candidate, dict)
                    and candidate.get("iteration") == 0
                    and candidate.get("phase") == "unguided_baseline"
                    and candidate.get("datasets") == baseline_datasets
                ):
                    baseline_summary = candidate
            except (OSError, ValueError, TypeError):
                baseline_summary = None

        if baseline_summary is not None:
            print(
                "Iteration 0 summary already exists and matches this protocol; "
                "skipping the unguided baseline"
            )
        else:
            baseline_problems: Dict[str, List[Dict]] = {}
            for dataset_name in baseline_datasets:
                problems = self.load_math_dataset(dataset_name)
                limit = (
                    train_max_problems
                    if dataset_name in train_datasets
                    else eval_max_problems
                )
                baseline_problems[dataset_name] = (
                    problems[:limit] if limit is not None else problems
                )

            print("\n" + "=" * 80)
            print("ITERATION 0: UNGUIDED BASELINE (NO TRAINING / NO INSIGHT LIBRARY)")
            print("=" * 80)
            baseline_summary = self.run_eval_only(
                dataset_list=baseline_datasets,
                max_problems=None,
                encyclopedia_paths=None,
                problem_overrides=baseline_problems,
                output_subdir="iteration_0_baseline",
                summary_name="iteration_0_baseline_summary.json",
                summary_metadata={
                    "iteration": 0,
                    "phase": "unguided_baseline",
                    "training_performed": False,
                    "insight_library_used": False,
                },
            )

        print("\n" + "=" * 80)
        print("ITERATION 1+: INSIGHT TRAINING")
        print("=" * 80)
        print(f"Training insight library on: {', '.join(train_datasets)}")
        print(f"Held-out evaluation datasets: {', '.join(eval_datasets)}")
        train_summary = self.run_iterative_pipeline(
            dataset_list=train_datasets,
            max_problems=train_max_problems,
            r1=r1,
            r2=r2,
            start_from_step=start_from_step,
            individual=False,
        )

        encyclopedia_name = (
            "encyclopedia_all.json" if self.mode == "text" else "encyclopedia_all.txt"
        )
        encyclopedia_path = str(
            Path(self.output_dir, encyclopedia_name).resolve()
        )
        if not os.path.isfile(encyclopedia_path):
            raise FileNotFoundError(
                "Training completed without the expected combined encyclopedia: "
                f"{encyclopedia_path}"
            )

        iteration_encyclopedias = []
        for item in train_summary.get("iteration_history", []):
            path = item.get("iteration_encyclopedia")
            if path and os.path.isfile(path):
                iteration_encyclopedias.append(os.path.abspath(path))
        if len(iteration_encyclopedias) != self.num_iterations:
            raise FileNotFoundError(
                "Expected one saved insight library per training iteration, but found "
                f"{len(iteration_encyclopedias)} of {self.num_iterations}."
            )

        heldout_eval_by_iteration = []
        for iteration, iteration_encyclopedia in enumerate(
            iteration_encyclopedias, 1
        ):
            iteration_summary_name = (
                f"heldout_eval_iter_{iteration:02d}_summary.json"
            )
            iteration_summary_path = Path(
                self.output_dir, iteration_summary_name
            )
            iteration_eval_summary = None
            if iteration_summary_path.is_file():
                try:
                    candidate = json.loads(
                        iteration_summary_path.read_text(encoding="utf-8")
                    )
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("iteration") == iteration
                        and candidate.get("phase")
                        == "heldout_iteration_matched"
                        and candidate.get("datasets") == eval_datasets
                        and os.path.abspath(
                            candidate.get("iteration_encyclopedia", "")
                        )
                        == os.path.abspath(iteration_encyclopedia)
                    ):
                        iteration_eval_summary = candidate
                except (OSError, ValueError, TypeError):
                    iteration_eval_summary = None

            if iteration_eval_summary is not None:
                print(
                    f"Held-out iteration {iteration} summary already exists; "
                    "skipping this completed evaluation"
                )
            else:
                print("\n" + "=" * 80)
                print(
                    f"HELD-OUT EVALUATION ITERATION {iteration}/{self.num_iterations}: "
                    "ONE ITERATION-MATCHED LIBRARY"
                )
                print("=" * 80)
                iteration_eval_summary = self.run_eval_only(
                    dataset_list=eval_datasets,
                    max_problems=eval_max_problems,
                    encyclopedia_paths=[iteration_encyclopedia],
                    output_subdir=f"heldout_eval_iter_{iteration:02d}",
                    summary_name=iteration_summary_name,
                    summary_metadata={
                        "iteration": iteration,
                        "phase": "heldout_iteration_matched",
                        "iteration_encyclopedia": iteration_encyclopedia,
                    },
                )
            heldout_eval_by_iteration.append(iteration_eval_summary)

        eval_summary = heldout_eval_by_iteration[-1]

        summary = {
            "mode": "cross_dataset_train_eval",
            "train_datasets": train_datasets,
            "eval_datasets": eval_datasets,
            "train_max_problems": train_max_problems,
            "eval_max_problems": eval_max_problems,
            "iteration_0_baseline": baseline_summary,
            "encyclopedia": encyclopedia_path,
            "iteration_encyclopedias": iteration_encyclopedias,
            "heldout_evaluation_count": len(heldout_eval_by_iteration),
            "heldout_eval_by_iteration": heldout_eval_by_iteration,
            "train_summary": train_summary,
            "eval_summary": eval_summary,
            "total_time_seconds": time.time() - start_time,
        }
        summary_path = os.path.join(
            self.output_dir, "cross_dataset_train_eval_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Cross-dataset summary saved: {summary_path}")
        return summary

    def run_split_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        split: float,
        seed: int,
        r1: float = 0.95,
        r2: float = 0.4,
        individual: bool = False,
    ) -> Dict:
        """Train an insight library on a random split, then eval on held-out problems.

        The training split runs the normal Step 1/2/3 flow. The evaluation split
        only solves with the generated library and computes accuracy/performance.
        """
        if not 0.0 < split < 1.0:
            raise ValueError("--split must be a float strictly between 0 and 1")
        if not dataset_list:
            raise ValueError("Provide at least one dataset for split mode")

        base_output_dir = self.output_dir
        train_output_dir = os.path.join(base_output_dir, "split_train")
        eval_output_dir = os.path.join(base_output_dir, "split_eval")

        split_manifest = {
            "mode": "split_individual" if individual else "split",
            "split": split,
            "seed": seed,
            "max_problems": max_problems,
            "individual": individual,
            "train_output_dir": train_output_dir,
            "eval_output_dir": eval_output_dir,
            "datasets": {},
        }
        train_problem_map: Dict[str, List[Dict]] = {}
        eval_problem_map: Dict[str, List[Dict]] = {}

        for dataset_index, dataset_name in enumerate(dataset_list):
            problems = self.load_math_dataset(dataset_name)
            worklist = problems[:max_problems] if max_problems else problems
            indices = list(range(len(worklist)))
            rng = random.Random(seed + dataset_index)
            rng.shuffle(indices)
            train_size = int(len(indices) * split)
            if len(indices) > 1:
                train_size = max(1, min(train_size, len(indices) - 1))
            train_indices = indices[:train_size]
            eval_indices = indices[train_size:]
            train_problem_map[dataset_name] = [worklist[i] for i in train_indices]
            eval_problem_map[dataset_name] = [worklist[i] for i in eval_indices]
            split_manifest["datasets"][dataset_name] = {
                "total": len(worklist),
                "train": len(train_indices),
                "eval": len(eval_indices),
                "train_indices": train_indices,
                "eval_indices": eval_indices,
                "train_ids": [worklist[i].get("id", i + 1) for i in train_indices],
                "eval_ids": [worklist[i].get("id", i + 1) for i in eval_indices],
            }

        os.makedirs(base_output_dir, exist_ok=True)
        split_path = os.path.join(base_output_dir, "split_manifest.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2, ensure_ascii=False)
        print(f"Split manifest saved: {split_path}")

        orig_output_dir = self.output_dir
        iteration_history = []
        encyclopedia_paths: Optional[List[str]] = None
        encyclopedia_map: Dict[str, List[str]] = {}
        combined_ency_path = None
        per_dataset_ency_paths: Dict[str, str] = {}
        try:
            for iteration in range(1, self.num_iterations + 1):
                iter_train_dir = os.path.join(train_output_dir, f"iter_{iteration:02d}")
                iter_eval_dir = os.path.join(eval_output_dir, f"iter_{iteration:02d}")

                # --- TRAIN: extract insights on training split ---
                self.output_dir = iter_train_dir
                os.makedirs(self.output_dir, exist_ok=True)
                print("\n" + "=" * 80)
                print(
                    f"SPLIT ITERATION {iteration}/{self.num_iterations} "
                    f"TRAIN: Step 1/2/3 on {split:.0%} split"
                )
                print("=" * 80)
                train_results_map: Dict[str, List[Dict]] = {}
                for dataset_name in dataset_list:
                    dataset_encyclopedia_paths = (
                        encyclopedia_map.get(dataset_name) if individual else encyclopedia_paths
                    )
                    _, train_results = self._extract_insights_for_dataset(
                        dataset_name=dataset_name,
                        problems=train_problem_map[dataset_name],
                        max_problems=None,
                        encyclopedia_paths=dataset_encyclopedia_paths,
                        iteration=iteration,
                    )
                    train_results_map[dataset_name] = train_results

                train_accuracy_map = {
                    ds: (sum(1 for r in res if r["is_correct"]) / len(res) if res else 0.0)
                    for ds, res in train_results_map.items()
                }
                print(f"\nIteration {iteration} TRAIN accuracy:")
                for ds, acc in train_accuracy_map.items():
                    print(f"  - {ds}: {acc:.2%}")

                if individual:
                    print(
                        f"\nIteration {iteration}: Generating individual encyclopedias from training split..."
                    )
                    per_dataset_ency_paths = self.aggregate_insights(
                        dataset_list, r1=r1, r2=r2, iteration=iteration
                    )
                    encyclopedia_map = {
                        dataset: [path] for dataset, path in per_dataset_ency_paths.items()
                    }
                    combined_ency_path = None
                    encyclopedia_paths = None
                else:
                    print(f"\nIteration {iteration}: Generating combined encyclopedia from training split...")
                    combined_ency_path = self.generate_combined_encyclopedia(
                        dataset_list, r1=r1, r2=r2, iteration=iteration
                    )
                    encyclopedia_paths = [combined_ency_path] if combined_ency_path else []

                # --- EVAL: eval-only on held-out split ---
                self.output_dir = iter_eval_dir
                os.makedirs(self.output_dir, exist_ok=True)
                print("\n" + "=" * 80)
                print(f"SPLIT ITERATION {iteration}/{self.num_iterations} EVAL: held-out split")
                print("=" * 80)
                eval_summary = self.run_eval_only(
                    dataset_list=dataset_list,
                    max_problems=None,
                    encyclopedia_paths=encyclopedia_paths,
                    encyclopedia_map=encyclopedia_map if individual else None,
                    problem_overrides=eval_problem_map,
                    output_subdir="problems",
                    summary_name="split_eval_summary.json",
                )

                iteration_history.append({
                    "iteration": iteration,
                    "train_accuracy": train_accuracy_map,
                    "eval_summary": eval_summary,
                    "combined_encyclopedia": combined_ency_path,
                    "per_dataset_encyclopedias": per_dataset_ency_paths,
                })

                print(f"\nIteration {iteration} EVAL accuracy:")
                for ds, acc in eval_summary.get("accuracy_per_dataset", {}).items():
                    print(f"  - {ds}: {acc:.2%}")
        finally:
            self.output_dir = orig_output_dir

        summary = {
            "mode": "split_individual" if individual else "split",
            "split": split,
            "seed": seed,
            "num_iterations": self.num_iterations,
            "individual": individual,
            "datasets": dataset_list,
            "split_manifest": split_manifest,
            "combined_encyclopedia": combined_ency_path,
            "per_dataset_encyclopedias": per_dataset_ency_paths,
            "iteration_history": iteration_history,
        }
        summary_path = os.path.join(base_output_dir, "split_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSplit summary saved: {summary_path}")
        return summary

    # ------------------------------------------------------------------
    # Iterative Learning Pipeline
    # ------------------------------------------------------------------
    def run_iterative_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step: int = 1,
        individual: bool = False,
    ) -> Dict:
        """Run iterative learning pipeline.

        Each iteration:
        1. Use encyclopedia (from previous iteration) to solve problems and log accuracy
        2. Extract insights from the same problems
        3. Aggregate insights into new encyclopedia
        4. Repeat

        Args:
            dataset_list: List of datasets to train on
            max_problems: Max problems per dataset
            r1: Similarity threshold for aggregation
            r2: Similarity threshold for aggregation
            start_from_step: Start from step 1 (extract) or 2 (aggregate only)

        Returns:
            Summary dict with iteration history
        """
        if not dataset_list:
            raise ValueError("Provide at least one dataset for iterative learning")

        start_time = time.time()
        iteration_history = []

        encyclopedia_paths: Optional[List[str]] = None
        encyclopedia_map: Dict[str, List[str]] = {}
        combined_ency_path: Optional[str] = None
        per_dataset_ency_paths: Dict[str, str] = {}

        if individual:
            encyclopedia_map = self._find_existing_individual_encyclopedias(dataset_list)
            if encyclopedia_map:
                print(f"Found {len(encyclopedia_map)} individual per-dataset encyclopedias:")
                for dataset_name, paths in encyclopedia_map.items():
                    print(f"  - {dataset_name}: {paths[0]}")
        else:
            # Check if combined encyclopedia already exists (from previous runs)
            # Use encyclopedia_all.json/txt instead of per-dataset encyclopedias
            if self.mode == "text":
                combined_ency_path = os.path.join(self.output_dir, "encyclopedia_all.json")
            else:
                combined_ency_path = os.path.join(self.output_dir, "encyclopedia_all.txt")

            if os.path.exists(combined_ency_path):
                encyclopedia_paths = [combined_ency_path]
                print(f"Found existing combined encyclopedia: {combined_ency_path}")
            else:
                # Fallback: check for per-dataset encyclopedias
                per_dataset_encyclopedias = []
                for dataset_name in dataset_list:
                    dataset_ency = self._dataset_encyclopedia_path(dataset_name)
                    if os.path.exists(dataset_ency):
                        per_dataset_encyclopedias.append(dataset_ency)

                if per_dataset_encyclopedias:
                    encyclopedia_paths = per_dataset_encyclopedias
                    print(f"Found {len(per_dataset_encyclopedias)} per-dataset encyclopedias:")
                    for ep in per_dataset_encyclopedias:
                        print(f"  - {ep}")

        print(f"\n{'='*80}")
        print(f"Starting Iterative Learning Pipeline: {self.num_iterations} iterations")
        print(f"Individual per-dataset mode: {'yes' if individual else 'no'}")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        if individual and encyclopedia_map:
            print(
                f"Using {len(encyclopedia_map)} individual encyclopedias for iteration 1"
            )
        elif encyclopedia_paths:
            print(
                f"Using {len(encyclopedia_paths)} existing encyclopedias for iteration 1"
            )
        print(f"{'='*80}\n")

        for iteration in range(1, self.num_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{self.num_iterations}")
            print(f"{'='*80}")

            if not individual:
                completed_snapshot = self._iteration_encyclopedia_path(iteration)
                if os.path.isfile(completed_snapshot):
                    completed_snapshot = os.path.abspath(completed_snapshot)
                    encyclopedia_paths = [completed_snapshot]
                    combined_ency_path = completed_snapshot
                    resumed_accuracy = self._resumable_extraction_accuracy(
                        iteration, dataset_list, max_problems
                    ) or {name: 0.0 for name in dataset_list}
                    iteration_history.append(
                        {
                            "iteration": iteration,
                            "datasets": dataset_list,
                            "accuracy_per_dataset": resumed_accuracy,
                            "combined_encyclopedia": completed_snapshot,
                            "iteration_encyclopedia": completed_snapshot,
                            "per_dataset_encyclopedias": {},
                            "resumed": True,
                        }
                    )
                    print(
                        f"Iteration {iteration} library already exists; "
                        "skipping completed extraction and aggregation"
                    )
                    continue

            # STEP 1: Extract insights (and solve if encyclopedia exists)
            if start_from_step == 1:
                resumed_accuracy = (
                    None
                    if individual
                    else self._resumable_extraction_accuracy(
                        iteration, dataset_list, max_problems
                    )
                )
                if resumed_accuracy is not None:
                    accuracy_map = resumed_accuracy
                    insights_map = {
                        name: os.path.join(self.output_dir, name)
                        for name in dataset_list
                    }
                    print(
                        f"Iteration {iteration} extraction is complete; "
                        "resuming directly at aggregation/profiling"
                    )
                elif individual:
                    insights_map: Dict[str, str] = {}
                    accuracy_map: Dict[str, float] = {}
                    for dataset_name in dataset_list:
                        dataset_paths = encyclopedia_map.get(dataset_name)
                        dataset_insights, dataset_accuracy = self.learn_insights_from_datasets(
                            [dataset_name], max_problems, dataset_paths, iteration
                        )
                        insights_map.update(dataset_insights)
                        accuracy_map.update(dataset_accuracy)
                else:
                    insights_map, accuracy_map = self.learn_insights_from_datasets(
                        dataset_list, max_problems, encyclopedia_paths, iteration
                    )
            else:
                # Starting from step 2: check if insights exist from previous run
                print(
                    f"\nSkipping Step 1 (insight extraction) - assuming insights already exist"
                )
                insights_exist = all(
                    os.path.isdir(os.path.join(self.output_dir, name))
                    for name in dataset_list
                )
                if not insights_exist:
                    raise FileNotFoundError(
                        f"Cannot start from step 2: Insight directories not found in {self.output_dir}. "
                        "Run with --start-from-step 1 first to extract insights."
                    )
                accuracy_map = {name: 0.0 for name in dataset_list}

            if individual:
                print(f"\nIteration {iteration}: Generating individual encyclopedias per dataset...")
                per_dataset_ency_paths = self.aggregate_insights(
                    dataset_list, r1=r1, r2=r2, iteration=iteration
                )
                encyclopedia_map = {
                    dataset: [path] for dataset, path in per_dataset_ency_paths.items()
                }
                combined_ency_path = None
                encyclopedia_paths = None
            else:
                # STEP 2: Generate combined encyclopedia from all datasets at once
                # Collect all skills from all available datasets instead of processing each individually
                print(f"\nIteration {iteration}: Generating combined encyclopedia from all datasets...")
                combined_ency_path = self.generate_combined_encyclopedia(
                    dataset_list, r1=r1, r2=r2, iteration=iteration
                )
                # Use only the combined encyclopedia for next iteration's Step 1
                encyclopedia_paths = [combined_ency_path] if combined_ency_path else []

            iteration_encyclopedia_path = None
            if not individual:
                candidate = self._iteration_encyclopedia_path(iteration)
                if os.path.isfile(candidate):
                    iteration_encyclopedia_path = os.path.abspath(candidate)

            # Save iteration results
            iteration_summary = {
                "iteration": iteration,
                "datasets": dataset_list,
                "accuracy_per_dataset": accuracy_map,
                "combined_encyclopedia": combined_ency_path,
                "iteration_encyclopedia": iteration_encyclopedia_path,
                "per_dataset_encyclopedias": per_dataset_ency_paths,
            }
            iteration_history.append(iteration_summary)

            print(f"\nIteration {iteration} Summary:")
            for dataset, acc in accuracy_map.items():
                print(f"  - {dataset}: {acc:.2%}")
            if individual and encyclopedia_map:
                print("  Individual encyclopedias:")
                for dataset_name, paths in encyclopedia_map.items():
                    print(f"    - {dataset_name}: {paths[0]}")
            elif encyclopedia_paths:
                print(f"  Combined encyclopedia: {encyclopedia_paths[0]}")

        # Final summary
        final_summary = {
            "mode": "iterative_individual" if individual else "iterative",
            "num_iterations": self.num_iterations,
            "individual": individual,
            "datasets": dataset_list,
            "iteration_history": iteration_history,
            "total_time_seconds": time.time() - start_time,
        }

        summary_path = os.path.join(self.output_dir, "iterative_summary.json")
        # Persist final summary, merging with existing per-dataset entries
        try:
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []
            payload = {
                "final": final_summary,
            }
            # Store both the final aggregate and previously appended per-dataset entries
            combined = existing if isinstance(existing, list) else []
            combined.append(payload)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to write final iterative summary: {e}")

        print(f"\n{'='*80}")
        print("ITERATIVE LEARNING COMPLETE")
        print(f"{'='*80}")
        print(f"\nAccuracy per iteration:")
        for iter_sum in iteration_history:
            print(f"  Iteration {iter_sum['iteration']}:")
            for dataset, acc in iter_sum["accuracy_per_dataset"].items():
                print(f"    - {dataset}: {acc:.2%}")
        print(f"\nFinal summary saved: {summary_path}")

        return final_summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _extract_answer_from_solution(
        self, solution: str, dataset_name: str, problem_data: Dict
    ) -> Optional[str]:
        """Extract answer from solution text using dataset-specific strategies.

        Args:
            solution: The model's generated solution text
            dataset_name: Name of the dataset (e.g., 'aime25', 'livemathbench', 'gsm8k')
            problem_data: Original problem data dictionary

        Returns:
            Extracted answer string or None if extraction failed
        """
        if not solution:
            return None

        if dataset_name == "hle":
            from hle_datasets.hle import extract_hle_final_answer

            return extract_hle_final_answer(solution)

        def _extract_boxed_balanced(text: str) -> Optional[str]:
            marker = "\\boxed{"
            start = text.rfind(marker)
            if start == -1:
                return None

            i = start + len(marker)
            depth = 1
            while i < len(text) and depth > 0:
                char = text[i]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                i += 1

            if depth != 0:
                return None

            return text[start + len(marker) : i - 1].strip()

        # Strategy 1: Extract from \boxed{} format (with or without LaTeX math mode)
        # Handles: \boxed{204}, \(\boxed{204}\), \[\boxed{204}\]
        boxed_answer = _extract_boxed_balanced(solution)
        if boxed_answer:
            boxed_answer = boxed_answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
            boxed_answer = boxed_answer.replace("\\text{", "").replace("}", "")
            return boxed_answer

        boxed_patterns = [
            r"\\\(\\boxed\{([^}]+)\}\\\)",  # \(\boxed{answer}\)
            r"\\\[\\boxed\{([^}]+)\}\\\]",  # \[\boxed{answer}\]
            r"\\boxed\{([^}]+)\}",  # \boxed{answer}
        ]

        for pattern in boxed_patterns:
            match = re.search(pattern, solution)
            if match:
                answer = match.group(1).strip()
                # Clean up LaTeX formatting from inside boxed
                answer = answer.replace("\\,", "").replace("\\:", "").replace("\\;", "")
                answer = answer.replace("\\text{", "").replace("}", "")
                return answer

        # Strategy 2: Extract from ## Answer: section (structured format)
        if "## Answer:" in solution:
            start_idx = solution.find("## Answer:") + len("## Answer:")
            end_idx = solution.find("## End of Answer:")
            if end_idx == -1:
                # Try to find next ## heading or use rest of text
                next_heading = solution.find("##", start_idx)
                end_idx = next_heading if next_heading != -1 else len(solution)
            answer = solution[start_idx:end_idx].strip()

            boxed_answer = _extract_boxed_balanced(answer)
            if boxed_answer:
                return boxed_answer

            # Try to extract boxed from this section
            for pattern in boxed_patterns:
                match = re.search(pattern, answer)
                if match:
                    return match.group(1).strip()
            return answer

        # Strategy 3: Dataset-specific extraction strategies
        if dataset_name:
            # AIME/Math competition formats: look for "the answer is" patterns
            if dataset_name.startswith("aime") or dataset_name.startswith("imo"):
                # Look for common answer phrases near the end
                answer_patterns = [
                    r"(?:the answer is|answer:|final answer:?)\s*\$?([^.$\n]+)\$?",
                    r"(?:therefore|thus|so),?\s+(?:the answer is)?\s*\$?([^.$\n]+)\$?",
                ]
                # Search in last 1000 characters for efficiency
                search_text = solution[-1000:] if len(solution) > 1000 else solution
                for pattern in answer_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    # Get the last match
                    last_match = None
                    for match in matches:
                        last_match = match
                    if last_match:
                        answer = last_match.group(1).strip()
                        # Clean LaTeX and extract number
                        answer = answer.replace("\\,", "").replace("$", "")
                        numbers = extract_numbers(answer)
                        if numbers:
                            num = numbers[-1]
                            return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

            # GSM8K: typically ends with #### answer format in ground truth,
            # but model should use boxed or numeric answer
            elif dataset_name == "gsm8k":
                # GSM8K answers are typically simple numbers
                # Look for last number in solution
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

            # LiveMathBench: various formats depending on sub-benchmark
            elif "livemathbench" in dataset_name:
                # Try numeric extraction first
                numbers = extract_numbers(solution)
                if numbers:
                    num = numbers[-1]
                    return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

        # Strategy 4: Generic fallback - extract last number from solution
        numbers = extract_numbers(solution)
        if numbers:
            num = numbers[-1]
            return str(int(num)) if math.isfinite(num) and num == int(num) else str(num)

        # Strategy 5: Last resort - return last non-empty line (cleaned)
        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            # Remove common suffixes
            last_line = re.sub(r"\.$", "", last_line)
            last_line = last_line.replace("\\)", "").replace("\\(", "")
            return last_line[:100]  # Limit length

        return None

    def _normalize_answer_for_comparison(self, answer: str) -> str:
        """Normalize answer text for comparison.

        Removes LaTeX delimiters, whitespace, and other formatting that doesn't
        affect mathematical equivalence.
        """
        if not answer:
            return ""

        # Remove LaTeX delimiters
        answer = answer.replace("$", "").replace("\\(", "").replace("\\)", "")
        answer = answer.replace("\\[", "").replace("\\]", "")
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

        # Remove common LaTeX commands that don't affect meaning
        answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)  # \text{...} -> ...
        answer = re.sub(r"\\displaystyle\s*", "", answer)
        answer = re.sub(r"^\s*[a-zA-Z]\w*\s*=\s*", "", answer)
        answer = re.sub(r"^\s*(?:answer|finalanswer|ans)\s*[:=]\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(
            r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", answer
        )  # \frac{a}{b} -> (a)/(b)

        # Remove all whitespace (spaces, tabs, newlines)
        answer = re.sub(r"\s+", "", answer)

        # Normalize case
        answer = answer.lower()

        return answer.strip()

    def _check_answer_match(
        self,
        predicted: str,
        ground_truth: str,
        dataset_name: Optional[str] = None,
        problem_text: Optional[str] = None,
    ) -> bool:
        """Check if predicted answer matches ground truth using dataset-specific evaluators.

        Routes evaluation based on dataset:
        - LiveMathBench: livemathbench_evaluator
        - IMOBench: imo_evaluator (optionally uses Gemini if enabled)
        - Others: numeric/string/symbolic comparison

        Returns True if answers match.
        """
        if not predicted or not ground_truth:
            return False

        if dataset_name == "hle":
            from hle_datasets.hle import is_hle_answer_correct

            return is_hle_answer_correct(predicted, ground_truth)

        # LiveMathBench datasets
        if dataset_name and "livemathbench" in dataset_name:
            return livemathbench_evaluator(
                predicted,
                ground_truth,
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # IMOBench datasets
        if dataset_name and dataset_name.startswith("imo"):
            return imo_evaluator(
                prediction=predicted,
                ground_truth=ground_truth,
                benchmark=dataset_name,
                problem_text=problem_text,
                use_gemini=self.use_api and self.client is not None,
                client=self.client,
            )

        # GPQA datasets (multiple choice)
        if dataset_name and dataset_name.startswith("gpqa"):
            from science_datasets.gpqa import gpqa_evaluator

            return gpqa_evaluator(
                predicted,
                ground_truth,
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # LiveCodeBench datasets (code generation)
        if dataset_name and "livecodebench" in dataset_name:
            from code_datasets.livecodebench import livecodebench_evaluator

            return livecodebench_evaluator(
                predicted,
                ground_truth,  # This will be test_cases dict
                dataset_name=dataset_name,
                problem_text=problem_text,
            )

        # Standard datasets: try multiple comparison strategies

        # Strategy 1: Normalized symbolic comparison (for algebraic expressions like 10^{2^n-n-1})
        pred_normalized = self._normalize_answer_for_comparison(predicted)
        gt_normalized = self._normalize_answer_for_comparison(ground_truth)

        if pred_normalized and gt_normalized and pred_normalized == gt_normalized:
            return True

        # Strategy 2: Numeric comparison (for numeric answers)
        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)

        if pred_nums and gt_nums:
            # Check if any predicted number matches any ground truth number
            return any(abs(p - g) < 1e-6 for p in pred_nums for g in gt_nums)

        # Strategy 3: Fallback to case-insensitive string comparison
        return predicted.strip().lower() == ground_truth.strip().lower()

    # (Legacy IMO evaluation helpers removed; using math_datasets.imo_benchmark)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_list_arg(raw: Optional[List[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    normalized: List[str] = []
    for item in raw:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        normalized.extend(parts)
    return normalized or None


def main():
    parser = argparse.ArgumentParser(
        description="Iterative benchmark learning pipeline: solve (if encyclopedia available) + extract insights + aggregate → repeat"
    )
    parallel_utils.add_num_workers_argument(parser)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help=(
            "Datasets for iterative insight learning (space- or comma-separated). "
            "With --eval-datasets, these are training datasets only."
        ),
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=None,
        help=(
            "Disjoint held-out datasets to evaluate after building the combined "
            "encyclopedia from --datasets."
        ),
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset per iteration.",
    )
    parser.add_argument(
        "--eval-max-problems",
        type=int,
        default=None,
        help="Optional held-out evaluation limit per --eval-datasets dataset.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (HF).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--r1", type=float, default=0.95, help="r1 threshold for same insights."
    )
    parser.add_argument(
        "--r2", type=float, default=0.6, help="r2 threshold for linked insights."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help=(
            "Optional train/eval split fraction. If set, randomly use this "
            "fraction of each dataset for Step 1/2/3 insight generation and "
            "evaluate the remaining problems with Step 1 only."
        ),
    )
    parser.add_argument(
        "--use-api", action="store_true", help="Use an API provider instead of HuggingFace model."
    )
    parser.add_argument(
        "--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"],
        help="Which API provider to use (default: gemini).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for the chosen provider.",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help=(
            "Provider model name. For Gemini through OpenRouter, use a slug "
            "such as google/gemini-2.5-flash-lite."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Enable OpenRouter reasoning for every default-client generation "
            "and text-mode insight aggregation call."
        ),
    )
    parser.add_argument(
        "--load-in-8bit",
        type=bool,
        default=False,
        help="Load model with 8-bit quantization (default: False, uses FP16 instead)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["normal", "text"],
        help="Aggregation/inference mode (normal=GraphRAG, text=text-based). Default: text",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of iterations (default: 3)",
    )
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=[1, 2],
        help="Start from step: 1=extract insights (default), 2=aggregate only (assumes insights already exist)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Eval-only mode: solve problems and check accuracy without extracting traces or aggregating.",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to encyclopedia file(s) to use for eval-only or iterative mode.",
    )
    parser.add_argument(
        "--client",
        type=str,
        default="default",
        choices=_CLIENT_CHOICES,
        help=(
            "Client algorithm to use for insight extraction. "
            "default=ChainOfThoughtReader (client.py), "
            "metacognitive=MetacognitiveClient, "
            "trt=TRTClient, "
            "hyperagents=HyperAgentsClient, "
            "evolveprompt=EvolvePromptClient, "
            "ace=ACEClient."
        ),
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help=(
            "Use per-dataset insight isolation. Each dataset builds its own "
            "encyclopedia from only that dataset's problems, and later solves "
            "that dataset using only its own encyclopedia. In --split mode, "
            "each dataset trains on its train split and evaluates on its eval "
            "split without sharing insights with other datasets."
        ),
    )

    args = parser.parse_args()

    # Normalize dataset lists
    datasets = _parse_list_arg(args.datasets)
    eval_datasets = _parse_list_arg(args.eval_datasets)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed}")

    pipeline = BenchmarkDomainPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        reasoning_enabled=True if args.thinking else None,
        mode=args.mode,
        num_iterations=args.num_iterations,
        load_in_8bit=args.load_in_8bit,
        client_type=args.client,
        num_workers=args.num_workers,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")
        if args.thinking and args.api_provider != "openrouter":
            raise ValueError("--thinking currently requires --api-provider openrouter")
        if args.thinking and args.client != "default":
            raise ValueError("--thinking currently requires --client default")

        if eval_datasets:
            if args.split is not None:
                raise ValueError("--eval-datasets cannot be combined with --split")
            if args.eval_only:
                raise ValueError("--eval-datasets cannot be combined with --eval-only")
            if args.individual:
                raise ValueError(
                    "--eval-datasets requires a shared combined encyclopedia; "
                    "it cannot be combined with --individual"
                )
            pipeline.run_cross_dataset_pipeline(
                train_datasets=datasets,
                eval_datasets=eval_datasets,
                train_max_problems=args.max_problems,
                eval_max_problems=args.eval_max_problems,
                r1=args.r1,
                r2=args.r2,
                start_from_step=args.start_from_step,
            )
        elif args.split is not None and args.eval_only:
            # Eval-only on the held-out (eval) portion of a deterministic split.
            # Uses the same seed+split fraction as run_split_pipeline so the eval
            # subset is identical to what that pipeline would have evaluated.
            if not 0.0 < args.split < 1.0:
                raise ValueError("--split must be a float strictly between 0 and 1")
            eval_problem_map: Dict[str, List[Dict]] = {}
            for dataset_index, dataset_name in enumerate(datasets):
                problems = pipeline.load_math_dataset(dataset_name)
                worklist = problems[: args.max_problems] if args.max_problems else problems
                indices = list(range(len(worklist)))
                rng = random.Random(args.seed + dataset_index)
                rng.shuffle(indices)
                train_size = int(len(indices) * args.split)
                if len(indices) > 1:
                    train_size = max(1, min(train_size, len(indices) - 1))
                eval_indices = indices[train_size:]
                eval_problem_map[dataset_name] = [worklist[i] for i in eval_indices]
                print(
                    f"[split-eval] {dataset_name}: {len(eval_indices)}/{len(worklist)} "
                    f"problems selected as eval set (split={args.split}, seed={args.seed})"
                )
            pipeline.run_eval_only(
                dataset_list=datasets,
                max_problems=None,
                encyclopedia_paths=args.encyclopedia,
                encyclopedia_map=(
                    pipeline._find_existing_individual_encyclopedias(datasets)
                    if args.individual and not args.encyclopedia
                    else None
                ),
                problem_overrides=eval_problem_map,
            )
        elif args.split is not None:
            pipeline.run_split_pipeline(
                dataset_list=datasets,
                max_problems=args.max_problems,
                split=args.split,
                seed=args.seed,
                r1=args.r1,
                r2=args.r2,
                individual=args.individual,
            )
        elif args.eval_only:
            # Eval-only mode: solve + accuracy, no trace extraction or aggregation
            pipeline.run_eval_only(
                dataset_list=datasets,
                max_problems=args.max_problems,
                encyclopedia_paths=args.encyclopedia,
                encyclopedia_map=(
                    pipeline._find_existing_individual_encyclopedias(datasets)
                    if args.individual and not args.encyclopedia
                    else None
                ),
            )
        else:
            pipeline.run_iterative_pipeline(
                dataset_list=datasets,
                max_problems=args.max_problems,
                r1=args.r1,
                r2=args.r2,
                start_from_step=args.start_from_step,
                individual=args.individual,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        print("\nExamples:")
        print(
            "  python task_benchmark_domain.py --datasets aime25 --max-problems 10 --num-iterations 3"
        )
        print(
            "  python task_benchmark_domain.py --datasets gsm8k math500 --max-problems 20 --mode text"
        )
        print(
            "  python task_benchmark_domain.py --datasets imo_answerbench --max-problems 30 --num-iterations 5"
        )
        print(
            "  python task_benchmark_domain.py --datasets gpqa_diamond --max-problems 50 --use-api"
        )


if __name__ == "__main__":
    main()
