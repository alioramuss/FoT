"""
Paper Insight Reading Pipeline
Extracts insights from scientific papers using the client + server_text pipeline.

This file handles the paper-reading specific task:
- Reading papers from papers_dir
- Extracting and formatting paper content
- Generating insights using client.solve_problem()
- Aggregating insights using server_text pipeline

The generic 3-step pipeline (Solution → Reflection → Behavior) stays in client.py
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

from client import ChainOfThoughtReader
from client_metacognitive import MetacognitiveClient
from client_trt import TRTClient
from client_hyperagents import HyperAgentsClient
from client_evolveprompt import EvolvePromptClient
from client_ace import ACEClient
from server_text import TextBasedInsightAggregationServer
from utils import OpenRouterInFlightBudgetError, sanitize_unicode_text

_CLIENT_CHOICES = ["default", "metacognitive", "trt", "hyperagents", "evolveprompt", "ace"]


@contextmanager
def _quiet_pdf_parser_diagnostics():
    """Hide non-fatal malformed-PDF diagnostics while preserving exceptions."""
    logger = logging.getLogger("PyPDF2")
    previous_level = logger.level
    try:
        logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(previous_level)


def _paper_file_sort_key(path: Path):
    """Sort by leading numeric prefix first (e.g., 001_), then filename/path."""
    filename = path.name
    match = re.match(r"^(\d+)_", filename)
    has_prefix = 0 if match else 1
    prefix_value = int(match.group(1)) if match else 10**9
    return (has_prefix, prefix_value, filename.lower(), str(path).lower())


def _read_paper_file(paper_path: str) -> Optional[str]:
    """Read a supported paper file without depending on a reader instance."""
    try:
        path = Path(paper_path)
        if path.suffix in {".txt", ".md"}:
            return sanitize_unicode_text(path.read_text(encoding="utf-8"))
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            content = data.get("content") or data.get("text") or data.get("abstract")
            return sanitize_unicode_text(content) if content is not None else None
        if path.suffix == ".pdf":
            if not HAS_PYPDF2:
                print("  Warning: PyPDF2 not installed. Install with: pip install PyPDF2")
                return None
            with _quiet_pdf_parser_diagnostics(), path.open("rb") as handle:
                reader = PyPDF2.PdfReader(handle)
                parts = [(page.extract_text() or "") for page in reader.pages]
            return sanitize_unicode_text("\n".join(parts).strip())
        print(f"  Warning: Unsupported file format {path.suffix}")
        return None
    except Exception as exc:
        print(f"  Error reading {paper_path}: {exc}")
        return None


def _format_paper_items(paper_items: List[Tuple[str, str]]) -> str:
    sections = ["Complete following tasks one by one:"]
    for paper_name, paper_content in paper_items:
        paper_name = sanitize_unicode_text(paper_name)
        paper_content = sanitize_unicode_text(paper_content)
        sections.append(
            f"\n            ### Answer research question of the paper: "
            f"{paper_name} based on paper content:{paper_content}\n            "
        )
    sections.append(" All papers need to be considered in the analysis. ")
    return "\n\n".join(sections)


def _sanitize_json_strings(value: Any) -> Any:
    """Recursively make model-produced JSON safe for UTF-8 output."""
    if isinstance(value, str):
        return sanitize_unicode_text(value)
    if isinstance(value, dict):
        return {
            sanitize_unicode_text(key): _sanitize_json_strings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_strings(item) for item in value]
    return value


_PAPER_PROCESS_CONFIG: Optional[Dict[str, Any]] = None
_PAPER_PROCESS_CLIENT = None


def _initialize_paper_process(config: Dict[str, Any]) -> None:
    """Initialize process-local configuration; API clients remain lazy."""
    global _PAPER_PROCESS_CONFIG, _PAPER_PROCESS_CLIENT
    _PAPER_PROCESS_CONFIG = config
    _PAPER_PROCESS_CLIENT = None


def _get_paper_process_client():
    global _PAPER_PROCESS_CLIENT
    if _PAPER_PROCESS_CONFIG is None:
        raise RuntimeError("Paper worker process was not initialized")
    if _PAPER_PROCESS_CLIENT is None:
        _PAPER_PROCESS_CLIENT = _build_client(
            client_type=_PAPER_PROCESS_CONFIG["client_type"],
            model_name=_PAPER_PROCESS_CONFIG["model_name"],
            device=_PAPER_PROCESS_CONFIG["device"],
            use_api=_PAPER_PROCESS_CONFIG["use_api"],
            api_key=_PAPER_PROCESS_CONFIG["api_key"],
            api_provider=_PAPER_PROCESS_CONFIG["api_provider"],
            output_dir=_PAPER_PROCESS_CONFIG["output_dir"],
            load_in_8bit=_PAPER_PROCESS_CONFIG["load_in_8bit"],
        )
    return _PAPER_PROCESS_CLIENT


def _load_cached_batch(spec, output_dir: str):
    """Return a completed batch checkpoint, or None when it must be processed.

    Callable from the coordinator process so already-finished batches never
    spawn a worker or build an API client. Worker processes reuse it too, so
    both paths agree on what counts as a valid checkpoint.
    """
    batch_start, batch_end, batch_label, batch_file_names = spec
    output_file = os.path.join(output_dir, f"paper_{batch_label}.json")
    if not os.path.isfile(output_file):
        return None
    try:
        with open(output_file, encoding="utf-8") as handle:
            cached_book = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached_book, dict) or not cached_book:
        return None
    return {
        "paper_names": [Path(path).stem for path in batch_file_names],
        "insight_count": len(cached_book),
        "batch_range": [batch_start + 1, batch_end],
        "skipped_papers": [],
        "cached": True,
        "worker_pid": os.getpid(),
    }


def _process_paper_batch(spec):
    """Process one paper batch inside an isolated OS process."""
    if _PAPER_PROCESS_CONFIG is None:
        raise RuntimeError("Paper worker process was not initialized")
    batch_start, batch_end, batch_label, batch_file_names = spec
    output_file = os.path.join(
        _PAPER_PROCESS_CONFIG["output_dir"], f"paper_{batch_label}.json"
    )

    cached = _load_cached_batch(spec, _PAPER_PROCESS_CONFIG["output_dir"])
    if cached is not None:
        return cached

    paper_items = []
    skipped_names = []
    for paper_file_name in batch_file_names:
        paper_name = Path(paper_file_name).stem
        paper_content = _read_paper_file(paper_file_name)
        if not paper_content:
            skipped_names.append(paper_name)
            continue
        paper_items.append((paper_name, paper_content))
    if not paper_items:
        return None

    result = _get_paper_process_client().solve_problem(
        task=_format_paper_items(paper_items)
    )
    insight_book = result.get("insight_book", {})
    if not isinstance(insight_book, dict) or not insight_book:
        raise ValueError(f"Batch {batch_label} returned an empty insight_book")
    insight_book = _sanitize_json_strings(insight_book)

    temporary_file = f"{output_file}.tmp.{os.getpid()}"
    with open(temporary_file, "w", encoding="utf-8") as handle:
        json.dump(insight_book, handle, indent=2, ensure_ascii=False)
    os.replace(temporary_file, output_file)
    return {
        "paper_names": [name for name, _ in paper_items],
        "insight_count": len(insight_book),
        "batch_range": [batch_start + 1, batch_end],
        "skipped_papers": skipped_names,
        "cached": False,
        "worker_pid": os.getpid(),
    }


def _build_client(
    client_type: str,
    model_name: str,
    device,
    use_api: bool,
    api_key,
    api_provider: str,
    output_dir: str,
    load_in_8bit: bool,
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
        client = TRTClient(api_model=model_name, **common)
    elif client_type == "hyperagents":
        client = HyperAgentsClient(**common)
    elif client_type == "evolveprompt":
        client = EvolvePromptClient(**common)
    elif client_type == "ace":
        client = ACEClient(api_model=model_name, **common)
    else:  # "default"
        return ChainOfThoughtReader(
            model_name=model_name,
            device=device,
            use_api=use_api,
            api_key=api_key,
            api_provider=api_provider,
            api_model=model_name,
            load_in_8bit=load_in_8bit,
        )

    # These legacy clients do not expose api_model in their constructors, but
    # their OpenRouter dispatch reads api_model_name on each request.
    if use_api and api_provider == "openrouter":
        client.api_model_name = model_name
    return client


class PaperInsightReader:
    """
    Pipeline for extracting insights from scientific papers.
    Similar structure to BenchmarkDomainPipeline but for paper reading.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        papers_dir: str = "papers",
        output_dir: str = "paper_insights",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        load_in_8bit: bool = False,
        client_type: str = "default",
        num_insights: int = 200,
    ):
        """
        Initialize Paper Insight Reader.

        Args:
            model_name: Model for insight extraction
            device: Device for model (cuda/cpu)
            papers_dir: Directory containing papers to read
            output_dir: Directory to save extracted insights
            use_api: Whether to use an API provider
            api_key: API key for the chosen provider
            api_provider: Which API provider to use (gemini/openrouter)
            load_in_8bit: Whether to load model in 8-bit
            client_type: Which client algorithm to use (default/metacognitive/trt/hyperagents/evolveprompt/ace)
            num_insights: Exact number of entries requested in the final library
        """
        self.model_name = model_name
        self.device = device
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.use_api = use_api
        self.api_key = api_key
        self.api_provider = api_provider
        self.load_in_8bit = load_in_8bit
        self.client_type = client_type
        if num_insights < 1:
            raise ValueError("num_insights must be at least 1")
        self.num_insights = num_insights

        # Initialize client (generic pipeline)
        self.client = None

        # Initialize server_text (generic aggregation)
        self.server_text = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_client(self):
        """Lazy load client"""
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
            )
            print(f"[PaperInsightReader] Using client: {self.client_type} ({type(self.client).__name__})")

    def _read_paper_content(self, paper_path: str) -> Optional[str]:
        """
        Read paper content from file.

        Supports:
        - .txt: Plain text
        - .md: Markdown
        - .pdf: PDF (using PyPDF2)
        - .json: JSON with 'content' field

        Args:
            paper_path: Path to paper file

        Returns:
            Paper content as string, or None if failed
        """
        return _read_paper_file(paper_path)

    def _format_paper_batch_for_insight_extraction(
        self, paper_items: List[tuple]
    ) -> str:
        """
        Format a batch of papers into one extraction prompt.

        Args:
            paper_items: List of (paper_name, paper_content) pairs

        Returns:
            Combined prompt for the batch
        """
        return _format_paper_items(paper_items)

    def extract_insights_from_papers(
        self,
        max_papers: Optional[int] = None,
        file_pattern: str = "*.txt",
        agent_read_num: int = 1,
        num_workers: int = 1,
    ) -> Dict:
        """
        Extract insights from papers in papers_dir.

        Similar to task_benchmark_domain._extract_insights_for_dataset()

        Args:
            max_papers: Maximum number of papers to process
            file_pattern: Glob pattern for paper files (e.g., "*.txt", "*.pdf")
            agent_read_num: Number of papers to concatenate into one step-1 read
            num_workers: Number of paper-agent batches to run concurrently

        Returns:
            Dictionary with extraction results
        """
        # Find paper files
        papers_path = Path(self.papers_dir)
        paper_files = sorted(papers_path.rglob(file_pattern), key=_paper_file_sort_key)

        if not paper_files:
            print(f"No papers found in {self.papers_dir} matching pattern {file_pattern}")
            return {"papers_processed": 0, "insights_extracted": 0}

        if max_papers:
            paper_files = paper_files[:max_papers]

        if agent_read_num < 1:
            raise ValueError("agent_read_num must be at least 1")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if num_workers > 1 and not self.use_api:
            raise ValueError(
                "Parallel paper agents require --use-api. A local model cannot "
                "safely be replicated across worker processes."
            )

        print(f"Found {len(paper_files)} papers to process")
        if agent_read_num > 1:
            print(f"Batching papers in groups of {agent_read_num}")
        print(f"Paper-agent API processes: {num_workers}")
        print("=" * 80)

        batch_specs = []
        for batch_start in range(0, len(paper_files), agent_read_num):
            batch_files = paper_files[batch_start : batch_start + agent_read_num]
            batch_end = batch_start + len(batch_files)
            batch_label = f"{batch_start + 1:04d}_{batch_end:04d}"
            batch_specs.append(
                (
                    batch_start,
                    batch_end,
                    batch_label,
                    [str(path) for path in batch_files],
                )
            )

        worker_config = {
            "client_type": self.client_type,
            "model_name": self.model_name,
            "device": self.device,
            "use_api": self.use_api,
            "api_key": self.api_key,
            "api_provider": self.api_provider,
            "output_dir": self.output_dir,
            "load_in_8bit": self.load_in_8bit,
        }

        # Resolve completed batches in the coordinator before touching the
        # process pool. Otherwise a fully-resumed run still spawns
        # num_workers processes, builds an API client in each, and walks all
        # batches from the first one just to read cached JSON.
        results = []
        failures = []
        pending_specs = []
        for spec in batch_specs:
            cached = _load_cached_batch(spec, self.output_dir)
            if cached is None:
                pending_specs.append(spec)
            else:
                results.append(cached)
        if results:
            print(
                f"Reusing {len(results)}/{len(batch_specs)} cached batch(es) "
                f"from {self.output_dir}"
            )
        if not pending_specs:
            print(
                "Every batch is already cached; skipping extraction entirely "
                "and proceeding to insight-library aggregation."
            )
        else:
            print(
                f"Processing {len(pending_specs)} remaining batch(es) "
                f"with {min(num_workers, len(pending_specs))} worker(s)"
            )

        completed_outputs = []
        if not pending_specs:
            pass
        elif num_workers == 1:
            _initialize_paper_process(worker_config)
            for spec in pending_specs:
                try:
                    completed_outputs.append((spec, _process_paper_batch(spec), None))
                except OpenRouterInFlightBudgetError:
                    raise
                except Exception as exc:
                    completed_outputs.append((spec, None, exc))
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(num_workers, len(pending_specs)),
                initializer=_initialize_paper_process,
                initargs=(worker_config,),
            )
            future_to_spec = {
                executor.submit(_process_paper_batch, spec): spec
                for spec in pending_specs
            }
            capacity_error = None
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    completed_outputs.append((spec, future.result(), None))
                except OpenRouterInFlightBudgetError as exc:
                    capacity_error = exc
                    for pending_future in future_to_spec:
                        pending_future.cancel()
                    break
                except Exception as exc:
                    completed_outputs.append((spec, None, exc))
            executor.shutdown(wait=True, cancel_futures=True)
            if capacity_error is not None:
                raise capacity_error

        for completed, (spec, batch_result, error) in enumerate(
            completed_outputs, 1
        ):
            batch_start, batch_end, batch_label, _ = spec
            if error is not None:
                failures.append((batch_label, error))
                print(
                    f"[{completed}/{len(pending_specs)}] Batch {batch_label} "
                    f"failed: {error}"
                )
                continue
            if batch_result is None:
                print(
                    f"[{completed}/{len(pending_specs)}] Batch {batch_label} "
                    "contained no readable papers"
                )
                continue
            results.append(batch_result)
            print(
                f"[{completed}/{len(pending_specs)}] Extracted "
                f"{batch_result['insight_count']} insights for papers "
                f"{batch_start + 1}-{batch_end} in PID "
                f"{batch_result['worker_pid']}"
            )

        if failures:
            labels = ", ".join(label for label, _ in failures[:10])
            causes = "; ".join(
                f"{label}: {type(error).__name__}: {error}"
                for label, error in failures[:5]
            )
            if not results:
                # Nothing succeeded, so there is no corpus to aggregate at all.
                raise RuntimeError(
                    f"All {len(failures)} paper-agent batch(es) failed "
                    f"({labels}). First failure causes: {causes}. "
                    "No insight library can be built; rerun the same command "
                    "to retry every batch."
                )
            # Transient per-batch failures (connection errors, an occasional
            # empty extraction) should not discard the batches that did
            # succeed. Successful batch files are cached on disk, so rerunning
            # the same command retries only the missing/invalid batches while
            # reusing everything already completed.
            print("\n" + "!" * 80)
            print(
                f"WARNING: skipping {len(failures)} failed paper-agent "
                f"batch(es) ({labels})."
            )
            print(f"First failure causes: {causes}")
            print(
                f"Continuing with the {len(results)} successful batch(es). "
                "Rerun the same command to retry only the skipped batches "
                "(completed batches are reused from disk)."
            )
            print("!" * 80)

        results.sort(key=lambda item: item["batch_range"][0])
        insights_count = sum(item["insight_count"] for item in results)
        # batch_range is 1-indexed and inclusive on both ends, so a batch
        # covering papers 1051-1075 contains 25 papers, not 24.
        papers_covered = sum(
            item["batch_range"][1] - item["batch_range"][0] + 1 for item in results
        )

        print("\n" + "=" * 80)
        print(
            f"Processed {papers_covered}/{len(paper_files)} papers in "
            f"{len(results)}/{len(batch_specs)} batches"
            + (f" ({len(failures)} batch(es) skipped)" if failures else "")
        )
        print(f"Total insights extracted: {insights_count}")

        return {
            "papers_processed": papers_covered,
            "papers_total": len(paper_files),
            "batches_processed": len(results),
            "batches_total": len(batch_specs),
            "batches_skipped": [label for label, _ in failures],
            "insights_extracted": insights_count,
            "results": results,
        }

    def aggregate_insights(self, r1: float = 0.95, r2: float = 0.4) -> str:
        """
        Aggregate insights from all paper results using server_text pipeline.

        Similar to task_benchmark_domain.generate_combined_encyclopedia()

        Args:
            r1: Threshold for insight aggregation
            r2: Threshold for insight relationships

        Returns:
            Path to combined encyclopedia
        """
        print("\n" + "=" * 80)
        print("Aggregating Insights from All Papers")
        print("=" * 80)

        # Find all paper result files
        # Only consume step-1 batch artifacts.  A broad ``paper_*.json`` glob
        # also matches ``paper_encyclopedia.json`` on a resumed run and would
        # recursively train on the previous final library.
        json_files = sorted(
            path
            for path in Path(self.output_dir).glob("paper_*.json")
            if re.fullmatch(r"paper_\d{4}_\d{4}\.json", path.name)
        )

        if not json_files:
            print("No paper result files found!")
            return None

        print(f"Found {len(json_files)} paper result files")

        # Initialize server_text (generic aggregation pipeline)
        self.server_text = TextBasedInsightAggregationServer(
            model_name=self.model_name,
            device=self.device,
            input_dirs=[self.output_dir],
            use_api=self.use_api,
            api_key=self.api_key,
            api_provider=self.api_provider,
            api_model=self.model_name,
            num_insights=self.num_insights,
        )

        # Run aggregation pipeline
        result = self.server_text.aggregate_and_build_encyclopedia(
            json_files=[str(f) for f in json_files], output_dir=self.output_dir
        )

        # Save encyclopedia
        encyclopedia_path = os.path.join(self.output_dir, "paper_encyclopedia.json")
        encyclopedia_dict = self.server_text._try_parse_json(
            self.server_text.encyclopedia
        )

        if encyclopedia_dict is None:
            json_content = self.server_text._extract_json_only(
                self.server_text.encyclopedia
            )
            encyclopedia_dict = self.server_text._try_parse_json(json_content)

        if encyclopedia_dict is None:
            print("Warning: Could not parse encyclopedia as JSON")
            # Save as text instead
            encyclopedia_path = encyclopedia_path.replace(".json", ".txt")
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                f.write(self.server_text.encyclopedia)
        else:
            if len(encyclopedia_dict) != self.num_insights:
                # --num-insights is a target, not a contract. The aggregation
                # model saturates before an arbitrary count, and discarding a
                # library full of valid insights would waste the whole
                # extraction run it was built from.
                print(
                    f"Note: library contains {len(encyclopedia_dict)} insights "
                    f"({self.num_insights} requested); writing it as-is."
                )
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(
                    _sanitize_json_strings(encyclopedia_dict),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        print(f"\nPaper encyclopedia saved to: {encyclopedia_path}")
        return encyclopedia_path

    def run_pipeline(
        self,
        max_papers: Optional[int] = None,
        file_pattern: str = "*.txt",
        agent_read_num: int = 1,
        num_workers: int = 1,
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step2: bool = False,
    ):
        """
        Run the complete paper insight extraction pipeline.

        Args:
            max_papers: Maximum number of papers to process
            file_pattern: Glob pattern for paper files
            agent_read_num: Number of papers to concatenate into one step-1 read
            num_workers: Number of concurrent step-1 paper-agent batches
            r1: Aggregation threshold
            r2: Relationship threshold
            start_from_step2: If True, skip step 1 and start from aggregation
        """
        start_time = time.time()

        extraction_result = None
        
        # Step 1: Extract insights from papers
        if not start_from_step2:
            print("=" * 80)
            print("STEP 1: Extracting Insights from Papers")
            print("=" * 80)
            extraction_result = self.extract_insights_from_papers(
                max_papers=max_papers,
                file_pattern=file_pattern,
                agent_read_num=agent_read_num,
                num_workers=num_workers,
            )
        else:
            print("Skipping STEP 1 (start_from_step2=True)")

        # Step 2: Aggregate insights
        print("\n" + "=" * 80)
        print("STEP 2: Aggregating Insights")
        print("=" * 80)
        encyclopedia_path = self.aggregate_insights(r1=r1, r2=r2)

        # Summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        if extraction_result:
            print(f"Papers processed: {extraction_result['papers_processed']}")
            print(f"Insights extracted: {extraction_result['insights_extracted']}")
        print(f"Encyclopedia: {encyclopedia_path}")
        print(f"Total time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract insights from scientific papers using client + server_text pipeline"
    )
    parser.add_argument(
        "-p",
        "--papers-dir",
        type=str,
        default="papers",
        help="Directory containing papers to read",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="paper_insights",
        help="Directory to save extracted insights",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name for insight extraction",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=None, help="Maximum number of papers to process"
    )
    parser.add_argument(
        "--agent-read-num",
        type=int,
        default=1,
        help="Number of papers to concatenate into one agent read for step 1",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Concurrent OS processes for paper-agent batches in step 1 "
            "(default: 1). "
            "Values above 1 require --use-api."
        ),
    )
    parser.add_argument(
        "--num-insights",
        type=int,
        default=200,
        help="Exact number of entries in the final insight library (default: 200)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.txt",
        help="Glob pattern for paper files (e.g., *.txt, *.pdf)",
    )
    parser.add_argument("--use-api", action="store_true", help="Use an API provider instead of HuggingFace model")
    parser.add_argument("--api-provider", type=str, default="gemini", choices=["gemini", "openrouter"], help="Which API provider to use (default: gemini)")
    parser.add_argument("--api-key", type=str, help="API key for the chosen provider")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument(
        "--r1", type=float, default=0.95, help="Aggregation threshold"
    )
    parser.add_argument(
        "--r2", type=float, default=0.4, help="Relationship threshold"
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip insight extraction and start from aggregation (step 2)",
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

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PaperInsightReader(
        model_name=args.model,
        device=args.device,
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        load_in_8bit=args.load_in_8bit,
        client_type=args.client,
        num_insights=args.num_insights,
    )

    # Run pipeline
    pipeline.run_pipeline(
        max_papers=args.max_papers,
        file_pattern=args.file_pattern,
        agent_read_num=args.agent_read_num,
        num_workers=args.num_workers,
        r1=args.r1,
        r2=args.r2,
        start_from_step2=args.start_from_step2,
    )


if __name__ == "__main__":
    main()
