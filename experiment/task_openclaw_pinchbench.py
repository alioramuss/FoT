"""
OpenClaw PinchBench Pipeline
Runs PinchBench tasks with OpenClaw agents, extracts reasoning traces from
transcripts, and aggregates them into an insight library via server_text.py,
server_cod.py, or server_claude_compact.py.

Structure mirrors task_paper_insight_reading.py / task_benchmark_domain.py:

  Iteration N:
    Step 1: Run each pinchbench task through the OpenClaw agent
            (calls pinchbench scripts/lib_agent.py execute_openclaw_task)
    Step 2: Reflection — extract procedural knowledge from transcript
    Step 3: Insight extraction — package as reusable traces (JSON)
    Save:   problem_XXXX.json  (same format as other pipelines)

  Aggregation:
    By default call server_text.py to build an encyclopedia from all extracted
    insights. Use --cod for server_cod.py or --compact for
    server_claude_compact.py; these flags do not change the client stages.

  Next Iteration:
    Write the encyclopedia as INSIGHTS.md into the agent workspace.
    prepare_task_workspace() in lib_agent.py preserves it across cleanups
    and injects a mandatory "read INSIGHTS.md first" instruction into
    BOOTSTRAP.md, so the agent is hardcoded to read and apply the insights.

Usage:
    python task_openclaw_pinchbench.py \\
        --model anthropic/claude-sonnet-4 \\
        --output-dir pinchbench_output \\
        --suite automated-only \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Start from aggregation step (skip task execution):
    python task_openclaw_pinchbench.py --start-from-step2 \\
        --output-dir pinchbench_output --use-api --api-provider gemini --api-key YOUR_KEY

    # Start from evaluation with existing encyclopedia:
    python task_openclaw_pinchbench.py \\
        --model anthropic/claude-sonnet-4 \\
        --encyclopedia pinchbench_output/encyclopedia.json \\
        --output-dir pinchbench_output_iter2 \\
        --use-api --api-provider gemini --api-key YOUR_KEY

Resume behavior:
    Rerun the same command with the same --output-dir. Completed training and
    held-out tasks, relationship profiling, and final aggregation are validated
    and skipped automatically; only incomplete work is executed.
"""

import argparse
import collections
import datetime
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import pinchbench internals (must be on sys.path)
# ---------------------------------------------------------------------------
_PINCHBENCH_SCRIPTS = Path(__file__).parent / "pinchbench" / "scripts"
if str(_PINCHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PINCHBENCH_SCRIPTS))

from lib_agent import (
    cleanup_agent_sessions,
    configure_bench_models,
    ensure_agent_exists,
    execute_openclaw_task,
    slugify_model,
    validate_openrouter_model,
    ModelValidationError,
    _get_agent_workspace,
)
from lib_grading import grade_task
from lib_tasks import TaskLoader
from lib_direct_agent import execute_direct_task

# ---------------------------------------------------------------------------
# Import our local pipeline pieces
# ---------------------------------------------------------------------------
from client import ChainOfThoughtReader
import parallel_utils
import server_claude_compact
import server_cod
from server_text import TextBasedInsightAggregationServer
from trace_pooling import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_EMBEDDING_MODEL,
    build_pooling_references,
    discover_reasoning_trace_files,
    load_cumulative_raw_transcript_traces,
    load_individual_library_sources,
    load_individual_library_traces,
    load_reasoning_traces,
    resolve_iteration_dir,
)
from utils import normalize_api_model, resolve_api_key
from utils import call_gemini_thinking, call_openrouter


def _validate_grading_api() -> None:
    """Fail before task execution when a stale PinchBench grader is imported."""
    parameters = inspect.signature(grade_task).parameters
    required = {"judge_provider", "judge_api_key"}
    missing = sorted(required - set(parameters))
    if missing:
        source = inspect.getsourcefile(grade_task) or "unknown module"
        raise RuntimeError(
            "Incompatible PinchBench grading module loaded from "
            f"{source}: grade_task() is missing {missing}. Sync the complete "
            "pinchbench/scripts directory together with task_openclaw_pinchbench.py."
        )


def _generated_output_tokens(token_info: Dict[str, Any]) -> int:
    """Return all newly generated tokens, including provider reasoning tokens."""
    visible = int(token_info.get("output_tokens", 0) or 0)
    thinking = int(token_info.get("thinking_tokens", 0) or 0)
    return visible + thinking


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Predefined OpenClaw skills activated by --openclaw-skill
# ---------------------------------------------------------------------------
_PREDEFINED_OPENCLAW_SKILLS: List[str] = [
    "nano-pdf",
    "summarize",
    "github",
    "notion",
    "slack",
    "weather",
    "coding-agent",
]


def _resolve_openclaw_skill_source(
    openclaw_bin: str,
    skill: str,
) -> Optional[Path]:
    """Return the skill directory reported by the active OpenClaw CLI."""
    try:
        result = subprocess.run(
            [openclaw_bin, "skills", "info", skill, "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
        file_path = Path(str(payload.get("filePath", ""))).expanduser()
        if file_path.is_file() and file_path.name == "SKILL.md":
            return file_path.parent.resolve()
        base_dir = Path(str(payload.get("baseDir", ""))).expanduser()
        if (base_dir / "SKILL.md").is_file():
            return base_dir.resolve()
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return None
    return None


# V1 tasks from pinchbench/skill at 417bfce28ab55aad09386ec08caf75d7d3b5827a
# (Apr 9 tree), mapped to the current renamed task IDs.
_PINCHBENCH_V1_TASK_IDS = {
    "task_sanity",
    "task_calendar",
    "task_stock",
    "task_blog",
    "task_weather",
    "task_summary",
    "task_events",
    "task_email",
    "task_memory",
    "task_files",
    "task_workflow",
    "task_clawdhub",
    "task_skill_search",
    "task_image_gen",
    "task_humanizer",
    "task_daily_summary",
    "task_email_triage",
    "task_email_search",
    "task_market_research",
    "task_spreadsheet_summary",
    "task_eli5_pdf_summary",
    "task_openclaw_comprehension",
    "task_second_brain",
}


def _install_openclaw_skills(skills: List[str]) -> None:
    """
    Install OpenClaw skills into the main workspace so that
    prepare_task_workspace() copies them to every task workspace.

    Runs `openclaw install <skill>` for each name. Failures are logged
    as warnings rather than aborting the benchmark.
    """
    openclaw_bin = shutil.which("openclaw") or os.environ.get("OPENCLAW_PATH", "openclaw")
    for skill in skills:
        source = _resolve_openclaw_skill_source(openclaw_bin, skill)
        if source:
            print(f"  [skill] Available: {skill} ({source})")
            continue
        print(f"  [skill] Installing: {skill}")
        try:
            result = subprocess.run(
                [openclaw_bin, "skills", "install", skill],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                print(f"  [skill] Installed: {skill}")
            else:
                print(f"  [skill] Warning: install failed for '{skill}' (rc={result.returncode}): {result.stderr.strip()[:200]}")
        except Exception as exc:
            print(f"  [skill] Warning: could not install '{skill}': {exc}")


def _materialize_openclaw_skills(
    skills: List[str],
    output_dir: Path,
) -> Path:
    """Copy exactly the curated skills into a stable run-scoped directory."""
    openclaw_bin = shutil.which("openclaw") or os.environ.get(
        "OPENCLAW_PATH", "openclaw"
    )
    source_root = Path.home() / ".openclaw" / "workspace" / "skills"
    resolved_sources = {
        skill: (
            _resolve_openclaw_skill_source(openclaw_bin, skill)
            or (
                source_root / skill
                if (source_root / skill / "SKILL.md").is_file()
                else None
            )
        )
        for skill in skills
    }
    missing = [
        skill
        for skill in skills
        if resolved_sources[skill] is None
    ]
    if missing:
        raise RuntimeError(
            "The curated OpenClaw skills were not installed correctly; missing "
            "SKILL.md for: " + ", ".join(missing)
        )
    destination_root = output_dir / "curated_openclaw_skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_skills = []
    for skill in skills:
        source = resolved_sources[skill]
        assert source is not None
        destination = destination_root / skill
        shutil.copytree(source, destination, dirs_exist_ok=True)
        manifest_skills.append(
            {
                "name": skill,
                "source": str(source.resolve()),
                "path": str((destination / "SKILL.md").resolve()),
            }
        )
    manifest_path = destination_root / "manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "mode": "curated_openclaw_skills",
                "skill_count": len(manifest_skills),
                "skills": manifest_skills,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(
        f"Curated skill bundle ready: {destination_root} "
        f"({len(manifest_skills)} skills)"
    )
    return destination_root.resolve()


def _check_openclaw() -> str:
    """
    Verify OpenClaw is available and return the command used to invoke it.
    Raises SystemExit with install instructions if not found.
    """
    path = shutil.which("openclaw")
    if path:
        return path

    override_path = os.environ.get("OPENCLAW_PATH")
    if override_path and Path(override_path).exists():
        return override_path

    print(
        "\n" + "=" * 70 + "\n"
        "ERROR: OpenClaw CLI not found.\n\n"
        "PinchBench requires the OpenClaw binary to be installed and available.\n"
        "Install it from: https://github.com/openclaw/openclaw\n\n"
        "Typical install:\n"
        "  npm install -g openclaw        # Node.js / npm\n"
        "  # or follow the instructions at https://openclaw.dev\n\n"
        "After installing, make sure 'openclaw' is on your PATH:\n"
        "  which openclaw   # should print a path\n"
        "Or set OPENCLAW_PATH to an absolute openclaw binary path.\n"
        + "=" * 70 + "\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_transcript_text(transcript: List[Dict[str, Any]]) -> str:
    """
    Extract human-readable text from an OpenClaw transcript.

    Transcript entries look like:
        {"type": "message", "message": {"role": "assistant", "content": "..."}}

    We concatenate all assistant messages (and optionally tool results)
    to form the "solution" that will be reflected on.
    """
    parts = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if isinstance(content, list):
            # Content can be a list of blocks
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        parts.append(f"[{role}]: {text}")
        elif role == "assistant":
            parts.append(f"[assistant]: {content}")
    return "\n\n".join(parts)


def _write_insights_to_workspace(
    agent_id: str,
    encyclopedia_path: str,
    workspace: Optional[Path] = None,
) -> bool:
    """
    Write the encyclopedia as INSIGHTS.md.

    Writes directly to the OpenClaw agent workspace.
    Pass *workspace* to skip the openclaw-agents-list query (preferred).

    Returns True if the workspace write succeeded.
    """
    try:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  Warning: failed to load encyclopedia: {exc}")
        return False

    # Format into readable markdown
    if isinstance(data, dict) and set(data.keys()) == {"insight"}:
        body = data["insight"]
    elif isinstance(data, dict):
        lines = []
        for name, desc in data.items():
            lines.append(f"### {name}\n{desc}\n")
        body = "\n".join(lines)
    else:
        body = str(data)

    content = (
        "# Insight Library\n\n"
        "This file contains reasoning traces and techniques extracted from solving "
        "tasks similar to yours. Read this file carefully before starting each task "
        "and apply the relevant insights.\n\n"
        f"{body}\n"
    )

    # Use provided workspace or fall back to querying openclaw
    if workspace is None:
        workspace = _get_agent_workspace(agent_id)
    if workspace is None:
        print("  Warning: failed to resolve OpenClaw workspace")
        return False

    workspace.mkdir(parents=True, exist_ok=True)
    insights_path = workspace / "INSIGHTS.md"
    insights_path.write_text(content, encoding="utf-8")
    print(f"  Written INSIGHTS.md to agent workspace: {insights_path}")

    return insights_path.exists()


def _iter_dict_nodes(node: Any):
    """Yield all dict nodes recursively from arbitrary JSON-like structures."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dict_nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dict_nodes(value)


def _analyze_tool_calls(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best-effort analysis of tool call usage from an OpenClaw transcript.

    Tracks:
      - tool names used
      - number of calls
      - per-call success/error status (error = explicit tool-return error)
    """
    call_type_markers = {
        "tool_call",
        "tool_use",
        "function_call",
        "tool_invocation",
        "tool-request",
    }
    result_type_markers = {
        "tool_result",
        "tool_return",
        "function_result",
        "tool_response",
        "tool-output",
    }

    def _extract_name(node: Dict[str, Any]) -> Optional[str]:
        for key in ("name", "tool_name", "tool", "function", "function_name"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested_fn = node.get("function")
        if isinstance(nested_fn, dict):
            fn_name = nested_fn.get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                return fn_name.strip()
        return None

    def _extract_call_id(node: Dict[str, Any]) -> Optional[str]:
        for key in ("call_id", "tool_call_id", "toolUseId", "id", "tool_use_id"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _looks_like_error(node: Dict[str, Any]) -> bool:
        if node.get("is_error") is True or node.get("isError") is True:
            return True
        status = node.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
        if node.get("error"):
            return True
        return False

    calls: List[Dict[str, Any]] = []
    calls_by_id: Dict[str, Dict[str, Any]] = {}

    for entry in transcript:
        for node in _iter_dict_nodes(entry):
            node_type = node.get("type")
            node_type_lower = node_type.lower() if isinstance(node_type, str) else ""

            if node_type_lower in call_type_markers:
                name = _extract_name(node) or "unknown_tool"
                call = {
                    "name": name,
                    "call_id": _extract_call_id(node),
                    "status": "unknown",
                    "error": None,
                }
                calls.append(call)
                if call["call_id"]:
                    calls_by_id[call["call_id"]] = call
                continue

            if node_type_lower in result_type_markers:
                call_id = _extract_call_id(node)
                has_error = _looks_like_error(node)
                error_msg = node.get("error")
                if not isinstance(error_msg, str) and error_msg is not None:
                    error_msg = str(error_msg)

                linked = calls_by_id.get(call_id) if call_id else None
                if linked is not None:
                    linked["status"] = "error" if has_error else "ok"
                    linked["error"] = error_msg if has_error else None
                else:
                    name = _extract_name(node) or "unknown_tool"
                    calls.append(
                        {
                            "name": name,
                            "call_id": call_id,
                            "status": "error" if has_error else "ok",
                            "error": error_msg if has_error else None,
                        }
                    )

    # Fallback for transcripts that don't expose explicit call/result typing:
    # infer from message content blocks that include tool names.
    if not calls:
        for entry in transcript:
            if entry.get("type") != "message":
                continue
            message = entry.get("message", {})
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", "")).lower()
                if "tool" not in block_type and block_type not in {"function_call", "function_result"}:
                    continue
                name = _extract_name(block) or "unknown_tool"
                status = "error" if _looks_like_error(block) else "ok"
                error_msg = block.get("error")
                if not isinstance(error_msg, str) and error_msg is not None:
                    error_msg = str(error_msg)
                calls.append(
                    {
                        "name": name,
                        "call_id": _extract_call_id(block),
                        "status": status,
                        "error": error_msg if status == "error" else None,
                    }
                )

    tool_names = sorted({call["name"] for call in calls if call.get("name")})
    tool_counter = collections.Counter(call["name"] for call in calls if call.get("name"))
    ok_calls = sum(1 for call in calls if call.get("status") == "ok")
    error_calls = sum(1 for call in calls if call.get("status") == "error")
    unknown_calls = sum(1 for call in calls if call.get("status") == "unknown")

    return {
        "tool_names": tool_names,
        "tool_name_counts": dict(sorted(tool_counter.items())),
        "total_tool_calls": len(calls),
        "successful_tool_calls": ok_calls,
        "error_tool_calls": error_calls,
        "unknown_status_tool_calls": unknown_calls,
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class OpenClawPinchBenchPipeline:
    """
    Pipeline for running PinchBench tasks, extracting reasoning traces,
    and aggregating them into an insight library.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "pinchbench_output",
        suite: str = "all",
        pinchbench_dir: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-2.5-flash-lite",
        base_url: Optional[str] = None,
        openclaw_api_key: Optional[str] = None,
        timeout_multiplier: float = 1.0,
        encyclopedia_path: Optional[str] = None,
        trace_folder: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_provider: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        thinking_level: Optional[str] = None,
        local_reflect_round: int = 1,
        openclaw_skill: bool = False,
        v1_only: bool = False,
        exclude_v1: bool = False,
        individual: bool = False,
        isolated: bool = False,
        pooling: bool = False,
        pooling_dir: Optional[str] = None,
        pooling_raw_transcript: bool = False,
        participate: Optional[int] = None,
        rag: bool = False,
        rag_api_key: Optional[str] = None,
        rag_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        pooling_context_window: int = DEFAULT_CONTEXT_WINDOW,
        num_workers: int = parallel_utils.DEFAULT_NUM_WORKERS,
        agent_backend: str = "openclaw",
        aggregation_mode: str = "text",
        aggregation_chunk_size: int = 50_000,
        cod_summary_words: int = server_cod.DEFAULT_SUMMARY_WORDS,
    ):
        _validate_grading_api()
        self.model_id = model_id
        self.output_dir = output_dir
        self.suite = suite
        self.use_api = use_api
        self.api_provider = api_provider.strip().lower()
        self.api_key = resolve_api_key(self.api_provider, api_key)
        self.api_model = normalize_api_model(self.api_provider, api_model)
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.timeout_multiplier = timeout_multiplier
        self.encyclopedia_path = encyclopedia_path
        self.trace_folder = trace_folder
        # Judge model for LLM-judge tasks; defaults to the same model as the agent
        self.judge_model = judge_model or model_id
        self.judge_provider = (judge_provider or self.api_provider).strip().lower()
        self.judge_api_key = resolve_api_key(
            self.judge_provider,
            judge_api_key
            or (self.api_key if self.judge_provider == self.api_provider else None),
        )
        self.thinking_level = thinking_level  # "low" / "medium" / "high" / None
        self.openclaw_skill = openclaw_skill
        self.curated_skills_dir: Optional[Path] = None
        self.v1_only = v1_only
        self.exclude_v1 = exclude_v1
        self.individual = individual
        self.isolated = isolated
        self.pooling = pooling
        self.pooling_dir = pooling_dir
        self.pooling_raw_transcript = pooling_raw_transcript
        self.participate = int(participate) if participate is not None else None
        self.rag = rag
        self.rag_api_key = rag_api_key
        self.rag_embedding_model = rag_embedding_model
        self.pooling_context_window = int(pooling_context_window)
        self._pooling_reference_paths: Dict[str, str] = {}
        self._pooling_manifest_path: Optional[str] = None
        self.num_workers = parallel_utils.worker_count(num_workers)
        self.agent_backend = agent_backend
        if aggregation_mode not in {"text", "cod", "compact"}:
            raise ValueError(
                "aggregation_mode must be one of: text, cod, compact"
            )
        if aggregation_chunk_size < 1:
            raise ValueError("aggregation_chunk_size must be positive")
        if cod_summary_words < 1:
            raise ValueError("cod_summary_words must be positive")
        if local_reflect_round < 1:
            raise ValueError("local_reflect_round must be at least 1")
        self.aggregation_mode = aggregation_mode
        self.aggregation_chunk_size = int(aggregation_chunk_size)
        self.cod_summary_words = int(cod_summary_words)
        self.local_reflect_round = int(local_reflect_round)

        # Ensure downstream judge/API helpers that read GEMINI_API_KEY from
        # environment can use the CLI-provided key.
        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key
        elif self.api_provider == "openrouter" and self.api_key:
            os.environ["OPENROUTER_API_KEY"] = self.api_key

        # Pinchbench skill root
        if pinchbench_dir:
            self.skill_dir = Path(pinchbench_dir)
        else:
            self.skill_dir = Path(__file__).parent / "pinchbench"

        self.tasks_dir = self.skill_dir / "tasks"

        # Agent identifier — includes a short hash of output_dir so each run
        # gets its own workspace and INSIGHTS.md does not bleed across runs
        # that share the same model but differ in encyclopedia / output path.
        self.model_slug = slugify_model(model_id)
        import hashlib as _hashlib
        _dir_hash = _hashlib.md5(os.path.abspath(output_dir).encode()).hexdigest()[:8]
        self.agent_id = f"bench-{self.model_slug}-{_dir_hash}"

        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy-loaded insight extractor client
        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache_by_output_dir: Dict[str, Dict[str, Any]] = {}

    def _metrics_output_key(self) -> str:
        return str(Path(self.output_dir).resolve())

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        """Write a JSON checkpoint without exposing a partially written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)

    @staticmethod
    def _file_sha256(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        candidate = Path(path)
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _phase_resume_manifest(
        self,
        tasks: List[Any],
        *,
        phase: str,
        insight_library_path: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "phase": phase,
            "task_ids": [str(task.task_id) for task in tasks],
            "model": self.model_id,
            "agent_backend": self.agent_backend,
            "api_provider": getattr(self, "api_provider", None),
            "api_model": getattr(self, "api_model", None),
            "base_url": getattr(self, "base_url", None),
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider,
            "thinking_level": getattr(self, "thinking_level", None),
            "timeout_multiplier": getattr(self, "timeout_multiplier", 1.0),
            "openclaw_skill": getattr(self, "openclaw_skill", False),
            "v1_only": getattr(self, "v1_only", False),
            "individual": getattr(self, "individual", False),
            "isolated": getattr(self, "isolated", False),
            "pooling": getattr(self, "pooling", False),
            "pooling_raw_transcript": getattr(
                self, "pooling_raw_transcript", False
            ),
            "rag": getattr(self, "rag", False),
            "pooling_context_window": getattr(
                self, "pooling_context_window", DEFAULT_CONTEXT_WINDOW
            ),
            "insight_library_sha256": self._file_sha256(insight_library_path),
        }

    def _validate_resume_phase(
        self,
        tasks: List[Any],
        *,
        phase: str,
        insight_library_path: Optional[str],
    ) -> None:
        """Refuse to mix task checkpoints produced by incompatible runs."""
        manifest_path = Path(self.output_dir) / "resume_manifest.json"
        expected = self._phase_resume_manifest(
            tasks,
            phase=phase,
            insight_library_path=insight_library_path,
        )
        if manifest_path.exists():
            try:
                actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid resume manifest at {manifest_path}: {exc}"
                ) from exc
            if isinstance(actual, dict) and actual.get("version") == 1:
                # Backward-compatible default for checkpoints created before
                # the V1-only baseline selector was added.
                actual.setdefault("v1_only", False)
                actual.setdefault("individual", False)
                actual.setdefault("isolated", False)
                actual.setdefault("pooling", False)
                actual.setdefault("pooling_raw_transcript", False)
                actual.setdefault("rag", False)
                actual.setdefault("pooling_context_window", DEFAULT_CONTEXT_WINDOW)
            if actual != expected:
                differing = sorted(
                    key
                    for key in set(actual) | set(expected)
                    if actual.get(key) != expected.get(key)
                )
                raise RuntimeError(
                    "Existing checkpoints are incompatible with this run in "
                    f"{self.output_dir}; differing fields: {', '.join(differing)}. "
                    "Use a new --output-dir."
                )
            return

        # Older output directories have no manifest. Validate the configuration
        # fields that were already recorded before adopting their checkpoints.
        metrics_path = self._metrics_log_path()
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid metrics checkpoint at {metrics_path}: {exc}"
                ) from exc
            legacy_checks = {
                "model": expected["model"],
                "agent_backend": expected["agent_backend"],
                "judge_model": expected["judge_model"],
                "judge_provider": expected["judge_provider"],
            }
            differing = [
                key
                for key, value in legacy_checks.items()
                if metrics.get(key) is not None and metrics.get(key) != value
            ]
            if differing:
                raise RuntimeError(
                    "Existing metrics are incompatible with this run in "
                    f"{self.output_dir}; differing fields: {', '.join(differing)}. "
                    "Use a new --output-dir."
                )
        self._write_json_atomic(manifest_path, expected)

    def _load_checkpoint_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics_path = self._metrics_log_path()
        if not metrics_path.exists():
            return {}
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid metrics checkpoint at {metrics_path}: {exc}"
            ) from exc
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            raise RuntimeError(f"Invalid task list in metrics checkpoint {metrics_path}")
        return {
            str(item.get("task_id")): item
            for item in tasks
            if isinstance(item, dict) and item.get("task_id")
        }

    def _load_problem_checkpoints(self) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
        records: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
        for path in sorted(Path(self.output_dir).glob("problem_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if task_id:
                records[str(task_id)] = (path, payload)
        return records

    def _checkpoint_output_path(self, task_id: str, preferred_index: int) -> Path:
        existing = self._load_problem_checkpoints().get(str(task_id))
        if existing:
            return existing[0]
        preferred = Path(self.output_dir) / f"problem_{preferred_index:04d}.json"
        if not preferred.exists():
            return preferred
        used = [
            int(match.group(1))
            for path in Path(self.output_dir).glob("problem_*.json")
            if (match := re.fullmatch(r"problem_(\d+)\.json", path.name))
        ]
        next_index = max(used, default=0) + 1
        return Path(self.output_dir) / f"problem_{next_index:04d}.json"

    def _resume_task_checkpoint(
        self,
        *,
        task: Any,
        require_insights: bool,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        metric = self._load_checkpoint_metrics().get(str(task.task_id))
        if not metric:
            return None
        status = metric.get("execution_status")
        grade = metric.get("grade", {}) or {}
        failure_type = metric.get("failure_type")
        if status != "success" and failure_type in {
            "timeout",
            "agent_execution_error",
        }:
            if grade.get("score") != 0 and grade.get("score") != 0.0:
                return None
            result = {
                "task_id": task.task_id,
                "task_name": task.name,
                "status": status,
                "failure_type": failure_type,
                "score": 0.0,
                "max_score": float(grade.get("max_score", 1.0) or 1.0),
                "insights_extracted": 0,
                "output_file": None,
                "resumed": True,
            }
            return metric, result
        if status != "success":
            return None
        if not isinstance(grade.get("score"), (int, float)) or not isinstance(
            grade.get("max_score"), (int, float)
        ):
            return None
        problem = self._load_problem_checkpoints().get(str(task.task_id))
        if not problem:
            return None
        output_path, payload = problem
        if payload.get("execution_status") != "success":
            return None
        insight_book = payload.get("insight_book")
        if require_insights and not insight_book:
            return None
        result = {
            "task_id": task.task_id,
            "task_name": task.name,
            "status": "success",
            "score": float(grade["score"]),
            "max_score": float(grade["max_score"]),
            "insights_extracted": int(metric.get("insights_extracted", 0) or 0),
            "output_file": str(output_path),
            "resumed": True,
        }
        return metric, result

    def _build_metrics_summary(
        self,
        task_metrics: List[Dict[str, Any]],
        library_output_tokens: Optional[int],
    ) -> Dict[str, Any]:
        graded_tasks = [
            task for task in task_metrics if task.get("grade", {}).get("score") is not None
        ]
        graded_score_sum = sum(float(task["grade"].get("score", 0.0)) for task in graded_tasks)
        graded_max_score_sum = sum(float(task["grade"].get("max_score", 0.0)) for task in graded_tasks)
        overall_accuracy_pct = (
            (graded_score_sum / graded_max_score_sum * 100.0)
            if graded_max_score_sum > 0
            else None
        )

        tool_name_counts: Dict[str, int] = {}
        total_tool_calls = 0
        successful_tool_calls = 0
        error_tool_calls = 0
        unknown_tool_calls = 0
        for task in task_metrics:
            tools = task.get("tools", {})
            total_tool_calls += int(tools.get("total_calls", 0) or 0)
            successful_tool_calls += int(tools.get("successful_calls", 0) or 0)
            error_tool_calls += int(tools.get("error_calls", 0) or 0)
            unknown_tool_calls += int(tools.get("unknown_status_calls", 0) or 0)
            for name, count in (tools.get("name_counts", {}) or {}).items():
                tool_name_counts[name] = tool_name_counts.get(name, 0) + int(count)

        total_execution_time = sum(
            float(task.get("execution_time_seconds", 0.0) or 0.0) for task in task_metrics
        )
        total_agent_output_tokens = sum(
            int((task.get("output_tokens") or {}).get("agent", 0) or 0) for task in task_metrics
        )
        total_extraction_output_tokens = sum(
            int((task.get("output_tokens") or {}).get("extraction", 0) or 0)
            for task in task_metrics
        )
        total_agent_requests = sum(
            int((task.get("agent_usage") or {}).get("request_count", 0) or 0)
            for task in task_metrics
        )
        task_output_tokens = total_agent_output_tokens + total_extraction_output_tokens
        iteration_output_tokens = (
            task_output_tokens + int(library_output_tokens)
            if library_output_tokens is not None
            else None
        )

        return {
            "tasks_total": len(task_metrics),
            "tasks_succeeded": sum(
                task.get("execution_status") == "success" for task in task_metrics
            ),
            "tasks_failed": sum(
                task.get("execution_status") != "success" for task in task_metrics
            ),
            "tasks_timed_out": sum(
                task.get("execution_status") == "timeout" for task in task_metrics
            ),
            "tasks_agent_errors": sum(
                task.get("failure_type") == "agent_execution_error"
                for task in task_metrics
            ),
            "tasks_graded": len(graded_tasks),
            "graded_score_sum": graded_score_sum,
            "graded_max_score_sum": graded_max_score_sum,
            "overall_accuracy_pct": overall_accuracy_pct,
            "output_tokens_agent_total": total_agent_output_tokens,
            "output_tokens_extraction_total": total_extraction_output_tokens,
            "output_tokens_total": task_output_tokens,
            "output_tokens_iteration_total": iteration_output_tokens,
            "agent_requests_total": total_agent_requests,
            "tool_names": sorted(tool_name_counts.keys()),
            "tool_name_counts": dict(sorted(tool_name_counts.items())),
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "error_tool_calls": error_tool_calls,
            "unknown_status_tool_calls": unknown_tool_calls,
            "execution_time_total_seconds": total_execution_time,
            "library_output_tokens": int(library_output_tokens) if library_output_tokens is not None else None,
        }

    @staticmethod
    def _is_fatal_agent_execution_error(exec_result: Dict[str, Any]) -> bool:
        """Return True for systemic failures that must abort the benchmark run."""
        detail = "\n".join(
            str(exec_result.get(field) or "") for field in ("stderr", "stdout")
        ).lower()
        fatal_markers = (
            "requires more credits",
            "insufficient credit",
            "openrouter_api_key",
            "gemini_api_key",
            "api key not set",
            "invalid api key",
            "authentication",
            "unauthorized",
            "error code: 401",
            "error code: 402",
            "error code: 429",
            "rate limit",
            "invalid model",
            "model not found",
            "connection error",
            "name resolution",
            "service unavailable",
        )
        return any(marker in detail for marker in fatal_markers)

    def _record_failed_task(
        self,
        *,
        task: Any,
        exec_result: Dict[str, Any],
        execution_status: str,
        failure_type: str,
        failure_note: str,
        output_file: Optional[str] = None,
        eval_only: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build a checkpointed zero-score record without invoking the judge."""
        transcript = exec_result.get("transcript", []) or []
        usage = exec_result.get("usage", {}) or {}
        agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
        execution_time_seconds = exec_result.get("execution_time")
        if execution_time_seconds is not None:
            execution_time_seconds = float(execution_time_seconds)
        tool_stats = _analyze_tool_calls(transcript)
        grade_info = {
            "score": 0.0,
            "max_score": 1.0,
            "accuracy_pct": 0.0,
            "grading_type": task.grading_type,
            "notes": failure_note,
        }
        task_metric = {
            "task_id": task.task_id,
            "task_name": task.name,
            "execution_status": execution_status,
            "failure_type": failure_type,
            "failure_detail": failure_note,
            "execution_time_seconds": execution_time_seconds,
            "grade": grade_info,
            "output_tokens": {
                "agent": agent_output_tokens,
                "extraction": 0,
                "total": agent_output_tokens,
            },
            "agent_usage": {
                "request_count": int(usage.get("request_count", 0) or 0),
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": agent_output_tokens,
                "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
                "processed_tokens": int(usage.get("total_tokens", 0) or 0),
                "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
            },
            "tools": {
                "names": tool_stats.get("tool_names", []),
                "name_counts": tool_stats.get("tool_name_counts", {}),
                "total_calls": tool_stats.get("total_tool_calls", 0),
                "successful_calls": tool_stats.get("successful_tool_calls", 0),
                "error_calls": tool_stats.get("error_tool_calls", 0),
                "unknown_status_calls": tool_stats.get("unknown_status_tool_calls", 0),
                "calls": tool_stats.get("calls", []),
            },
            "insights_extracted": 0,
        }
        result = {
            "task_id": task.task_id,
            "task_name": task.name,
            "status": execution_status,
            "failure_type": failure_type,
            "score": 0.0,
            "max_score": 1.0,
            "insights_extracted": 0,
            "output_file": output_file,
        }
        if output_file:
            save_data = {
                "task_id": task.task_id,
                "task_name": task.name,
                "task_prompt": task.prompt,
                "execution_status": execution_status,
                "failure_type": failure_type,
                "failure_detail": failure_note,
                "output_tokens": task_metric["output_tokens"],
                "grade": {
                    "score": 0.0,
                    "max_score": 1.0,
                    "grading_type": task.grading_type,
                    "notes": failure_note,
                },
                "eval_only": eval_only,
            }
            self._write_json_atomic(Path(output_file), save_data)
        return task_metric, result

    def _record_timed_out_task(
        self,
        *,
        task: Any,
        exec_result: Dict[str, Any],
        output_file: Optional[str] = None,
        eval_only: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build a zero-score record for a timeout without invoking the judge."""
        return self._record_failed_task(
            task=task,
            exec_result=exec_result,
            execution_status="timeout",
            failure_type="timeout",
            failure_note="Task exceeded its execution timeout.",
            output_file=output_file,
            eval_only=eval_only,
        )

    def _record_agent_execution_error(
        self,
        *,
        task: Any,
        exec_result: Dict[str, Any],
        output_file: Optional[str] = None,
        eval_only: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Record a task-local agent failure as zero without invoking the judge."""
        status = str(exec_result.get("status") or "error")
        stdout = (exec_result.get("stdout") or "").strip()
        stderr = (exec_result.get("stderr") or "").strip()
        detail = stderr or stdout or "no error detail"
        note = f"Agent execution returned status={status!r}: {detail[:2000]}"
        return self._record_failed_task(
            task=task,
            exec_result=exec_result,
            execution_status=status,
            failure_type="agent_execution_error",
            failure_note=note,
            output_file=output_file,
            eval_only=eval_only,
        )

    def _write_metrics_log(
        self,
        task_metrics: List[Dict[str, Any]],
        *,
        library_output_tokens: Optional[int] = None,
    ) -> None:
        log_path = self._metrics_log_path()
        if library_output_tokens is None:
            cached = self._metrics_cache_by_output_dir.get(self._metrics_output_key())
            if cached:
                library_output_tokens = cached.get("summary", {}).get("library_output_tokens")

        payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(Path(self.output_dir).resolve()),
            "model": self.model_id,
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider,
            "suite": self.suite,
            "v1_only": getattr(self, "v1_only", False),
            "exclude_v1": self.exclude_v1,
            "individual": self.individual,
            "isolated": getattr(self, "isolated", False),
            "pooling": getattr(self, "pooling", False),
            "rag": getattr(self, "rag", False),
            "pooling_dir": getattr(self, "pooling_dir", None),
            "use_api": self.use_api,
            "agent_backend": self.agent_backend,
            "thinking_level": self.thinking_level,
            "summary": self._build_metrics_summary(task_metrics, library_output_tokens),
            "tasks": task_metrics,
        }
        self._write_json_atomic(log_path, payload)
        self._metrics_cache_by_output_dir[self._metrics_output_key()] = payload

    def _write_overall_metrics_log(self, root_output_dir: str, elapsed_seconds: float) -> None:
        root = Path(root_output_dir)
        iter_logs = sorted(root.glob("iter_*/metrics_log.json"))
        if not iter_logs:
            return

        iteration_summaries = []
        for log_path in iter_logs:
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = payload.get("summary", {})
            iteration_summaries.append(
                {
                    "iteration": log_path.parent.name,
                    "metrics_log": str(log_path),
                    "summary": summary,
                }
            )

        if not iteration_summaries:
            return

        total_tasks = sum(int(item["summary"].get("tasks_total", 0) or 0) for item in iteration_summaries)
        total_graded = sum(int(item["summary"].get("tasks_graded", 0) or 0) for item in iteration_summaries)
        graded_score_sum = sum(
            float(item["summary"].get("graded_score_sum", 0.0) or 0.0)
            for item in iteration_summaries
        )
        graded_max_score_sum = sum(
            float(item["summary"].get("graded_max_score_sum", 0.0) or 0.0)
            for item in iteration_summaries
        )
        overall_accuracy_pct = (
            (graded_score_sum / graded_max_score_sum * 100.0)
            if graded_max_score_sum > 0
            else None
        )

        overall_tool_counts: Dict[str, int] = {}
        for item in iteration_summaries:
            for name, count in (item["summary"].get("tool_name_counts", {}) or {}).items():
                overall_tool_counts[name] = overall_tool_counts.get(name, 0) + int(count)

        overall_payload = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "output_dir": str(root.resolve()),
            "iterations": iteration_summaries,
            "summary": {
                "iterations": len(iteration_summaries),
                "tasks_total": total_tasks,
                "tasks_graded": total_graded,
                "graded_score_sum": graded_score_sum,
                "graded_max_score_sum": graded_max_score_sum,
                "overall_accuracy_pct": overall_accuracy_pct,
                "output_tokens_agent_total": sum(
                    int(item["summary"].get("output_tokens_agent_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "output_tokens_extraction_total": sum(
                    int(item["summary"].get("output_tokens_extraction_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "output_tokens_total": sum(
                    int(item["summary"].get("output_tokens_total", 0) or 0)
                    for item in iteration_summaries
                ),
                "tool_names": sorted(overall_tool_counts.keys()),
                "tool_name_counts": dict(sorted(overall_tool_counts.items())),
                "total_tool_calls": sum(
                    int(item["summary"].get("total_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "successful_tool_calls": sum(
                    int(item["summary"].get("successful_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "error_tool_calls": sum(
                    int(item["summary"].get("error_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "unknown_status_tool_calls": sum(
                    int(item["summary"].get("unknown_status_tool_calls", 0) or 0)
                    for item in iteration_summaries
                ),
                "execution_time_total_seconds": sum(
                    float(item["summary"].get("execution_time_total_seconds", 0.0) or 0.0)
                    for item in iteration_summaries
                ),
                "library_output_tokens_total": sum(
                    int(item["summary"].get("library_output_tokens", 0) or 0)
                    for item in iteration_summaries
                ),
                "output_tokens_iteration_total": sum(
                    int(item["summary"].get("output_tokens_total", 0) or 0)
                    + int(item["summary"].get("library_output_tokens", 0) or 0)
                    for item in iteration_summaries
                ),
                "pipeline_elapsed_seconds": elapsed_seconds,
            },
        }

        (root / "metrics_log_overall.json").write_text(
            json.dumps(overall_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure_client(self) -> ChainOfThoughtReader:
        if self._client is None:
            self._client = ChainOfThoughtReader(
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                api_model=self.api_model,
            )
        return self._client

    def _parallel_worker(
        self,
        task: Any,
        index: int,
        worker_root: Path,
    ) -> "OpenClawPinchBenchPipeline":
        """Create isolated task state for one concurrent OpenClaw client."""
        worker = copy.copy(self)
        worker.num_workers = 1
        worker.output_dir = str(worker_root / f"task_{index:04d}")
        worker.agent_id = (
            f"{self.agent_id}-{index:04d}-{slugify_model(str(task.task_id))}"[:120]
        )
        worker._client = None
        worker._metrics_cache_by_output_dir = {}
        worker.openclaw_skill = False
        os.makedirs(worker.output_dir, exist_ok=True)
        return worker

    def _ensure_curated_skill_bundle(self) -> Optional[Path]:
        """Install and snapshot the predefined skill set once for this run."""
        if not self.openclaw_skill:
            return None
        if self.curated_skills_dir and all(
            (self.curated_skills_dir / skill / "SKILL.md").is_file()
            for skill in _PREDEFINED_OPENCLAW_SKILLS
        ):
            return self.curated_skills_dir
        _check_openclaw()
        _install_openclaw_skills(_PREDEFINED_OPENCLAW_SKILLS)
        self.curated_skills_dir = _materialize_openclaw_skills(
            _PREDEFINED_OPENCLAW_SKILLS,
            Path(self.output_dir),
        )
        return self.curated_skills_dir

    def _run_tasks_and_extract_parallel(
        self,
        tasks: List[Any],
        run_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run one isolated OpenClaw agent per task and merge ordered outputs."""
        self._validate_resume_phase(
            tasks,
            phase="training",
            insight_library_path=self.encyclopedia_path,
        )
        if self.openclaw_skill:
            self._ensure_curated_skill_bundle()

        worker_root = Path(self.output_dir) / "parallel_workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        indexed_tasks = list(enumerate(tasks, 1))

        def process(item):
            index, task = item
            worker = self._parallel_worker(task, index, worker_root)
            results = worker.run_tasks_and_extract(
                run_id=f"{run_id or 'fot'}-{index}",
                tasks=[task],
            )
            metrics_path = Path(worker.output_dir) / "metrics_log.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {"tasks": []}
            )
            problem_files = sorted(Path(worker.output_dir).glob("problem_*.json"))
            return results, metrics.get("tasks", []), problem_files

        outcomes = parallel_utils.parallel_map_ordered(
            process, indexed_tasks, num_workers=self.num_workers
        )
        results: List[Dict[str, Any]] = []
        task_metrics: List[Dict[str, Any]] = []
        for (index, _task), (worker_results, metrics, problem_files) in zip(
            indexed_tasks, outcomes
        ):
            results.extend(worker_results)
            task_metrics.extend(metrics)
            if problem_files:
                shutil.copy2(
                    problem_files[0],
                    Path(self.output_dir) / f"problem_{index:04d}.json",
                )
        self._write_metrics_log(task_metrics)
        return results

    def _run_tasks_eval_parallel(
        self,
        tasks: List[Any],
        run_id: Optional[str],
        encyclopedia_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run held-out tasks with isolated concurrent OpenClaw agents."""
        self._validate_resume_phase(
            tasks,
            phase="held_out_eval",
            insight_library_path=(
                getattr(self, "_pooling_manifest_path", None) or encyclopedia_path
            ),
        )
        if self.openclaw_skill:
            self._ensure_curated_skill_bundle()

        worker_root = Path(self.output_dir) / "parallel_workers_eval"
        worker_root.mkdir(parents=True, exist_ok=True)
        indexed_tasks = list(enumerate(tasks, 1))

        def process(item):
            index, task = item
            worker = self._parallel_worker(task, index, worker_root)
            results = worker.run_tasks_eval_only(
                [task],
                run_id=f"{run_id or 'fot-eval'}-{index}",
                encyclopedia_path=encyclopedia_path,
            )
            metrics_path = Path(worker.output_dir) / "metrics_log.json"
            metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.exists()
                else {"tasks": []}
            )
            return results, metrics.get("tasks", [])

        outcomes = parallel_utils.parallel_map_ordered(
            process, indexed_tasks, num_workers=self.num_workers
        )
        results: List[Dict[str, Any]] = []
        task_metrics: List[Dict[str, Any]] = []
        for worker_results, metrics in outcomes:
            results.extend(worker_results)
            task_metrics.extend(metrics)
        self._write_metrics_log(task_metrics)
        return results

    def _setup_agent(self) -> None:
        """Validate openclaw install, validate model, and ensure agent exists."""
        if self.agent_backend == "direct":
            if self.openclaw_skill:
                self._ensure_curated_skill_bundle()
            if not (self.openclaw_api_key or self.api_key):
                raise ValueError(
                    "Direct agent execution requires --api-key or the provider API-key environment variable"
                )
            print(
                f"Using direct OpenAI-compatible agent backend for model '{self.model_id}' "
                "(OpenClaw CLI is not required)"
            )
            return

        # Fail fast if openclaw is not installed — don't silently run 25 tasks
        _check_openclaw()

        if self.openclaw_skill:
            print(f"\nInstalling predefined OpenClaw skills ({len(_PREDEFINED_OPENCLAW_SKILLS)} skills)...")
            self._ensure_curated_skill_bundle()
            print("Skill installation complete.\n")
        # Determine effective model ID and connection config.
        # For google/gemini models, use OpenClaw's native Google provider
        # (reads GEMINI_API_KEY from env) rather than a custom OpenAI-compat
        # endpoint — the custom provider triggers gateway/pairing mode.
        effective_base_url = self.base_url
        effective_api_key = self.openclaw_api_key
        effective_model_id = self.model_id
        if self.use_api and self.api_provider == "gemini" and not self.base_url:
            # Normalise to google/ prefix so OpenClaw uses its native Gemini provider
            if self.model_id.startswith("gemini/"):
                effective_model_id = "google/" + self.model_id.split("/", 1)[1]
            elif not self.model_id.startswith("google/"):
                effective_model_id = "google/" + self.model_id
            else:
                effective_model_id = self.model_id
            # Ensure GEMINI_API_KEY is in env so OpenClaw can authenticate
            gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
                os.environ.setdefault("GOOGLE_AI_STUDIO_KEY", gemini_key)
            else:
                print("Warning: GEMINI_API_KEY is not set; OpenClaw Gemini calls may fail")
            print(f"Using OpenClaw native Google provider for model '{effective_model_id}'")

        if not effective_base_url:
            if self.api_provider != "gemini":
                print("No custom OpenClaw base URL provided, using default openrouter.ai endpoints")
                try:
                    print(f"Validating model: {self.model_id}")
                    validate_openrouter_model(self.model_id)
                except ModelValidationError as exc:
                    print(f"Warning: {exc}")
        else:
            print(f"Using custom OpenClaw base URL: {effective_base_url}")

        print(f"Ensuring OpenClaw agent exists for model '{self.model_id}' with ID '{self.agent_id}'")
        agent_workspace = _get_agent_workspace(self.agent_id)
        if agent_workspace is None:
            # Deterministic fallback aligned with OpenClaw convention
            normalized_id = self.agent_id.replace(":", "-").lower()
            agent_workspace = (
                Path.home() / ".openclaw" / "agents" / normalized_id / "workspace"
            )
        # Retry once on ConfigMutationConflictError (openclaw config race)
        agent_ready = False
        for _attempt in range(2):
            ok = ensure_agent_exists(
                self.agent_id,
                effective_model_id,
                agent_workspace,
                base_url=effective_base_url,
                api_key=effective_api_key,
            )
            if ok is not False:
                agent_ready = True
                break
            # Re-query in case the agent was created by a concurrent process
            resolved = _get_agent_workspace(self.agent_id)
            if resolved is not None:
                agent_workspace = resolved
                agent_ready = True
                break
            import time as _time; _time.sleep(1)
        if not agent_ready:
            raise RuntimeError(
                f"Failed to create or find OpenClaw agent '{self.agent_id}'. "
                "Run `openclaw agents list` and retry after any concurrent OpenClaw command finishes."
            )
        # Store resolved workspace so downstream calls don't need to re-query
        self._agent_workspace: Path = agent_workspace
        # For truly custom (non-native) endpoints, re-write models.json after a
        # brief delay to overwrite any async re-initialisation openclaw may do.
        if effective_base_url:
            import time as _time
            _time.sleep(2)
            configure_bench_models(
                self.agent_id, effective_model_id, effective_base_url, effective_api_key
            )
        cleanup_agent_sessions(self.agent_id)

    def _read_insight_library(self, path: Optional[str]) -> Optional[str]:
        """Read the complete active insight library for direct-agent injection."""
        if not path:
            return None
        library_path = Path(path)
        if not library_path.exists():
            return None
        return library_path.read_text(encoding="utf-8")

    def _execute_task(
        self,
        *,
        task: Any,
        run_id: str,
        insight_library_path: Optional[str],
    ) -> Dict[str, Any]:
        """Execute one task through the selected OpenClaw or direct backend."""
        if self.agent_backend == "direct":
            if self.api_provider == "openrouter":
                base_url = self.base_url or "https://openrouter.ai/api/v1"
                model_id = self.model_id
            else:
                base_url = self.base_url or (
                    "https://generativelanguage.googleapis.com/v1beta/openai"
                )
                model_id = self.model_id
                if model_id.startswith("google/") or model_id.startswith("gemini/"):
                    model_id = model_id.split("/", 1)[1]
            curated_skills_dir = getattr(self, "curated_skills_dir", None)
            skills_dir = curated_skills_dir or (
                self.skill_dir / ".agents" / "skills"
            )
            return execute_direct_task(
                task=task,
                model_id=model_id,
                run_id=run_id,
                timeout_multiplier=self.timeout_multiplier,
                skill_dir=self.skill_dir,
                api_key=self.openclaw_api_key or self.api_key or "",
                base_url=base_url,
                output_dir=Path(self.output_dir) / "transcripts",
                workspace_root=Path(self.output_dir) / "direct_workspaces",
                extra_skills_dir=skills_dir if skills_dir.exists() else None,
                include_openclaw_workspace_skills=curated_skills_dir is None,
                thinking_level=self.thinking_level,
                insight_library_text=self._read_insight_library(insight_library_path),
            )

        return execute_openclaw_task(
            task=task,
            agent_id=self.agent_id,
            model_id=self.model_id,
            run_id=run_id,
            timeout_multiplier=self.timeout_multiplier,
            skill_dir=self.skill_dir,
            output_dir=Path(self.output_dir) / "transcripts",
            verbose=False,
            thinking_level=self.thinking_level,
        )

    def _write_local_reflect_library(
        self,
        *,
        base_insight_path: Optional[str],
        local_insight_book: Dict[str, str],
        task: Any,
        local_round: int,
    ) -> str:
        """Merge the base insight library with this task's own --local-reflect-round
        traces accumulated so far, for injection into the *next* local round.
        """
        merged: Dict[str, Any] = {}
        if base_insight_path and Path(base_insight_path).exists():
            try:
                base_data = json.loads(Path(base_insight_path).read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    f"  Warning: failed to load base insight library for "
                    f"local reflection: {exc}"
                )
                base_data = None
            if isinstance(base_data, dict):
                merged.update(base_data)
        merged.update(local_insight_book)

        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.task_id)).strip("_") or "task"
        scratch_path = (
            Path(self.output_dir)
            / "local_reflect"
            / f"{safe_task_id}_round{local_round:02d}.json"
        )
        self._write_json_atomic(scratch_path, merged)
        return str(scratch_path)

    def _execute_task_with_local_reflection(
        self,
        *,
        task: Any,
        run_id_prefix: str,
        base_insight_path: Optional[str],
    ) -> Dict[str, Any]:
        """Run --local-reflect-round warm-up attempts on this one task before
        the final attempt, whose result feeds grading + Steps 2/3 as usual.

        Each warm-up round executes the task, extracts a reasoning trace from
        it (Steps 2 & 3), and folds that trace into the insight library used
        by the *next* round — so round r+1 sees round r's own freshly
        extracted trace on top of whatever base library/encyclopedia was
        already active. This never mixes across tasks or global rounds: the
        --iterations loop is unaffected, and each task starts its own warm-up
        sequence from the same shared base_insight_path.
        """
        rounds = max(1, self.local_reflect_round)
        if rounds == 1:
            return self._execute_task(
                task=task, run_id=run_id_prefix, insight_library_path=base_insight_path
            )

        current_path = base_insight_path
        local_insight_book: Dict[str, str] = {}
        exec_result: Dict[str, Any] = {}
        for local_round in range(1, rounds + 1):
            if self.agent_backend == "openclaw" and current_path:
                _write_insights_to_workspace(
                    self.agent_id, current_path, self._agent_workspace
                )
            print(f"  Local reflect round {local_round}/{rounds}")
            try:
                exec_result = self._execute_task(
                    task=task,
                    run_id=f"{run_id_prefix}-local{local_round}",
                    insight_library_path=current_path,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Task execution failed for {task.task_id} "
                    f"(local reflect round {local_round}/{rounds}): {exc}"
                ) from exc

            if local_round == rounds:
                # Final round: return as-is so the caller's existing
                # grading/extraction/save logic runs on it unchanged.
                return exec_result

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            if status != "success" or not transcript:
                # A warm-up round failed — surface it exactly like a normal
                # single-round failure would: return it as if it were the
                # final result, so the caller's existing status check
                # records the failure/timeout for this task.
                return exec_result

            agent_response = _extract_transcript_text(transcript)
            if not agent_response.strip():
                raise RuntimeError(
                    f"Task execution produced no extractable solution for "
                    f"{task.task_id} (local reflect round {local_round}/{rounds})"
                )

            extraction = self._apply_reflection_and_extraction(
                task_prompt=task.prompt, agent_response=agent_response,
            )
            insight_book = extraction.get("insight_book", {})
            if not insight_book:
                raise RuntimeError(
                    f"Insight extraction returned an empty library for "
                    f"{task.task_id} (local reflect round {local_round}/{rounds})"
                )
            local_insight_book.update(insight_book)
            current_path = self._write_local_reflect_library(
                base_insight_path=base_insight_path,
                local_insight_book=local_insight_book,
                task=task,
                local_round=local_round,
            )

        return exec_result

    def _load_tasks(self, *, exclude_v1: Optional[bool] = None) -> list:
        loader = TaskLoader(self.tasks_dir)
        tasks = loader.load_all_tasks()

        if self.suite == "all":
            selected = tasks
        elif self.suite == "automated-only":
            selected = [t for t in tasks if t.grading_type == "automated"]
        else:
            # Comma-separated list of task IDs
            ids = {tid.strip() for tid in self.suite.split(",") if tid.strip()}
            selected = [t for t in tasks if t.task_id in ids]

        should_exclude_v1 = self.exclude_v1 if exclude_v1 is None else exclude_v1
        if self.v1_only:
            before = len(selected)
            selected = [t for t in selected if t.task_id in _PINCHBENCH_V1_TASK_IDS]
            print(
                f"Selected {len(selected)} PinchBench V1 tasks "
                f"from {before} suite tasks ({len(_PINCHBENCH_V1_TASK_IDS)} configured)"
            )
        elif should_exclude_v1:
            before = len(selected)
            selected = [t for t in selected if t.task_id not in _PINCHBENCH_V1_TASK_IDS]
            excluded = before - len(selected)
            print(
                f"Excluded {excluded} PinchBench V1 tasks "
                f"({len(_PINCHBENCH_V1_TASK_IDS)} configured)"
            )

        return selected

    def _load_v1_train_eval_tasks(self) -> Tuple[List[Any], List[Any]]:
        """Return V1 train tasks and non-V1 eval tasks after applying --suite."""
        selected = self._load_tasks(exclude_v1=False)
        train_tasks = [t for t in selected if t.task_id in _PINCHBENCH_V1_TASK_IDS]
        eval_tasks = [t for t in selected if t.task_id not in _PINCHBENCH_V1_TASK_IDS]
        return train_tasks, eval_tasks

    def _call_for_extraction(
        self,
        prompt: str,
        max_new_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """Call model for reflection/extraction, using ThinkingConfig when set."""
        if self.use_api and self.api_provider == "gemini" and self.thinking_level:
            return call_gemini_thinking(
                api_key=self.api_key,
                model_name=self.api_model,
                prompt=prompt,
                thinking_level=self.thinking_level,
                max_new_tokens=max_new_tokens,
            )
        if self.use_api and self.api_provider == "openrouter":
            return call_openrouter(
                api_key=self.api_key,
                model_name=self.api_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                response_format=response_format,
                reasoning_enabled=False,
            )
        client = self._ensure_client()
        return client._call_model(prompt, max_new_tokens=max_new_tokens)

    @staticmethod
    def _parse_insight_extraction_response(response: str) -> Dict[str, str]:
        """Parse and validate the flat JSON object required by Prompt 3."""
        start = response.find("{")
        if start == -1:
            raise ValueError("response contains no JSON object")

        brace_count = 0
        in_string = False
        escape_next = False
        json_str = None
        for index in range(start, len(response)):
            char = response[index]
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start : index + 1]
                        break

        if json_str is None:
            raise ValueError("response contains no complete JSON object")

        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        raw = json.loads(json_str)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("insight-extraction JSON must be a non-empty object")

        insights: Dict[str, str] = {}
        invalid_fields = []
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                invalid_fields.append(str(key))
                continue
            name = key if key.startswith("insight_") else f"insight_{key}"
            description = re.sub(r"\s+", " ", value).strip()
            if len(description) < 20:
                invalid_fields.append(key)
                continue
            insights[name] = description

        if invalid_fields:
            raise ValueError(
                "invalid or too-short insight fields: "
                + ", ".join(invalid_fields[:5])
            )
        if not insights:
            raise ValueError("no valid reasoning traces were extracted")
        return insights

    @staticmethod
    def _raw_insight_fallback(response: str) -> Optional[str]:
        """Return a usable verbatim fallback, never an empty/None library value."""
        if not isinstance(response, str):
            return None
        text = response.strip()
        if not text:
            return None
        unfenced = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE
        ).strip()
        if unfenced.lower() in {"", "{}", "null", "none"}:
            return None
        return text

    def _apply_reflection_and_extraction(
        self, task_prompt: str, agent_response: str
    ) -> Dict[str, Any]:
        """
        Apply Steps 2 & 3 from client.py (reflection + insight extraction)
        to the agent's response transcript.

        Returns the insight_book dict extracted from the agent's solution.
        """
        client = self._ensure_client()

        # Step 2: Reflection
        print("  Step 2: Reflecting on agent solution...")
        reflection_prompt = client._get_reflection_prompt(task_prompt, agent_response)
        reflection, reflection_token_info = self._call_for_extraction(
            reflection_prompt, max_new_tokens=8192
        )
        if not isinstance(reflection, str) or not reflection.strip():
            raise RuntimeError(
                "Reflection model returned empty content "
                f"(finish_reason={reflection_token_info.get('finish_reason')!r}, "
                f"output_tokens={reflection_token_info.get('output_tokens', 0)}, "
                f"reasoning_chars={reflection_token_info.get('reasoning_chars', 0)})"
            )
        print(f"  Reflection length: {len(reflection)} chars")

        # Step 3: Insight extraction
        print("  Step 3: Extracting reasoning traces...")
        behavior_prompt = client._get_behavior_prompt(task_prompt, agent_response, reflection)
        extraction_output_tokens = 0
        insights: Dict[str, str] = {}
        extraction_error: Optional[Exception] = None
        raw_fallback_candidates: List[str] = []
        extraction_attempts = 2
        for attempt in range(1, extraction_attempts + 1):
            retry_instruction = ""
            if attempt > 1:
                retry_instruction = (
                    "\n\nCRITICAL JSON REGENERATION: The prior response was invalid. "
                    "Regenerate the complete answer from the source material above. "
                    "Return exactly one non-empty flat JSON object with trace names as "
                    "keys and strings as values. Escape every double quote inside a "
                    "description as \\\". Do not use markdown or nested values."
                )
                print(
                    f"  Retrying insight extraction after invalid JSON "
                    f"({attempt}/{extraction_attempts})"
                )
            extraction_response, token_info = self._call_for_extraction(
                behavior_prompt + retry_instruction,
                max_new_tokens=8192,
                response_format={"type": "json_object"},
            )
            extraction_output_tokens += _generated_output_tokens(token_info)
            raw_fallback = self._raw_insight_fallback(extraction_response)
            if raw_fallback is not None:
                raw_fallback_candidates.append(raw_fallback)
            try:
                if (
                    not isinstance(extraction_response, str)
                    or not extraction_response.strip()
                ):
                    raise ValueError("model returned empty content")
                insights = self._parse_insight_extraction_response(
                    extraction_response
                )
                extraction_error = None
                break
            except (ValueError, json.JSONDecodeError) as exc:
                extraction_error = exc

        if extraction_error is not None:
            if raw_fallback_candidates:
                complete_raw_response = max(
                    raw_fallback_candidates, key=len
                )
                insights = {
                    "insight_unparsed_extraction_response": complete_raw_response
                }
                print(
                    "  Warning: extraction JSON remained invalid; preserving the "
                    "complete non-empty response as one insight-library string"
                )
            else:
                raise RuntimeError(
                    "Could not parse insight-extraction JSON after "
                    f"{extraction_attempts} attempts: {extraction_error}"
                ) from extraction_error

        print(f"  Extracted {len(insights)} insights")
        reflection_output_tokens = _generated_output_tokens(reflection_token_info)
        trace_extraction_output_tokens = extraction_output_tokens
        return {
            "insight_book": insights,
            "output_tokens": reflection_output_tokens + trace_extraction_output_tokens,
            "output_token_breakdown": {
                "reflection": reflection_output_tokens,
                "trace_extraction": trace_extraction_output_tokens,
            },
        }

    def _create_trace_insight_library(self) -> Optional[str]:
        """Create an INSIGHTS-style JSON file from trace folder reasoning traces."""
        if not self.trace_folder:
            return None

        trace_path = Path(self.trace_folder)
        if not trace_path.exists():
            raise RuntimeError(f"Trace folder not found: {trace_path}")

        trace_files = sorted(trace_path.rglob("problem*.json")) + sorted(trace_path.rglob("paper*.json"))
        if not trace_files:
            raise RuntimeError(f"No trace files found under {trace_path}")

        insight_items: Dict[str, str] = {}
        for idx, trace_file in enumerate(trace_files, start=1):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read trace file {trace_file}: {exc}"
                ) from exc

            problem_text = trace_data.get("problem") or trace_data.get("task_prompt") or trace_data.get("question") or ""
            solution_text = trace_data.get("solution") or trace_data.get("reasoning") or trace_data.get("agent_response") or ""
            if not solution_text:
                raise RuntimeError(f"Trace file has no solution/reasoning: {trace_file}")

            key = f"trace_{idx:04d}"
            desc = (
                f"Source: {trace_file.relative_to(trace_path)}\n"
                f"Problem: {problem_text}\n\n"
                f"Reasoning Trace:\n{solution_text}"
            )
            insight_items[key] = desc

        if not insight_items:
            raise RuntimeError(f"No usable trace content found in {trace_path}")

        trace_insights_path = Path(self.output_dir) / "trace_insights.json"
        with open(trace_insights_path, "w", encoding="utf-8") as f:
            json.dump(insight_items, f, indent=2, ensure_ascii=False)

        print(f"Created trace insight library: {trace_insights_path} ({len(insight_items)} entries)")
        return str(trace_insights_path)

    def run_tasks_and_extract(
        self,
        run_id: Optional[str] = None,
        tasks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step 1 + Steps 2/3: For each pinchbench task, run the OpenClaw agent
        and extract reasoning traces from the transcript.

        Saves each task's insights as problem_XXXX.json in output_dir.
        Returns a list of per-task result dicts.
        """
        tasks = tasks if tasks is not None else self._load_tasks()
        if not tasks:
            raise RuntimeError(f"No tasks found in {self.tasks_dir}")
        if self.num_workers > 1 and len(tasks) > 1:
            return self._run_tasks_and_extract_parallel(tasks, run_id)

        # Write trace folder content or encyclopedia as INSIGHTS.md into the agent workspace.
        # prepare_task_workspace() preserves this file and injects a
        # mandatory read instruction into BOOTSTRAP.md for every task.
        trace_insights_path = self._create_trace_insight_library()
        active_insight_path = trace_insights_path or self.encyclopedia_path
        self._validate_resume_phase(
            tasks,
            phase="training",
            insight_library_path=active_insight_path,
        )
        all_tasks_checkpointed = all(
            self._resume_task_checkpoint(task=task, require_insights=True)
            is not None
            for task in tasks
        )
        if all_tasks_checkpointed:
            print("\nAll training tasks have valid checkpoints; agent setup is skipped")
        else:
            self._setup_agent()
            if self.agent_backend == "direct":
                if active_insight_path:
                    print(f"\nInjecting insight library into direct agent: {active_insight_path}")
                else:
                    print("\nNo encyclopedia or trace-folder insights — running without prior insights")
            elif trace_insights_path:
                print(f"\nWriting trace-folder insights to agent workspace as INSIGHTS.md")
                _write_insights_to_workspace(self.agent_id, trace_insights_path, self._agent_workspace)
            elif self.encyclopedia_path and Path(self.encyclopedia_path).exists():
                print(f"\nWriting encyclopedia to agent workspace as INSIGHTS.md")
                _write_insights_to_workspace(self.agent_id, self.encyclopedia_path, self._agent_workspace)
            else:
                if self.trace_folder:
                    print(f"\nTrace folder configured but no usable trace insights were created from {self.trace_folder}")
                print("\nNo encyclopedia or trace-folder insights — running without prior insights")

        print(f"\nLoaded {len(tasks)} tasks (suite='{self.suite}')")
        print("=" * 80)

        if run_id is None:
            run_id = f"fot_{int(time.time())}"

        results = []
        task_metrics: List[Dict[str, Any]] = []

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Task: {task.task_id} — {task.name}")
            print("-" * 60)

            resumed = self._resume_task_checkpoint(
                task=task,
                require_insights=True,
            )
            if resumed is not None:
                task_metric, task_result = resumed
                task_metrics.append(task_metric)
                results.append(task_result)
                print(
                    f"  Resumed completed checkpoint: status={task_result['status']}, "
                    f"score={task_result['score']}/{task_result['max_score']}"
                )
                continue

            # -----------------------------------------------------------------
            # Step 1: Execute the task with the OpenClaw agent (optionally
            # preceded by --local-reflect-round warm-up attempts on this same
            # question; see _execute_task_with_local_reflection)
            # -----------------------------------------------------------------
            try:
                exec_result = self._execute_task_with_local_reflection(
                    task=task,
                    run_id_prefix=f"{run_id}-{i}",
                    base_insight_path=active_insight_path,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Task execution failed for {task.task_id}: {exc}"
                ) from exc

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            print(f"  Execution status: {status} | transcript entries: {len(transcript)}")
            if status != "success":
                command = exec_result.get("command")
                if command:
                    print(f"  OpenClaw command: {command}")
                print(f"  Exit code: {exec_result.get('exit_code')} | timed_out: {exec_result.get('timed_out')}")
                stdout = (exec_result.get("stdout") or "").strip()
                stderr = (exec_result.get("stderr") or "").strip()
                if stdout:
                    print(f"  Stdout (first 500 chars): {stdout[:500]}")
                if stderr:
                    print(f"  Stderr (first 500 chars): {stderr[:500]}")
                if status == "timeout" or exec_result.get("timed_out"):
                    task_metric, timeout_result = self._record_timed_out_task(
                        task=task,
                        exec_result=exec_result,
                    )
                    task_metrics.append(task_metric)
                    results.append(timeout_result)
                    self._write_metrics_log(task_metrics)
                    print("  Marked failed with score 0/1; continuing to the next task")
                    continue
                if self._is_fatal_agent_execution_error(exec_result):
                    raise RuntimeError(
                        f"Task execution returned systemic status={status!r} "
                        f"for {task.task_id}: "
                        f"{stderr or stdout or 'no error detail'}"
                    )
                task_metric, failure_result = self._record_agent_execution_error(
                    task=task,
                    exec_result=exec_result,
                )
                task_metrics.append(task_metric)
                results.append(failure_result)
                self._write_metrics_log(task_metrics)
                print(
                    "  Agent failed before producing a solution; "
                    "recorded score 0/1 and continuing to the next task"
                )
                continue

            if not transcript:
                raise RuntimeError(
                    f"Task execution produced an empty transcript for {task.task_id}"
                )

            # -----------------------------------------------------------------
            # Grade the task (automated or LLM-judge using the same model)
            # -----------------------------------------------------------------
            try:
                grade = grade_task(
                    task=task,
                    execution_result=exec_result,
                    skill_dir=self.skill_dir,
                    judge_model=self.judge_model,
                    judge_backend="api",
                    judge_provider=self.judge_provider,
                    judge_api_key=self.judge_api_key,
                )
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({score_pct:.0f}%)"
                      f" [{grade.grading_type}]")
                if grade.breakdown:
                    for criterion, val in grade.breakdown.items():
                        print(f"    {criterion}: {val}")
                if grade.notes:
                    print(f"  Notes: {grade.notes}")
            except Exception as exc:
                raise RuntimeError(f"Grading failed for {task.task_id}: {exc}") from exc

            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
            extraction_output_tokens = 0
            execution_time_seconds = exec_result.get("execution_time")
            if execution_time_seconds is not None:
                execution_time_seconds = float(execution_time_seconds)

            tool_stats = _analyze_tool_calls(transcript)
            task_metric = {
                "task_id": task.task_id,
                "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": execution_time_seconds,
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (
                        (grade.score / grade.max_score * 100.0)
                        if grade and grade.max_score > 0
                        else None
                    ),
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {
                    "agent": agent_output_tokens,
                    "extraction": extraction_output_tokens,
                    "total": agent_output_tokens,
                },
                "agent_usage": {
                    "request_count": int(usage.get("request_count", 0) or 0),
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": agent_output_tokens,
                    "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
                    "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
                    "processed_tokens": int(usage.get("total_tokens", 0) or 0),
                    "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
                },
                "tools": {
                    "names": tool_stats.get("tool_names", []),
                    "name_counts": tool_stats.get("tool_name_counts", {}),
                    "total_calls": tool_stats.get("total_tool_calls", 0),
                    "successful_calls": tool_stats.get("successful_tool_calls", 0),
                    "error_calls": tool_stats.get("error_tool_calls", 0),
                    "unknown_status_calls": tool_stats.get("unknown_status_tool_calls", 0),
                    "calls": tool_stats.get("calls", []),
                },
                "insights_extracted": 0,
            }

            # -----------------------------------------------------------------
            # Extract readable solution text from transcript
            # -----------------------------------------------------------------
            agent_response = _extract_transcript_text(transcript)
            if not agent_response.strip():
                raise RuntimeError(
                    f"Task transcript contains no extractable solution for {task.task_id}"
                )

            # -----------------------------------------------------------------
            # Steps 2 & 3: Reflection + insight extraction
            # -----------------------------------------------------------------
            try:
                extraction = self._apply_reflection_and_extraction(
                    task_prompt=task.prompt,
                    agent_response=agent_response,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Insight extraction failed for {task.task_id}: {exc}"
                ) from exc

            insight_book = extraction.get("insight_book", {})
            if not insight_book:
                raise RuntimeError(
                    f"Insight extraction returned an empty library for {task.task_id}"
                )

            extraction_output_tokens = int(extraction.get("output_tokens", 0) or 0)
            extraction_breakdown = extraction.get("output_token_breakdown", {}) or {}
            task_metric["output_tokens"] = {
                "agent": agent_output_tokens,
                "extraction": extraction_output_tokens,
                "reflection": int(extraction_breakdown.get("reflection", 0) or 0),
                "trace_extraction": int(
                    extraction_breakdown.get("trace_extraction", 0) or 0
                ),
                "total": agent_output_tokens + extraction_output_tokens,
            }
            task_metric["insights_extracted"] = len(insight_book)

            # -----------------------------------------------------------------
            # Save as problem_XXXX.json (matches server_text expected format)
            # -----------------------------------------------------------------
            output_file = str(self._checkpoint_output_path(task.task_id, i))
            grade_info = {}
            if grade is not None:
                grade_info = {
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "grading_type": grade.grading_type,
                    "notes": grade.notes,
                }

            save_data = {
                "task_id": task.task_id,
                "task_name": task.name,
                "task_prompt": task.prompt,
                "execution_status": status,
                "output_tokens": task_metric["output_tokens"],
                "grade": grade_info,
                "insight_book": insight_book,
            }
            self._write_json_atomic(Path(output_file), save_data)

            print(f"  Saved {len(insight_book)} insights → {output_file}")
            results.append({
                "task_id": task.task_id,
                "task_name": task.name,
                "status": status,
                "score": grade.score if grade else None,
                "max_score": grade.max_score if grade else None,
                "insights_extracted": len(insight_book),
                "output_file": output_file,
            })
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        print("\n" + "=" * 80)
        total_insights = sum(int(t.get("insights_extracted", 0) or 0) for t in task_metrics)
        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        print(f"Tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            score_sum = sum(float(t["grade"].get("score", 0.0) or 0.0) for t in graded)
            max_sum = sum(float(t["grade"].get("max_score", 0.0) or 0.0) for t in graded)
            overall = score_sum / max_sum * 100 if max_sum > 0 else 0.0
            print(f"Overall score:   {overall:.1f}%  ({len(graded)} tasks graded, judge={self.judge_model})")
        print(f"Total insights extracted: {total_insights}")
        self._write_metrics_log(task_metrics)
        return results

    @staticmethod
    def _collect_summary_server_insights(
        json_files: List[Path],
    ) -> Dict[str, str]:
        """Collect selected trace files for the CoD/compact aggregators.

        The standalone summary servers scan an entire directory.  The pipeline
        has already filtered failed and empty tasks, so preserve that exact file
        selection instead of accidentally including stale checkpoints.
        """
        metadata_keys = {
            "paper_name",
            "problem",
            "problem_id",
            "iteration",
            "is_correct",
            "number_output_tokens",
            "loop_count",
        }
        insights: Dict[str, str] = {}
        counter = 0
        for path in json_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "insight_book" in payload:
                book = payload["insight_book"]
            elif "behavior_book" in payload:
                book = payload["behavior_book"]
            else:
                book = payload
            if not isinstance(book, dict):
                continue
            for name, description in book.items():
                if name in metadata_keys:
                    continue
                counter += 1
                insights[f"{name}_{counter:06d}"] = str(description)
        return insights

    def _aggregate_with_summary_server(
        self,
        json_files: List[Path],
    ) -> Tuple[Dict[str, str], int]:
        """Run the selected CoD or Claude-compact aggregation algorithm."""
        if not self.api_key:
            raise ValueError(
                f"--{self.aggregation_mode} requires an API key for "
                f"{self.api_provider}"
            )

        insights = self._collect_summary_server_insights(json_files)
        if not insights:
            raise RuntimeError(
                f"No insights were available for server_{self.aggregation_mode}"
            )

        model = None
        summary_call = None
        if self.api_provider == "gemini":
            setup_summary_model = (
                server_cod.setup_gemini
                if self.aggregation_mode == "cod"
                else server_claude_compact.setup_gemini
            )
            model = setup_summary_model(
                api_key=self.api_key,
                model_name=self.api_model,
            )
        elif self.api_provider == "openrouter":
            def summary_call(prompt: str, max_output_tokens: int):
                return call_openrouter(
                    api_key=self.api_key,
                    model_name=self.api_model,
                    prompt=prompt,
                    max_new_tokens=max_output_tokens,
                    temperature=0.0,
                    reasoning_enabled=False,
                )
        else:
            raise ValueError(
                f"Unsupported summary API provider: {self.api_provider}"
            )

        if self.aggregation_mode == "cod":
            formatted = server_cod.format_insights_as_text(insights)
            summary, output_tokens = server_cod.cod_summarize_with_chunking(
                formatted,
                model=model,
                chunk_size=self.aggregation_chunk_size,
                call_model=summary_call,
                summary_words=self.cod_summary_words,
            )
        elif self.aggregation_mode == "compact":
            formatted = server_claude_compact.format_insights_as_text(insights)
            summary, output_tokens = (
                server_claude_compact.summarize_with_chunking(
                    formatted,
                    model=model,
                    chunk_size=self.aggregation_chunk_size,
                    call_model=summary_call,
                )
            )
        else:
            raise ValueError(
                f"Unsupported summary aggregation mode: {self.aggregation_mode}"
            )

        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError(
                f"server_{self.aggregation_mode} returned an empty summary"
            )
        encyclopedia = {"insight_summary": summary.strip()}
        # Keep exactly the same flat insight_* -> non-empty text protocol as
        # server_text so every existing library consumer can load this file.
        TextBasedInsightAggregationServer._validate_insight_library(encyclopedia)
        return encyclopedia, int(output_tokens or 0)

    def aggregate_insights(self) -> Optional[str]:
        """
        Aggregate all problem_XXXX.json files with the selected server and save
        the resulting encyclopedia.json.

        Returns the path to the encyclopedia file, or None on failure.
        """
        aggregation_mode = getattr(self, "aggregation_mode", "text")
        aggregation_chunk_size = int(
            getattr(self, "aggregation_chunk_size", 50_000)
        )
        server_label = {
            "text": "server_text",
            "cod": "server_cod",
            "compact": "server_claude_compact",
        }[aggregation_mode]
        print("\n" + "=" * 80)
        print(f"Aggregating Insights via {server_label}")
        print("=" * 80)

        all_json_files = sorted(Path(self.output_dir).glob("problem_*.json"))
        metric_records = self._load_checkpoint_metrics()
        successful_task_ids = {
            task_id
            for task_id, metric in metric_records.items()
            if metric.get("execution_status") == "success"
        }
        json_files = []
        for path in all_json_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict) or not payload.get("insight_book"):
                continue
            if successful_task_ids and str(payload.get("task_id")) not in successful_task_ids:
                continue
            json_files.append(path)
        if not json_files:
            raise RuntimeError("No problem_*.json files found; aggregation cannot run")

        print(f"Found {len(json_files)} insight files")

        input_fingerprint = [
            {
                "file": path.name,
                "sha256": self._file_sha256(str(path)),
            }
            for path in json_files
        ]
        encyclopedia_path = Path(self.output_dir) / "encyclopedia.json"
        checkpoint_path = Path(self.output_dir) / "aggregation_checkpoint.json"
        aggregation_config = {
            "mode": aggregation_mode,
            "chunk_size": (
                aggregation_chunk_size if aggregation_mode != "text" else None
            ),
            "api_provider": self.api_provider,
            "api_model": self.api_model,
        }
        if encyclopedia_path.exists() and checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                encyclopedia_payload = json.loads(
                    encyclopedia_path.read_text(encoding="utf-8")
                )
            except Exception:
                checkpoint = None
                encyclopedia_payload = None
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("version") == 2
                and checkpoint.get("input_fingerprint") == input_fingerprint
                and checkpoint.get("aggregation_config") == aggregation_config
                and checkpoint.get("encyclopedia_sha256")
                == self._file_sha256(str(encyclopedia_path))
                and isinstance(encyclopedia_payload, dict)
                and encyclopedia_payload
            ):
                print(
                    "Aggregation checkpoint matches all insight files; "
                    "reusing existing encyclopedia"
                )
                existing_tasks = list(self._load_checkpoint_metrics().values())
                self._write_metrics_log(
                    existing_tasks,
                    library_output_tokens=int(
                        checkpoint.get("total_output_tokens", 0) or 0
                    ),
                )
                return str(encyclopedia_path)

        if aggregation_mode == "text":
            server = TextBasedInsightAggregationServer(
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                api_model=self.api_model,
                input_dirs=[self.output_dir],
            )

            result = server.aggregate_and_build_encyclopedia(
                json_files=[str(f) for f in json_files],
                output_dir=self.output_dir,
            )

            enc_dict = server._try_parse_json(server.encyclopedia)
            if enc_dict is None:
                enc_dict = server._try_parse_json(
                    server._extract_json_only(server.encyclopedia)
                )
            total_output_tokens = result.get("total_output_tokens", 0)
        else:
            enc_dict, total_output_tokens = self._aggregate_with_summary_server(
                json_files
            )

        if not isinstance(enc_dict, dict) or not enc_dict:
            raise RuntimeError(
                "Aggregation returned an empty or invalid encyclopedia JSON object"
            )
        else:
            self._write_json_atomic(encyclopedia_path, enc_dict)
            print(f"Encyclopedia saved: {encyclopedia_path}")

        self._write_json_atomic(
            checkpoint_path,
            {
                "version": 2,
                "input_fingerprint": input_fingerprint,
                "aggregation_config": aggregation_config,
                "encyclopedia_sha256": self._file_sha256(str(encyclopedia_path)),
                "total_output_tokens": int(total_output_tokens or 0),
            },
        )
        print(f"Insight library output tokens: {total_output_tokens}")
        metrics_key = self._metrics_output_key()
        existing_payload = self._metrics_cache_by_output_dir.get(metrics_key)
        existing_tasks = (
            existing_payload.get("tasks", [])
            if existing_payload
            else list(self._load_checkpoint_metrics().values())
        )
        self._write_metrics_log(existing_tasks, library_output_tokens=int(total_output_tokens or 0))
        return str(encyclopedia_path)

    def run_tasks_eval_only(
        self,
        tasks: List[Any],
        *,
        run_id: Optional[str] = None,
        encyclopedia_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run PinchBench tasks with an existing insight library, but do not extract or aggregate."""
        if not tasks:
            raise RuntimeError("No held-out evaluation tasks found")
        if self.num_workers > 1 and len(tasks) > 1:
            return self._run_tasks_eval_parallel(
                tasks, run_id, encyclopedia_path
            )

        self._validate_resume_phase(
            tasks,
            phase="held_out_eval",
            insight_library_path=(
                getattr(self, "_pooling_manifest_path", None) or encyclopedia_path
            ),
        )
        all_tasks_checkpointed = all(
            self._resume_task_checkpoint(task=task, require_insights=False)
            is not None
            for task in tasks
        )
        if all_tasks_checkpointed:
            print("\nAll held-out tasks have valid checkpoints; agent setup is skipped")
        else:
            self._setup_agent()
            if getattr(self, "_pooling_reference_paths", {}):
                mode = (
                    "RAG retrieval"
                    if getattr(self, "rag", False)
                    else "appended trace pooling"
                )
                print(f"\nUsing task-specific {mode} references")
            elif self.agent_backend == "direct":
                if encyclopedia_path and Path(encyclopedia_path).exists():
                    print(f"\nInjecting split-train encyclopedia into direct agent: {encyclopedia_path}")
                else:
                    print("\nEvaluating direct agent without prior insights")
            elif encyclopedia_path and Path(encyclopedia_path).exists():
                print(f"\nWriting split-train encyclopedia to agent workspace as INSIGHTS.md")
                _write_insights_to_workspace(self.agent_id, encyclopedia_path, self._agent_workspace)
            else:
                # Remove any stale INSIGHTS.md so this run sees no prior insights
                stale = self._agent_workspace / "INSIGHTS.md"
                if stale.exists():
                    stale.unlink()
                    print("\nRemoved stale INSIGHTS.md — evaluating without prior insights")
                else:
                    print("\nNo split-train encyclopedia found — evaluating without prior insights")

        print(f"\nLoaded {len(tasks)} held-out eval tasks")
        print("=" * 80)

        if run_id is None:
            run_id = f"fot_split_eval_{int(time.time())}"

        results = []
        task_metrics: List[Dict[str, Any]] = []

        for i, task in enumerate(tasks, 1):
            print(f"\n[eval {i}/{len(tasks)}] Task: {task.task_id} — {task.name}")
            print("-" * 60)

            resumed = self._resume_task_checkpoint(
                task=task,
                require_insights=False,
            )
            if resumed is not None:
                task_metric, task_result = resumed
                task_metrics.append(task_metric)
                results.append(task_result)
                print(
                    f"  Resumed completed checkpoint: status={task_result['status']}, "
                    f"score={task_result['score']}/{task_result['max_score']}"
                )
                continue

            try:
                active_reference_path = getattr(
                    self, "_pooling_reference_paths", {}
                ).get(
                    str(task.task_id), encyclopedia_path
                )
                if (
                    getattr(self, "_pooling_reference_paths", {})
                    and self.agent_backend == "openclaw"
                    and active_reference_path
                ):
                    _write_insights_to_workspace(
                        self.agent_id,
                        active_reference_path,
                        self._agent_workspace,
                    )
                exec_result = self._execute_task(
                    task=task,
                    run_id=f"{run_id}-{i}",
                    insight_library_path=active_reference_path,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Held-out task execution failed for {task.task_id}: {exc}"
                ) from exc

            status = exec_result.get("status", "error")
            transcript = exec_result.get("transcript", [])
            print(f"  Execution status: {status} | transcript entries: {len(transcript)}")
            if status != "success":
                command = exec_result.get("command")
                if command:
                    print(f"  OpenClaw command: {command}")
                print(f"  Exit code: {exec_result.get('exit_code')} | timed_out: {exec_result.get('timed_out')}")
                stdout = (exec_result.get("stdout") or "").strip()
                stderr = (exec_result.get("stderr") or "").strip()
                if stdout:
                    print(f"  Stdout (first 500 chars): {stdout[:500]}")
                if stderr:
                    print(f"  Stderr (first 500 chars): {stderr[:500]}")
                if status == "timeout" or exec_result.get("timed_out"):
                    output_file = str(self._checkpoint_output_path(task.task_id, i))
                    task_metric, timeout_result = self._record_timed_out_task(
                        task=task,
                        exec_result=exec_result,
                        output_file=output_file,
                        eval_only=True,
                    )
                    task_metrics.append(task_metric)
                    results.append(timeout_result)
                    self._write_metrics_log(task_metrics)
                    print("  Marked failed with score 0/1; continuing to the next task")
                    continue
                if self._is_fatal_agent_execution_error(exec_result):
                    raise RuntimeError(
                        f"Held-out task returned systemic status={status!r} "
                        f"for {task.task_id}: "
                        f"{stderr or stdout or 'no error detail'}"
                    )
                output_file = str(self._checkpoint_output_path(task.task_id, i))
                task_metric, failure_result = self._record_agent_execution_error(
                    task=task,
                    exec_result=exec_result,
                    output_file=output_file,
                    eval_only=True,
                )
                task_metrics.append(task_metric)
                results.append(failure_result)
                self._write_metrics_log(task_metrics)
                print(
                    "  Agent failed before producing a solution; "
                    "recorded score 0/1 and continuing to the next task"
                )
                continue

            if not transcript:
                raise RuntimeError(
                    f"Held-out task produced an empty transcript for {task.task_id}"
                )
            if not _extract_transcript_text(transcript).strip():
                raise RuntimeError(
                    f"Held-out task transcript contains no solution for {task.task_id}"
                )

            try:
                grade = grade_task(
                    task=task,
                    execution_result=exec_result,
                    skill_dir=self.skill_dir,
                    judge_model=self.judge_model,
                    judge_backend="api",
                    judge_provider=self.judge_provider,
                    judge_api_key=self.judge_api_key,
                )
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                print(f"  Grade: {grade.score:.2f}/{grade.max_score:.2f} ({score_pct:.0f}%)"
                      f" [{grade.grading_type}]")
                if grade.breakdown:
                    for criterion, val in grade.breakdown.items():
                        print(f"    {criterion}: {val}")
                if grade.notes:
                    print(f"  Notes: {grade.notes}")
            except Exception as exc:
                raise RuntimeError(
                    f"Held-out grading failed for {task.task_id}: {exc}"
                ) from exc

            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
            execution_time_seconds = exec_result.get("execution_time")
            if execution_time_seconds is not None:
                execution_time_seconds = float(execution_time_seconds)

            tool_stats = _analyze_tool_calls(transcript)
            task_metric = {
                "task_id": task.task_id,
                "task_name": task.name,
                "execution_status": status,
                "execution_time_seconds": execution_time_seconds,
                "grade": {
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "accuracy_pct": (
                        (grade.score / grade.max_score * 100.0)
                        if grade and grade.max_score > 0
                        else None
                    ),
                    "grading_type": grade.grading_type if grade else task.grading_type,
                    "notes": grade.notes if grade else "",
                },
                "output_tokens": {
                    "agent": agent_output_tokens,
                    "extraction": 0,
                    "total": agent_output_tokens,
                },
                "agent_usage": {
                    "request_count": int(usage.get("request_count", 0) or 0),
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": agent_output_tokens,
                    "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
                    "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
                    "processed_tokens": int(usage.get("total_tokens", 0) or 0),
                    "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
                },
                "tools": {
                    "names": tool_stats.get("tool_names", []),
                    "name_counts": tool_stats.get("tool_name_counts", {}),
                    "total_calls": tool_stats.get("total_tool_calls", 0),
                    "successful_calls": tool_stats.get("successful_tool_calls", 0),
                    "error_calls": tool_stats.get("error_tool_calls", 0),
                    "unknown_status_calls": tool_stats.get("unknown_status_tool_calls", 0),
                    "calls": tool_stats.get("calls", []),
                },
                "insights_extracted": 0,
            }

            output_file = str(self._checkpoint_output_path(task.task_id, i))
            grade_info = {}
            if grade is not None:
                grade_info = {
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "grading_type": grade.grading_type,
                    "notes": grade.notes,
                }
            save_data = {
                "task_id": task.task_id,
                "task_name": task.name,
                "task_prompt": task.prompt,
                "execution_status": status,
                "output_tokens": agent_output_tokens,
                "grade": grade_info,
                "eval_only": True,
            }
            self._write_json_atomic(Path(output_file), save_data)

            results.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "status": status,
                    "score": grade.score if grade else None,
                    "max_score": grade.max_score if grade else None,
                    "output_file": output_file,
                }
            )
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        graded = [t for t in task_metrics if t.get("grade", {}).get("score") is not None]
        print("\n" + "=" * 80)
        print(f"Held-out eval tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            score_sum = sum(float(t["grade"].get("score", 0.0) or 0.0) for t in graded)
            max_sum = sum(float(t["grade"].get("max_score", 0.0) or 0.0) for t in graded)
            overall = score_sum / max_sum * 100 if max_sum > 0 else 0.0
            print(f"Held-out eval score: {overall:.1f}% ({len(graded)} tasks graded)")
        self._write_metrics_log(task_metrics)
        return results

    def _prepare_pooling_iteration(
        self,
        *,
        tasks: List[Any],
        iteration: int,
        iteration_output_dir: Path,
    ) -> Dict[str, Any]:
        if not self.pooling_dir:
            raise ValueError("--pooling-dir is required with --pooling")
        individual_library_pooling = False
        if self.pooling_raw_transcript:
            traces, source_iterations = load_cumulative_raw_transcript_traces(
                self.pooling_dir, iteration
            )
        else:
            source_iteration = resolve_iteration_dir(self.pooling_dir, iteration)
            source_iterations = [source_iteration]
            # Auto-detect the per-client library format rather than gating on
            # --individual: real --individual training runs vary in shape
            # (some write appended_individual_encyclopedia.json, others only
            # ever produced flat problem_*.json per client), so pick whichever
            # this round's directory actually has.
            individual_library_pooling = (
                source_iteration / "appended_individual_encyclopedia.json"
            ).is_file()
            traces = (
                load_individual_library_traces(
                    source_iteration, limit=self.participate
                )
                if individual_library_pooling
                else load_reasoning_traces(source_iteration, limit=self.participate)
            )
        task_prompts = {
            str(task.task_id): str(task.prompt)
            for task in tasks
        }
        reference_paths, manifest = build_pooling_references(
            traces=traces,
            task_prompts=task_prompts,
            output_dir=iteration_output_dir,
            context_window=self.pooling_context_window,
            rag=self.rag,
            rag_api_key=self.rag_api_key,
            rag_embedding_model=self.rag_embedding_model,
        )
        manifest["source_iterations"] = [
            str(source_iteration.resolve()) for source_iteration in source_iterations
        ]
        manifest["source_type"] = (
            "individual_client_encyclopedia"
            if individual_library_pooling
            else (
                "raw_transcript_jsonl"
                if self.pooling_raw_transcript
                else "extracted_reasoning_trace"
            )
        )
        manifest["source_scope"] = (
            "cumulative_through_iteration"
            if self.pooling_raw_transcript
            else (
                "all_clients_in_matching_iteration"
                if individual_library_pooling
                else "matching_iteration"
            )
        )
        manifest["participate"] = self.participate
        manifest_path = Path(manifest["manifest_path"])
        self._write_json_atomic(manifest_path, manifest)
        self._pooling_reference_paths = reference_paths
        # Resume compatibility must depend only on experiment inputs, not on
        # whether embeddings happened to come from the local cache this run.
        resume_manifest = {
            key: value
            for key, value in manifest.items()
            if key not in {"manifest_path", "rag_embedding_input_tokens"}
        }
        resume_manifest_path = iteration_output_dir / "pooling_resume_manifest.json"
        self._write_json_atomic(resume_manifest_path, resume_manifest)
        self._pooling_manifest_path = str(resume_manifest_path.resolve())
        print(
            f"Prepared {len(reference_paths)} task-specific pooling references "
            f"from {len(traces)} traces across {len(source_iterations)} "
            "source iteration(s)"
        )
        return manifest

    def _aggregate_participate_round(
        self,
        *,
        iteration: int,
        iteration_output_dir: Path,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Merge the first ``--participate`` clients' reasoning traces into one encyclopedia.

        Reads this round's client reasoning traces (post-reflection, never
        raw transcripts) from ``--pooling-dir``, restricts them to the first
        ``self.participate`` clients, and re-runs the selected aggregation
        server (server_text/--cod/--compact) over just those to build one
        merged ``encyclopedia.json`` — reusing ``aggregate_insights`` by
        staging the selected clients' ``problem_*.json`` inputs.

        Auto-detects the client-source format per round, same as
        ``_prepare_pooling_iteration``: a per-client
        ``appended_individual_encyclopedia.json`` library if one exists,
        otherwise the flat ``problem_*.json`` files each training client
        actually produced.
        """
        if not self.pooling_dir:
            raise ValueError("--pooling-dir is required with --participate")
        source_iteration = resolve_iteration_dir(self.pooling_dir, iteration)
        individual_library_pooling = (
            source_iteration / "appended_individual_encyclopedia.json"
        ).is_file()

        scratch_dir = iteration_output_dir / "participate_sources"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        if individual_library_pooling:
            sources = load_individual_library_sources(
                source_iteration, limit=self.participate
            )
            client_labels = [source["client_index"] for source in sources]
            for source in sources:
                problem_path = scratch_dir / f"problem_{source['client_index']:04d}.json"
                self._write_json_atomic(
                    problem_path,
                    {
                        "task_id": source["task_id"],
                        "insight_book": source["encyclopedia"],
                    },
                )
        else:
            problem_files = discover_reasoning_trace_files(
                source_iteration, limit=self.participate
            )
            client_labels = [problem_file.name for problem_file in problem_files]
            for index, problem_file in enumerate(problem_files, 1):
                shutil.copyfile(
                    problem_file, scratch_dir / f"problem_{index:04d}.json"
                )

        original_output_dir = self.output_dir
        try:
            self.output_dir = str(scratch_dir)
            encyclopedia_path = self.aggregate_insights()
        finally:
            self.output_dir = original_output_dir

        manifest = {
            "participate": self.participate,
            "source_iteration": str(source_iteration.resolve()),
            "source_type": (
                "individual_client_encyclopedia"
                if individual_library_pooling
                else "extracted_reasoning_trace"
            ),
            "clients": client_labels,
            "client_count": len(client_labels),
            "encyclopedia_path": encyclopedia_path,
        }
        manifest_path = iteration_output_dir / "participate_manifest.json"
        self._write_json_atomic(manifest_path, manifest)
        print(
            f"Aggregated {len(client_labels)} client reasoning trace(s) "
            f"(--participate {self.participate}) from {source_iteration} "
            f"into {encyclopedia_path}"
        )
        return encyclopedia_path, manifest

    def run_eval_only_iterations(
        self,
        tasks: List[Any],
        *,
        iterations: int,
        encyclopedia_path: Optional[str] = None,
    ) -> None:
        """Run independent eval-only rounds, optionally using matched trace pools."""
        if iterations < 1:
            raise ValueError("--iterations must be at least 1")
        start_time = time.time()
        base_output_dir = str(Path(self.output_dir).resolve())
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        original_output_dir = self.output_dir
        original_reference_paths = self._pooling_reference_paths
        original_manifest_path = self._pooling_manifest_path
        history: List[Dict[str, Any]] = []
        try:
            for iteration in range(1, iterations + 1):
                use_iteration_dir = iterations > 1 or self.pooling or bool(self.participate)
                iteration_output_dir = (
                    Path(base_output_dir) / f"iter_{iteration:02d}"
                    if use_iteration_dir
                    else Path(base_output_dir)
                )
                iteration_output_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = str(iteration_output_dir)
                self._pooling_reference_paths = {}
                self._pooling_manifest_path = None
                pooling_manifest = None
                participate_manifest = None
                round_encyclopedia_path = encyclopedia_path
                if self.pooling:
                    pooling_manifest = self._prepare_pooling_iteration(
                        tasks=tasks,
                        iteration=iteration,
                        iteration_output_dir=iteration_output_dir,
                    )
                elif self.participate:
                    round_encyclopedia_path, participate_manifest = (
                        self._aggregate_participate_round(
                            iteration=iteration,
                            iteration_output_dir=iteration_output_dir,
                        )
                    )
                    # Aggregation temporarily repoints self.output_dir at a
                    # scratch folder; restore it for this round's eval.
                    self.output_dir = str(iteration_output_dir)
                print("\n" + "=" * 80)
                print(f"EVAL-ONLY ITERATION {iteration}/{iterations}")
                print("=" * 80)
                self.run_tasks_eval_only(
                    tasks,
                    run_id=f"eval-only-iteration-{iteration}",
                    encyclopedia_path=round_encyclopedia_path,
                )
                history.append(
                    {
                        "iteration": iteration,
                        "output_dir": str(iteration_output_dir.resolve()),
                        "pooling": pooling_manifest,
                        "participate": participate_manifest,
                    }
                )
        finally:
            self.output_dir = original_output_dir
            self._pooling_reference_paths = original_reference_paths
            self._pooling_manifest_path = original_manifest_path

        elapsed = time.time() - start_time
        if iterations > 1 or self.pooling or self.participate:
            self._write_overall_metrics_log(base_output_dir, elapsed)
        self._write_json_atomic(
            Path(base_output_dir) / "eval_only_summary.json",
            {
                "iterations": iterations,
                "pooling": self.pooling,
                "participate": self.participate,
                "rag": self.rag,
                "history": history,
                "elapsed_seconds": elapsed,
            },
        )

    def run_split_pipeline(self, split: float, seed: int) -> None:
        """Run train split through Step 1/2/3, then eval held-out split with Step 1 only."""
        if not 0.0 < split < 1.0:
            raise ValueError("--split must be a float strictly between 0 and 1")

        start_time = time.time()
        base_output_dir = self.output_dir
        tasks = self._load_tasks()
        if len(tasks) < 2:
            raise ValueError("Split mode requires at least two tasks")

        indices = list(range(len(tasks)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        train_size = int(len(indices) * split)
        train_size = max(1, min(train_size, len(indices) - 1))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_tasks = [tasks[i] for i in train_indices]
        eval_tasks = [tasks[i] for i in eval_indices]

        split_manifest = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "suite": self.suite,
            "exclude_v1": self.exclude_v1,
            "total_tasks": len(tasks),
            "train": len(train_tasks),
            "eval": len(eval_tasks),
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "train_tasks": [{"task_id": t.task_id, "name": t.name} for t in train_tasks],
            "eval_tasks": [{"task_id": t.task_id, "name": t.name} for t in eval_tasks],
        }
        os.makedirs(base_output_dir, exist_ok=True)
        split_manifest_path = Path(base_output_dir) / "split_manifest.json"
        split_manifest_path.write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Split manifest saved: {split_manifest_path}")

        train_dir = os.path.join(base_output_dir, "split_train")
        eval_dir = os.path.join(base_output_dir, "split_eval")

        orig_output_dir = self.output_dir
        orig_encyclopedia = self.encyclopedia_path
        try:
            print("\n" + "=" * 80)
            print(f"SPLIT TRAIN: Step 1/2/3 on {len(train_tasks)} tasks ({split:.0%})")
            print("=" * 80)
            self.output_dir = train_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.encyclopedia_path = orig_encyclopedia
            self.run_tasks_and_extract(run_id="split-train", tasks=train_tasks)
            encyclopedia_path = self.aggregate_insights()

            print("\n" + "=" * 80)
            print(f"SPLIT EVAL: Step 1 only on {len(eval_tasks)} held-out tasks")
            print("=" * 80)
            self.output_dir = eval_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.encyclopedia_path = encyclopedia_path
            self.run_tasks_eval_only(
                eval_tasks,
                run_id="split-eval",
                encyclopedia_path=encyclopedia_path,
            )
        finally:
            self.output_dir = orig_output_dir
            self.encyclopedia_path = orig_encyclopedia

        elapsed = time.time() - start_time
        summary = {
            "mode": "split",
            "split": split,
            "seed": seed,
            "train_dir": train_dir,
            "eval_dir": eval_dir,
            "train_encyclopedia": encyclopedia_path if "encyclopedia_path" in locals() else None,
            "elapsed_seconds": elapsed,
        }
        summary_path = Path(base_output_dir) / "split_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 80)
        print("Split Pipeline Complete")
        print("=" * 80)
        print(f"Summary saved: {summary_path}")

    @staticmethod
    def _client_directory_name(index: int, task: Any) -> str:
        task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.task_id)).strip("_")
        return f"client_{index:04d}_{task_slug or 'task'}"

    def _export_client_reasoning_traces(
        self,
        *,
        client_dir: Path,
        task: Any,
    ) -> Optional[str]:
        """Save one client's current extracted traces without server aggregation."""
        for problem_file in sorted(client_dir.glob("problem_*.json")):
            payload = json.loads(problem_file.read_text(encoding="utf-8"))
            if str(payload.get("task_id")) != str(task.task_id):
                continue
            insight_book = payload.get("insight_book")
            if not isinstance(insight_book, dict) or not insight_book:
                raise RuntimeError(
                    f"Client trace checkpoint is empty for {task.task_id}: {problem_file}"
                )
            trace_path = client_dir / "reasoning_traces.json"
            self._write_json_atomic(trace_path, insight_book)
            return str(trace_path.resolve())
        return None

    def _write_appended_individual_encyclopedia(
        self,
        *,
        entries: List[Dict[str, Any]],
        output_path: Path,
        scope: str,
    ) -> Dict[str, Any]:
        """Append independent client encyclopedias without merging their keys.

        Each source library remains a separate object so identically named
        insights from different clients or rounds are preserved.  The resulting
        JSON file can be injected directly by both the PinchBench direct agent
        and the ClawEval runner.
        """
        sources: List[Dict[str, Any]] = []
        total_insights = 0
        for entry in entries:
            state_path = entry.get("state")
            if not state_path:
                continue
            source_path = Path(str(state_path))
            if not source_path.is_file():
                raise RuntimeError(
                    f"Individual encyclopedia does not exist: {source_path}"
                )
            try:
                encyclopedia = json.loads(source_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not parse individual encyclopedia {source_path}: {exc}"
                ) from exc
            if not isinstance(encyclopedia, dict) or not encyclopedia:
                raise RuntimeError(
                    f"Individual encyclopedia is empty or invalid: {source_path}"
                )
            total_insights += len(encyclopedia)
            sources.append(
                {
                    "iteration": int(entry["iteration"]),
                    "client_index": int(entry["client_index"]),
                    "task_id": str(entry["task_id"]),
                    "task_name": str(entry["task_name"]),
                    "state_updated_this_iteration": bool(
                        entry.get("state_updated", True)
                    ),
                    "source_path": str(source_path.resolve()),
                    "insight_count": len(encyclopedia),
                    "encyclopedia": encyclopedia,
                }
            )

        if not sources:
            raise RuntimeError(
                f"No individual encyclopedias were available for {scope}"
            )

        payload = {
            "format": "pinchbench-appended-individual-encyclopedia-v1",
            "scope": scope,
            "merge_policy": (
                "lossless_append; each client/iteration encyclopedia is kept "
                "separate and no insight keys are deduplicated"
            ),
            "source_count": len(sources),
            "insight_count": total_insights,
            "sources": sources,
        }
        self._write_json_atomic(output_path, payload)
        return {
            "path": str(output_path.resolve()),
            "source_count": len(sources),
            "insight_count": total_insights,
            "size_bytes": output_path.stat().st_size,
        }

    def run_per_client_v1_pipeline(
        self,
        *,
        mode: str,
        iterations: int = 1,
        start_from_step2: bool = False,
    ) -> None:
        """Run persistent V1 clients without sharing traces across clients.

        ``isolated`` reuses each client's most recent extracted reasoning traces
        directly. ``individual`` invokes one independent server aggregation per
        client and reuses only that client's resulting encyclopedia.  Individual
        mode also writes losslessly appended per-round and all-round
        encyclopedias for later, separately invoked evaluation commands.
        """
        if mode not in {"isolated", "individual"}:
            raise ValueError(f"Unsupported per-client mode: {mode}")
        if iterations < 1:
            raise ValueError("--iterations must be at least 1")
        if self.encyclopedia_path:
            raise ValueError(
                f"--encyclopedia cannot be combined with --{mode}; "
                "each client must begin without shared guidance"
            )

        all_selected = self._load_tasks(exclude_v1=False)
        client_tasks = [
            task
            for task in all_selected
            if task.task_id in _PINCHBENCH_V1_TASK_IDS
        ]
        if not client_tasks:
            raise ValueError(f"No V1 client tasks found for --{mode} mode")

        start_time = time.time()
        base_output_dir = str(Path(self.output_dir).resolve())
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": f"{mode}_v1_clients",
            "suite": self.suite,
            "iterations": iterations,
            "client_count": len(client_tasks),
            "client_definition": "one selected PinchBench V1 task per client",
            "server_aggregation": mode == "individual",
            "cross_client_trace_collection": False,
            "append_individual_encyclopedias": mode == "individual",
            "non_v1_evaluation": False,
            "curated_skills": (
                list(_PREDEFINED_OPENCLAW_SKILLS)
                if self.openclaw_skill
                else []
            ),
            "clients": [
                {
                    "client_index": index,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "directory": self._client_directory_name(index, task),
                }
                for index, task in enumerate(client_tasks, 1)
            ],
        }
        manifest_path = Path(base_output_dir) / f"{mode}_manifest.json"
        self._write_json_atomic(manifest_path, manifest)
        print(f"{mode.title()} client manifest saved: {manifest_path}")

        if self.openclaw_skill:
            self._ensure_curated_skill_bundle()

        original_output_dir = self.output_dir
        original_encyclopedia = self.encyclopedia_path
        original_trace_folder = self.trace_folder
        client_states: Dict[str, Optional[str]] = {
            str(task.task_id): None for task in client_tasks
        }
        iteration_history: List[Dict[str, Any]] = []

        try:
            for iteration in range(1, iterations + 1):
                iter_dir = Path(base_output_dir) / f"iter_{iteration:02d}"
                clients_root = iter_dir / "clients"
                clients_root.mkdir(parents=True, exist_ok=True)
                print("\n" + "=" * 80)
                print(
                    f"{mode.upper()} V1 CLIENT ROUND {iteration}/{iterations}: "
                    f"{len(client_tasks)} clients, no held-out evaluation"
                )
                print("=" * 80)

                indexed_tasks = list(enumerate(client_tasks, 1))

                def process(item):
                    index, task = item
                    client_id = str(task.task_id)
                    prior_state = client_states.get(client_id)
                    client_dir = clients_root / self._client_directory_name(index, task)
                    client_dir.mkdir(parents=True, exist_ok=True)

                    worker = copy.copy(self)
                    worker.num_workers = 1
                    worker.output_dir = str(client_dir)
                    worker.encyclopedia_path = prior_state
                    worker.trace_folder = None
                    worker.openclaw_skill = False
                    worker._client = None
                    worker._metrics_cache_by_output_dir = {}
                    worker.agent_id = (
                        f"{self.agent_id}-{index:04d}-{slugify_model(client_id)}"[:120]
                    )

                    skip_execution = start_from_step2 and iteration == 1
                    if skip_execution:
                        print(
                            f"[{client_id}] Skipping execution and using existing "
                            "client trace checkpoint (start_from_step2=True)"
                        )
                        results: List[Dict[str, Any]] = []
                    else:
                        results = worker.run_tasks_and_extract(
                            run_id=f"{mode}-client-{index}-round-{iteration}",
                            tasks=[task],
                        )

                    problem_files = sorted(client_dir.glob("problem_*.json"))
                    successful = any(
                        result.get("status") == "success" for result in results
                    )
                    if skip_execution:
                        successful = bool(problem_files)
                    if successful and not problem_files:
                        raise RuntimeError(
                            f"Successful client {client_id} produced no reasoning-trace checkpoint"
                        )

                    new_state: Optional[str] = None
                    if problem_files:
                        if mode == "isolated":
                            new_state = worker._export_client_reasoning_traces(
                                client_dir=client_dir,
                                task=task,
                            )
                        else:
                            new_state = worker.aggregate_insights()
                    elif skip_execution:
                        raise RuntimeError(
                            f"Cannot start from server phase for client {client_id}; "
                            f"no problem_*.json exists in {client_dir}"
                        )

                    active_state = new_state or prior_state
                    metrics_path = client_dir / "metrics_log.json"
                    metrics_payload = (
                        json.loads(metrics_path.read_text(encoding="utf-8"))
                        if metrics_path.exists()
                        else {"tasks": [], "summary": {}}
                    )
                    return {
                        "client_index": index,
                        "task_id": client_id,
                        "task_name": task.name,
                        "client_dir": str(client_dir.resolve()),
                        "prior_state": prior_state,
                        "state": active_state,
                        "state_updated": new_state is not None,
                        "state_kind": (
                            "reasoning_traces"
                            if mode == "isolated"
                            else "encyclopedia"
                        ),
                        "results": results,
                        "metrics": metrics_payload,
                    }

                outcomes = parallel_utils.parallel_map_ordered(
                    process,
                    indexed_tasks,
                    num_workers=self.num_workers,
                )

                merged_metrics: List[Dict[str, Any]] = []
                library_output_tokens = 0
                round_clients = []
                for outcome in outcomes:
                    client_states[outcome["task_id"]] = outcome["state"]
                    merged_metrics.extend(outcome["metrics"].get("tasks", []))
                    library_output_tokens += int(
                        outcome["metrics"].get("summary", {}).get(
                            "library_output_tokens", 0
                        )
                        or 0
                    )
                    round_clients.append({
                        key: outcome[key]
                        for key in (
                            "client_index",
                            "task_id",
                            "task_name",
                            "client_dir",
                            "prior_state",
                            "state",
                            "state_updated",
                            "state_kind",
                        )
                    })

                summary_writer = copy.copy(self)
                summary_writer.output_dir = str(iter_dir)
                summary_writer._metrics_cache_by_output_dir = {}
                summary_writer._write_metrics_log(
                    merged_metrics,
                    library_output_tokens=library_output_tokens,
                )
                round_summary = {
                    "iteration": iteration,
                    "mode": mode,
                    "client_count": len(client_tasks),
                    "client_states_updated": sum(
                        bool(client["state_updated"]) for client in round_clients
                    ),
                    "server_aggregations": (
                        sum(bool(client["state_updated"]) for client in round_clients)
                        if mode == "individual"
                        else 0
                    ),
                    "non_v1_evaluation": False,
                    "clients": round_clients,
                }
                if mode == "individual":
                    round_entries = [
                        {
                            **client,
                            "iteration": iteration,
                        }
                        for client in round_clients
                        if client.get("state")
                    ]
                    round_summary["appended_encyclopedia"] = (
                        self._write_appended_individual_encyclopedia(
                            entries=round_entries,
                            output_path=(
                                iter_dir / "appended_individual_encyclopedia.json"
                            ),
                            scope=f"iteration_{iteration:02d}",
                        )
                    )
                self._write_json_atomic(
                    iter_dir / "client_round_summary.json", round_summary
                )
                iteration_history.append(round_summary)
        finally:
            self.output_dir = original_output_dir
            self.encyclopedia_path = original_encyclopedia
            self.trace_folder = original_trace_folder

        elapsed = time.time() - start_time
        self._write_overall_metrics_log(base_output_dir, elapsed)
        appended_encyclopedia = None
        if mode == "individual":
            cumulative_entries = []
            for round_summary in iteration_history:
                for client in round_summary["clients"]:
                    if client.get("state"):
                        cumulative_entries.append(
                            {
                                **client,
                                "iteration": round_summary["iteration"],
                            }
                        )
            appended_encyclopedia = self._write_appended_individual_encyclopedia(
                entries=cumulative_entries,
                output_path=(
                    Path(base_output_dir)
                    / "appended_individual_encyclopedia_all_iterations.json"
                ),
                scope="all_iterations",
            )
        final_summary = {
            "mode": f"{mode}_v1_clients",
            "iterations": iterations,
            "client_count": len(client_tasks),
            "non_v1_evaluation": False,
            "curated_skills": (
                list(_PREDEFINED_OPENCLAW_SKILLS)
                if self.openclaw_skill
                else []
            ),
            "appended_encyclopedia_all_iterations": appended_encyclopedia,
            "final_client_states": client_states,
            "iteration_history": iteration_history,
            "elapsed_seconds": elapsed,
        }
        summary_path = Path(base_output_dir) / f"{mode}_summary.json"
        self._write_json_atomic(summary_path, final_summary)

        print("\n" + "=" * 80)
        print(f"{mode.title()} V1 Client Pipeline Complete")
        print("=" * 80)
        print(f"Summary saved: {summary_path}")

    def run_individual_v1_pipeline(
        self,
        iterations: int = 1,
        start_from_step2: bool = False,
    ) -> None:
        """Backward-compatible entry point for per-client individual mode."""
        self.run_per_client_v1_pipeline(
            mode="individual",
            iterations=iterations,
            start_from_step2=start_from_step2,
        )

    def run_pipeline(
        self,
        iterations: int = 1,
        start_from_step2: bool = False,
    ) -> None:
        """
        Run the full pipeline for N iterations.

        Iteration 1: no encyclopedia, collect insights, aggregate.
        Iteration 2+: inject encyclopedia, collect insights, aggregate.
        """
        start_time = time.time()
        base_output_dir = self.output_dir

        current_encyclopedia: Optional[str] = self.encyclopedia_path

        for iteration in range(1, iterations + 1):
            iter_label = f"Iteration {iteration}/{iterations}"
            print("\n" + "=" * 80)
            print(iter_label)
            print("=" * 80)

            # Use separate sub-directory per iteration when doing multi-iteration
            if iterations > 1:
                iter_dir = os.path.join(self.output_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                orig_output_dir = self.output_dir
                self.output_dir = iter_dir
            else:
                orig_output_dir = None

            # Set current encyclopedia for this iteration
            if current_encyclopedia:
                self.encyclopedia_path = current_encyclopedia

            if not start_from_step2 or iteration > 1:
                print(f"\n--- Step 1/2/3: Task Execution + Insight Extraction ---")
                self.run_tasks_and_extract()
            else:
                print("Skipping task execution (start_from_step2=True)")

            print(f"\n--- Aggregation ---")
            encyclopedia_path = self.aggregate_insights()

            if encyclopedia_path:
                current_encyclopedia = encyclopedia_path

            if orig_output_dir is not None:
                self.output_dir = orig_output_dir

        elapsed = time.time() - start_time
        if iterations > 1:
            self._write_overall_metrics_log(base_output_dir, elapsed)
        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        print(f"Total time: {elapsed:.1f}s")
        if current_encyclopedia:
            print(f"Final encyclopedia: {current_encyclopedia}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw PinchBench Pipeline — extract reasoning traces from benchmark tasks"
    )
    parallel_utils.add_num_workers_argument(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="OpenClaw model identifier (e.g., anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--agent-backend",
        choices=["openclaw", "direct"],
        default="openclaw",
        help=(
            "Task agent executor: 'openclaw' uses the OpenClaw CLI; 'direct' uses "
            "the repository's OpenAI-compatible tool loop (default: openclaw)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pinchbench_output",
        help="Directory to save insights and encyclopedia (default: pinchbench_output)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        help='Tasks to run: "all", "automated-only", or comma-separated task IDs (default: all)',
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help=(
            "Judge model for LLM-judge tasks. Defaults to the same model as --model. "
            "Use OpenRouter format (e.g. google/gemini-2.5-pro-preview) or an "
            "Anthropic model ID."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "openrouter"],
        default=None,
        help=(
            "API provider used for judging. Defaults to --api-provider. This is "
            "authoritative even when the model name has a provider-like prefix."
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help=(
            "Optional judge API key. Defaults to the selected provider's environment "
            "key, or --api-key when judge and extraction use the same provider."
        ),
    )
    parser.add_argument(
        "--pinchbench-dir",
        type=str,
        default=None,
        help="Path to pinchbench skill root (default: ./pinchbench)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations (default: 1). Iteration 2+ uses the encyclopedia from iteration 1.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help=(
            "Optional train/eval split fraction. If set, randomly use this "
            "fraction of PinchBench tasks for Step 1/2/3 insight generation "
            "and evaluate the remaining held-out tasks with Step 1 only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --split partitioning (default: 42).",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        default=None,
        help="Path to existing encyclopedia.json to inject into agent for the first iteration",
    )
    parser.add_argument(
        "--trace-folder",
        type=str,
        default=None,
        help="Path to a trace folder containing problem*.json or paper*.json for RAG-style agent insights",
    )
    parser.add_argument(
        "--pooling",
        action="store_true",
        help=(
            "Evaluate with reference context from the matching iter_XX under "
            "--pooling-dir. "
            "With --individual, pool that round's individual client libraries."
        ),
    )
    parser.add_argument(
        "--pooling-dir",
        "--pooling_dir",
        "--participate-dir",
        dest="pooling_dir",
        type=str,
        default=None,
        help=(
            "Root containing iter_01, iter_02, ... reasoning-trace "
            "directories. Also accepted as --participate-dir, which reads "
            "identically — use that spelling with --participate when no "
            "--pooling is involved, so the flag name doesn't imply pooling."
        ),
    )
    parser.add_argument(
        "--pooling-raw-transcript",
        metavar="RUN_DIR",
        type=str,
        default=None,
        help=(
            "Evaluate with cumulative semantic RAG over raw transcript JSONL "
            "from a previous run. This implies --eval-only, --pooling, --rag, "
            "and --pooling-dir RUN_DIR. Iteration N uses transcripts from "
            "iter_01 through iter_N."
        ),
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help=(
            "With --pooling, index traces through OpenRouter embeddings and "
            "retrieve at most 4096 estimated tokens per task."
        ),
    )
    parser.add_argument(
        "--rag-api-key",
        type=str,
        default=None,
        help="OpenRouter key for --rag (default: OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--rag-embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenRouter embedding model for --rag (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--pooling-context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Context window used to budget appended pooling references "
            f"(default: {DEFAULT_CONTEXT_WINDOW})."
        ),
    )
    parser.add_argument(
        "--participate",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Restrict --pooling-dir to the first N clients (by client_index) "
            "of each matching iter_XX round, and skip this run's own "
            "training (implies --eval-only). Without --pooling, the N "
            "clients' post-reflection reasoning traces (their individual "
            "encyclopedias, not raw transcripts) are merged into one "
            "encyclopedia via the selected aggregation server "
            "(server_text/--cod/--compact) before eval. With --individual "
            "--pooling, the N clients' individual libraries are instead "
            "pooled directly as reference context (same as --individual "
            "--pooling, capped to N clients)."
        ),
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip task execution and start from aggregation",
    )
    parser.add_argument(
        "--local-reflect-round",
        "--local_reflect_round",
        dest="local_reflect_round",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Per-question warm-up rounds during training (default: 1, i.e. "
            "no warm-up). For N > 1, each question runs the agent, extracts "
            "a reasoning trace from that attempt (Steps 2 & 3), and folds it "
            "into the insight library used for the *next* attempt on the "
            "same question — repeated N times before the final attempt's "
            "result is graded and saved as usual. This is local to one "
            "question within one global round; --iterations still controls "
            "the global round count and is unaffected."
        ),
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use an API provider for reflection/extraction steps",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        default="gemini",
        choices=["gemini", "openrouter"],
        help="Which API provider to use for extraction (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the chosen extraction provider (or set GEMINI_API_KEY / OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Model name for extraction and aggregation (default: gemini-2.5-flash-lite)",
    )
    aggregation_group = parser.add_mutually_exclusive_group()
    aggregation_group.add_argument(
        "--cod",
        action="store_true",
        help=(
            "Aggregate reasoning traces with server_cod.py instead of "
            "server_text.py. Client task execution and trace extraction are unchanged."
        ),
    )
    aggregation_group.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Aggregate reasoning traces with server_claude_compact.py instead of "
            "server_text.py. Client task execution and trace extraction are unchanged."
        ),
    )
    parser.add_argument(
        "--aggregation-chunk-size",
        "--chunk-size",
        dest="aggregation_chunk_size",
        type=int,
        default=50_000,
        help=(
            "Maximum input characters per CoD/compact aggregation chunk "
            "(default: 50000). This does not directly control final library length."
        ),
    )
    parser.add_argument(
        "--cod-summary-words",
        type=int,
        default=server_cod.DEFAULT_SUMMARY_WORDS,
        help=(
            "Target word count for each Chain-of-Density summary when using "
            f"--cod (default: {server_cod.DEFAULT_SUMMARY_WORDS})."
        ),
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="default",
        choices=["default", "low", "medium", "high"],
        help=(
            "Thinking level for models that support reasoning controls. "
            "Use 'default' to omit the reasoning override and keep the provider default. "
            "Choices: default, low, medium, high (default: default). "
            "Uses the new google-genai SDK with types.ThinkingConfig."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible API base URL for the OpenClaw agent",
    )
    parser.add_argument(
        "--openclaw-api-key",
        type=str,
        default=None,
        help="API key for custom OpenClaw agent endpoint (default: $OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Scale all task timeouts (default: 1.0)",
    )
    parser.add_argument(
        "--openclaw-skill",
        action="store_true",
        help=(
            "Pre-install all predefined OpenClaw skills before running tasks. "
            f"Skills installed: {', '.join(_PREDEFINED_OPENCLAW_SKILLS)}. "
            "Each skill is installed with 'openclaw skills install <skill>', "
            "snapshotted into a run-scoped curated bundle, and made available "
            "to PinchBench and ClawEval task prompts."
        ),
    )
    parser.add_argument(
        "--V1-only",
        "--v1-only",
        dest="v1_only",
        action="store_true",
        help="Run only the configured 23-task PinchBench V1 roster.",
    )
    parser.add_argument(
        "--exclude-V1",
        dest="exclude_v1",
        action="store_true",
        help=(
            "Exclude the 23 V1 PinchBench tasks from the Apr 9 "
            "pinchbench/skill tree, using their current renamed task IDs."
        ),
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help=(
            "Run only the selected V1 clients with one independent server "
            "aggregation per client per round. Each client uses only its own "
            "encyclopedia in the next round. Per-round and all-round appended "
            "encyclopedias are written for separate evaluation commands. With "
            "--pooling, it instead selects individual-library evaluation."
        ),
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help=(
            "Run only the selected V1 clients without server aggregation. "
            "Each client uses only its own previous-round reasoning traces as "
            "guidance in the next round; no non-V1 evaluation is run."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Eval-only mode: run tasks and grade them without extracting or aggregating insights. "
            "Use --encyclopedia to inject an existing insight library into the agent workspace."
        ),
    )

    args = parser.parse_args()

    if args.pooling_raw_transcript:
        args.eval_only = True
        args.pooling = True
        args.rag = True
        args.pooling_dir = args.pooling_raw_transcript
    if args.pooling:
        args.eval_only = True
    if args.individual and args.pooling:
        if args.pooling_raw_transcript:
            parser.error(
                "--individual selects encyclopedia pooling and cannot be "
                "combined with --pooling-raw-transcript"
            )
    if args.participate is not None:
        args.eval_only = True

    if args.v1_only and args.exclude_v1:
        parser.error("--V1-only and --exclude-V1 are mutually exclusive")
    if args.individual and args.isolated:
        parser.error("--individual and --isolated are mutually exclusive")
    if (args.individual or args.isolated) and args.encyclopedia:
        parser.error(
            "--encyclopedia cannot be combined with --individual or --isolated"
        )
    if args.pooling and not args.pooling_dir:
        parser.error("--pooling-dir is required with --pooling")
    if args.pooling_dir and not args.pooling and args.participate is None:
        parser.error("--pooling-dir requires --pooling or --participate")
    if args.rag and not args.pooling:
        parser.error("--rag requires --pooling")
    if args.pooling and args.encyclopedia:
        parser.error("--pooling cannot be combined with --encyclopedia")
    if args.pooling and not (args.v1_only or args.exclude_v1):
        parser.error("--pooling requires either --V1-only or --exclude-V1")
    if args.pooling and args.isolated:
        parser.error("--pooling cannot be combined with --isolated")
    if args.participate is not None:
        if args.participate <= 0:
            parser.error("--participate must be a positive integer")
        if not args.pooling_dir:
            parser.error("--participate requires --pooling-dir")
        if args.encyclopedia:
            parser.error("--participate cannot be combined with --encyclopedia")
        if (args.individual or args.isolated) and not args.pooling:
            parser.error(
                "--participate with --individual/--isolated also requires "
                "--pooling (otherwise --participate would be silently "
                "ignored by per-client training mode)"
            )
    if args.pooling_context_window < 1:
        parser.error("--pooling-context-window must be positive")
    if args.aggregation_chunk_size < 1:
        parser.error("--aggregation-chunk-size must be positive")
    if args.cod_summary_words < 1:
        parser.error("--cod-summary-words must be positive")
    if args.local_reflect_round < 1:
        parser.error("--local-reflect-round must be at least 1")
    if (
        not args.model
        and not args.start_from_step2
        and (not args.eval_only or args.individual or args.isolated)
    ):
        parser.error("--model is required unless using --start-from-step2 or --eval-only")

    # Default model for aggregation-only runs
    model_id = args.model or "anthropic/claude-sonnet-4"

    # Judge defaults to the same model as the agent
    judge_model = args.judge or model_id
    rag_api_key = (
        args.rag_api_key
        or os.getenv("OPENROUTER_API_KEY")
        or (args.api_key if args.api_provider == "openrouter" else None)
        or args.openclaw_api_key
    )
    if args.rag and not rag_api_key:
        parser.error(
            "RAG requires --rag-api-key or OPENROUTER_API_KEY"
        )

    pipeline = OpenClawPinchBenchPipeline(
        model_id=model_id,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        base_url=args.base_url,
        openclaw_api_key=args.openclaw_api_key,
        timeout_multiplier=args.timeout_multiplier,
        encyclopedia_path=args.encyclopedia,
        trace_folder=args.trace_folder,
        judge_model=judge_model,
        judge_provider=args.judge_provider,
        judge_api_key=args.judge_api_key,
        thinking_level=(
            None
            if not args.use_api or args.thinking_level == "default"
            else args.thinking_level
        ),
        local_reflect_round=args.local_reflect_round,
        openclaw_skill=args.openclaw_skill,
        v1_only=args.v1_only,
        exclude_v1=args.exclude_v1,
        individual=args.individual,
        isolated=args.isolated,
        pooling=args.pooling,
        pooling_dir=args.pooling_dir,
        pooling_raw_transcript=bool(args.pooling_raw_transcript),
        participate=args.participate,
        rag=args.rag,
        rag_api_key=rag_api_key,
        rag_embedding_model=args.rag_embedding_model,
        pooling_context_window=args.pooling_context_window,
        num_workers=args.num_workers,
        agent_backend=args.agent_backend,
        aggregation_mode=(
            "cod" if args.cod else "compact" if args.compact else "text"
        ),
        aggregation_chunk_size=args.aggregation_chunk_size,
        cod_summary_words=args.cod_summary_words,
    )

    if args.pooling:
        tasks = pipeline._load_tasks()
        pipeline.run_eval_only_iterations(
            tasks,
            iterations=args.iterations,
            encyclopedia_path=args.encyclopedia,
        )
    elif args.individual or args.isolated:
        if args.eval_only:
            print(
                "Warning: per-client V1 mode has no held-out evaluation; "
                "ignoring --eval-only"
            )
        if args.split is not None:
            print("Warning: per-client V1 mode ignores --split")
        pipeline.run_per_client_v1_pipeline(
            mode="individual" if args.individual else "isolated",
            iterations=args.iterations,
            start_from_step2=args.start_from_step2,
        )
    elif args.eval_only:
        tasks = pipeline._load_tasks()
        pipeline.run_eval_only_iterations(
            tasks,
            iterations=args.iterations,
            encyclopedia_path=args.encyclopedia,
        )
    elif args.split is not None:
        pipeline.run_split_pipeline(split=args.split, seed=args.seed)
    else:
        pipeline.run_pipeline(
            iterations=args.iterations,
            start_from_step2=args.start_from_step2,
        )


if __name__ == "__main__":
    main()
