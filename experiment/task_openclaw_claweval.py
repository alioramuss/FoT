"""
OpenClaw ClawEval Pipeline
Runs claw-eval benchmark tasks with an OpenAI-compatible model, extracts
reasoning traces from the JSONL transcripts, and aggregates them into an
insight library via server_text.py.

Structure mirrors task_openclaw_pinchbench.py:

  Iteration N:
    Step 1: Run each task through ClawEval's official in-process runner
            (CLI subprocess only as a fallback) and save its JSONL trace
    Step 2: Reflection — extract procedural knowledge from transcript
    Step 3: Insight extraction — package as reusable traces (JSON)
    Save:   problem_XXXX.json  (same format as other pipelines)

  Aggregation:
    Call server_text.py TextBasedInsightAggregationServer to build an
    encyclopedia from all extracted insights.

  Next Iteration:
    Write the encyclopedia as INSIGHTS.md into each task trace, so the
    model can read it as context in the next iteration.

Usage:
    python task_openclaw_claweval.py \\
        --model google/gemini-3-pro-preview \\
        --output-dir claweval_output \\
        --tag general \\
        --use-api --api-provider gemini --api-key YOUR_KEY \\
        --iterations 2

    # Start from aggregation step (skip task execution):
    python task_openclaw_claweval.py --start-from-step2 \\
        --output-dir claweval_output --use-api --api-provider gemini --api-key YOUR_KEY

    # Start from evaluation with existing encyclopedia:
    python task_openclaw_claweval.py \\
        --model google/gemini-3-pro-preview \\
        --encyclopedia claweval_output/encyclopedia.json \\
        --output-dir claweval_output_iter2 \\
        --use-api --api-provider gemini --api-key YOUR_KEY
"""

import argparse
import collections
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Locate the vendored claw-eval source tree next to this script.
# ---------------------------------------------------------------------------
_CLAWEVAL_DIR = Path(__file__).parent / "claw-eval"
_CLAWEVAL_SRC = _CLAWEVAL_DIR / "src"
if _CLAWEVAL_SRC.exists():
    # Always give the checkout beside this pipeline priority over a separately
    # installed claw-eval.  Merely checking membership is insufficient when an
    # editable/site-packages entry occurs earlier in sys.path.
    _claweval_src_text = str(_CLAWEVAL_SRC.resolve())
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) != _claweval_src_text
    ]
    sys.path.insert(0, _claweval_src_text)

# ---------------------------------------------------------------------------
# Import ClawEval's official per-task runner directly.  This bypasses the CLI
# subprocess, but deliberately keeps ClawEval's task loading, prompt builder,
# tools, services, user-agent simulation, runtime, trace format, and graders.
# ---------------------------------------------------------------------------
try:
    import claw_eval as _claw_eval_package
    from claw_eval.cli import _run_single_task as _run_official_claweval_task
    _HAS_CLAWEVAL_RUNNER = True
    _LOADED_CLAWEVAL_PATH = Path(_claw_eval_package.__file__).resolve()
except ImportError:
    _HAS_CLAWEVAL_RUNNER = False
    _LOADED_CLAWEVAL_PATH = None

# ---------------------------------------------------------------------------
# Import our local pipeline pieces
# ---------------------------------------------------------------------------
from client import ChainOfThoughtReader
import parallel_utils
from server_text import TextBasedInsightAggregationServer
from trace_pooling import (
    DEFAULT_EMBEDDING_MODEL,
    build_pooling_references,
    load_cumulative_raw_transcript_traces,
    load_individual_library_traces,
    load_reasoning_traces,
    resolve_iteration_dir,
)
from utils import call_gemini_thinking


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

_PREDEFINED_OPENCLAW_SKILLS = (
    "nano-pdf",
    "summarize",
    "github",
    "notion",
    "slack",
    "weather",
    "coding-agent",
)


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


def _install_openclaw_skills(output_dir: Path) -> Path:
    """Resolve/install and snapshot the exact curated OpenClaw skill set."""
    openclaw_bin = shutil.which("openclaw") or os.environ.get("OPENCLAW_PATH")
    if not openclaw_bin:
        raise RuntimeError(
            "--openclaw-skill requires the OpenClaw CLI on PATH or OPENCLAW_PATH"
        )
    resolved_sources: Dict[str, Path] = {}
    for skill in _PREDEFINED_OPENCLAW_SKILLS:
        source = _resolve_openclaw_skill_source(openclaw_bin, skill)
        if source:
            resolved_sources[skill] = source
            print(f"  [skill] Available: {skill} ({source})")
            continue
        print(f"  [skill] Installing: {skill}")
        try:
            result = subprocess.run(
                [openclaw_bin, "skills", "install", skill],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(
                    f"  [skill] Install returned {result.returncode} for "
                    f"{skill!r}; checking the existing installation"
                )
        except Exception as exc:
            print(f"  [skill] Install warning for {skill!r}: {exc}")

        source = _resolve_openclaw_skill_source(openclaw_bin, skill)
        if source:
            resolved_sources[skill] = source

    missing = [
        skill
        for skill in _PREDEFINED_OPENCLAW_SKILLS
        if skill not in resolved_sources
    ]
    if missing:
        raise RuntimeError(
            "Installed OpenClaw skills are missing SKILL.md: " + ", ".join(missing)
        )

    destination_root = output_dir / "curated_openclaw_skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_skills = []
    for skill in _PREDEFINED_OPENCLAW_SKILLS:
        source = resolved_sources[skill]
        destination = destination_root / skill
        shutil.copytree(source, destination, dirs_exist_ok=True)
        manifest_skills.append(
            {
                "name": skill,
                "source": str(source),
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


def _check_claw_eval():
    """Verify the vendored ClawEval runner is importable."""
    if _HAS_CLAWEVAL_RUNNER:
        expected_package_dir = (_CLAWEVAL_SRC / "claw_eval").resolve()
        try:
            _LOADED_CLAWEVAL_PATH.relative_to(expected_package_dir)
        except (AttributeError, ValueError):
            raise RuntimeError(
                "Imported claw_eval from the wrong checkout: "
                f"{_LOADED_CLAWEVAL_PATH}; expected it under "
                f"{expected_package_dir}. Restart Python after updating the repo."
            )
        print(f"ClawEval source: {_LOADED_CLAWEVAL_PATH}")
        return
    # Fallback: check if package is on sys.path directly
    try:
        import importlib
        importlib.import_module("claw_eval")
        return
    except ImportError:
        pass

    print(
        "\n" + "=" * 70 + "\n"
        "ERROR: claw-eval package not importable.\n\n"
        "Install the vendored package dependencies:\n"
        "  cd claw-eval\n"
        "  pip install -e .\n"
        "  # or: uv pip install -e .\n\n"
        "The source tree is at:\n"
        f"  {_CLAWEVAL_SRC}\n"
        + "=" * 70 + "\n"
    )
    sys.exit(1)


def _check_docker_sandbox() -> None:
    """Validate Docker once and discover a rootless DS-cluster socket."""
    if not os.environ.get("DOCKER_HOST"):
        socket_candidates = []
        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if xdg_runtime_dir:
            socket_candidates.append(Path(xdg_runtime_dir) / "docker.sock")
        if hasattr(os, "getuid"):
            socket_candidates.append(Path(f"/run/user/{os.getuid()}/docker.sock"))
        socket_candidates.append(Path("/var/run/docker.sock"))
        for socket_path in socket_candidates:
            if socket_path.exists():
                os.environ["DOCKER_HOST"] = f"unix://{socket_path}"
                break

    try:
        import docker

        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:
        docker_host = os.environ.get("DOCKER_HOST", "default /var/run/docker.sock")
        raise RuntimeError(
            "ClawEval Docker sandbox was requested, but no reachable Docker "
            f"daemon is available at {docker_host}: {exc}. On a DS compute "
            "node without Docker, omit --sandbox; the wrapper already enables "
            "ClawEval's local --sandbox-tools mode."
        ) from exc


# ---------------------------------------------------------------------------
# Task discovery (scan tasks/ YAML files, filter by tag)
# ---------------------------------------------------------------------------

def _load_task_yamls(tasks_dir: Path, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scan tasks_dir for subdirectories containing task.yaml.
    Returns a list of task metadata dicts, filtered by tag if given.

    Each dict has at minimum: task_id, task_name, task_dir, tags, prompt, category.
    """
    tasks = []
    for entry in sorted(tasks_dir.iterdir()):
        yaml_path = entry / "task.yaml"
        if not entry.is_dir() or not yaml_path.exists():
            continue
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue

            task_tags = data.get("tags") or []
            if isinstance(task_tags, str):
                task_tags = [task_tags]

            if tag and tag not in task_tags:
                continue

            prompt_field = data.get("prompt") or {}
            if isinstance(prompt_field, dict):
                prompt_text = prompt_field.get("text") or ""
            else:
                prompt_text = str(prompt_field)

            tasks.append({
                "task_id": data.get("task_id") or entry.name,
                "task_name": data.get("task_name") or data.get("name") or entry.name,
                "task_dir": entry,
                "tags": task_tags,
                "category": data.get("category") or "",
                "difficulty": data.get("difficulty") or "",
                "prompt": prompt_text,
                "yaml": data,
            })
        except Exception as exc:
            print(f"  Warning: could not load {yaml_path}: {exc}")

    return tasks


# ---------------------------------------------------------------------------
# Transcript / trace parsing
# ---------------------------------------------------------------------------

def _extract_transcript_text(events: List[Dict[str, Any]]) -> str:
    """
    Extract human-readable text from a list of claw-eval JSONL trace events.

    Event types: trace_start, message, tool_dispatch, trace_end, grading_result
    We concatenate assistant messages and tool dispatch summaries.
    """
    parts = []
    for event in events:
        etype = event.get("type", "")

        if etype == "message":
            msg = event.get("message") or event
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content") or ""
                        if text:
                            parts.append(f"[{role}]: {text}")
            elif role in ("assistant", "user"):
                parts.append(f"[{role}]: {content}")

        elif etype == "tool_dispatch":
            tool_name = event.get("tool_name") or event.get("name") or "tool"
            inputs = event.get("request_body") or event.get("inputs") or event.get("input") or {}
            outputs = event.get("response_body") or event.get("outputs") or event.get("output") or {}
            inputs_str = json.dumps(inputs)[:300] if inputs else ""
            outputs_str = json.dumps(outputs)[:300] if outputs else ""
            parts.append(
                f"[tool:{tool_name}] inputs={inputs_str} → outputs={outputs_str}"
            )

    return "\n\n".join(parts)


def _parse_grading_result(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract the grading_result event from trace events, if present.

    GradingResult has task_score/passed at the top level and a nested
    'scores' dict (DimensionScores) with completion/robustness/communication/safety.
    Return the full event so all fields are accessible.
    """
    for event in events:
        if event.get("type") == "grading_result":
            return event
    # Fallback: task_score/passed are also on trace_end when no grading_result
    for event in events:
        if event.get("type") == "trace_end" and event.get("task_score") is not None:
            return {
                "task_score": event.get("task_score"),
                "passed": event.get("passed"),
                "scores": event.get("scores", {}),
            }
    return None


def _parse_trace_file(trace_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL trace file and return list of parsed event dicts."""
    events = []
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        print(f"  Warning: failed to read trace {trace_path}: {exc}")
    return events


def _find_latest_trace(trace_dir: Path, task_id: str) -> Optional[Path]:
    """Find the most recently written JSONL trace for the given task_id."""
    candidates = list(trace_dir.glob(f"{task_id}*.jsonl"))
    if not candidates:
        # Fallback: any JSONL in the dir
        candidates = list(trace_dir.glob("*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_context_length_error(error: Any) -> bool:
    """Identify provider rejections caused by an oversized agent context."""
    text = str(error or "").lower()
    markers = (
        "maximum context length",
        "context length exceeded",
        "context_length_exceeded",
        "prompt is too long",
        "too many tokens",
        "reduce the length",
    )
    return any(marker in text for marker in markers)


def _is_systemic_execution_error(error: Any) -> bool:
    """Return True when continuing would invalidate many benchmark tasks."""
    text = str(error or "").lower()
    markers = (
        "requires more credits",
        "insufficient credit",
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
        "internal server error",
    )
    return any(marker in text for marker in markers)


def _classify_execution_failure(exec_result: Dict[str, Any]) -> str:
    """Classify a failed ClawEval run without blaming the agent by default.

    Newer ClawEval runners report the stage directly. Marker-based fallbacks
    retain compatibility with older installed versions and the CLI path.
    """
    stage = str(exec_result.get("failure_stage") or "").lower()
    detail = "\n".join(
        str(exec_result.get(field) or "")
        for field in ("error", "stderr", "stdout")
    )
    detail_lower = detail.lower()
    error_type = str(exec_result.get("error_type") or "").lower()

    if _is_systemic_execution_error(detail) or error_type in {
        "apiconnectionerror",
        "apitimeouterror",
        "internalservererror",
    }:
        return "systemic_error"
    if stage == "service_setup":
        return "infrastructure_error"
    if stage == "grading":
        return "evaluator_error"
    if stage == "agent_execution" or _is_context_length_error(detail):
        return "agent_execution_error"
    if exec_result.get("status") == "timeout":
        return "agent_timeout"

    infrastructure_markers = (
        "servicestarterror",
        "service '",
        "failed to become ready",
        "did not become ready",
        "address already in use",
        "connection refused",
        "docker daemon",
        "no such file or directory",
    )
    if any(marker in detail_lower for marker in infrastructure_markers):
        return "infrastructure_error"

    # A partial trace proves that the task reached the agent loop. Unknown
    # failures before any trace are kept unscored for investigation.
    if exec_result.get("events") or exec_result.get("trace_path"):
        return "agent_execution_error"
    return "unknown_execution_error"


def _parse_cli_score(stdout: str) -> Dict[str, Any]:
    """Parse authoritative grading lines printed by `claw-eval run`.

    The JSONL trace has a trace_end event written before grading, so its
    task_score can be stale/zero. The CLI output and .result.json are written
    after grading and should override trace_end.
    """
    parsed: Dict[str, Any] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^(completion|robustness|communication|safety|task_score):\s*([0-9.]+)", line)
        if m:
            parsed[m.group(1)] = float(m.group(2))
            continue

        m = re.search(r"\btask_score=([0-9.]+)\b", line)
        if m:
            parsed["task_score"] = float(m.group(1))

        m = re.search(r"\bpassed=(True|False|true|false)\b", line)
        if m:
            parsed["passed"] = m.group(1).lower() == "true"

        m = re.match(r"^passed:\s*(True|False|true|false)", line)
        if m:
            parsed["passed"] = m.group(1).lower() == "true"

    if "task_score" in parsed and "passed" not in parsed:
        parsed["passed"] = parsed["task_score"] >= 0.8
    return parsed


def _analyze_tool_calls(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tool call usage from a list of claw-eval trace events."""
    calls: List[Dict[str, Any]] = []

    for event in events:
        if event.get("type") == "tool_dispatch":
            name = (
                event.get("tool_name")
                or event.get("name")
                or "unknown_tool"
            )
            has_error = bool(event.get("error") or event.get("is_error"))
            status_code = event.get("response_status")
            if isinstance(status_code, int) and status_code >= 400:
                has_error = True
            error_msg = event.get("error")
            if error_msg is None and has_error:
                body = event.get("response_body")
                if isinstance(body, dict):
                    error_msg = body.get("error") or body.get("message")
                elif body is not None:
                    error_msg = str(body)
            if not isinstance(error_msg, str) and error_msg is not None:
                error_msg = str(error_msg)
            calls.append({
                "name": name,
                "status": "error" if has_error else "ok",
                "error": error_msg if has_error else None,
            })

    tool_counter = collections.Counter(c["name"] for c in calls)
    return {
        "tool_names": sorted(tool_counter.keys()),
        "tool_name_counts": dict(sorted(tool_counter.items())),
        "total_calls": len(calls),
        "successful_calls": sum(1 for c in calls if c["status"] == "ok"),
        "error_calls": sum(1 for c in calls if c["status"] == "error"),
        "unknown_status_calls": 0,
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Config YAML writing
# ---------------------------------------------------------------------------

def _write_runtime_config(
    base_config_path: Path,
    model_id: str,
    api_key: Optional[str],
    base_url: Optional[str],
    judge_model: Optional[str],
    trace_dir: str,
    dest_path: Path,
    judge_api_key: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    insight_library_text: Optional[str] = None,
    thinking_level: Optional[str] = None,
    context_window: Optional[int] = None,
    skill_entries: Optional[List[Dict[str, str]]] = None,
) -> None:
    """
    Load the base config YAML and overlay model/api_key/base_url/judge fields,
    writing a runtime copy to dest_path.
    """
    try:
        with open(base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}

    if "model" not in config:
        config["model"] = {}
    config["model"]["model_id"] = model_id
    if context_window is not None:
        config["model"]["context_window"] = int(context_window)
    if api_key:
        config["model"]["api_key"] = api_key
    if base_url:
        config["model"]["base_url"] = base_url
        if "generativelanguage.googleapis.com" in base_url:
            extra_body = config["model"].get("extra_body")
            if isinstance(extra_body, dict):
                extra_body.pop("reasoning", None)
                if not extra_body:
                    config["model"].pop("extra_body", None)

    if thinking_level is None:
        extra_body = config["model"].get("extra_body")
        if isinstance(extra_body, dict):
            extra_body.pop("reasoning", None)
            if not extra_body:
                config["model"].pop("extra_body", None)
    elif base_url and "generativelanguage.googleapis.com" not in base_url:
        config["model"].setdefault("extra_body", {})["reasoning"] = {
            "effort": thinking_level
        }

    if insight_library_text:
        config["model"]["system_prompt_prefix"] = (
            "## Insight Library (from prior PinchBench experience)\n"
            "Use the following reusable insights when solving the task.\n\n"
            f"{insight_library_text}"
        )

    if skill_entries:
        prompt_config = config.setdefault("prompt", {})
        skills_config = prompt_config.setdefault("skills", {})
        skills_config["default"] = skill_entries
        skills_config["load_via_tool_call"] = True
        skills_config["read_tool_name"] = "Read"

    if judge_model:
        if "judge" not in config:
            config["judge"] = {}
        effective_judge_api_key = judge_api_key or api_key
        effective_judge_base_url = judge_base_url or base_url
        if (
            effective_judge_base_url
            and "generativelanguage.googleapis.com" in effective_judge_base_url
        ):
            if judge_model.startswith("google/"):
                judge_model = judge_model.split("/", 1)[1]
            elif judge_model.startswith("gemini/"):
                judge_model = judge_model.split("/", 1)[1]
        config["judge"]["model_id"] = judge_model
        if effective_judge_api_key:
            config["judge"]["api_key"] = effective_judge_api_key
        if effective_judge_base_url:
            config["judge"]["base_url"] = effective_judge_base_url

    if base_url and "generativelanguage.googleapis.com" in base_url:
        config["user_agent_model"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model_id": model_id,
        }

    if "defaults" not in config:
        config["defaults"] = {}
    config["defaults"]["trace_dir"] = trace_dir

    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class OpenClawClawEvalPipeline:
    """
    Pipeline for running claw-eval benchmark tasks, extracting reasoning
    traces, and aggregating them into an insight library.

    Mirrors OpenClawPinchBenchPipeline but calls ClawEval's official per-task
    runner in-process.  The subprocess CLI is retained only as a compatibility
    fallback when the vendored package cannot be imported.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "claweval_output",
        tag: str = "general",
        claweval_dir: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "gemini",
        api_model: str = "gemini-3-pro-preview",
        base_url: Optional[str] = None,
        openclaw_api_key: Optional[str] = None,
        timeout: int = 300,
        trials: int = 1,
        parallel: int = 1,
        encyclopedia_path: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_provider: Optional[str] = None,
        sandbox: bool = False,
        thinking_level: Optional[str] = None,
        pooling: bool = False,
        pooling_dir: Optional[str] = None,
        pooling_raw_transcript: bool = False,
        participate: Optional[int] = None,
        rag: bool = False,
        rag_api_key: Optional[str] = None,
        rag_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        pooling_context_window: Optional[int] = None,
        openclaw_skill: bool = False,
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.tag = tag
        self.use_api = use_api
        self.api_provider = api_provider
        self.api_key = api_key or (
            os.getenv("GEMINI_API_KEY") if api_provider == "gemini" else os.getenv("OPENROUTER_API_KEY")
        )
        self.api_model = api_model
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.timeout = timeout
        self.trials = trials
        self.parallel = int(parallel)
        if self.parallel < 1:
            raise ValueError("parallel must be at least 1")
        self.encyclopedia_path = encyclopedia_path
        self.judge_model = judge_model or model_id
        self.judge_provider = judge_provider or api_provider
        self.judge_api_key = (
            self.api_key
            if self.judge_provider == self.api_provider
            else (
                os.getenv("GEMINI_API_KEY")
                if self.judge_provider == "gemini"
                else os.getenv("OPENROUTER_API_KEY")
            )
        )
        self.judge_base_url = (
            "https://generativelanguage.googleapis.com/v1beta/openai"
            if self.judge_provider == "gemini"
            else "https://openrouter.ai/api/v1"
        )
        self.sandbox = sandbox
        self.thinking_level = thinking_level  # "low" / "medium" / "high" / None
        self.pooling = pooling
        self.pooling_dir = pooling_dir
        self.pooling_raw_transcript = pooling_raw_transcript
        self.participate = int(participate) if participate is not None else None
        self.rag = rag
        self.rag_api_key = rag_api_key
        self.rag_embedding_model = rag_embedding_model
        self.pooling_context_window = pooling_context_window
        self._pooling_reference_paths: Dict[str, str] = {}
        self.openclaw_skill = openclaw_skill
        self._curated_skills_root = (
            _install_openclaw_skills(Path(self.output_dir))
            if openclaw_skill
            else None
        )

        if self.api_provider == "gemini" and self.api_key:
            os.environ["GEMINI_API_KEY"] = self.api_key

        self.claweval_dir = Path(claweval_dir) if claweval_dir else _CLAWEVAL_DIR
        self.tasks_dir = self.claweval_dir / "tasks"
        self.base_config = self.claweval_dir / "config_general.yaml"

        os.makedirs(self.output_dir, exist_ok=True)

        # Lazy-loaded insight extractor client (same as pinchbench)
        self._client: Optional[ChainOfThoughtReader] = None
        self._metrics_cache_by_output_dir: Dict[str, Dict[str, Any]] = {}
        self._claw_cmd: Optional[List[str]] = None

    def _load_skill_entries(self) -> List[Dict[str, str]]:
        """Return ClawEval-native descriptors for a directory of SKILL.md files."""
        if not self._curated_skills_root:
            return []
        if not self._curated_skills_root.is_dir():
            raise RuntimeError(
                "ClawEval curated skills root does not exist: "
                f"{self._curated_skills_root}"
            )
        entries: List[Dict[str, str]] = []
        for skill_name in _PREDEFINED_OPENCLAW_SKILLS:
            skill_path = self._curated_skills_root / skill_name / "SKILL.md"
            text = skill_path.read_text(encoding="utf-8", errors="replace")
            description_match = re.search(
                r"(?mi)^description:\s*[\"']?(.+?)[\"']?\s*$", text[:8000]
            )
            entries.append(
                {
                    "name": skill_path.parent.name,
                    "description": (
                        description_match.group(1).strip()
                        if description_match
                        else "Specialized task instructions"
                    ),
                    "path": str(skill_path.resolve()),
                }
            )
        if not entries:
            raise RuntimeError(
                "No curated OpenClaw SKILL.md files were available to ClawEval"
            )
        return entries

    # ------------------------------------------------------------------
    # ClawEval task invocation
    # ------------------------------------------------------------------

    def _run_single_task(
        self,
        task: Dict[str, Any],
        config_path: Path,
        trace_dir: Path,
        port_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Run a task through ClawEval's official programmatic runner.

        Falls back to the official CLI subprocess if the runner is not importable.
        Returns a dict with status, events (parsed JSONL), and usage info.
        """
        task_dir = task["task_dir"]

        # Determine base_url for the model provider
        if self.api_provider == "gemini":
            base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            agent_model_id = self.api_model or self.model_id
            if agent_model_id.startswith("google/"):
                agent_model_id = agent_model_id.split("/", 1)[1]
            elif agent_model_id.startswith("gemini/"):
                agent_model_id = agent_model_id.split("/", 1)[1]
        else:
            base_url = self.base_url or "https://openrouter.ai/api/v1"
            agent_model_id = self.model_id

        resolved_api_key = self.api_key or (
            os.getenv("GEMINI_API_KEY") if self.api_provider == "gemini"
            else os.getenv("OPENROUTER_API_KEY")
        )

        if _HAS_CLAWEVAL_RUNNER:
            execution_mode = "Docker sandbox" if self.sandbox else "local sandbox-tools"
            print(
                "  [claweval] Running official in-process task runner "
                f"({execution_mode}): {task['task_id']}"
            )
            result = _run_official_claweval_task(
                task_dir=str(task_dir),
                config_path=str(config_path) if config_path.exists() else None,
                model=agent_model_id,
                api_key=resolved_api_key or "",
                base_url=base_url,
                trace_dir=str(trace_dir),
                port_offset=port_offset,
                no_judge=False,
                judge_model=self.judge_model,
                trials=self.trials,
                proxy=None,
                sandbox=self.sandbox,
                sandbox_image=None,
                # This is ClawEval's built-in local tool dispatcher when
                # sandbox=False; no Docker daemon is involved.
                sandbox_tools=True,
            )

            trials = result.get("trials") or []
            trial_errors = [str(t.get("error")) for t in trials if t.get("error")]
            task_error = result.get("error")
            if task_error or trial_errors:
                details = str(task_error or "; ".join(trial_errors))
                return {
                    "status": "error",
                    "events": [],
                    "usage": {},
                    "execution_time": 0.0,
                    "trace_path": None,
                    "trace_paths": [],
                    "task_score": 0.0,
                    "passed": False,
                    "scores": {},
                    "error": details,
                    "error_type": result.get("error_type"),
                    "failure_stage": result.get("failure_stage"),
                }

            valid_trials = [t for t in trials if t.get("trace")]
            if not valid_trials:
                return {
                    "status": "error",
                    "events": [],
                    "usage": {},
                    "execution_time": 0.0,
                    "trace_path": None,
                    "trace_paths": [],
                    "task_score": 0.0,
                    "passed": False,
                    "scores": {},
                    "error": "Official ClawEval runner returned no completed trial trace",
                }

            trace_paths = [Path(t["trace"]) for t in valid_trials]
            events: List[Dict[str, Any]] = []
            for completed_trace in trace_paths:
                events.extend(_parse_trace_file(completed_trace))

            def _trial_sum(field: str) -> float:
                return sum(float(t.get(field) or 0) for t in valid_trials)

            def _trial_average(field: str) -> float:
                return _trial_sum(field) / len(valid_trials)

            scores = {
                "completion": _trial_average("completion"),
                "robustness": _trial_average("robustness"),
                "communication": _trial_average("communication"),
                "safety": _trial_average("safety"),
            }
            return {
                "status": "success",
                "events": events,
                "usage": {
                    "input_tokens": int(_trial_sum("model_input_tokens")),
                    "output_tokens": int(_trial_sum("model_output_tokens")),
                },
                "execution_time": _trial_sum("wall_time_s"),
                "trace_path": str(trace_paths[-1]),
                "trace_paths": [str(path) for path in trace_paths],
                "task_score": result.get("avg_score"),
                "passed": result.get("avg_passed"),
                "scores": scores,
                "error": None,
            }

        # ---- Fallback: subprocess (when package not importable in-process) ----
        import subprocess as _sp
        # Strip provider prefix (e.g. "google/", "gemini/") that Gemini API rejects
        subprocess_model_id = self.model_id
        if self.api_provider == "gemini":
            for prefix in ("google/", "gemini/"):
                if subprocess_model_id.startswith(prefix):
                    subprocess_model_id = subprocess_model_id.split("/", 1)[1]
                    break
        cmd = [sys.executable, "-m", "claw_eval.cli", "run",
               "--task", str(task_dir.resolve()),
               "--config", str(config_path.resolve()),
               "--trace-dir", str(trace_dir.resolve()),
               "--model", subprocess_model_id,
               "--api-key", resolved_api_key or "",
               "--base-url", base_url,
               "--trials", str(self.trials)]
        if port_offset:
            cmd.extend(["--port-offset", str(port_offset)])
        cmd.append("--sandbox-tools")
        if self.sandbox:
            cmd.append("--sandbox")

        print(f"  [claweval] Running via subprocess: {' '.join(cmd[:6])} ...")
        t0 = time.time()
        try:
            proc = _sp.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout * self.trials + 60,
                cwd=str(self.claweval_dir),
            )
        except _sp.TimeoutExpired:
            return {"status": "timeout", "events": [], "usage": {}, "execution_time": self.timeout}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "events": [], "usage": {}, "execution_time": None}

        elapsed = time.time() - t0
        status = "success" if proc.returncode == 0 else "failed"

        # Always print stdout for visibility; it contains "Trace: <path>" and scores
        if proc.stdout:
            print(f"  stdout: {proc.stdout[:2000]}")
        if proc.stderr:
            print(f"  stderr: {proc.stderr[:3000]}")

        # --- Parse the actual trace path from CLI stdout ---
        # claw-eval prints "Trace: /abs/path/to/<task_id>_<uuid8>.jsonl"
        trace_path: Optional[Path] = None
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("Trace:"):
                candidate = Path(line.split("Trace:", 1)[1].strip())
                if candidate.exists():
                    trace_path = candidate
                    break

        # Fallback: glob trace_dir (handles edge cases / future CLI changes)
        if trace_path is None:
            trace_path = _find_latest_trace(trace_dir.resolve(), task["task_id"])
            if not trace_path:
                existing = list(trace_dir.resolve().glob("*.jsonl")) if trace_dir.resolve().exists() else []
                print(f"  [debug] trace_dir={trace_dir.resolve()} task_id={task['task_id']} found={[p.name for p in existing]}")

        events = _parse_trace_file(trace_path) if trace_path else []

        # --- Read grade + tokens from .result.json written by CLI ---
        # claw-eval writes <trace_stem>.result.json alongside the JSONL
        result_json: Dict[str, Any] = {}
        if trace_path:
            result_json_path = trace_path.with_suffix(".result.json")
            if not result_json_path.exists():
                # CLI also prints "Result: <path>"
                for line in (proc.stdout or "").splitlines():
                    line = line.strip()
                    if line.startswith("Result:"):
                        rp = Path(line.split("Result:", 1)[1].strip())
                        if rp.exists():
                            result_json_path = rp
                            break
            if result_json_path.exists():
                try:
                    result_json = json.loads(result_json_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"  Warning: failed to read result.json: {exc}")

        stdout_grade = _parse_cli_score(proc.stdout or "")
        authoritative_grade: Dict[str, Any] = result_json or stdout_grade

        # Prefer result.json for tokens (most accurate); fall back to trace_end event
        if result_json:
            usage = {
                "input_tokens": result_json.get("model_input_tokens") or result_json.get("input_tokens") or 0,
                "output_tokens": result_json.get("model_output_tokens") or result_json.get("output_tokens") or 0,
            }
        else:
            usage = {}
            for event in events:
                if event.get("type") == "trace_end":
                    usage = {
                        "input_tokens": event.get("model_input_tokens") or event.get("input_tokens") or 0,
                        "output_tokens": event.get("model_output_tokens") or event.get("output_tokens") or 0,
                    }
                    break

        # Inject post-grading score so _parse_grading_result does not fall back
        # to stale trace_end.task_score=0.0 from before grading.
        if authoritative_grade and not any(e.get("type") == "grading_result" for e in events):
            events.append({
                "type": "grading_result",
                "task_score": authoritative_grade.get("task_score"),
                "passed": authoritative_grade.get("passed"),
                "scores": {
                    "completion": authoritative_grade.get("completion"),
                    "robustness": authoritative_grade.get("robustness"),
                    "communication": authoritative_grade.get("communication"),
                    "safety": authoritative_grade.get("safety"),
                },
            })

        return {
            "status": status,
            "events": events,
            "usage": usage,
            "execution_time": elapsed,
            "trace_path": str(trace_path) if trace_path else None,
            "task_score": authoritative_grade.get("task_score") if authoritative_grade else None,
            "passed": authoritative_grade.get("passed") if authoritative_grade else None,
            "scores": {
                "completion": authoritative_grade.get("completion"),
                "robustness": authoritative_grade.get("robustness"),
                "communication": authoritative_grade.get("communication"),
                "safety": authoritative_grade.get("safety"),
            } if authoritative_grade else None,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": None if proc.returncode == 0 else (
                (proc.stderr or proc.stdout or f"subprocess exit code {proc.returncode}").strip()
            ),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics_output_key(self) -> str:
        return str(Path(self.output_dir).resolve())

    def _metrics_log_path(self) -> Path:
        return Path(self.output_dir) / "metrics_log.json"

    def _build_metrics_summary(
        self,
        task_metrics: List[Dict[str, Any]],
        library_output_tokens: Optional[int],
    ) -> Dict[str, Any]:
        graded = [t for t in task_metrics if t.get("grade", {}).get("task_score") is not None]
        score_sum = sum(float(t["grade"].get("task_score", 0.0)) for t in graded)
        passed_count = sum(1 for t in graded if t["grade"].get("passed"))

        tool_name_counts: Dict[str, int] = {}
        total_tool_calls = 0
        successful_tool_calls = 0
        error_tool_calls = 0
        for task in task_metrics:
            tools = task.get("tools", {})
            total_tool_calls += int(tools.get("total_calls", 0) or 0)
            successful_tool_calls += int(tools.get("successful_calls", 0) or 0)
            error_tool_calls += int(tools.get("error_calls", 0) or 0)
            for name, count in (tools.get("tool_name_counts", {}) or {}).items():
                tool_name_counts[name] = tool_name_counts.get(name, 0) + int(count)

        total_time = sum(float(t.get("execution_time_seconds") or 0.0) for t in task_metrics)
        total_agent_tokens = sum(int((t.get("output_tokens") or {}).get("agent", 0) or 0) for t in task_metrics)
        total_extraction_tokens = sum(int((t.get("output_tokens") or {}).get("extraction", 0) or 0) for t in task_metrics)

        return {
            "tasks_total": len(task_metrics),
            "tasks_succeeded": sum(
                t.get("execution_status") == "success" for t in task_metrics
            ),
            "tasks_failed": sum(
                t.get("execution_status") != "success" for t in task_metrics
            ),
            "tasks_graded": len(graded),
            "tasks_ungraded": len(task_metrics) - len(graded),
            "tasks_passed": passed_count,
            "tasks_agent_errors": sum(
                t.get("failure_type") in {
                    "agent_execution_error",
                    "agent_timeout",
                }
                for t in task_metrics
            ),
            "tasks_infrastructure_errors": sum(
                t.get("failure_type") == "infrastructure_error"
                for t in task_metrics
            ),
            "tasks_evaluator_errors": sum(
                t.get("failure_type") == "evaluator_error"
                for t in task_metrics
            ),
            "avg_task_score": (score_sum / len(graded)) if graded else None,
            "pass_rate_pct": (passed_count / len(graded) * 100.0) if graded else None,
            "output_tokens_agent_total": total_agent_tokens,
            "output_tokens_extraction_total": total_extraction_tokens,
            "output_tokens_total": total_agent_tokens + total_extraction_tokens,
            "tool_names": sorted(tool_name_counts.keys()),
            "tool_name_counts": dict(sorted(tool_name_counts.items())),
            "total_tool_calls": total_tool_calls,
            "successful_tool_calls": successful_tool_calls,
            "error_tool_calls": error_tool_calls,
            "execution_time_total_seconds": total_time,
            "library_output_tokens": int(library_output_tokens) if library_output_tokens is not None else None,
        }

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
            "tag": self.tag,
            "trials": self.trials,
            "use_api": self.use_api,
            "pooling": getattr(self, "pooling", False),
            "rag": getattr(self, "rag", False),
            "pooling_dir": getattr(self, "pooling_dir", None),
            "summary": self._build_metrics_summary(task_metrics, library_output_tokens),
            "tasks": task_metrics,
        }
        log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
            iteration_summaries.append({
                "iteration": log_path.parent.name,
                "metrics_log": str(log_path),
                "summary": payload.get("summary", {}),
            })

        if not iteration_summaries:
            return

        tasks_total = sum(int(s["summary"].get("tasks_total", 0) or 0) for s in iteration_summaries)
        tasks_passed = sum(int(s["summary"].get("tasks_passed", 0) or 0) for s in iteration_summaries)
        scores = [s["summary"].get("avg_task_score") for s in iteration_summaries if s["summary"].get("avg_task_score") is not None]

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
                "tasks_total": tasks_total,
                "tasks_passed": tasks_passed,
                "avg_task_score_across_iterations": (sum(scores) / len(scores)) if scores else None,
                "output_tokens_total": sum(int(s["summary"].get("output_tokens_total", 0) or 0) for s in iteration_summaries),
                "tool_names": sorted(overall_tool_counts.keys()),
                "tool_name_counts": dict(sorted(overall_tool_counts.items())),
                "total_tool_calls": sum(int(s["summary"].get("total_tool_calls", 0) or 0) for s in iteration_summaries),
                "execution_time_total_seconds": sum(float(s["summary"].get("execution_time_total_seconds", 0.0) or 0.0) for s in iteration_summaries),
                "library_output_tokens_total": sum(int(s["summary"].get("library_output_tokens", 0) or 0) for s in iteration_summaries),
                "pipeline_elapsed_seconds": elapsed_seconds,
            },
        }
        (root / "metrics_log_overall.json").write_text(
            json.dumps(overall_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Insight extraction (same as pinchbench)
    # ------------------------------------------------------------------

    def _ensure_client(self) -> ChainOfThoughtReader:
        if self._client is None:
            self._client = ChainOfThoughtReader(
                use_api=self.use_api,
                api_key=self.api_key,
                api_provider=self.api_provider,
                api_model=self.api_model,
            )
        return self._client

    def _call_for_extraction(self, prompt: str, max_new_tokens: int) -> tuple:
        """Call model for reflection/extraction, using ThinkingConfig when set."""
        if self.use_api and self.api_provider == "gemini" and self.thinking_level:
            return call_gemini_thinking(
                api_key=self.api_key,
                model_name=self.api_model,
                prompt=prompt,
                thinking_level=self.thinking_level,
                max_new_tokens=max_new_tokens,
            )
        client = self._ensure_client()
        return client._call_model(prompt, max_new_tokens=max_new_tokens)

    def _apply_reflection_and_extraction(
        self, task_prompt: str, agent_response: str
    ) -> Dict[str, Any]:
        """
        Apply Steps 2 & 3 from client.py (reflection + insight extraction)
        to the agent's response transcript.
        """
        client = self._ensure_client()

        # Step 2: Reflection
        print("  Step 2: Reflecting on agent solution...")
        reflection_prompt = client._get_reflection_prompt(task_prompt, agent_response)
        reflection, _ = self._call_for_extraction(reflection_prompt, max_new_tokens=4096)
        print(f"  Reflection length: {len(reflection)} chars")

        # Step 3: Insight extraction
        print("  Step 3: Extracting reasoning traces...")
        behavior_prompt = client._get_behavior_prompt(task_prompt, agent_response, reflection)
        extraction_response, token_info = self._call_for_extraction(
            behavior_prompt, max_new_tokens=8192
        )

        # Parse insight JSON (same logic as client._step_behavior_extraction)
        insights = {}
        try:
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", extraction_response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                start = extraction_response.find("{")
                json_str = None
                if start != -1:
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    for i in range(start, len(extraction_response)):
                        ch = extraction_response[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if ch == "\\":
                            escape_next = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            continue
                        if not in_string:
                            if ch == "{":
                                brace_count += 1
                            elif ch == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = extraction_response[start: i + 1]
                                    break

            if json_str:
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                raw = json.loads(json_str)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        name = k if k.startswith("insight_") else f"insight_{k}"
                        desc = v if isinstance(v, str) else str(v)
                        desc = re.sub(r"\s+", " ", desc).strip()
                        if len(desc) >= 20:
                            insights[name] = desc
        except Exception as exc:
            print(f"  Warning: insight extraction parse error: {exc}")

        print(f"  Extracted {len(insights)} insights")
        return {
            "insight_book": insights,
            "output_tokens": token_info.get("output_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_tasks_and_extract(self, *, extract_insights: bool = True) -> List[Dict[str, Any]]:
        """
        Step 1 + Steps 2/3: For each claw-eval task (filtered by tag),
        run the claw-eval agent and extract reasoning traces from the
        JSONL transcript.

        Saves each task's insights as problem_XXXX.json in output_dir.
        Returns a list of per-task result dicts.
        """
        _check_claw_eval()
        if self.sandbox:
            _check_docker_sandbox()
            print("ClawEval sandbox mode: Docker")
        else:
            print(
                "ClawEval sandbox mode: local --sandbox-tools "
                "(compatible with DS nodes, but not container-isolated)"
            )

        tasks = _load_task_yamls(self.tasks_dir, tag=self.tag)
        if not tasks:
            print(f"No tasks found in {self.tasks_dir} with tag='{self.tag}'")
            return []

        print(f"\nLoaded {len(tasks)} tasks (tag='{self.tag}')")
        print("=" * 80)

        # Write runtime config YAML with model/api/judge settings
        trace_dir = Path(self.output_dir) / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        config_dest = Path(self.output_dir) / "runtime_config.yaml"
        effective_api_key = self.api_key if self.use_api else self.openclaw_api_key
        effective_base_url = self.base_url
        effective_model_id = self.model_id

        if self.use_api and self.api_provider == "gemini":
            effective_api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            effective_base_url = effective_base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            if self.model_id.startswith("google/"):
                effective_model_id = self.model_id.split("/", 1)[1]
            elif self.model_id.startswith("gemini/"):
                effective_model_id = self.model_id.split("/", 1)[1]
            print("Using direct Gemini endpoint for claw-eval agent (bypassing OpenRouter)")
        elif self.use_api and self.api_provider == "openrouter":
            effective_base_url = (
                effective_base_url or "https://openrouter.ai/api/v1"
            )

        insight_library_text = None
        if (
            not getattr(self, "_pooling_reference_paths", {})
            and self.encyclopedia_path
            and Path(self.encyclopedia_path).exists()
        ):
            insight_library_text = Path(self.encyclopedia_path).read_text(encoding="utf-8")
        skill_entries = self._load_skill_entries()

        _write_runtime_config(
            base_config_path=self.base_config,
            model_id=effective_model_id,
            api_key=effective_api_key,
            base_url=effective_base_url,
            judge_model=self.judge_model,
            judge_api_key=self.judge_api_key,
            judge_base_url=self.judge_base_url,
            trace_dir=str(trace_dir),
            dest_path=config_dest,
            insight_library_text=insight_library_text,
            thinking_level=self.thinking_level,
            context_window=(
                self._effective_pooling_context_window()
                if getattr(self, "_pooling_reference_paths", {})
                else None
            ),
            skill_entries=skill_entries,
        )
        print(f"Runtime config written to: {config_dest}")

        if getattr(self, "_pooling_reference_paths", {}):
            mode = (
                "RAG retrieval"
                if getattr(self, "rag", False)
                else "appended trace pooling"
            )
            print(f"\nUsing task-specific {mode} references")
        elif self.encyclopedia_path and Path(self.encyclopedia_path).exists():
            print(f"\nEncyclopedia loaded: {self.encyclopedia_path}")
        else:
            print("\nNo encyclopedia — running without prior insights")

        # Prepare task-specific configs before starting workers. Config files
        # are immutable during execution, so workers only share read-only state.
        prepared_tasks = []
        for i, task in enumerate(tasks, 1):
            task_config_dest = config_dest
            reference_path = getattr(
                self, "_pooling_reference_paths", {}
            ).get(str(task["task_id"]))
            if reference_path:
                reference_text = Path(reference_path).read_text(encoding="utf-8")
                task_configs_dir = Path(self.output_dir) / "pooling_runtime_configs"
                task_configs_dir.mkdir(parents=True, exist_ok=True)
                safe_task_id = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_", str(task["task_id"])
                ).strip("_")
                task_config_dest = task_configs_dir / f"{safe_task_id or 'task'}.yaml"
                _write_runtime_config(
                    base_config_path=self.base_config,
                    model_id=effective_model_id,
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                    judge_model=self.judge_model,
                    judge_api_key=self.judge_api_key,
                    judge_base_url=self.judge_base_url,
                    trace_dir=str(trace_dir),
                    dest_path=task_config_dest,
                    insight_library_text=reference_text,
                    thinking_level=self.thinking_level,
                    context_window=self._effective_pooling_context_window(),
                    skill_entries=skill_entries,
                )

            # ClawEval reserves roughly 30 service ports per task. Its own
            # batch runner uses a stride of 50, which we mirror here.
            prepared_tasks.append((i, task, task_config_dest, (i - 1) * 50))

        workers = min(getattr(self, "parallel", 1), len(prepared_tasks))
        print(f"\nExecuting {len(prepared_tasks)} ClawEval tasks with {workers} worker(s)")

        def execute_task(prepared):
            _i, task, task_config_dest, port_offset = prepared
            try:
                return self._run_single_task(
                    task,
                    task_config_dest,
                    trace_dir,
                    port_offset=port_offset,
                )
            except Exception as exc:
                # Return task-local setup failures to the ordered result loop
                # so one broken task does not cancel all parallel futures.
                # Systemic failures are still identified and raised there.
                return {
                    "status": "error",
                    "events": [],
                    "usage": {},
                    "execution_time": None,
                    "trace_path": None,
                    "trace_paths": [],
                    "task_score": None,
                    "passed": None,
                    "scores": {},
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "failure_stage": "service_setup",
                }

        exec_results = parallel_utils.parallel_map_ordered(
            execute_task,
            prepared_tasks,
            num_workers=workers,
        )

        results = []
        task_metrics: List[Dict[str, Any]] = []
        task_counter = 0

        # Process scores, extraction, and output files in task order. This
        # keeps filenames and metrics stable regardless of completion order.
        for prepared, exec_result in zip(prepared_tasks, exec_results):
            i, task, _task_config_dest, _port_offset = prepared
            print(f"\n[{i}/{len(tasks)}] Task: {task['task_id']} — {task['task_name']}")
            print("-" * 60)

            # -----------------------------------------------------------------
            # Step 1 result: task execution completed in the worker pool above
            # -----------------------------------------------------------------
            status = exec_result.get("status", "error")
            events = exec_result.get("events", [])
            print(f"  Execution status: {status} | events: {len(events)}")

            if status != "success":
                stderr = (exec_result.get("stderr") or "").strip()
                if stderr:
                    print(f"  Stderr (first 400): {stderr[:400]}")
                error_detail = exec_result.get("error") or stderr or "no error detail"
                error_type = exec_result.get("error_type")
                failure_stage = exec_result.get("failure_stage")
                print(
                    f"  Error detail: {str(error_detail)[:2000]}"
                    + (f" | type={error_type}" if error_type else "")
                    + (f" | stage={failure_stage}" if failure_stage else "")
                )
                failure_type = _classify_execution_failure(exec_result)
                if failure_type in {"agent_execution_error", "agent_timeout"}:
                    # Agent-local failures count as failed task attempts. They
                    # must affect the benchmark average and must not stop the
                    # remaining independent tasks.
                    failure_grade = {
                        "task_score": 0.0,
                        "passed": False,
                        "scores": {},
                        "notes": (
                            "Agent failed to complete task execution: "
                            f"{str(error_detail)[:2000]}"
                        ),
                    }
                    usage = exec_result.get("usage", {}) or {}
                    agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
                    execution_time = exec_result.get("execution_time")
                    if execution_time is not None:
                        execution_time = float(execution_time)
                    task_metric = {
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "category": task["category"],
                        "tags": task["tags"],
                        "execution_status": status,
                        "failure_type": failure_type,
                        "failure_detail": str(error_detail)[:2000],
                        "execution_time_seconds": execution_time,
                        "trace_path": exec_result.get("trace_path"),
                        "trace_paths": exec_result.get("trace_paths") or [],
                        "grade": failure_grade,
                        "output_tokens": {
                            "agent": agent_output_tokens,
                            "extraction": 0,
                            "total": agent_output_tokens,
                        },
                        "tools": _analyze_tool_calls(events),
                        "insights_extracted": 0,
                    }
                    task_metrics.append(task_metric)
                    results.append({
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "status": status,
                        "failure_type": failure_type,
                        "task_score": 0.0,
                        "passed": False,
                        "insights_extracted": 0,
                    })
                    self._write_metrics_log(task_metrics)
                    print(
                        "  Agent execution failure: recorded task_score=0.0, "
                        "passed=False; continuing to the next task"
                    )
                    continue
                if failure_type != "systemic_error":
                    # Infrastructure, evaluator, and unknown pre-trace errors
                    # must not be charged to the agent. Preserve them as
                    # ungraded task records so completed work is not discarded.
                    usage = exec_result.get("usage", {}) or {}
                    agent_output_tokens = int(usage.get("output_tokens", 0) or 0)
                    execution_time = exec_result.get("execution_time")
                    if execution_time is not None:
                        execution_time = float(execution_time)
                    task_metric = {
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "category": task["category"],
                        "tags": task["tags"],
                        "execution_status": status,
                        "failure_type": failure_type,
                        "failure_detail": str(error_detail)[:2000],
                        "execution_time_seconds": execution_time,
                        "trace_path": exec_result.get("trace_path"),
                        "trace_paths": exec_result.get("trace_paths") or [],
                        "grade": {},
                        "output_tokens": {
                            "agent": agent_output_tokens,
                            "extraction": 0,
                            "total": agent_output_tokens,
                        },
                        "tools": _analyze_tool_calls(events),
                        "insights_extracted": 0,
                    }
                    task_metrics.append(task_metric)
                    results.append({
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "status": status,
                        "failure_type": failure_type,
                        "task_score": None,
                        "passed": None,
                        "insights_extracted": 0,
                    })
                    self._write_metrics_log(task_metrics)
                    print(
                        f"  {failure_type}: recorded as ungraded; "
                        "continuing to the next task"
                    )
                    continue
                raise RuntimeError(
                    f"ClawEval task returned {failure_type} with "
                    f"status={status!r} for "
                    f"{task['task_id']}: "
                    f"{error_detail}"
                )

            # -----------------------------------------------------------------
            # Prefer the official runner's aggregate result (authoritative),
            # and fall back to parsing the grading event for CLI compatibility.
            # -----------------------------------------------------------------
            exec_task_score = exec_result.get("task_score")
            exec_passed = exec_result.get("passed")
            exec_scores = exec_result.get("scores")  # DimensionScores object or None

            grade = _parse_grading_result(events)

            # The in-process path may aggregate multiple trials, so it overrides
            # any single grading event parsed from the concatenated traces.
            if exec_task_score is not None:
                task_score = exec_task_score
                passed = exec_passed
                dim: Any = exec_scores or (grade.get("scores") if grade else {}) or {}
                completion = dim.get("completion") if isinstance(dim, dict) else getattr(dim, "completion", None)
                safety = dim.get("safety") if isinstance(dim, dict) else getattr(dim, "safety", None)
                # Rebuild grade dict so downstream code (task_metric["grade"]) is populated
                grade = {
                    "task_score": task_score,
                    "passed": passed,
                    "scores": dim if isinstance(dim, dict) else (dim.model_dump() if hasattr(dim, "model_dump") else {}),
                }
            elif grade:
                task_score = grade.get("task_score")
                passed = grade.get("passed")
                dim = grade.get("scores") or {}
                completion = dim.get("completion") if isinstance(dim, dict) else getattr(dim, "completion", None)
                safety = dim.get("safety") if isinstance(dim, dict) else getattr(dim, "safety", None)
            else:
                raise RuntimeError(
                    f"ClawEval grading produced no score for {task['task_id']}"
                )

            if task_score is not None:
                print(f"  Score: task_score={task_score} | passed={passed} | completion={completion} | safety={safety}")
            else:
                print("  Grade: not available")

            # Usage from the in-process adapter is summed across all trials.
            # token_info remains as a compatibility fallback for older results.
            token_info = exec_result.get("token_info") or {}
            usage = exec_result.get("usage", {}) or {}
            agent_output_tokens = int(
                token_info.get("model_output_tokens") or usage.get("output_tokens") or 0
            )
            extraction_output_tokens = 0
            execution_time = exec_result.get("execution_time")
            if execution_time is not None:
                execution_time = float(execution_time)

            tool_stats = _analyze_tool_calls(events)
            task_metric = {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "category": task["category"],
                "tags": task["tags"],
                "execution_status": status,
                "execution_time_seconds": execution_time,
                "trace_path": exec_result.get("trace_path"),
                "trace_paths": exec_result.get("trace_paths")
                or ([exec_result.get("trace_path")] if exec_result.get("trace_path") else []),
                "grade": grade or {},
                "output_tokens": {
                    "agent": agent_output_tokens,
                    "extraction": extraction_output_tokens,
                    "total": agent_output_tokens,
                },
                "tools": tool_stats,
                "insights_extracted": 0,
            }

            if not extract_insights:
                results.append({
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "status": status,
                    "task_score": task_score,
                    "passed": passed,
                    "insights_extracted": 0,
                })
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            if not events:
                print("  Skipping insight extraction — no trace events")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # -----------------------------------------------------------------
            # Extract readable solution text from trace events
            # -----------------------------------------------------------------
            agent_response = _extract_transcript_text(events)
            if not agent_response.strip():
                print("  Skipping insight extraction — no extractable text in trace")
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            # Prepend encyclopedia as context if available
            task_prompt = task["prompt"]
            if self.encyclopedia_path and Path(self.encyclopedia_path).exists():
                try:
                    with open(self.encyclopedia_path, "r", encoding="utf-8") as f:
                        enc = json.load(f)
                    enc_text = json.dumps(enc, indent=2)[:8000]
                    task_prompt = (
                        f"## Insight Library (from prior experience)\n{enc_text}\n\n"
                        f"## Task\n{task_prompt}"
                    )
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Steps 2 & 3: Reflection + insight extraction
            # -----------------------------------------------------------------
            try:
                extraction = self._apply_reflection_and_extraction(
                    task_prompt=task_prompt,
                    agent_response=agent_response,
                )
            except Exception as exc:
                print(f"  Error during insight extraction: {exc}")
                task_metric["extraction_error"] = str(exc)
                task_metrics.append(task_metric)
                self._write_metrics_log(task_metrics)
                continue

            insight_book = extraction.get("insight_book", {})
            if not insight_book:
                print("  Warning: no insights extracted for this task")
                insight_book = {}

            extraction_output_tokens = int(extraction.get("output_tokens", 0) or 0)
            task_metric["output_tokens"] = {
                "agent": agent_output_tokens,
                "extraction": extraction_output_tokens,
                "total": agent_output_tokens + extraction_output_tokens,
            }
            task_metric["insights_extracted"] = len(insight_book)

            # -----------------------------------------------------------------
            # Save as problem_XXXX.json (matches server_text expected format)
            # -----------------------------------------------------------------
            task_counter += 1
            output_file = os.path.join(self.output_dir, f"problem_{task_counter:04d}.json")

            save_data = {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "category": task["category"],
                "tags": task["tags"],
                "task_prompt": task["prompt"],
                "execution_status": status,
                "grade": grade or {},
                "output_tokens": extraction.get("output_tokens", 0),
                "insight_book": insight_book,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"  Saved {len(insight_book)} insights → {output_file}")
            results.append({
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "status": status,
                "task_score": task_score,
                "passed": passed,
                "insights_extracted": len(insight_book),
                "output_file": output_file,
            })
            task_metrics.append(task_metric)
            self._write_metrics_log(task_metrics)

        print("\n" + "=" * 80)
        total_insights = sum(int(t.get("insights_extracted", 0) or 0) for t in task_metrics)
        graded = [t for t in task_metrics if t.get("grade", {}).get("task_score") is not None]
        print(f"Tasks processed: {len(task_metrics)}/{len(tasks)}")
        if graded:
            avg_score = sum(float(t["grade"].get("task_score", 0.0) or 0.0) for t in graded) / len(graded)
            passed_count = sum(1 for t in graded if t["grade"].get("passed"))
            print(f"Avg task score:  {avg_score:.3f}  ({passed_count}/{len(graded)} passed)")
        print(f"Total insights extracted: {total_insights}")
        self._write_metrics_log(task_metrics)
        return results

    def _effective_pooling_context_window(self) -> int:
        if self.pooling_context_window is not None:
            return int(self.pooling_context_window)
        try:
            config = yaml.safe_load(self.base_config.read_text(encoding="utf-8")) or {}
            configured = int((config.get("model") or {}).get("context_window", 0) or 0)
            if configured > 0:
                return configured
        except Exception:
            pass
        return 200_000

    def _prepare_pooling_iteration(
        self,
        *,
        tasks: List[Dict[str, Any]],
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
            str(task["task_id"]): str(task["prompt"])
            for task in tasks
        }
        reference_paths, manifest = build_pooling_references(
            traces=traces,
            task_prompts=task_prompts,
            output_dir=iteration_output_dir,
            context_window=self._effective_pooling_context_window(),
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
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._pooling_reference_paths = reference_paths
        print(
            f"Prepared {len(reference_paths)} task-specific pooling references "
            f"from {len(traces)} traces across {len(source_iterations)} "
            "source iteration(s)"
        )
        return manifest

    def run_eval_only_iterations(self, *, iterations: int) -> None:
        """Run ClawEval repeatedly, matching each round to one trace-pool round."""
        if iterations < 1:
            raise ValueError("--iterations must be at least 1")
        tasks = _load_task_yamls(self.tasks_dir, tag=self.tag)
        if not tasks:
            raise RuntimeError(
                f"No tasks found in {self.tasks_dir} with tag={self.tag!r}"
            )
        start_time = time.time()
        base_output_dir = str(Path(self.output_dir).resolve())
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        original_output_dir = self.output_dir
        original_reference_paths = self._pooling_reference_paths
        history: List[Dict[str, Any]] = []
        try:
            for iteration in range(1, iterations + 1):
                use_iteration_dir = iterations > 1 or self.pooling
                iteration_output_dir = (
                    Path(base_output_dir) / f"iter_{iteration:02d}"
                    if use_iteration_dir
                    else Path(base_output_dir)
                )
                iteration_output_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = str(iteration_output_dir)
                self._pooling_reference_paths = {}
                pooling_manifest = None
                if self.pooling:
                    pooling_manifest = self._prepare_pooling_iteration(
                        tasks=tasks,
                        iteration=iteration,
                        iteration_output_dir=iteration_output_dir,
                    )
                print("\n" + "=" * 80)
                print(f"CLAW-EVAL EVAL-ONLY ITERATION {iteration}/{iterations}")
                print("=" * 80)
                self.run_tasks_and_extract(extract_insights=False)
                history.append(
                    {
                        "iteration": iteration,
                        "output_dir": str(iteration_output_dir.resolve()),
                        "pooling": pooling_manifest,
                    }
                )
        finally:
            self.output_dir = original_output_dir
            self._pooling_reference_paths = original_reference_paths

        elapsed = time.time() - start_time
        if iterations > 1 or self.pooling:
            self._write_overall_metrics_log(base_output_dir, elapsed)
        summary_path = Path(base_output_dir) / "eval_only_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "iterations": iterations,
                    "pooling": self.pooling,
                    "participate": self.participate,
                    "rag": self.rag,
                    "history": history,
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def aggregate_insights(self) -> Optional[str]:
        """
        Aggregate all problem_XXXX.json files using TextBasedInsightAggregationServer
        and save the resulting encyclopedia.json.
        """
        print("\n" + "=" * 80)
        print("Aggregating Insights via server_text")
        print("=" * 80)

        json_files = sorted(Path(self.output_dir).glob("problem_*.json"))
        if not json_files:
            print("No problem_*.json files found — nothing to aggregate")
            return None

        print(f"Found {len(json_files)} insight files")

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

        encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
        enc_dict = server._try_parse_json(server.encyclopedia)
        if enc_dict is None:
            enc_dict = server._try_parse_json(server._extract_json_only(server.encyclopedia))

        if enc_dict is None:
            encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.txt")
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                f.write(server.encyclopedia)
            print(f"Warning: encyclopedia saved as plain text: {encyclopedia_path}")
        else:
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(enc_dict, f, indent=2, ensure_ascii=False)
            print(f"Encyclopedia saved: {encyclopedia_path}")

        total_output_tokens = result.get("total_output_tokens", 0)
        print(f"Insight library output tokens: {total_output_tokens}")
        metrics_key = self._metrics_output_key()
        existing_payload = self._metrics_cache_by_output_dir.get(metrics_key)
        existing_tasks = existing_payload.get("tasks", []) if existing_payload else []
        self._write_metrics_log(existing_tasks, library_output_tokens=int(total_output_tokens or 0))
        return encyclopedia_path

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

            if iterations > 1:
                iter_dir = os.path.join(self.output_dir, f"iter_{iteration:02d}")
                os.makedirs(iter_dir, exist_ok=True)
                orig_output_dir = self.output_dir
                self.output_dir = iter_dir
            else:
                orig_output_dir = None

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
        description="OpenClaw ClawEval Pipeline — extract reasoning traces from claw-eval benchmark tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Model identifier (e.g., google/gemini-3-pro-preview, anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="claweval_output",
        help="Directory to save insights and encyclopedia (default: claweval_output)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="general",
        help=(
            'Filter tasks by YAML tag: "general" (default, 199 tasks: '
            '161 standard + 38 user-agent), "multimodal" (101 tasks), '
            'or "user_agent" (38 tasks)'
        ),
    )
    parser.add_argument(
        "--claweval-dir",
        type=str,
        default=None,
        help="Path to claw-eval repo root (default: ./claw-eval)",
    )
    parser.add_argument(
        "--openclaw-skill",
        action="store_true",
        help=(
            "Install and expose the curated OpenClaw skill set to each "
            "ClawEval task agent."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations (default: 1). Iteration 2+ uses the encyclopedia from iteration 1.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per task for claw-eval (default: 1)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum concurrent ClawEval task workers (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run tasks in Docker sandbox (requires claw-eval Docker setup)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Judge model for LLM-judge tasks (default: same as --model)",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["gemini", "openrouter"],
        help="Judge API provider (default: same as --api-provider)",
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        default=None,
        help="Path to existing encyclopedia.json to inject as context for the first iteration",
    )
    parser.add_argument(
        "--pooling",
        action="store_true",
        help=(
            "Evaluate with reference context from the matching iter_XX under "
            "--pooling-dir. Appended individual libraries are detected automatically."
        ),
    )
    parser.add_argument(
        "--pooling-dir",
        "--pooling_dir",
        dest="pooling_dir",
        type=str,
        default=None,
        help="Root containing iter_01, iter_02, ... reasoning-trace directories.",
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
        default=None,
        help=(
            "Override the ClawEval model context window used for pooling "
            "budgets (default: runtime config value)."
        ),
    )
    parser.add_argument(
        "--participate",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --pooling against an --individual --pooling-dir training "
            "run, restrict each matching iter_XX round to the first N "
            "clients (by client_index) instead of pooling all of them. Has "
            "no effect without --pooling — use task_openclaw_pinchbench.py "
            "--participate to build a merged --encyclopedia for a plain "
            "(non-pooling) eval-only run instead."
        ),
    )
    parser.add_argument(
        "--start-from-step2",
        action="store_true",
        help="Skip task execution and start from aggregation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Evaluate ClawEval tasks with the optional --encyclopedia context, "
            "without reflection, trace extraction, or library aggregation."
        ),
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use an API provider for reflection/extraction steps (and as agent endpoint if model is gemini/*)",
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
        default="gemini-3-pro-preview",
        help="Model name for extraction (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom OpenAI-compatible API base URL for the claw-eval agent",
    )
    parser.add_argument(
        "--openclaw-api-key",
        type=str,
        default=None,
        help="API key for custom OpenClaw agent endpoint (default: $OPENAI_API_KEY or $OPENROUTER_API_KEY)",
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

    args = parser.parse_args()

    if args.pooling_raw_transcript:
        args.eval_only = True
        args.pooling = True
        args.rag = True
        args.pooling_dir = args.pooling_raw_transcript
    if args.pooling:
        args.eval_only = True

    if args.pooling and not args.pooling_dir:
        parser.error("--pooling-dir is required with --pooling")
    if args.pooling_dir and not args.pooling:
        parser.error("--pooling-dir requires --pooling")
    if args.rag and not args.pooling:
        parser.error("--rag requires --pooling")
    if args.pooling and args.encyclopedia:
        parser.error("--pooling cannot be combined with --encyclopedia")
    if args.participate is not None:
        if args.participate <= 0:
            parser.error("--participate must be a positive integer")
        if not args.pooling:
            parser.error("--participate requires --pooling")
    if args.pooling_context_window is not None and args.pooling_context_window < 1:
        parser.error("--pooling-context-window must be positive")
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")

    if not args.model and not args.start_from_step2:
        parser.error("--model is required unless using --start-from-step2")

    model_id = args.model or "google/gemini-3-pro-preview"
    judge_model = args.judge or model_id

    # Resolve openclaw agent API key: CLI > env vars
    openclaw_api_key = args.openclaw_api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    rag_api_key = (
        args.rag_api_key
        or os.getenv("OPENROUTER_API_KEY")
        or (args.api_key if args.api_provider == "openrouter" else None)
        or args.openclaw_api_key
    )
    if args.rag and not rag_api_key:
        parser.error("--rag requires --rag-api-key or OPENROUTER_API_KEY")

    pipeline = OpenClawClawEvalPipeline(
        model_id=model_id,
        output_dir=args.output_dir,
        tag=args.tag,
        claweval_dir=args.claweval_dir,
        use_api=args.use_api,
        api_key=args.api_key,
        api_provider=args.api_provider,
        api_model=args.api_model,
        base_url=args.base_url,
        openclaw_api_key=openclaw_api_key,
        timeout=args.timeout,
        trials=args.trials,
        parallel=args.parallel,
        encyclopedia_path=args.encyclopedia,
        judge_model=judge_model,
        judge_provider=args.judge_provider,
        sandbox=args.sandbox,
        thinking_level=(
            None
            if not args.use_api or args.thinking_level == "default"
            else args.thinking_level
        ),
        pooling=args.pooling,
        pooling_dir=args.pooling_dir,
        pooling_raw_transcript=bool(args.pooling_raw_transcript),
        participate=args.participate,
        rag=args.rag,
        rag_api_key=rag_api_key,
        rag_embedding_model=args.rag_embedding_model,
        pooling_context_window=args.pooling_context_window,
        openclaw_skill=args.openclaw_skill,
    )

    if args.eval_only:
        pipeline.run_eval_only_iterations(iterations=args.iterations)
    else:
        pipeline.run_pipeline(
            iterations=args.iterations,
            start_from_step2=args.start_from_step2,
        )


if __name__ == "__main__":
    main()
