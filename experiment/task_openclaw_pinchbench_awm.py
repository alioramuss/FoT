"""
Agent Workflow Memory (AWM) runner for PinchBench.

This is an adaptation of Wang, et al., "Agent Workflow Memory"
(arXiv:2409.07429) to the PinchBench/OpenClaw harness.  Unlike ExpeL, AWM
does not learn a list of general rules or retrieve whole trajectories.  It
induces reusable, parameterized multi-step workflows from successful agent
trajectories and exposes those workflows as a portable agent skill. Tasks can
run through a direct API tool loop (no OpenClaw installation) or OpenClaw.

Two modes are supported:

* offline: run the first half of the selected tasks once, induce workflows
  from full-credit trajectories grouped by task category, and evaluate on
  the second half with the resulting workflow skill.
* online: process every selected task as a stream.  A full-credit trajectory
  induces workflows that become available to all subsequent tasks.

The learned memory is saved as both JSON and a human-readable SKILL.md.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_PINCHBENCH_SCRIPTS = Path(__file__).parent / "pinchbench" / "scripts"
if str(_PINCHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PINCHBENCH_SCRIPTS))

from lib_agent import (  # type: ignore  # noqa: E402
    ModelValidationError,
    _get_agent_workspace,
    cleanup_agent_sessions,
    ensure_agent_exists,
    execute_openclaw_task,
    slugify_model,
    validate_openrouter_model,
)
from lib_grading import GradeResult, grade_task  # type: ignore  # noqa: E402
from lib_tasks import Task, TaskLoader  # type: ignore  # noqa: E402
from lib_direct_agent import execute_direct_task  # type: ignore  # noqa: E402
from utils import normalize_api_model, resolve_api_key  # noqa: E402
import parallel_utils  # noqa: E402

from task_openclaw_pinchbench_expel import (  # noqa: E402
    ExpelTrajectory,
    GeminiCaller,
    OpenRouterCaller,
    _check_openclaw,
    _clear_fot_insights,
    _clone_task_with_prompt,
    _extract_transcript_text,
    _full_credit,
    _grade_to_dict,
    _safe_write_json,
    _split_chunks,
    _task_full_prompt,
    _trim,
)


AWM_PAPER_URL = "https://arxiv.org/abs/2409.07429"


@dataclass
class WorkflowStep:
    """One AWM step: environment state, reasoning, and executable action."""

    state: str
    reasoning: str
    action: str


@dataclass
class AgentWorkflow:
    """A reusable parameterized workflow induced from successful experience."""

    workflow_id: str
    name: str
    description: str
    category: str
    applicability: str
    parameters: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    source_task_ids: List[str] = field(default_factory=list)


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:64] or "workflow"


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _json_from_model(text: str) -> Any:
    """Parse JSON even when a model surrounds it with prose or a code fence."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
        if not starts:
            raise
        start = min(starts)
        decoder = json.JSONDecoder()
        value, _ = decoder.raw_decode(cleaned[start:])
        return value


def _normalize_workflows(
    raw: Any,
    *,
    category: str,
    source_task_ids: Sequence[str],
) -> List[AgentWorkflow]:
    if isinstance(raw, dict):
        raw = raw.get("workflows", raw.get("items", []))
    if not isinstance(raw, list):
        return []

    workflows: List[AgentWorkflow] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("title") or "").strip()
        description = str(item.get("description") or item.get("goal") or "").strip()
        raw_steps = item.get("steps") or item.get("trajectory") or []
        if not name or not description or not isinstance(raw_steps, list):
            continue
        steps: List[WorkflowStep] = []
        for raw_step in raw_steps:
            if isinstance(raw_step, str):
                action = raw_step.strip()
                if action:
                    steps.append(
                        WorkflowStep(
                            state="The previous workflow step has completed.",
                            reasoning="Continue the reusable subroutine.",
                            action=action,
                        )
                    )
                continue
            if not isinstance(raw_step, dict):
                continue
            state = str(
                raw_step.get("state")
                or raw_step.get("environment_state")
                or raw_step.get("observation")
                or ""
            ).strip()
            reasoning = str(
                raw_step.get("reasoning")
                or raw_step.get("thought")
                or raw_step.get("rationale")
                or ""
            ).strip()
            action = str(
                raw_step.get("action")
                or raw_step.get("executable_action")
                or raw_step.get("command")
                or ""
            ).strip()
            if state and reasoning and action:
                steps.append(WorkflowStep(state=state, reasoning=reasoning, action=action))
        # The paper requires workflows to contain at least two steps.
        if len(steps) < 2:
            continue
        params = _string_list(item.get("parameters") or item.get("variables"))
        params = [
            param if param.startswith("{") and param.endswith("}") else f"{{{param}}}"
            for param in params
        ]
        if not params:
            joined = " ".join(
                [description] + [s.state + " " + s.reasoning + " " + s.action for s in steps]
            )
            params = sorted(set(re.findall(r"\{[a-zA-Z][a-zA-Z0-9_-]*\}", joined)))
        stable = hashlib.sha1(
            (category + "\n" + name + "\n" + description).lower().encode("utf-8")
        ).hexdigest()[:10]
        workflows.append(
            AgentWorkflow(
                workflow_id=f"{_slug(name)}-{stable}",
                name=name,
                description=description,
                category=str(item.get("category") or category or "general").strip(),
                applicability=str(
                    item.get("applicability")
                    or item.get("when_to_use")
                    or description
                ).strip(),
                parameters=params,
                steps=steps,
                source_task_ids=sorted(set(source_task_ids)),
            )
        )
    return workflows


def _workflow_signature(workflow: AgentWorkflow) -> str:
    tokens = re.findall(
        r"[a-z0-9]+",
        f"{workflow.category} {workflow.name} {workflow.description}".lower(),
    )
    return " ".join(tokens)


def _token_set(text: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "the", "this", "to", "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", text.lower())
        if token not in stop
    }


def _transcript_for_induction(transcript: Sequence[Dict[str, Any]]) -> str:
    """Preserve reasoning and tool actions from OpenClaw JSONL messages."""
    lines: List[str] = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        message = entry.get("message") or {}
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                lines.append(f"[{role}] {content.strip()}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            text = block.get("text") or block.get("content")
            if isinstance(text, str) and text.strip():
                lines.append(f"[{role}:{block_type or 'text'}] {text.strip()}")
            tool_name = block.get("name") or block.get("toolName")
            tool_input = (
                block.get("input")
                or block.get("arguments")
                or block.get("args")
            )
            if tool_name:
                try:
                    rendered = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    rendered = str(tool_input)
                lines.append(f"[{role}:action] {tool_name}({rendered})")
    return "\n".join(lines)


class OpenClawPinchBenchAWM:
    def __init__(
        self,
        *,
        model_id: str,
        output_dir: str,
        suite: str,
        pinchbench_dir: Optional[str],
        judge_model: Optional[str],
        api_key: Optional[str],
        api_model: str,
        api_provider: str,
        thinking_level: str,
        agent_thinking_level: Optional[str],
        mode: str,
        induction_batch_size: int,
        workflow_top_k: int,
        max_workflows: int,
        workflow_memory: Optional[str],
        skill_name: str,
        timeout_multiplier: float,
        base_url: Optional[str],
        openclaw_api_key: Optional[str],
        num_workers: int = parallel_utils.DEFAULT_NUM_WORKERS,
        agent_backend: str = "direct",
        direct_max_steps: int = 40,
    ) -> None:
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.suite = suite
        self.pinchbench_dir = (
            Path(pinchbench_dir)
            if pinchbench_dir
            else Path(__file__).parent / "pinchbench"
        )
        self.tasks_dir = self.pinchbench_dir / "tasks"
        self.skill_dir = self.pinchbench_dir
        self.api_provider = api_provider.strip().lower()
        if self.api_provider not in {"gemini", "openrouter"}:
            raise ValueError("api_provider must be 'gemini' or 'openrouter'")
        self.api_key = resolve_api_key(self.api_provider, api_key)
        if not self.api_key:
            env_name = "OPENROUTER_API_KEY" if self.api_provider == "openrouter" else "GEMINI_API_KEY"
            raise ValueError(f"{env_name} is required via --api-key or the environment")

        if self.api_provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = self.api_key
            if "/" not in api_model.removeprefix("openrouter/"):
                api_model = f"google/{api_model}"
            self.api_model = normalize_api_model("openrouter", api_model)
            requested_judge = judge_model or model_id
            self.judge_model = (
                requested_judge
                if requested_judge.startswith("openrouter/")
                else f"openrouter/{requested_judge}"
            )
        else:
            os.environ["GEMINI_API_KEY"] = self.api_key
            self.api_model = api_model.removeprefix("google/")
            self.judge_model = judge_model or model_id

        self.thinking_level = thinking_level
        self.agent_thinking_level = agent_thinking_level
        self.mode = mode
        self.induction_batch_size = max(1, induction_batch_size)
        self.workflow_top_k = max(0, workflow_top_k)
        self.max_workflows = max(1, max_workflows)
        self.workflow_memory = Path(workflow_memory) if workflow_memory else None
        self.skill_name = _slug(skill_name)
        self.timeout_multiplier = timeout_multiplier
        self.base_url = base_url
        self.openclaw_api_key = openclaw_api_key
        self.num_workers = parallel_utils.worker_count(num_workers)
        self.agent_backend = agent_backend
        if self.agent_backend not in {"direct", "openclaw"}:
            raise ValueError("agent_backend must be 'direct' or 'openclaw'")
        self.direct_max_steps = max(1, direct_max_steps)
        self.agent_id = f"pinchbench-awm-{slugify_model(model_id)}"
        self.model_caller = (
            OpenRouterCaller(self.api_key, self.api_model)
            if self.api_provider == "openrouter"
            else GeminiCaller(self.api_key, self.api_model, self.thinking_level)
        )

        self.workflows: List[AgentWorkflow] = []
        self.trajectories: List[ExpelTrajectory] = []
        self.token_records: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[Dict[str, Any]]] = {"train": [], "eval": [], "online": []}

    def _run_config(self) -> Dict[str, Any]:
        return {
            "method": "AWM",
            "paper": AWM_PAPER_URL,
            "mode": self.mode,
            "model_id": self.model_id,
            "suite": self.suite,
            "judge_model": self.judge_model,
            "api_provider": self.api_provider,
            "api_model": self.api_model,
            "thinking_level": self.thinking_level,
            "agent_thinking_level": self.agent_thinking_level,
            "induction_batch_size": self.induction_batch_size,
            "workflow_top_k": self.workflow_top_k,
            "max_workflows": self.max_workflows,
            "workflow_memory": str(self.workflow_memory) if self.workflow_memory else None,
            "skill_name": self.skill_name,
            "timeout_multiplier": self.timeout_multiplier,
            "num_workers": self.num_workers,
            "agent_backend": self.agent_backend,
            "direct_max_steps": self.direct_max_steps,
        }

    def _record_tokens(
        self,
        stage: str,
        token_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.token_records.append(
            {
                "stage": stage,
                "metadata": metadata or {},
                "token_info": token_info or {},
                "timestamp": time.time(),
            }
        )
        self._write_token_usage()

    def _token_summary(self) -> Dict[str, Any]:
        keys = (
            "input_tokens", "output_tokens", "thinking_tokens", "total_tokens",
            "cache_read_tokens", "cache_write_tokens", "cost_usd", "request_count",
        )
        totals = {key: 0.0 for key in keys}
        by_stage: Dict[str, Dict[str, float]] = {}
        for record in self.token_records:
            stage = str(record.get("stage") or "unknown")
            stage_totals = by_stage.setdefault(stage, {key: 0.0 for key in keys})
            info = record.get("token_info") or {}
            for key in keys:
                try:
                    value = float(info.get(key, 0) or 0)
                except (TypeError, ValueError):
                    value = 0.0
                totals[key] += value
                stage_totals[key] += value
        return {"totals": totals, "by_stage": by_stage, "records": len(self.token_records)}

    def _write_token_usage(self) -> None:
        _safe_write_json(
            self.output_dir / "token_usage.json",
            {
                "config": self._run_config(),
                "summary": self._token_summary(),
                "records": self.token_records,
            },
        )

    def _setup_agent(self) -> None:
        if self.agent_backend == "direct":
            return
        _check_openclaw()
        effective_base_url = self.base_url
        effective_api_key = self.openclaw_api_key
        effective_model_id = self.model_id
        if self.api_provider == "gemini" and self.model_id.startswith(("google/", "gemini/")):
            effective_base_url = (
                effective_base_url
                or "https://generativelanguage.googleapis.com/v1beta/openai"
            )
            effective_api_key = effective_api_key or self.api_key
            effective_model_id = self.model_id.split("/", 1)[1]
        elif self.api_provider == "openrouter":
            effective_api_key = effective_api_key or self.api_key

        if not effective_base_url:
            try:
                validate_openrouter_model(self.model_id)
            except ModelValidationError as exc:
                print(f"Warning: {exc}")

        workspace = _get_agent_workspace(self.agent_id)
        if workspace is None:
            workspace = Path.home() / ".openclaw" / "agents" / self.agent_id.lower() / "workspace"
        ensure_agent_exists(
            self.agent_id,
            effective_model_id,
            workspace,
            base_url=effective_base_url,
            api_key=effective_api_key,
        )
        _clear_fot_insights(self.agent_id)
        cleanup_agent_sessions(self.agent_id)

    def _load_tasks(self) -> List[Task]:
        tasks = TaskLoader(self.tasks_dir).load_all_tasks()
        if self.suite == "all":
            return tasks
        if self.suite == "automated-only":
            return [task for task in tasks if task.grading_type == "automated"]
        wanted = {task_id.strip() for task_id in self.suite.split(",") if task_id.strip()}
        return [task for task in tasks if task.task_id in wanted]

    def _load_initial_memory(self) -> None:
        if not self.workflow_memory:
            return
        payload = json.loads(self.workflow_memory.read_text(encoding="utf-8"))
        raw_items = payload.get("workflows", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            raise ValueError("--workflow-memory must contain a JSON workflow list")
        loaded: List[AgentWorkflow] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            steps = [
                WorkflowStep(**step)
                for step in item.get("steps", [])
                if isinstance(step, dict)
                and all(key in step for key in ("state", "reasoning", "action"))
            ]
            if len(steps) < 2:
                continue
            loaded.append(
                AgentWorkflow(
                    workflow_id=str(item.get("workflow_id") or _slug(str(item.get("name", "workflow")))),
                    name=str(item.get("name") or "Unnamed workflow"),
                    description=str(item.get("description") or ""),
                    category=str(item.get("category") or "general"),
                    applicability=str(item.get("applicability") or item.get("description") or ""),
                    parameters=_string_list(item.get("parameters")),
                    steps=steps,
                    source_task_ids=_string_list(item.get("source_task_ids")),
                )
            )
        self._merge_workflows(loaded)

    def _workflow_markdown(self, workflows: Optional[Sequence[AgentWorkflow]] = None) -> str:
        selected = list(self.workflows if workflows is None else workflows)
        lines = [
            "---",
            f"name: {self.skill_name}",
            "description: Reusable parameterized workflows learned from successful OpenClaw tasks.",
            "---",
            "",
            "# Agent Workflow Memory",
            "",
            "Use a workflow only when its applicability matches the current task and observed state.",
            "Replace every `{placeholder}` with a value grounded in the current task or environment.",
            "Re-check the current state before each action; adapt element IDs, paths, and tool arguments.",
            "A workflow is guidance, not permission to skip validation or violate the user request.",
            "",
        ]
        if not selected:
            lines.extend(["No workflows have been learned yet.", ""])
            return "\n".join(lines)
        for workflow in selected:
            lines.extend(
                [
                    f"## {workflow.name}",
                    "",
                    f"- Category: {workflow.category}",
                    f"- When to use: {workflow.applicability}",
                    f"- Goal: {workflow.description}",
                    f"- Parameters: {', '.join(workflow.parameters) if workflow.parameters else '(none)'}",
                    "",
                ]
            )
            for idx, step in enumerate(workflow.steps, 1):
                lines.extend(
                    [
                        f"{idx}. State: {step.state}",
                        f"   Reasoning: {step.reasoning}",
                        f"   Action: `{step.action}`",
                    ]
                )
            lines.append("")
        return "\n".join(lines)

    def _persist_memory(self) -> None:
        payload = {
            "method": "AWM",
            "paper": AWM_PAPER_URL,
            "count": len(self.workflows),
            "workflows": [asdict(workflow) for workflow in self.workflows],
        }
        _safe_write_json(self.output_dir / "workflows.json", payload)
        skill_text = self._workflow_markdown()
        skill_out = self.output_dir / "skill" / self.skill_name / "SKILL.md"
        skill_out.parent.mkdir(parents=True, exist_ok=True)
        skill_out.write_text(skill_text, encoding="utf-8")

        # PinchBench copies skills from the main OpenClaw workspace into each
        # freshly prepared benchmark workspace.
        if self.agent_backend == "openclaw":
            installed = (
                Path.home()
                / ".openclaw"
                / "workspace"
                / "skills"
                / self.skill_name
                / "SKILL.md"
            )
            installed.parent.mkdir(parents=True, exist_ok=True)
            installed.write_text(skill_text, encoding="utf-8")

    def _select_workflows(self, task: Task) -> List[AgentWorkflow]:
        if not self.workflows or self.workflow_top_k == 0:
            return []
        query = _token_set(
            f"{task.category} {task.name} {_task_full_prompt(task)}"
        )
        scored: List[Tuple[float, int, AgentWorkflow]] = []
        for idx, workflow in enumerate(self.workflows):
            text = (
                f"{workflow.category} {workflow.name} {workflow.description} "
                f"{workflow.applicability} {' '.join(workflow.parameters)}"
            )
            tokens = _token_set(text)
            overlap = len(query & tokens) / max(1, len(query | tokens))
            category_bonus = 1.0 if workflow.category.lower() == task.category.lower() else 0.0
            scored.append((category_bonus + overlap, -idx, workflow))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[: self.workflow_top_k]]

    def _prompt_with_memory(self, task: Task, workflows: Sequence[AgentWorkflow]) -> str:
        if not workflows:
            return task.prompt
        relative_skill = f"skills/{self.skill_name}/SKILL.md"
        return (
            "Before starting, read the Agent Workflow Memory skill at "
            f"`{relative_skill}`. The relevant workflows are also shown below. "
            "Use them only when applicable, ground placeholders in the current "
            "environment, and validate the result.\n\n"
            f"{self._workflow_markdown(workflows)}\n"
            "Now complete the original task exactly as requested:\n\n"
            f"{task.prompt}"
        )

    def _execute_and_grade(
        self,
        *,
        task: Task,
        execution_task: Task,
        run_id: str,
        transcript_dir: Path,
    ) -> Tuple[Dict[str, Any], Optional[GradeResult]]:
        self._persist_memory()
        if self.agent_backend == "direct":
            if self.api_provider == "openrouter":
                direct_base_url = self.base_url or "https://openrouter.ai/api/v1"
                direct_model = self.model_id.removeprefix("openrouter/")
            else:
                direct_base_url = (
                    self.base_url
                    or "https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                direct_model = self.model_id.removeprefix("google/").removeprefix("gemini/")
            execution_result = execute_direct_task(
                task=execution_task,
                model_id=direct_model,
                run_id=run_id,
                timeout_multiplier=self.timeout_multiplier,
                skill_dir=self.skill_dir,
                api_key=self.openclaw_api_key or self.api_key,
                base_url=direct_base_url,
                output_dir=transcript_dir,
                workspace_root=self.output_dir / "direct_workspaces",
                extra_skills_dir=self.output_dir / "skill",
                max_steps=self.direct_max_steps,
                thinking_level=self.agent_thinking_level,
            )
        else:
            _clear_fot_insights(self.agent_id)
            execution_result = execute_openclaw_task(
                task=execution_task,
                agent_id=self.agent_id,
                model_id=self.model_id,
                run_id=run_id,
                timeout_multiplier=self.timeout_multiplier,
                skill_dir=self.skill_dir,
                output_dir=transcript_dir,
                verbose=False,
                thinking_level=self.agent_thinking_level,
            )
        try:
            grade = grade_task(
                task=task,
                execution_result=execution_result,
                skill_dir=self.skill_dir,
                judge_model=self.judge_model,
                judge_backend="api",
            )
        except Exception as exc:
            print(f"  Warning: grading failed for {task.task_id}: {exc}")
            grade = None
        return execution_result, grade

    def _make_trajectory(
        self,
        *,
        task: Task,
        split: str,
        execution_result: Dict[str, Any],
        grade: Optional[GradeResult],
        transcript_file: Optional[Path],
    ) -> ExpelTrajectory:
        usage = execution_result.get("usage") or {}
        token_usage = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_read_tokens": int(usage.get("cache_read_tokens", 0) or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
            "request_count": int(usage.get("request_count", 0) or 0),
        }
        transcript = execution_result.get("transcript") or []
        induction_text = _transcript_for_induction(transcript)
        return ExpelTrajectory(
            task_id=task.task_id,
            task_name=task.name,
            task_prompt=_task_full_prompt(task),
            attempt=1,
            split=split,
            status=str(execution_result.get("status", "unknown")),
            score=grade.score if grade else None,
            max_score=grade.max_score if grade else None,
            is_success=_full_credit(grade),
            transcript_text=induction_text or _extract_transcript_text(transcript),
            reflections=[],
            execution_time_seconds=(
                float(execution_result["execution_time"])
                if execution_result.get("execution_time") is not None
                else None
            ),
            output_tokens=token_usage["output_tokens"],
            input_tokens=token_usage["input_tokens"],
            total_tokens=token_usage["total_tokens"],
            cache_read_tokens=token_usage["cache_read_tokens"],
            cache_write_tokens=token_usage["cache_write_tokens"],
            cost_usd=token_usage["cost_usd"],
            request_count=token_usage["request_count"],
            token_usage=token_usage,
            workspace=str(execution_result.get("workspace", "")),
            transcript_file=str(transcript_file) if transcript_file else None,
            grade=_grade_to_dict(grade),
        )

    def _induction_prompt(
        self,
        trajectories: Sequence[ExpelTrajectory],
        category: str,
    ) -> str:
        examples = []
        for idx, trajectory in enumerate(trajectories, 1):
            examples.append(
                f"### Successful task {idx}: {trajectory.task_id}\n"
                f"Instruction:\n{_trim(trajectory.task_prompt, 5000)}\n\n"
                f"Successful trajectory:\n{_trim(trajectory.transcript_text, 14000)}"
            )
        existing = "\n".join(
            f"- {workflow.name}: {workflow.description}"
            for workflow in self.workflows[-30:]
        ) or "(none)"
        return f"""
You are the workflow-induction module for an autonomous OpenClaw web/tool agent.
Given one or more successful tasks and their action trajectories, extract
reusable multi-step subroutines as Agent Workflow Memory.

Follow the AWM representation:
- Each workflow has a concise high-level description and an applicability condition.
- Each workflow has at least TWO steps.
- Every step has: (1) current environment state/observation, (2) agent reasoning,
  and (3) one concrete executable action or tool call.
- Extract fine-grained subroutines, not an entire example-specific task.
- Replace non-fixed values, element IDs, paths, queries, names, and message text
  with descriptive placeholders such as {{product-name}} or {{target-path}}.
- Do not invent actions absent from the successful evidence.
- Do not emit similar or overlapping workflows.
- If no reusable multi-step subroutine is supported, return an empty list.

Task category/domain: {category}

Existing workflow catalog (do not duplicate it):
{existing}

Successful experiences:
{chr(10).join(examples)}

Return JSON only with this schema:
{{
  "workflows": [
    {{
      "name": "short reusable name",
      "description": "high-level goal",
      "category": "{category}",
      "applicability": "observable condition for using it",
      "parameters": ["{{descriptive-placeholder}}"],
      "steps": [
        {{
          "state": "observable state before the action",
          "reasoning": "why this action advances the subroutine",
          "action": "tool_name({{parameter}}) or an executable UI action"
        }}
      ]
    }}
  ]
}}
"""

    def _merge_workflows(self, candidates: Iterable[AgentWorkflow]) -> int:
        existing = {_workflow_signature(workflow) for workflow in self.workflows}
        added = 0
        for workflow in candidates:
            signature = _workflow_signature(workflow)
            if signature in existing:
                continue
            self.workflows.append(workflow)
            existing.add(signature)
            added += 1
            if len(self.workflows) >= self.max_workflows:
                break
        return added

    def _induce(
        self,
        trajectories: Sequence[ExpelTrajectory],
        category: str,
        stage: str,
    ) -> int:
        if not trajectories or len(self.workflows) >= self.max_workflows:
            return 0
        prompt = self._induction_prompt(trajectories, category)
        text, token_info = self.model_caller.generate(
            prompt,
            max_output_tokens=4096,
            temperature=0.0,
        )
        self._record_tokens(
            f"{self.api_provider}_workflow_induction",
            token_info,
            {
                "stage": stage,
                "category": category,
                "source_task_ids": [trajectory.task_id for trajectory in trajectories],
            },
        )
        try:
            parsed = _json_from_model(text)
            candidates = _normalize_workflows(
                parsed,
                category=category,
                source_task_ids=[trajectory.task_id for trajectory in trajectories],
            )
        except Exception as exc:
            print(f"  Warning: could not parse induced workflows: {exc}")
            _safe_write_json(
                self.output_dir / "induction_parse_errors" / f"{stage}-{int(time.time())}.json",
                {"prompt_category": category, "response": text, "error": str(exc)},
            )
            return 0
        added = self._merge_workflows(candidates)
        self._persist_memory()
        return added

    def _metric(self, trajectory: ExpelTrajectory, used: Sequence[AgentWorkflow]) -> Dict[str, Any]:
        return {
            "task_id": trajectory.task_id,
            "task_name": trajectory.task_name,
            "status": trajectory.status,
            "score": trajectory.score,
            "max_score": trajectory.max_score,
            "is_success": trajectory.is_success,
            "workflows_available": len(self.workflows),
            "workflows_used": [workflow.workflow_id for workflow in used],
            "input_tokens": trajectory.input_tokens,
            "output_tokens": trajectory.output_tokens,
            "total_tokens": trajectory.total_tokens,
            "cost_usd": trajectory.cost_usd,
            "request_count": trajectory.request_count,
            "execution_time_seconds": trajectory.execution_time_seconds,
        }

    def _write_metrics(self, split: str) -> None:
        metrics = self.metrics[split]
        graded = [item for item in metrics if item.get("score") is not None and item.get("max_score")]
        score = sum(float(item["score"] or 0) for item in graded)
        max_score = sum(float(item["max_score"] or 0) for item in graded)
        _safe_write_json(
            self.output_dir / split / "metrics.json",
            {
                "tasks": metrics,
                "summary": {
                    "tasks": len(metrics),
                    "graded": len(graded),
                    "score": score,
                    "max_score": max_score,
                    "accuracy_pct": score / max_score * 100 if max_score else 0.0,
                    "successes": sum(1 for item in metrics if item.get("is_success")),
                    "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in metrics),
                    "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in metrics),
                    "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in metrics),
                    "cost_usd": sum(float(item.get("cost_usd", 0) or 0) for item in metrics),
                },
            },
        )

    def _run_task(self, task: Task, split: str, index: int) -> ExpelTrajectory:
        selected = self._select_workflows(task)
        prompt = self._prompt_with_memory(task, selected)
        execution_task = _clone_task_with_prompt(task, prompt)
        transcript_dir = self.output_dir / split / "transcripts"
        execution_result, grade = self._execute_and_grade(
            task=task,
            execution_task=execution_task,
            run_id=f"awm-{split}-{index}",
            transcript_dir=transcript_dir,
        )
        transcript_file = transcript_dir / f"{task.task_id}.jsonl"
        trajectory = self._make_trajectory(
            task=task,
            split=split,
            execution_result=execution_result,
            grade=grade,
            transcript_file=transcript_file if transcript_file.exists() else None,
        )
        self.trajectories.append(trajectory)
        self._record_tokens(
            f"openclaw_{split}_attempt",
            trajectory.token_usage,
            {"task_id": task.task_id, "success": trajectory.is_success},
        )
        self.metrics[split].append(self._metric(trajectory, selected))
        self._write_metrics(split)
        _safe_write_json(
            self.output_dir / split / f"problem_{index:04d}.json",
            {
                **asdict(trajectory),
                "selected_workflows": [asdict(workflow) for workflow in selected],
            },
        )
        score = (
            f"{grade.score:.2f}/{grade.max_score:.2f}"
            if grade
            else "not graded"
        )
        print(f"  status={trajectory.status} grade={score} success={trajectory.is_success}")
        return trajectory

    def _parallel_task_worker(
        self,
        task: Task,
        split: str,
        index: int,
    ) -> Tuple[ExpelTrajectory, List[AgentWorkflow]]:
        """Run one task with isolated OpenClaw agent/session state."""
        worker = copy.copy(self)
        worker.agent_id = f"{self.agent_id}-task-{index:04d}"
        worker.token_records = []
        worker.trajectories = []
        worker.metrics = {"train": [], "eval": [], "online": []}
        # The main runner installs the frozen skill before workers start.
        # Avoid concurrent writes to the same global SKILL.md.
        worker._persist_memory = lambda: None  # type: ignore[method-assign]
        worker._setup_agent()

        selected = worker._select_workflows(task)
        execution_task = _clone_task_with_prompt(
            task,
            worker._prompt_with_memory(task, selected),
        )
        transcript_dir = (
            self.output_dir
            / split
            / "transcripts"
            / f"task_{index:04d}"
        )
        execution_result, grade = worker._execute_and_grade(
            task=task,
            execution_task=execution_task,
            run_id=f"awm-{split}-{index}",
            transcript_dir=transcript_dir,
        )
        transcript_file = transcript_dir / f"{task.task_id}.jsonl"
        trajectory = worker._make_trajectory(
            task=task,
            split=split,
            execution_result=execution_result,
            grade=grade,
            transcript_file=transcript_file if transcript_file.exists() else None,
        )
        return trajectory, selected

    def _save_parallel_result(
        self,
        *,
        trajectory: ExpelTrajectory,
        selected: Sequence[AgentWorkflow],
        split: str,
        index: int,
    ) -> None:
        """Merge an isolated worker result into deterministic main artifacts."""
        self.trajectories.append(trajectory)
        self._record_tokens(
            f"openclaw_{split}_attempt",
            trajectory.token_usage,
            {"task_id": trajectory.task_id, "success": trajectory.is_success},
        )
        self.metrics[split].append(self._metric(trajectory, selected))
        self._write_metrics(split)
        _safe_write_json(
            self.output_dir / split / f"problem_{index:04d}.json",
            {
                **asdict(trajectory),
                "selected_workflows": [asdict(workflow) for workflow in selected],
            },
        )
        grade = trajectory.grade
        score = (
            f"{float(grade['score']):.2f}/{float(grade['max_score']):.2f}"
            if grade.get("score") is not None and grade.get("max_score") is not None
            else "not graded"
        )
        print(
            f"  [{split} {index}] {trajectory.task_id}: "
            f"status={trajectory.status} grade={score} success={trajectory.is_success}"
        )

    def _run_tasks_parallel(
        self,
        tasks: Sequence[Task],
        split: str,
    ) -> List[ExpelTrajectory]:
        """Run a frozen-memory phase concurrently and preserve task order."""
        indexed_tasks = list(enumerate(tasks, 1))
        self._persist_memory()

        def process(item: Tuple[int, Task]) -> Tuple[ExpelTrajectory, List[AgentWorkflow]]:
            index, task = item
            return self._parallel_task_worker(task, split, index)

        print(
            f"Running {len(indexed_tasks)} {split} tasks with "
            f"{min(self.num_workers, len(indexed_tasks))} workers"
        )
        outcomes = parallel_utils.parallel_map_ordered(
            process,
            indexed_tasks,
            num_workers=self.num_workers,
        )
        trajectories: List[ExpelTrajectory] = []
        for (index, _task), (trajectory, selected) in zip(indexed_tasks, outcomes):
            self._save_parallel_result(
                trajectory=trajectory,
                selected=selected,
                split=split,
                index=index,
            )
            trajectories.append(trajectory)
        return trajectories

    def run_offline(self, tasks: Sequence[Task]) -> None:
        if len(tasks) < 2:
            raise ValueError("AWM offline mode requires at least two tasks")
        split_at = len(tasks) // 2
        train_tasks = list(tasks[:split_at])
        eval_tasks = list(tasks[split_at:])
        _safe_write_json(
            self.output_dir / "split.json",
            {
                "mode": "offline",
                "split_policy": "deterministic loader-order half split",
                "train_tasks": [{"task_id": task.task_id, "name": task.name} for task in train_tasks],
                "eval_tasks": [{"task_id": task.task_id, "name": task.name} for task in eval_tasks],
            },
        )
        print(f"Loaded {len(tasks)} tasks; train={len(train_tasks)} eval={len(eval_tasks)}")
        successes_by_category: Dict[str, List[ExpelTrajectory]] = {}
        if self.num_workers > 1 and len(train_tasks) > 1:
            train_trajectories = self._run_tasks_parallel(train_tasks, "train")
        else:
            train_trajectories = []
            for index, task in enumerate(train_tasks, 1):
                print(f"\n[train {index}/{len(train_tasks)}] {task.task_id} — {task.name}")
                train_trajectories.append(self._run_task(task, "train", index))

        for task, trajectory in zip(train_tasks, train_trajectories):
            if trajectory.is_success:
                successes_by_category.setdefault(task.category or "general", []).append(trajectory)

        for category, trajectories in successes_by_category.items():
            for batch_index, batch in enumerate(
                _split_chunks(trajectories, self.induction_batch_size),
                1,
            ):
                added = self._induce(
                    batch,
                    category,
                    stage=f"offline-{_slug(category)}-{batch_index}",
                )
                print(f"Induced {added} new workflow(s) from {category} batch {batch_index}")

        if self.num_workers > 1 and len(eval_tasks) > 1:
            self._run_tasks_parallel(eval_tasks, "eval")
        else:
            for index, task in enumerate(eval_tasks, 1):
                print(f"\n[eval {index}/{len(eval_tasks)}] {task.task_id} — {task.name}")
                self._run_task(task, "eval", index)

    def run_online(self, tasks: Sequence[Task]) -> None:
        if self.num_workers > 1:
            print(
                "Online AWM is causally sequential; ignoring --num-workers="
                f"{self.num_workers} and using one worker."
            )
        _safe_write_json(
            self.output_dir / "split.json",
            {
                "mode": "online",
                "stream_tasks": [{"task_id": task.task_id, "name": task.name} for task in tasks],
            },
        )
        print(f"Loaded {len(tasks)} online-stream tasks")
        for index, task in enumerate(tasks, 1):
            print(f"\n[online {index}/{len(tasks)}] {task.task_id} — {task.name}")
            trajectory = self._run_task(task, "online", index)
            if trajectory.is_success:
                added = self._induce(
                    [trajectory],
                    task.category or "general",
                    stage=f"online-{index}-{task.task_id}",
                )
                print(f"  Induced {added} new workflow(s); memory now has {len(self.workflows)}")

    def run(self) -> None:
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _safe_write_json(self.output_dir / "run_config.json", self._run_config())
        self._write_token_usage()
        self._setup_agent()
        self._load_initial_memory()
        self._persist_memory()
        tasks = self._load_tasks()
        if not tasks:
            raise ValueError("No tasks matched --suite")
        if self.mode == "offline":
            self.run_offline(tasks)
            summary_split = "eval"
        else:
            self.run_online(tasks)
            summary_split = "online"

        _safe_write_json(
            self.output_dir / "trajectories.json",
            [asdict(trajectory) for trajectory in self.trajectories],
        )
        self._persist_memory()
        summary_path = self.output_dir / summary_split / "metrics.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8")).get("summary", {})
        totals = self._token_summary()["totals"]
        print("\n" + "=" * 80)
        print("Agent Workflow Memory PinchBench Complete")
        print("=" * 80)
        print(f"Mode: {self.mode}")
        print(f"Workflows learned: {len(self.workflows)}")
        print(f"{summary_split.title()} accuracy: {float(summary.get('accuracy_pct', 0) or 0):.1f}%")
        print(
            "Token usage: "
            f"input={int(totals['input_tokens'])} "
            f"output={int(totals['output_tokens'])} "
            f"thinking={int(totals['thinking_tokens'])} "
            f"total={int(totals['total_tokens'])} "
            f"cost=${totals['cost_usd']:.4f}"
        )
        print(f"Output dir: {self.output_dir}")
        print(f"Elapsed: {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Workflow Memory (AWM) PinchBench runner"
    )
    parser.add_argument("--model", default=None, help="Task-agent model; defaults to --judge, then --api-model.")
    parser.add_argument("--output-dir", default="pinchbench_awm_output")
    parser.add_argument("--suite", default="all", help='"all", "automated-only", or comma-separated task IDs.')
    parser.add_argument("--pinchbench-dir", default=None)
    parser.add_argument("--judge", default=None, help="Judge model; defaults to --model.")
    parser.add_argument("--api-provider", choices=["gemini", "openrouter"], default=None)
    parser.add_argument("--api-key", default=None, help="Prefer GEMINI_API_KEY or OPENROUTER_API_KEY.")
    parser.add_argument("--api-model", default=None, help="Model used to induce workflows.")
    parser.add_argument("--thinking-level", default="high", choices=["low", "medium", "high"])
    parser.add_argument(
        "--agent-thinking-level",
        default=None,
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional task-agent thinking level; separate from workflow induction.",
    )
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--induction-batch-size", type=int, default=6)
    parser.add_argument("--workflow-top-k", type=int, default=8)
    parser.add_argument("--max-workflows", type=int, default=50)
    parser.add_argument("--workflow-memory", default=None, help="Optional existing workflows.json to warm-start memory.")
    parser.add_argument("--skill-name", default="awm-webagent")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1, help="Compatibility only; AWM performs one offline split or online stream.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible task-agent base URL.")
    parser.add_argument("--openclaw-api-key", default=None, help="Optional separate API key for the task agent.")
    parser.add_argument(
        "--agent-backend",
        choices=["direct", "openclaw"],
        default="direct",
        help=(
            "Task executor. 'direct' calls the model API with PinchBench tools "
            "and does not require OpenClaw (default)."
        ),
    )
    parser.add_argument(
        "--direct-max-steps",
        type=int,
        default=40,
        help="Maximum model/tool turns per task for --agent-backend direct.",
    )
    parallel_utils.add_num_workers_argument(parser)
    args = parser.parse_args()

    provider = args.api_provider
    if provider is None:
        provider = (
            "openrouter"
            if os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY")
            else "gemini"
        )
    api_model = args.api_model or (
        "google/gemini-3.1-pro-preview"
        if provider == "openrouter"
        else "gemini-3.1-pro-preview"
    )
    fallback_model = api_model.removeprefix("openrouter/")
    if provider == "gemini" and not fallback_model.startswith(("google/", "gemini/")):
        fallback_model = f"google/{fallback_model}"
    model_id = (args.model or args.judge or fallback_model).removeprefix("openrouter/")

    runner = OpenClawPinchBenchAWM(
        model_id=model_id,
        output_dir=args.output_dir,
        suite=args.suite,
        pinchbench_dir=args.pinchbench_dir,
        judge_model=args.judge,
        api_key=args.api_key,
        api_model=api_model,
        api_provider=provider,
        thinking_level=args.thinking_level,
        agent_thinking_level=args.agent_thinking_level,
        mode=args.mode,
        induction_batch_size=args.induction_batch_size,
        workflow_top_k=args.workflow_top_k,
        max_workflows=args.max_workflows,
        workflow_memory=args.workflow_memory,
        skill_name=args.skill_name,
        timeout_multiplier=args.timeout_multiplier,
        base_url=args.base_url,
        openclaw_api_key=args.openclaw_api_key,
        num_workers=args.num_workers,
        agent_backend=args.agent_backend,
        direct_max_steps=args.direct_max_steps,
    )
    runner.run()


if __name__ == "__main__":
    main()
