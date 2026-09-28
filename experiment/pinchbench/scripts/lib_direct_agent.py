"""Direct API tool-calling agent for PinchBench.

This executor deliberately has no OpenClaw dependency.  It prepares the same
task fixtures, gives an OpenAI-compatible model a compact set of local and web
tools, records a PinchBench-compatible transcript, and returns the standard
execution-result dictionary consumed by lib_grading.py.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import parse, request

from lib_tasks import Task


_BOOTSTRAP_FILES = (
    "SOUL.md",
    "BOOTSTRAP.md",
    "USER.md",
    "IDENTITY.md",
    "HEARTBEAT.md",
    "TOOLS.md",
)


_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories under a workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command with the task workspace as cwd. Use for tests, "
                "code execution, directory creation, and command-line utilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 120,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a public HTTP(S) URL and return its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 50000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web and return result titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _safe_workspace_path(workspace: Path, relative: str) -> Path:
    candidate = (workspace / relative).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes task workspace: {relative}")
    return candidate


def prepare_direct_workspace(
    *,
    skill_dir: Path,
    run_id: str,
    task: Task,
    workspace_root: Optional[Path] = None,
    extra_skills_dir: Optional[Path] = None,
    include_openclaw_workspace_skills: bool = True,
) -> Path:
    """Create a dedicated workspace without requiring the OpenClaw CLI."""
    base = workspace_root or (Path(tempfile.gettempdir()) / "pinchbench-direct")
    workspace = base / run_id / task.task_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            destination = _safe_workspace_path(workspace, file_spec["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(file_spec["content"]), encoding="utf-8")
            continue
        source = skill_dir / "assets" / file_spec["source"]
        destination = _safe_workspace_path(workspace, file_spec["dest"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # Mirror the persistent context that the OpenClaw executor preserves in its
    # agent workspace whenever a compatible ~/.openclaw/workspace exists.  This
    # is optional: a clean SLURM node can still use the direct backend without
    # installing OpenClaw.
    openclaw_workspace = Path.home() / ".openclaw" / "workspace"
    if openclaw_workspace.exists():
        for filename in _BOOTSTRAP_FILES:
            source = openclaw_workspace / filename
            if source.is_file():
                shutil.copy2(source, workspace / filename)

    skill_sources = []
    if include_openclaw_workspace_skills:
        skill_sources.append(openclaw_workspace / "skills")
    if extra_skills_dir:
        skill_sources.append(extra_skills_dir)
    for source in skill_sources:
        if source.exists():
            shutil.copytree(source, workspace / "skills", dirs_exist_ok=True)
    return workspace


def _build_system_prompt(workspace: Path, insight_library_text: Optional[str]) -> str:
    """Build and expose the complete direct-agent prompt deterministically."""
    sections = [
        (
            "You are an autonomous task agent. Complete the user's request with "
            "the provided tools. Work in the current task workspace, inspect the "
            "provided fixtures and relevant skills, perform the requested actions, "
            "validate resulting artifacts, and then give a concise final response."
        )
    ]
    for filename in _BOOTSTRAP_FILES:
        path = workspace / filename
        if path.is_file():
            sections.append(f"## {filename}\n{path.read_text(encoding='utf-8')}")
    available_skills = []
    skills_root = workspace / "skills"
    if skills_root.is_dir():
        for skill_path in sorted(skills_root.glob("*/SKILL.md")):
            text = skill_path.read_text(encoding="utf-8", errors="replace")
            description_match = re.search(
                r"(?mi)^description:\s*[\"']?(.+?)[\"']?\s*$", text[:8000]
            )
            available_skills.append(
                {
                    "name": skill_path.parent.name,
                    "description": (
                        description_match.group(1).strip()
                        if description_match
                        else "Specialized task instructions"
                    ),
                    "path": str(skill_path.relative_to(workspace)),
                }
            )
    if available_skills:
        rendered = [
            "## Available Skills (mandatory)",
            "Before acting, scan this list. If a skill clearly applies, read its "
            "SKILL.md with read_file and follow it. Do not claim to use a skill "
            "without reading it.",
        ]
        for skill in available_skills:
            rendered.append(
                f"- {skill['name']}: {skill['description']} "
                f"(path: {skill['path']})"
            )
        sections.append("\n".join(rendered))
    if insight_library_text:
        sections.append(
            "## Insight Library (from prior PinchBench experience)\n"
            "Apply these reusable insights when they are relevant to the task.\n\n"
            + insight_library_text
        )
    return "\n\n".join(sections)


def _strip_html(document: str) -> str:
    document = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", document)
    document = re.sub(r"(?s)<[^>]+>", " ", document)
    document = html.unescape(document)
    return re.sub(r"\s+", " ", document).strip()


def _execute_tool(name: str, arguments: Dict[str, Any], workspace: Path) -> str:
    if name == "read_file":
        path = _safe_workspace_path(workspace, str(arguments["path"]))
        offset = max(0, int(arguments.get("offset", 0) or 0))
        limit = min(50000, max(1, int(arguments.get("limit", 20000) or 20000)))
        text = path.read_text(encoding="utf-8")
        return text[offset : offset + limit]

    if name == "write_file":
        path = _safe_workspace_path(workspace, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(arguments.get("content", ""))
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path.relative_to(workspace)}"

    if name == "list_files":
        path = _safe_workspace_path(workspace, str(arguments.get("path", ".")))
        maximum = min(1000, max(1, int(arguments.get("max_entries", 200) or 200)))
        if path.is_file():
            return str(path.relative_to(workspace))
        entries = []
        for item in sorted(path.rglob("*")):
            suffix = "/" if item.is_dir() else ""
            entries.append(str(item.relative_to(workspace)) + suffix)
            if len(entries) >= maximum:
                entries.append("...[truncated]")
                break
        return "\n".join(entries) or "(empty workspace)"

    if name == "run_shell":
        command = str(arguments["command"])
        timeout = min(120, max(1, int(arguments.get("timeout_seconds", 30) or 30)))
        result = subprocess.run(
            ["/bin/zsh", "-lc", command],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return output[:50000]

    if name == "web_fetch":
        url = str(arguments["url"])
        if not url.startswith(("http://", "https://")):
            raise ValueError("web_fetch only supports HTTP(S) URLs")
        maximum = min(50000, max(1000, int(arguments.get("max_chars", 20000) or 20000)))
        req = request.Request(url, headers={"User-Agent": "PinchBench-Direct/1.0"})
        with request.urlopen(req, timeout=30) as response:
            raw = response.read(maximum * 4).decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
        text = _strip_html(raw) if "html" in content_type.lower() else raw
        return text[:maximum]

    if name == "web_search":
        query = str(arguments["query"])
        maximum = min(10, max(1, int(arguments.get("max_results", 5) or 5)))
        url = "https://html.duckduckgo.com/html/?" + parse.urlencode({"q": query})
        req = request.Request(url, headers={"User-Agent": "Mozilla/5.0 PinchBench"})
        with request.urlopen(req, timeout=30) as response:
            page = response.read(500000).decode("utf-8", errors="replace")
        matches = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            page,
            flags=re.I | re.S,
        )
        results = []
        for href, title in matches[:maximum]:
            parsed_href = parse.urlparse(html.unescape(href))
            query_values = parse.parse_qs(parsed_href.query)
            resolved = query_values.get("uddg", [href])[0]
            results.append(f"- {_strip_html(title)}\n  {resolved}")
        return "\n".join(results) or "No search results returned."

    raise ValueError(f"Unknown tool: {name}")


def _session_prompts(task: Task) -> List[tuple[str, bool]]:
    sessions = (task.frontmatter or {}).get("sessions")
    if not isinstance(sessions, list) or not sessions:
        return [(task.prompt, False)]
    prompts: List[tuple[str, bool]] = []
    for item in sessions:
        if isinstance(item, str):
            prompts.append((item, False))
        elif isinstance(item, dict):
            prompts.append(
                (
                    str(item.get("prompt") or item.get("message") or ""),
                    bool(item.get("new_session", False)),
                )
            )
    return prompts or [(task.prompt, False)]


def execute_direct_task(
    *,
    task: Task,
    model_id: str,
    run_id: str,
    timeout_multiplier: float,
    skill_dir: Path,
    api_key: str,
    base_url: str,
    output_dir: Optional[Path] = None,
    workspace_root: Optional[Path] = None,
    extra_skills_dir: Optional[Path] = None,
    include_openclaw_workspace_skills: bool = True,
    max_steps: Optional[int] = None,
    thinking_level: Optional[str] = None,
    insight_library_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a PinchBench task through an OpenAI-compatible tool loop."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Direct PinchBench execution requires `pip install openai`.") from exc

    start = time.time()
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    workspace = prepare_direct_workspace(
        skill_dir=skill_dir,
        run_id=run_id,
        task=task,
        workspace_root=workspace_root,
        extra_skills_dir=extra_skills_dir,
        include_openclaw_workspace_skills=include_openclaw_workspace_skills,
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    system = _build_system_prompt(workspace, insight_library_text)
    transcript: List[Dict[str, Any]] = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    status = "success"
    stderr = ""
    steps_used = 0
    execution_manifest_path: Optional[Path] = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        execution_manifest_path = output_dir / f"{task.task_id}.agent_config.json"
        execution_manifest_path.write_text(
            json.dumps(
                {
                    "agent_backend": "direct",
                    "model": model_id,
                    "base_url": base_url,
                    "timeout_seconds": timeout_seconds,
                    "max_steps": max_steps,
                    "provider_generation_defaults": True,
                    "thinking_level": thinking_level,
                    "bootstrap_files": [
                        name for name in _BOOTSTRAP_FILES if (workspace / name).is_file()
                    ],
                    "skill_files": sorted(
                        str(path.relative_to(workspace))
                        for path in (workspace / "skills").rglob("*")
                        if path.is_file()
                    ) if (workspace / "skills").exists() else [],
                    "system_prompt": system,
                    "tools": _TOOLS,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for session_prompt, new_session in _session_prompts(task):
        if new_session:
            messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": session_prompt})
        transcript.append(
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": session_prompt}],
                },
            }
        )
        session_finished = False
        while not session_finished:
            if time.time() - start >= timeout_seconds:
                status = "timeout"
                break
            if max_steps is not None and steps_used >= max_steps:
                status = "max_steps"
                break
            steps_used += 1
            kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "tools": _TOOLS,
                "timeout": max(1.0, timeout_seconds - (time.time() - start)),
            }
            if thinking_level and thinking_level not in {"off", "minimal"}:
                kwargs["extra_body"] = {"reasoning": {"effort": thinking_level}}
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                status = "error"
                stderr = str(exc)
                break

            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                total_usage["input_tokens"] += prompt_tokens
                total_usage["output_tokens"] += completion_tokens
                total_usage["total_tokens"] += total_tokens or prompt_tokens + completion_tokens
                usage_data = (
                    usage.model_dump()
                    if hasattr(usage, "model_dump")
                    else {}
                )
                prompt_details = usage_data.get("prompt_tokens_details") or {}
                total_usage["cache_read_tokens"] += int(
                    prompt_details.get("cached_tokens", 0) or 0
                )
                total_usage["cache_write_tokens"] += int(
                    prompt_details.get("cache_write_tokens", 0) or 0
                )
                total_usage["cost_usd"] += float(usage_data.get("cost", 0.0) or 0.0)
            total_usage["request_count"] += 1

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = list(message.tool_calls or [])
            assistant_blocks: List[Dict[str, Any]] = []
            if content:
                assistant_blocks.append({"type": "text", "text": content})
            api_tool_calls = []
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                assistant_blocks.append(
                    {
                        "type": "toolCall",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    }
                )
                api_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "{}",
                        },
                    }
                )
            transcript.append(
                {
                    "type": "message",
                    "message": {"role": "assistant", "content": assistant_blocks},
                }
            )
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if api_tool_calls:
                assistant_message["tool_calls"] = api_tool_calls
            messages.append(assistant_message)

            if not tool_calls:
                session_finished = True
                continue
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    result = _execute_tool(tool_call.function.name, arguments, workspace)
                except Exception as exc:
                    result = f"TOOL ERROR: {type(exc).__name__}: {exc}"
                result = result[:50000]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
                transcript.append(
                    {
                        "type": "message",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": tool_call.id,
                            "content": [{"type": "text", "text": result}],
                        },
                    }
                )
        if status != "success":
            break

    execution_time = time.time() - start
    if output_dir:
        transcript_path = output_dir / f"{task.task_id}.jsonl"
        transcript_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in transcript),
            encoding="utf-8",
        )
    return {
        "agent_id": "direct-api",
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "usage": total_usage,
        "workspace": str(workspace),
        "exit_code": 0 if status == "success" else 1,
        "timed_out": status == "timeout",
        "execution_time": execution_time,
        "stdout": "",
        "stderr": stderr,
        "steps_used": steps_used,
        "agent_config_path": (
            str(execution_manifest_path) if execution_manifest_path else None
        ),
    }
