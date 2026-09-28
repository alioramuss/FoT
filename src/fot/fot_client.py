"""FoT local reasoning (client) pipeline.

This module is the OpenClaw port of the current Federation over Text client
(``client.py`` / ``ChainOfThoughtReader`` in the research code). Each agent runs
three local stages on its own task and uploads only the distilled reasoning
traces, never the raw task instance:

1. Solution generation (Prompt 1). When an insight library is available the
   agent is told to review it and apply the relevant techniques.
2. Reflection on the solution (Prompt 2): extract procedural knowledge.
3. Reasoning-trace generation (Prompt 3): package the reflection as a flat JSON
   object ``{"trace_<name>": "<core idea + when to use>"}``.

Prompt numbers follow Appendix E.1 of "Federation over Text: Insight Sharing for
Multi-Agent Reasoning". The research code caps generation at 32,768 tokens for
solutions, 16,384 for reflection, and 8,192 for trace extraction; OpenClaw owns
those limits here, so they are recorded as constants for custom subclasses.

There are no fallbacks: an OpenClaw failure, an empty response, or a trace
response that is not a valid flat JSON object of ``trace_*`` strings raises.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fotclaw.openclaw_adapter import run_openclaw_prompt

SOLUTION_MAX_NEW_TOKENS = 32768
REFLECTION_MAX_NEW_TOKENS = 16384
TRACE_MAX_NEW_TOKENS = 8192

MIN_TRACE_DESCRIPTION_CHARS = 20
TRACE_PREFIXES = ("trace_", "insight_")
INSIGHT_LIBRARY_HEADER = "Available Insights to Guide Your Solution:"
INSIGHT_LIBRARY_INSTRUCTIONS = (
    "INSTRUCTIONS: Review the insights above and actively apply the relevant "
    "techniques from insights to solve this problem. Consider which insights can "
    "help you approach the problem more effectively."
)

def format_insights_section(insight_library: dict[str, str] | str | None) -> str:
    """Render Prompt 1's insight-library preamble.

    Returns an empty string when there is no library, in which case Prompt 1
    reduces to ``Problem: {problem}`` (isolated agents).
    """

    if not insight_library:
        return ""
    if isinstance(insight_library, dict):
        body = json.dumps(insight_library, indent=2, ensure_ascii=False)
    else:
        body = str(insight_library).strip()
    if not body:
        return ""
    return f"{INSIGHT_LIBRARY_HEADER}\n{body}\n\n---\n{INSIGHT_LIBRARY_INSTRUCTIONS}\n\n"


class LocalReasoningClient(ABC):
    """Abstract FoT local reasoning pipeline with overridable steps 1, 2, and 3."""

    def __init__(
        self,
        *,
        agent_name: str,
        workspace: str | Path,
        openclaw_path: str | None = None,
        output_dir: str = "output",
        timeout_seconds: float = 3600.0,
    ):
        self.agent_name = agent_name
        self.workspace = Path(workspace)
        self.openclaw_path = openclaw_path
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.reasoning_steps: list[dict[str, Any]] = []
        self.insight_book: dict[str, str] = {}

    def reset_state(self) -> None:
        self.reasoning_steps = []
        self.insight_book = {}

    @abstractmethod
    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        """Run local reasoning step 1 (solution generation) and return a step payload."""

    @abstractmethod
    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        """Run local reasoning step 2 (reflection) and return a step payload."""

    @abstractmethod
    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run local reasoning step 3 (reasoning-trace extraction) and return a step payload."""

    def build_existing_solution_step(self, *, problem: str, solution: str) -> dict[str, Any]:
        """Build a synthetic step 1 result when FoTClaw already has the solution transcript."""

        return {
            "step": 1,
            "name": "Existing Solution",
            "prompt": problem,
            "response": solution,
            "usage": {},
            "timestamp": time.time(),
            "source": "existing_solution",
        }

    def _record_step(
        self,
        result: dict[str, Any],
        *,
        step_number: int,
        default_name: str,
        require_response: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"FoT local step {step_number} must return a dict.")
        step = dict(result)
        step["step"] = step_number
        step.setdefault("name", default_name)
        step.setdefault("timestamp", time.time())
        if require_response:
            response = step.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"FoT local step {step_number} must provide a non-empty string `response`.")
        self.reasoning_steps.append(step)
        return step

    def _normalize_insight_book(self, step3_result: dict[str, Any]) -> dict[str, str]:
        """Validate step-3 output as the flat trace dictionary uploaded to the server.

        Prompt 3 names uploaded artifacts ``trace_*``. Names that already start
        with ``trace_`` or ``insight_`` are kept; anything else is prefixed with
        ``trace_``. Any non-string or too-short description raises ``ValueError``.
        """

        raw = (
            step3_result.get("valid_skills")
            or step3_result.get("skills_extracted")
            or step3_result.get("insight_book")
            or {}
        )
        if not isinstance(raw, dict):
            raise ValueError("Reasoning traces must be a dict of {trace_name: description}.")
        return _validate_traces(raw)

    def _finalize(
        self,
        *,
        problem: str,
        step1: dict[str, Any],
        step2: dict[str, Any],
        step3: dict[str, Any],
    ) -> dict[str, Any]:
        valid_skills = self._normalize_insight_book(step3)
        if not valid_skills:
            raise ValueError("Reasoning traces are empty: no valid traces were extracted.")
        self.insight_book.update(valid_skills)
        usage = {
            "reflection": step2.get("usage", {}),
            "insight_extraction": step3.get("usage", {}),
        }
        if step1.get("source") != "existing_solution":
            usage = {"solution": step1.get("usage", {}), **usage}
        return {
            "problem": problem,
            "task": problem,
            "solution": step1.get("response", ""),
            "reflection": step2.get("response", ""),
            "skills_extracted": valid_skills,
            "skills_used": list(valid_skills.keys()),
            "insight_book": self.insight_book,
            "total_steps": len(self.reasoning_steps),
            "usage": usage,
        }

    def solve_problem(
        self,
        task: str,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        """Run Solution -> Reflection -> Reasoning-trace extraction on one task."""

        self.reset_state()
        step1 = self._record_step(
            self.local_step_1(
                task,
                custom_solution_instruction=custom_solution_instruction,
                insights_section=insights_section,
            ),
            step_number=1,
            default_name="Local Step 1",
        )
        step2 = self._record_step(
            self.local_step_2(task, step1),
            step_number=2,
            default_name="Local Step 2",
        )
        step3 = self._record_step(
            self.local_step_3(task, step1, step2),
            step_number=3,
            default_name="Local Step 3",
            require_response=False,
        )
        return self._finalize(problem=task, step1=step1, step2=step2, step3=step3)

    def extract_from_trace(self, *, problem: str, solution: str) -> dict[str, Any]:
        """Run Reflection -> Reasoning-trace extraction on an existing OpenClaw transcript."""

        self.reset_state()
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("Solution transcript is empty.")
        step1 = self.build_existing_solution_step(problem=problem, solution=solution)
        step2 = self._record_step(
            self.local_step_2(problem, step1),
            step_number=2,
            default_name="Local Step 2",
        )
        step3 = self._record_step(
            self.local_step_3(problem, step1, step2),
            step_number=3,
            default_name="Local Step 3",
            require_response=False,
        )
        return self._finalize(problem=problem, step1=step1, step2=step2, step3=step3)

    def save_reasoning(self, reasoning_result: dict[str, Any], output_path: str | None = None) -> str:
        """Save only the trace book as a flat JSON object ``{"trace_name": "description"}``."""

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        insight_book = reasoning_result.get("insight_book", {})
        if not insight_book:
            raise ValueError("Reasoning traces are empty: no insight book can be saved.")
        if not output_path:
            safe_name = re.sub(r"[^\w\s-]", "", str(reasoning_result.get("problem", "reasoning"))[:50])
            safe_name = re.sub(r"[-\s]+", "_", safe_name).strip("_") or "reasoning"
            output_path = str(output_dir / f"{safe_name}.json")
        path = Path(output_path)
        if not path.is_absolute():
            path = output_dir / path
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.write_text(json.dumps(insight_book, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


class OpenClawFoTClient(LocalReasoningClient):
    """Default OpenClaw-backed implementation of the local FoT pipeline."""

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

    # ------------------------------------------------------------------ prompts

    def _get_solution_prompt(
        self,
        problem: str,
        custom_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> str:
        """Prompt 1: generating the solution, optionally guided by the insight library."""

        insights_text = insights_section or ""
        custom_section = f"\n\n{custom_instruction}" if custom_instruction else ""
        return f"{insights_text}Problem: {problem}{custom_section}"

    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        """Prompt 2: reflecting on the solution to extract procedural knowledge."""

        return f"""
Analyze the solution below to extract procedural knowledge that reflects the reasoning traces.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract the fundamental techniques used in resolution that can be packaged as reasoning traces. Focus on:

1. What step-by-step procedures were used? How can these be repeated?
2. What conditions determined which approach to use? When should each technique apply?
3. What methods, strategies, or workflows can be applied to similar problems?
4. What made this approach effective? What should someone know to use it correctly?
5. What types of problems would benefit from these techniques?

Output your analysis covering:

### I. Procedural Knowledge
- Break down the solution into clear, repeatable procedures
- The extracted traces should be concrete solutions rather than general principles.

### II. Reusable Techniques and Methods
- List specific techniques, strategies, or workflows used
- The techniques should be solid on practical questions rather than very general and high-level principles.
- For each technique, identify:
  * When it should be used (conditions/triggers)
  * How it was applied (concrete steps)
  * Why it was effective (insights)
  * What problems it could solve (applicability)

### III. Critical Insights and Guidelines
- What key insights made this solution work?
- What common pitfalls should be avoided?
- What variations or edge cases should be considered?

Focus on extracting actionable, procedural knowledge that can be packaged as reusable insights for similar problems.
""".strip()

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Prompt 3: generating reasoning traces as a flat ``trace_*`` JSON object."""

        prompt_template = """
Extract reasoning traces from the solution below. Analyze the solution and reflection to identify concrete, actionable traces so that similar problems can be solved via the traces

Problem: {problem}

Solution: {solution}

Reflection: {reflection}

**Your Task:**
Identify and extract all reusable reasoning traces, techniques, and methods used in the solution. Each trace should be a concrete procedure that can guide someone to solve similar problems.

**What Makes a Good Reasoning Trace:**
- A specific technique or method that was used in the solution.
- Something that can be applied to similar problems, not just this one.
- Includes guidance on when and how to use it with clear steps that can be followed if necessary.
- Not repetition of already well-known or commonly adopted techniques.
- Not too general and high-level but contains actionable procedural knowledge.

**Description Must Include:**
1. **Core idea**: The fundamental concept of what this trace is about. What is the main technique or method? What does it do?

2. **When to use**: Explain when this skill should be applied. What types of problems? What conditions must be met? What situations trigger this skill?

**Output Format (Simple JSON):**
Output a simple JSON object with skill names as keys and descriptions as string values:

{{"trace_name": "description"}}

Format Rules:
- Use valid JSON format
- Each trace name must start with "trace_"
- Keep JSON simple - no nested objects, just key-value pairs
- Escape quotes in descriptions with backslash: \\"

**Example:**
{{
  "trace_polynomialFactoring": "The major idea is how we can turn a polynomial into a product of simpler expressions. This skill is particularly useful for quadratic and higher-degree polynomial equations where factoring can simplify the problem. Factoring reduces complex polynomials to simpler equations. When solving equations with polynomial expressions that can be factored, especially when the polynomial has recognizable patterns like difference of squares (a²-b²), perfect square trinomials (a²±2ab+b²), or common factors.",
  "trace_depthFirstSearchImplementation": "This algorithm is essential for problems involving path finding, cycle detection, topological sorting, connected components, or exploring all possible solutions in a search space. DFS explores depth before breadth, using stack-based recursion or explicit stack. It is memory-efficient for deep structures and naturally handles backtracking. The visited set prevents infinite loops and redundant work. DFS is the foundation for many graph algorithms including topological sort, strongly connected components, and maze solving. When you need to explore or traverse a graph, tree, or nested structure systematically, going as deep as possible before backtracking. Use DFS when you need to visit all nodes in a connected component, find paths between nodes, detect cycles, or explore recursive structures like file systems, nested data, or game states. "
}}

**Output your response as a valid JSON object only:**
"""
        return prompt_template.format(problem=problem, solution=solution, reflection=reflection).strip()

    # -------------------------------------------------------------------- steps

    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._get_solution_prompt(problem, custom_solution_instruction, insights_section)
        response, usage = self._call_model(prompt, "solution")
        return {
            "name": "Solution Generation",
            "prompt": prompt,
            "response": response,
            "usage": usage,
        }

    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        prompt = self._get_reflection_prompt(problem, str(step1_result.get("response", "")))
        response, usage = self._call_model(prompt, "reflection")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Solution reflection is empty.")
        return {
            "name": "Reflection",
            "prompt": prompt,
            "response": response,
            "usage": usage,
        }

    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._get_behavior_prompt(
            problem,
            str(step1_result.get("response", "")),
            str(step2_result.get("response", "")),
        )
        response, usage = self._call_model(prompt, "insights")
        traces = parse_reasoning_traces(response)
        return {
            "name": "Insight Extraction",
            "prompt": prompt,
            "response": response,
            "skills": traces,
            "valid_skills": traces,
            "usage": usage,
        }


ChainOfThoughtReader = OpenClawFoTClient


# ---------------------------------------------------------------------- parsing


def _validate_traces(raw: dict[Any, Any]) -> dict[str, str]:
    if not raw:
        raise ValueError("Reasoning traces are empty: no traces were extracted.")
    traces: dict[str, str] = {}
    for raw_name, raw_description in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Reasoning trace has an empty name.")
        if not name.startswith(TRACE_PREFIXES):
            name = f"trace_{name}"
        if not isinstance(raw_description, str):
            raise ValueError(f"Reasoning trace '{name}' must be a string, got {type(raw_description).__name__}.")
        description = re.sub(r"\s+", " ", raw_description).strip()
        if len(description) < MIN_TRACE_DESCRIPTION_CHARS:
            raise ValueError(
                f"Reasoning trace '{name}' description is shorter than {MIN_TRACE_DESCRIPTION_CHARS} characters."
            )
        traces[name] = description
    return traces


def _find_json_object(text: str) -> str | None:
    """Return the fenced JSON object, or the first balanced ``{...}`` object, in ``text``."""

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1)
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


def parse_reasoning_traces(response: str) -> dict[str, str]:
    """Parse Prompt 3's response into ``{trace_name: description}``; raise on anything invalid."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Reasoning-trace extraction response is empty.")
    json_str = _find_json_object(response)
    if json_str is None:
        raise ValueError("Reasoning-trace extraction response contains no complete JSON object.")
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Reasoning-trace extraction response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Reasoning-trace extraction JSON must be an object.")
    return _validate_traces(payload)


def _extract_json_dict(text: str) -> dict[str, str]:
    """Backward-compatible alias for :func:`parse_reasoning_traces`."""

    return parse_reasoning_traces(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenClaw-backed FoT client (Prompts 1-3).")
    parser.add_argument("--agent", required=True, help="OpenClaw agent name to use")
    parser.add_argument("--workspace", required=True, help="Workspace path for the OpenClaw agent")
    parser.add_argument("--task", required=True, help="Problem or task prompt")
    parser.add_argument("--output", default="output", help="Output directory for reasoning trace JSON")
    parser.add_argument("--openclaw-path", default=None, help="Path to the openclaw binary")
    parser.add_argument(
        "--insight-library",
        default=None,
        help="Optional insight.json to prepend with Prompt 1 before solving",
    )
    args = parser.parse_args()

    insights_section = ""
    if args.insight_library:
        library = json.loads(Path(args.insight_library).read_text(encoding="utf-8"))
        insights_section = format_insights_section(library)

    reader = OpenClawFoTClient(
        agent_name=args.agent,
        workspace=args.workspace,
        openclaw_path=args.openclaw_path,
        output_dir=args.output,
    )
    result = reader.solve_problem(args.task, insights_section=insights_section or None)
    path = reader.save_reasoning(result)
    print(json.dumps({"output_path": path, "traces": result["insight_book"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
