from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fot.fot_client import (
    INSIGHT_LIBRARY_HEADER,
    OpenClawFoTClient,
    format_insights_section,
    parse_reasoning_traces,
)


class StubFoTClient(OpenClawFoTClient):
    def __init__(self, responses: dict[str, Any], tmp_path: Path):
        super().__init__(agent_name="test", workspace=tmp_path, output_dir=str(tmp_path / "out"))
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def _call_model(self, prompt: str, step_name: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((step_name, prompt))
        response = self.responses[step_name]
        if isinstance(response, list):
            return response.pop(0), {}
        return response, {}


TRACES = {
    "trace_boundaryCheck": "Check boundary conditions before trusting the general formula.",
    "polynomialFactoring": "Factor polynomials into simpler expressions to solve equations.",
}


def test_parse_reasoning_traces_prefixes_names() -> None:
    skills = parse_reasoning_traces("Here you go:\n```json\n" + json.dumps(TRACES) + "\n```")
    assert set(skills) == {"trace_boundaryCheck", "trace_polynomialFactoring"}


@pytest.mark.parametrize(
    "response, message",
    [
        ("", "empty"),
        ("no json here", "no complete JSON object"),
        ('{"trace_a": "A long enough description of a technique", "trace_b": oops}', "not valid JSON"),
        (json.dumps({"trace_short": "too short"}), "shorter than"),
        (json.dumps({"trace_list": ["a", "b"]}), "must be a string"),
        ("{}", "empty"),
    ],
)
def test_parse_reasoning_traces_raises_on_invalid(response: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_reasoning_traces(response)


def test_format_insights_section_matches_prompt_1() -> None:
    assert format_insights_section(None) == ""
    section = format_insights_section({"insight_x": "Use x."})
    assert section.startswith(INSIGHT_LIBRARY_HEADER)
    assert "actively apply the relevant techniques" in section
    assert section.endswith("\n\n")


def test_solve_problem_runs_prompts_1_to_3(tmp_path: Path) -> None:
    client = StubFoTClient(
        {"solution": "the answer is 4", "reflection": "I checked the boundary.", "insights": json.dumps(TRACES)},
        tmp_path,
    )
    result = client.solve_problem("2+2?", insights_section=format_insights_section({"insight_x": "Use x."}))

    assert [name for name, _ in client.calls] == ["solution", "reflection", "insights"]
    assert client.calls[0][1].endswith("Problem: 2+2?")
    assert '"trace_name": "description"' in client.calls[2][1]
    assert set(result["insight_book"]) == {"trace_boundaryCheck", "trace_polynomialFactoring"}
    saved = Path(client.save_reasoning(result, "problem_000001.json"))
    assert json.loads(saved.read_text(encoding="utf-8")) == result["insight_book"]


def test_extract_from_trace_raises_on_invalid_json_without_retry(tmp_path: Path) -> None:
    client = StubFoTClient({"reflection": "reflection text", "insights": "not json"}, tmp_path)
    with pytest.raises(ValueError, match="no complete JSON object"):
        client.extract_from_trace(problem="task", solution="transcript")
    assert [name for name, _ in client.calls] == ["reflection", "insights"]
