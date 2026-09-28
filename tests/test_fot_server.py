from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fot.fot_server import (
    OpenClawFoTServer,
    _extract_json_object,
    validate_insight_library,
    validate_profiling,
)


class StubFoTServer(OpenClawFoTServer):
    def __init__(self, responses: dict[str, Any], tmp_path: Path, **kwargs: Any):
        super().__init__(agent_name="test", workspace=tmp_path, **kwargs)
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def _call_model(self, prompt: str, step_name: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((step_name, prompt))
        response = self.responses[step_name]
        if isinstance(response, list):
            return response.pop(0), {}
        return response, {}


PROFILING = {
    "clusters": [
        {
            "cluster_id": 7,
            "cluster_name": "Verification",
            "traces": ["trace_check", "trace_prove"],
            "theme": "Validate a result",
        }
    ],
    "relationships": [
        {
            "trace_a": "trace_check",
            "trace_b": "trace_prove",
            "relationship_type": "complementary",
            "description": "Independent checks strengthen the proof.",
        }
    ],
}


def _write_trace(directory: Path, index: int, book: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"problem_{index:06d}.json"
    path.write_text(json.dumps({"problem": "p", "insight_book": book}), encoding="utf-8")
    return path


def test_extract_json_object_preserves_nested_values() -> None:
    response = """```json
    {
      "clusters": [{"cluster_id": 0, "traces": ["trace_a", "trace_b"]}],
      "relationships": [{"trace_a": "trace_a", "trace_b": "trace_b"}]
    }
    ```"""

    parsed = _extract_json_object(response)

    assert parsed is not None
    assert parsed["clusters"][0]["traces"] == ["trace_a", "trace_b"]
    assert parsed["relationships"][0]["trace_b"] == "trace_b"


def test_extract_json_object_finds_balanced_object_in_prose() -> None:
    parsed = _extract_json_object('Sure! {"insight_a": "uses {braces} inside"} trailing }')
    assert parsed == {"insight_a": "uses {braces} inside"}
    assert _extract_json_object('{"insight_a": "trailing comma",}') is None


def test_prompt_4_profiling_is_passed_to_prompt_5(tmp_path: Path) -> None:
    server = StubFoTServer({"profiling": json.dumps(PROFILING)}, tmp_path)
    collection = {
        "insight_store": {
            "trace_check": "Check the result numerically.",
            "trace_prove": "Prove the result symbolically.",
        }
    }

    step_2 = server.global_step_2(collection)
    prompt_5 = server._get_knowledge_extraction_prompt(collection["insight_store"], step_2["profiling"])

    assert step_2["profiling"] == PROFILING
    # The complete profiling is forwarded (not flattened) to Prompt 5.
    assert '"cluster_name": "Verification"' in prompt_5
    assert '"trace_check",' in prompt_5 and '"trace_prove"' in prompt_5
    assert '"theme": "Validate a result"' in prompt_5
    assert '"relationship_type": "complementary"' in prompt_5


def test_malformed_profiling_raises() -> None:
    validate_profiling(PROFILING)
    with pytest.raises(ValueError, match="Cluster 1"):
        validate_profiling({**PROFILING, "clusters": PROFILING["clusters"] + [{"cluster_id": 1, "traces": []}]})
    with pytest.raises(ValueError, match="Relationship 1"):
        validate_profiling({**PROFILING, "relationships": PROFILING["relationships"] + [{"trace_a": "x"}]})


def test_unparseable_profiling_raises(tmp_path: Path) -> None:
    server = StubFoTServer({"profiling": "clusters: verification traces belong together"}, tmp_path)
    with pytest.raises(ValueError, match="not a valid JSON object"):
        server.global_step_2({"insight_store": {"trace_a": "Check the result numerically."}})


def test_invalid_insight_library_raises() -> None:
    validate_insight_library({"insight_ok": "keep me"})
    with pytest.raises(ValueError, match="must start with 'insight_'"):
        validate_insight_library({"insight_ok": "keep me", "bad": "x"})
    with pytest.raises(ValueError, match="content is empty"):
        validate_insight_library({"insight_empty": " "})


def test_invalid_aggregation_response_raises_without_retry(tmp_path: Path) -> None:
    server = StubFoTServer({"profiling": json.dumps(PROFILING), "aggregate": "not json at all"}, tmp_path)
    with pytest.raises(ValueError, match="Knowledge-extraction response is not a valid JSON object"):
        server.aggregate_and_build_encyclopedia(
            json_files=[str(_write_trace(tmp_path / "t", 1, {"trace_a": "Check the result numerically."}))],
            output_dir=str(tmp_path / "out"),
        )
    assert [name for name, _ in server.calls] == ["profiling", "aggregate"]


def test_full_aggregation_merges_existing_library_and_checkpoints(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    files = [
        _write_trace(traces, 2, {"trace_prove": "Prove the result symbolically."}),
        _write_trace(traces, 1, {"trace_check": "Check the result numerically."}),
    ]
    output_dir = tmp_path / "aggregate"
    existing = tmp_path / "insight.json"
    existing.write_text(json.dumps({"insight_old": "Previous round insight."}), encoding="utf-8")
    library = {"insight_verify": "Verify results with two independent methods."}
    server = StubFoTServer(
        {"profiling": json.dumps(PROFILING), "aggregate": json.dumps(library)},
        tmp_path,
    )

    result = server.aggregate_and_build_encyclopedia(
        json_files=[str(path) for path in files],
        output_dir=str(output_dir),
        existing_encyclopedia_path=str(existing),
    )
    json_path, markdown_path = server.save_results(result, output_dir=str(output_dir))

    # Traces keep their names plus a global index, in file order.
    assert list(result["insight_store"]) == ["trace_prove_000001", "trace_check_000002"]
    assert result["encyclopedia_dict"] == library
    assert [name for name, _ in server.calls] == ["profiling", "aggregate"]
    assert "Previous round insight." in server.calls[1][1]
    assert json.loads(Path(json_path).read_text(encoding="utf-8")) == library
    assert "## insight_verify" in Path(markdown_path).read_text(encoding="utf-8")
    assert json.loads((output_dir / "profiling.json").read_text(encoding="utf-8")) == PROFILING

    # A resumed aggregation over the same traces reuses the Prompt-4 checkpoint.
    resumed = StubFoTServer({"aggregate": json.dumps(library)}, tmp_path)
    resumed.aggregate_and_build_encyclopedia(json_files=[str(path) for path in files], output_dir=str(output_dir))
    assert [name for name, _ in resumed.calls] == ["aggregate"]


def test_num_insights_requires_exact_count(tmp_path: Path) -> None:
    library = {f"insight_{i}": f"Insight number {i}." for i in range(5)}
    server = StubFoTServer(
        {"profiling": json.dumps(PROFILING), "aggregate": json.dumps(library)},
        tmp_path,
        num_insights=3,
    )
    with pytest.raises(ValueError, match="exactly 3 were requested"):
        server.aggregate_and_build_encyclopedia(
            json_files=[str(_write_trace(tmp_path / "t", 1, {"trace_a": "Check the result numerically."}))],
            output_dir=str(tmp_path / "out"),
        )
    assert "EXACTLY 3 top-level insight entries" in server.calls[-1][1]


def test_empty_traces_raise(tmp_path: Path) -> None:
    server = StubFoTServer({}, tmp_path)
    with pytest.raises(ValueError, match="contains no traces"):
        server.aggregate_and_build_encyclopedia(
            json_files=[str(_write_trace(tmp_path / "t", 1, {}))],
            output_dir=str(tmp_path / "out"),
        )
