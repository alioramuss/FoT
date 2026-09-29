<div align="center">

# FoT

[![arXiv](https://img.shields.io/badge/arXiv-2604.16778-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2604.16778)
[![Homepage](https://img.shields.io/badge/Homepage-Visit%20Site-2563eb?style=for-the-badge&logo=googlechrome&logoColor=white)](https://dixiyao.github.io/fot)

</div>

# Federation over Text
Federation over Text (FoT) is a federated-learning-like framework in which agents solving *different* tasks collectively build a shared library of metacognitive **insights**, without sharing their actual problem instances.

Instead of federating over gradients or model weights, FoT operates at the **semantic level** and needs no gradient optimization or supervision signal. In each round, every agent applies any local reasoning and self-improvement procedure to its own task and uploads concise **reasoning traces**. The server aggregates, distills, and consolidates knowledge across tasks and domains into a human-readable insight library, which it broadcasts back to current and future agents.

There are strong connections between FL and FoT. In FL, clients may adopt different optimization methods to solve local subproblems. Analogously, in FoT, each agent may use distinct local reasoning strategies and prompt designs to generate traces.

| | Federation over Gradients | Federation over Text (FoT) |
| --- | --- | --- |
| Learning objective | a global model that generalizes to new data or clients | an insight library that improves agents' reasoning |
| Local work | local training with any optimizer | local task solving with any self-improvement method |
| Local → server | model weights or gradients | abstracted reasoning traces (no raw instances) |
| Server aggregation | averaging weights or gradients | reasoning over traces with curated prompts |
| Server → local | current global model | current insight library |

### Highlights

Results from the paper (Gemini 2.5 Flash Lite as client and server unless noted):

- **Real-world daily tasks (OpenClaw).** Libraries built on PinchBench V1 lift success from 84.1% to 94.2% on V1, from 84.7% to 92.2% on held-out PinchBench tasks (V2 \ V1), and from 56.5 to 74.7 on ClawEval, compared with isolated agents.
- **Multi-domain collaboration.** A library built only from Humanity's Last Exam transfers to held-out domains: AIME26 43.3% → 56.7%, GPQA Diamond 53.5% → 69.0%, LiveCodeBench v6 32.3% → 43.3%.
- **Research insight discovery.** Insights aggregated from ICLR 2025 papers align with the core technical contributions of 26.93% more ICLR 2026 papers than insights generated directly by the LLM.
- Across daily tasks and multi-domain collaboration, FoT improves performance by **11.9 points** on average while reducing client-generated task-completion tokens by **5.5%**.
- Libraries transfer across models (including weak-to-strong: a Gemini 2.5 Flash Lite library improves DeepSeek V4 Flash), and uploaded traces expose little of the raw problem instances.

## The FoT Algorithm

```
Input: empty insight library, agent tasks, base LLM (may differ across agents)
for each round:
    for each agent (in parallel):
        local reasoning with the base LLM and the current insight library
        upload reasoning traces to the server
    server: gather all traces, aggregate them, update the insight library
    broadcast the new library to the agents
Output: insight library
```

**Local reasoning** (`src/fot/fot_client.py`). Each agent solves its task (Prompt 1: the library is shown and the agent is asked to actively apply the relevant insights), reflects on its solution to extract procedural knowledge (Prompt 2), and packages that reflection into a flat JSON object of `trace_*` reasoning traces, each stating the core idea and when to use it (Prompt 3). Only these traces leave the agent; the task and the raw trajectory stay local.

**Global aggregation** (`src/fot/fot_server.py`). The server collects every trace without deduplication, then:

1. clusters highly similar traces and builds a relationship graph (prerequisite, composition, alternative, complementary, derivation, similarity) within and across clusters (Prompt 4);
2. merges the previous library with the new traces, guided by the complete cluster and relationship profile, into fundamental, cross-domain `insight_*` entries (Prompt 5).

Because one trace can contribute to several insights, traces from different domains that share an underlying principle can be merged even when they look different on the surface.

FoT differs from existing experience-reuse frameworks in four ways: it is **validation-free** (no labels, rewards, test cases, or execution feedback), agents are **federated** rather than sequentially updating one memory, it performs **cross-domain merges**, and it is compatible with **arbitrary local reasoning** (for example Metacognitive Reuse, HyperAgents, ExpeL, or AWM can be plugged in as the local step).

## FoT Runtime
FoT is an orchestration framework for **Federation over Text (FoT)** built on top of `openclaw`.

It lets you run multiple OpenClaw agents in parallel, recover their reasoning traces after execution, and aggregate those traces into a persistent shared insight library. The result is a practical testbed for studying how agents can improve collectively through text-based reasoning exchange rather than parameter sharing.

- `⚡` Run multiple OpenClaw agents concurrently under FoT supervision.
- `🧠` Recover transcripts from finished or broken runs.
- `📝` Convert transcripts into structured local reasoning traces.
- `🔗` Aggregate traces into a persistent shared insight library.
- `📚` Inject the current insight library into new agent workspaces automatically.
- `🛠️` Subclass the local and global FoT reasoning interfaces to plug in custom trace extraction and aggregation algorithms.

## Table of Contents
- [Federation over Text](#federation-over-text)
- [The FoT Algorithm](#the-fot-algorithm)
- [FoT Runtime](#fot-runtime)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Overview](#command-overview)
- [How the FoT Pipeline Works](#how-the-fot-pipeline-works)
- [Algorithm Interfaces](#algorithm-interfaces)
- [Replacing the Default Algorithms](#replacing-the-default-algorithms)
- [Configuration](#configuration)
- [State and Artifacts](#state-and-artifacts)
- [Repository Layout](#repository-layout)
- [OpenClaw Integration Notes](#openclaw-integration-notes)
- [Research Framing](#research-framing)
- [Experiments](#experiments)
- [Citation](#citation)

## Architecture

The repository is split into two layers:

- `src/fot/`
  The FoT algorithm layer. This contains the local reasoning pipeline and the global aggregation pipeline.
- `src/fotclaw/`
  The orchestration layer. This contains the CLI, state management, OpenClaw integration, supervision, and persistence.

At a high level:

- `src/fotclaw` hosts and manages agents.
- `fot` defines how local reasoning traces are extracted and how global insights are aggregated.

## Installation

### Requirements

- Python `3.11+`
- OpenClaw installed separately and available as `openclaw`, or configured through `OPENCLAW_PATH`
- An OpenClaw setup that can run the model ids configured in the project root `setting.yaml`

### Setup

```bash
conda create -n fot python=3.12 -y
conda activate fot
python -m pip install --upgrade pip
python -m pip install -e .
```

Development extras:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Start a background agent:

```bash
fot agent --message "Solve the task in the current workspace."
```

Create or reuse a stable named agent shell:

```bash
fot agent --name math
```

Run work on a named agent:

```bash
fot agent --name math --message "Work on the math task."
```

Inspect agent state:

```bash
fot show agent --name math
```

List all FoT-managed agents:

```bash
fot list
```

Start aggregation:

```bash
fot aggregate
```

Inspect the aggregation worker and the shared insight library:

```bash
fot show agent --name aggregate
```

For detailed command usage:

```bash
fot --help
fot agent --help
fot show agent --help
```

## Command Overview

FoT provides these main commands:

- `fot agent`
- `fot list`
- `fot show agent`
- `fot stop`
- `fot delete agent`
- `fot aggregate`
- `fot clean`

The root CLI help now describes how each command is used, and command-specific help is available for the major subcommands.

## How the FoT Pipeline Works

When an agent finishes or breaks, FoT separates execution from FoT postprocessing:

1. **Execution**
   OpenClaw runs the task and produces a transcript. When a library exists, the prompt is prefixed with Prompt 1.
2. **Local FoT postprocessing**
   FoT reflects on the transcript (Prompt 2) and extracts `trace_*` reasoning traces (Prompt 3). If the response is not a valid flat JSON object of traces, postprocessing fails with an error.
3. **Trace persistence**
   The extracted reasoning traces are stored under the FoT state directory as `problem_XXXXXX.json`.
4. **Global aggregation**
   The server collects all traces, profiles their clusters and relationships (Prompt 4, checkpointed in `profiling.json`), and merges them with the previous `insight.json` into the new library (Prompt 5). There are no fallbacks: an unreadable trace file or a malformed model response stops the aggregation with an error, so a bad library is never written.

At the project level, FoT exposes two algorithm hooks:

- Local step: transform one task result or transcript into reusable local reasoning artifacts and insights.
- Server step: merge many local insight artifacts into the shared global insight library.

Every new FoT run copies the current shared library into the agent workspace as both `INSIGHTS.md` and `insight.md`, then prefixes the prompt so the agent is instructed to read and use it.

## Algorithm Interfaces

One of the main changes in the current codebase is that the FoT algorithm layer is now explicitly exposed through abstract interfaces.

Conceptually, users can think about FoT as having:

- a `local step` interface for per-task reasoning and insight extraction
- a `server step` interface for cross-task aggregation

Internally, the default implementation breaks each interface into staged abstract methods, but users do not need to think in terms of "step 1, step 2, step 3" when understanding the project at a high level.

### Local Reasoning Interface

The abstract base class is:

- `fot.fot_client.LocalReasoningClient`

Users can replace the default local FoT pipeline by subclassing this abstract local reasoning client.

The default implementation is:

- `fot.fot_client.OpenClawFoTClient`

### Global Aggregation Interface

The abstract base class is:

- `fot.fot_server.GlobalReasoningServer`

Users can replace the default global FoT aggregation pipeline by subclassing this abstract server-side reasoning interface.

The default implementation is:

- `fot.fot_server.OpenClawFoTServer`

### Return Format

The local reasoning interface and the server-side aggregation interface should each behave like a complete wrapper over their own algorithm.

Each abstract method should return a Python `dict`, but users should think in terms of:

- a local reasoning wrapper that turns one task result or transcript into reusable reasoning artifacts and insights
- an aggregation wrapper that merges many local reasoning artifacts into a final global insight library

FoT handles the orchestration around these interfaces; users only need to implement the algorithmic behavior for local reasoning and aggregation.

## Replacing the Default Algorithms

FoT loads the local and global algorithm classes from editable settings in the project root `setting.yaml`:

- `local_reasoning_class`
- `global_reasoning_class`

Default values:

- `fot.fot_client:OpenClawFoTClient`
- `fot.fot_server:OpenClawFoTServer`

Set custom implementations by editing:

```yaml
local_reasoning_class: mypkg.reasoning:MyLocalReasoner
global_reasoning_class: mypkg.reasoning:MyGlobalReasoner
```

Your module must be importable from the Python environment that runs `fot`.

### Minimal Example

```python
from typing import Any

from fot.fot_client import LocalReasoningClient
from fot.fot_server import GlobalReasoningServer


class MyLocalReasoner(LocalReasoningClient):
    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        return {"response": f"custom step 1 for {problem}", "usage": {}}

    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        return {"response": "custom local reflection", "usage": {}}

    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "valid_skills": {
                "trace_custom_local": "A custom local reasoning trace."
            },
            "usage": {},
        }


class MyGlobalReasoner(GlobalReasoningServer):
    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        return {"insight_store": {"trace_000001": "custom trace"}}

    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        return {"profiling": {"clusters": [], "relationships": []}, "usage": {}}

    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "encyclopedia_dict": {
                "insight_custom_global": "A custom aggregated insight."
            },
            "usage": {},
        }
```

## Configuration

FoT stores runtime state under project-local `.fot/` by default. Override this with `FOT_HOME`.

User-editable settings live in the project root `setting.yaml`:

- `default_model`
- `aggregation_model`
- `openclaw_path`
- `local_reasoning_class`
- `global_reasoning_class`
- `auto_aggregate_enabled`
- `auto_aggregate_trace_threshold`
- `auto_aggregate_min_interval_seconds`

Edit `setting.yaml` directly to change these values.

Runtime aggregation metadata is stored separately in `./.fot/config.json` by default.

## State and Artifacts

By default FoT stores:

- editable settings at `./setting.yaml`
- per-agent state under `./.fot/agents/<agent_id>/`
- extracted reasoning traces under `./.fot/reasoning_traces/problem_XXXXXX.json`
- aggregation workspace under `./.fot/aggregate/`, including the last `profiling.json` (Prompt 4 clusters and relationships), its resume checkpoint, and `aggregation_result.json`
- persistent shared insights at `./.fot/insight.json` and `./.fot/insight.md`

Each agent directory contains its record, logs, workspace, and any recovered transcript path.

`fot clean` removes FoT-managed per-agent state, transient traces, and aggregation scratch files while preserving the persistent shared insight library.

## Repository Layout

- `src/fot/`
  FoT algorithm package
- `src/fot/fot_client.py`
  local reasoning interfaces and default implementation
- `src/fot/fot_server.py`
  global aggregation interfaces and default implementation
- `src/fotclaw/`
  FoT host, CLI, supervisor, configuration, and OpenClaw integration
- `experiment/`
  research code and commands for the paper's experiments (see [Experiments](#experiments))
- `tests/`
  unit tests for the FoT client and server (`python -m pytest`)

## OpenClaw Integration Notes

- FoT creates a dedicated OpenClaw agent per background run so workspaces and transcripts stay isolated.
- FoT serializes only OpenClaw agent creation to avoid `agents add` races; actual task execution still runs in parallel.
- FoT uses OpenClaw for task execution, local FoT processing, and global aggregation so the full pipeline follows one model/runtime path.
- Global aggregation runs through the persistent OpenClaw agent `fotaggregation`.
- If `openclaw` is missing, FoT fails fast with an explicit error.

## Research Framing

An intuitive way to think about FoT is:

- each OpenClaw agent is like a researcher working on its own problem
- each local reasoning trace is that researcher's distilled procedural experience
- the aggregation stage is like a group meeting that consolidates those experiences
- `insight.md` is the shared lab notebook that future researchers can reuse

It is worth continuing to explore the design space of FoT, including personalization strategies, evaluation methodology, handling distribution drift across agents, and optimizing communication efficiency between agents and the server.

## Experiments

[`experiment/`](experiment/) contains the research code behind the paper's results, and [`experiment/example_commands.sh`](experiment/example_commands.sh) lists the commands. It uses the same prompts as `src/fot`, but calls model APIs directly so that benchmark sweeps can run in parallel.

| Application | Train (federation) | Evaluate | Runner |
| --- | --- | --- | --- |
| Multi-domain collaboration | Humanity's Last Exam | AIME26, GPQA Diamond, LiveCodeBench v6 | `task_benchmark_domain.py` |
| Real-world daily tasks | PinchBench V1 (23 tasks) | PinchBench V2 \ V1, ClawEval | `task_openclaw_pinchbench.py`, `task_openclaw_claweval.py` |
| Research insight discovery | ICLR 2025 accepted papers | ICLR 2026 accepted papers | `task_paper_insight_reading.py`, `checker_iclr.py` |

See [`experiment/README.md`](experiment/README.md) for setup, outputs, and ablations.

## Running Logs
We further release a subset of logs from our best-performing runs at [dixiyao/FoT_running_logs](https://huggingface.co/datasets/dixiyao/FoT_running_logs) for reference and future research, including studies of how malicious operations may affect reasoning traces, the insight library, and related behaviors. Access to the logs requires authentication and author approval. The logs are released under the CC BY-NC-ND 4.0 license and are restricted to non-commercial use. Please refer to the Hugging Face repository for detailed access requirements, usage instructions, and the code of conduct.

# Citation
```
@misc{yao2026federationtextinsightsharing,
      title={Federation over Text: Insight Sharing for Multi-Agent Reasoning}, 
      author={Dixi Yao and Tahseen Rabbani and Tian Li},
      year={2026},
      eprint={2604.16778},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.16778}, 
}
```
