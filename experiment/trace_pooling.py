"""Reasoning-trace pooling and OpenRouter-backed semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CONTEXT_WINDOW = 200_000
DEFAULT_COMPLETION_RESERVE = 65_536
DEFAULT_PROMPT_OVERHEAD = 8_192
DEFAULT_RAG_MAX_TOKENS = 4_096
DEFAULT_RAG_CHUNK_TOKENS = 512
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without a model-specific tokenizer."""
    if not text:
        return 0
    # Three UTF-8 bytes/token is deliberately more conservative than the
    # common four-English-characters/token approximation. This avoids filling
    # the provider context window to its exact edge.
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Return the longest prefix whose conservative estimate fits the budget."""
    if max_tokens <= 0 or not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def reference_token_budget(
    *,
    task_prompt: str,
    context_window: int,
    max_reference_tokens: Optional[int] = None,
    completion_reserve: int = DEFAULT_COMPLETION_RESERVE,
    prompt_overhead: int = DEFAULT_PROMPT_OVERHEAD,
) -> int:
    """Compute a safe reference budget while preserving the original prompt."""
    available = (
        int(context_window)
        - estimate_tokens(task_prompt)
        - int(completion_reserve)
        - int(prompt_overhead)
    )
    available = max(0, available)
    if max_reference_tokens is not None:
        available = min(available, int(max_reference_tokens))
    return available


def resolve_iteration_dir(pooling_dir: str, iteration: int) -> Path:
    """Resolve the requested source iteration without silently using another."""
    root = Path(pooling_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Pooling directory does not exist: {root}")
    candidates = [
        root / f"iter_{iteration:02d}",
        root / f"iter_{iteration}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise ValueError(
        f"Pooling iteration {iteration} is missing; expected {expected}"
    )


@dataclass(frozen=True)
class TraceChunk:
    chunk_id: str
    task_id: str
    trace_name: str
    source_file: str
    text: str

    def formatted(self) -> str:
        return (
            f"### Source task: {self.task_id} | Trace: {self.trace_name}\n"
            f"{self.text.strip()}"
        )


def _stringify_trace(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).strip()


def _trace_file_index(path: Path) -> Optional[int]:
    """Extract the numeric client/task suffix from e.g. ``problem_0007.json``."""
    match = re.search(r"_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def discover_reasoning_trace_files(
    iteration_dir: Path, limit: Optional[int] = None
) -> List[Path]:
    """Return validated, de-duplicated ``problem_*.json`` client files.

    Recurses under ``iteration_dir`` so this covers both a flat
    one-file-per-client layout and split train/eval directory layouts where
    parallel-worker scratch copies duplicate the canonical per-client file
    (e.g. ``v1_train/problem_0007.json`` and
    ``v1_train/parallel_workers/task_0007/problem_0001.json``). Exactly one
    file is kept per distinct ``(task_id, insight_book)`` content, ordered by
    the numeric suffix embedded in each file's name — the same "client N"
    ordering the training loop originally assigned. When ``limit`` is given,
    only the first ``limit`` clients in that order are returned (used by
    ``--participate``).
    """
    seen: Dict[str, Path] = {}
    for problem_file in iteration_dir.rglob("problem_*.json"):
        try:
            payload = json.loads(problem_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid pooling trace file {problem_file}: {exc}") from exc
        insight_book = payload.get("insight_book")
        if not isinstance(insight_book, dict) or not insight_book:
            continue
        task_id = str(payload.get("task_id") or problem_file.parent.name)
        content_key = hashlib.sha256(
            json.dumps(
                {"task_id": task_id, "insight_book": insight_book},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        # Parallel runs retain both worker and merged checkpoints. Treat
        # those byte-equivalent client files as one source client.
        if content_key in seen:
            continue
        seen[content_key] = problem_file

    def _sort_key(problem_file: Path) -> Tuple[float, str]:
        index = _trace_file_index(problem_file)
        return (index if index is not None else float("inf"), str(problem_file))

    ordered = sorted(seen.values(), key=_sort_key)
    if not ordered:
        raise RuntimeError(
            f"No non-empty reasoning traces were found under {iteration_dir}"
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        if limit > len(ordered):
            raise RuntimeError(
                f"--participate {limit} was requested but only {len(ordered)} "
                f"client(s) are available under {iteration_dir}"
            )
        ordered = ordered[:limit]
    return ordered


def load_reasoning_traces(
    iteration_dir: Path, limit: Optional[int] = None
) -> List[TraceChunk]:
    """Load and deduplicate successful ``problem_*.json`` insight books.

    When ``limit`` is given, restricts to the first ``limit`` clients
    (problem_*.json files, ordered by embedded file index) instead of every
    client found under ``iteration_dir``.
    """
    traces: List[TraceChunk] = []
    seen = set()
    for problem_file in discover_reasoning_trace_files(iteration_dir, limit=limit):
        payload = json.loads(problem_file.read_text(encoding="utf-8"))
        insight_book = payload["insight_book"]
        task_id = str(payload.get("task_id") or problem_file.parent.name)
        for trace_name, raw_value in insight_book.items():
            text = _stringify_trace(raw_value)
            if not text:
                continue
            identity = hashlib.sha256(
                f"{task_id}\0{trace_name}\0{text}".encode("utf-8")
            ).hexdigest()
            # Parallel runs retain both worker and merged checkpoints. Treat
            # those byte-equivalent trace records as one source trace.
            if identity in seen:
                continue
            seen.add(identity)
            traces.append(
                TraceChunk(
                    chunk_id=identity,
                    task_id=task_id,
                    trace_name=str(trace_name),
                    source_file=str(problem_file.resolve()),
                    text=text,
                )
            )
    if not traces:
        raise RuntimeError(
            f"No non-empty reasoning traces were found under {iteration_dir}"
        )
    return traces


def load_raw_transcript_traces(iteration_dir: Path) -> List[TraceChunk]:
    """Load complete raw agent transcript JSONL files for semantic retrieval.

    Unlike :func:`load_reasoning_traces`, this performs no reflection or insight
    extraction.  User messages, assistant messages, tool calls, and tool results
    are retained as structured JSON in the embedding corpus.
    """
    traces: List[TraceChunk] = []
    seen = set()
    transcript_files = sorted(iteration_dir.rglob("transcripts/*.jsonl"))
    for transcript_file in transcript_files:
        events: List[Any] = []
        try:
            for line_number, line in enumerate(
                transcript_file.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except Exception as exc:
                    raise RuntimeError(
                        f"invalid JSON on line {line_number}: {exc}"
                    ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Invalid raw transcript file {transcript_file}: {exc}"
            ) from exc
        if not events:
            continue
        text = json.dumps(events, ensure_ascii=False, sort_keys=True, indent=2)
        task_id = re.sub(r"_session\d+$", "", transcript_file.stem)
        relative_name = str(transcript_file.relative_to(iteration_dir))
        identity = hashlib.sha256(
            f"{task_id}\0{relative_name}\0{text}".encode("utf-8")
        ).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        traces.append(
            TraceChunk(
                chunk_id=identity,
                task_id=task_id,
                trace_name=f"raw_transcript:{relative_name}",
                source_file=str(transcript_file.resolve()),
                text=text,
            )
        )
    if not traces:
        raise RuntimeError(
            f"No non-empty raw transcript JSONL files were found under {iteration_dir}"
        )
    return traces


def load_cumulative_raw_transcript_traces(
    pooling_dir: str,
    iteration: int,
) -> Tuple[List[TraceChunk], List[Path]]:
    """Load and deduplicate raw transcripts from rounds 1 through ``iteration``."""
    traces_by_id: Dict[str, TraceChunk] = {}
    source_iterations: List[Path] = []
    for source_iteration_number in range(1, iteration + 1):
        source_iteration = resolve_iteration_dir(
            pooling_dir,
            source_iteration_number,
        )
        source_iterations.append(source_iteration)
        for trace in load_raw_transcript_traces(source_iteration):
            traces_by_id.setdefault(trace.chunk_id, trace)
    return list(traces_by_id.values()), source_iterations


def load_individual_library_sources(
    iteration_dir: Path, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load and validate the raw per-client entries from one individual round.

    Sources are sorted ascending by ``client_index`` and, when ``limit`` is
    given, trimmed to the first ``limit`` clients (used by ``--participate``
    to restrict pooling/aggregation to a fixed-size subset of clients).
    """
    library_path = iteration_dir / "appended_individual_encyclopedia.json"
    if not library_path.is_file():
        raise RuntimeError(
            "Individual pooled library is missing; expected "
            f"{library_path}. Run V1 with --individual first."
        )
    try:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Invalid individual pooled library {library_path}: {exc}"
        ) from exc
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(
            f"Individual pooled library contains no client sources: {library_path}"
        )

    entries: List[Dict[str, Any]] = []
    for source_index, source in enumerate(sources, 1):
        if not isinstance(source, dict) or not isinstance(
            source.get("encyclopedia"), dict
        ):
            raise RuntimeError(
                f"Invalid client library entry {source_index} in {library_path}"
            )
        entries.append(
            {
                "client_index": int(source.get("client_index") or source_index),
                "task_id": str(source.get("task_id") or f"client_{source_index:04d}"),
                "task_name": source.get("task_name"),
                "encyclopedia": source["encyclopedia"],
            }
        )
    entries.sort(key=lambda entry: entry["client_index"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        if limit > len(entries):
            raise RuntimeError(
                f"--participate {limit} was requested but only {len(entries)} "
                f"client(s) are available in {library_path}"
            )
        entries = entries[:limit]
    return entries


def load_individual_library_traces(
    iteration_dir: Path, limit: Optional[int] = None
) -> List[TraceChunk]:
    """Load every (or the first ``limit``) client encyclopedia from one round."""
    library_path = iteration_dir / "appended_individual_encyclopedia.json"
    entries = load_individual_library_sources(iteration_dir, limit=limit)

    traces: List[TraceChunk] = []
    for entry in entries:
        client_index = entry["client_index"]
        task_id = entry["task_id"]
        text = json.dumps(
            {
                "client_index": client_index,
                "task_id": task_id,
                "task_name": entry["task_name"],
                "encyclopedia": entry["encyclopedia"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        identity = hashlib.sha256(
            f"{library_path.resolve()}\0{client_index}\0{task_id}\0{text}".encode(
                "utf-8"
            )
        ).hexdigest()
        traces.append(
            TraceChunk(
                chunk_id=identity,
                task_id=task_id,
                trace_name=f"individual_library:client_{client_index:04d}:{task_id}",
                source_file=str(library_path.resolve()),
                text=text,
            )
        )
    return traces


def trace_corpus_digest(traces: Sequence[TraceChunk]) -> str:
    digest = hashlib.sha256()
    for trace in traces:
        digest.update(trace.chunk_id.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def append_trace_corpus(traces: Sequence[TraceChunk]) -> str:
    return "\n\n".join(trace.formatted() for trace in traces)


def _split_text(text: str, max_tokens: int) -> List[str]:
    """Split trace text into paragraph-aware embedding chunks."""
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]
    pieces: List[str] = []
    remaining = text.strip()
    while remaining:
        prefix = truncate_to_token_budget(remaining, max_tokens)
        if not prefix:
            break
        if len(prefix) < len(remaining):
            boundary = max(prefix.rfind("\n\n"), prefix.rfind("\n"), prefix.rfind(". "))
            if boundary >= max(1, len(prefix) // 2):
                prefix = prefix[: boundary + (2 if prefix[boundary:boundary + 2] == ". " else 0)]
        prefix = prefix.strip()
        if not prefix:
            prefix = remaining[:1]
        pieces.append(prefix)
        remaining = remaining[len(prefix):].lstrip()
    return pieces


def build_rag_chunks(
    traces: Sequence[TraceChunk],
    *,
    chunk_tokens: int = DEFAULT_RAG_CHUNK_TOKENS,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for trace in traces:
        for part_index, part in enumerate(_split_text(trace.text, chunk_tokens), 1):
            display = (
                f"### Source task: {trace.task_id} | Trace: {trace.trace_name} "
                f"| Part: {part_index}\n{part}"
            )
            chunk_id = hashlib.sha256(
                f"{trace.chunk_id}\0{part_index}\0{part}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "task_id": trace.task_id,
                    "trace_name": trace.trace_name,
                    "part": part_index,
                    "text": display,
                }
            )
    return chunks


class OpenRouterEmbeddingClient:
    """Minimal OpenRouter embeddings client with strict response validation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        batch_size: int = 64,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY or --rag-api-key is required for --rag")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, int(batch_size))
        self.input_tokens = 0

    def _request(self, texts: Sequence[str]) -> List[List[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(
                {"model": self.model, "input": list(texts)},
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"OpenRouter embeddings request failed ({exc.code}): {detail}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenRouter embeddings request failed: {exc}") from exc
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("OpenRouter embeddings response has an invalid item count")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if any(
            not isinstance(vector, list)
            or not vector
            or not all(isinstance(value, (int, float)) for value in vector)
            for vector in vectors
        ):
            raise RuntimeError("OpenRouter embeddings response contains an invalid vector")
        self.input_tokens += int((payload.get("usage") or {}).get("prompt_tokens", 0) or 0)
        return [[float(value) for value in vector] for vector in vectors]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[offset: offset + self.batch_size]))
        return vectors


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("RAG embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_pooling_references(
    *,
    traces: Sequence[TraceChunk],
    task_prompts: Dict[str, str],
    output_dir: Path,
    context_window: int,
    rag: bool,
    rag_api_key: Optional[str] = None,
    rag_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client: Optional[OpenRouterEmbeddingClient] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Create one context-budgeted reference file for every evaluation task."""
    references_dir = output_dir / "pooling_references"
    references_dir.mkdir(parents=True, exist_ok=True)
    corpus_digest = trace_corpus_digest(traces)
    reference_paths: Dict[str, str] = {}
    task_metadata: Dict[str, Any] = {}
    embedding_input_tokens = 0

    if rag:
        chunks = build_rag_chunks(traces)
        index_path = output_dir / "pooling_rag_index.json"
        index_payload: Dict[str, Any] = {}
        if index_path.exists():
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Invalid existing RAG index {index_path}: {exc}") from exc
        cache_valid = (
            index_payload.get("version") == 1
            and index_payload.get("corpus_sha256") == corpus_digest
            and index_payload.get("embedding_model") == rag_embedding_model
            and isinstance(index_payload.get("chunks"), list)
            and len(index_payload.get("chunks")) == len(chunks)
        )
        client = embedding_client or OpenRouterEmbeddingClient(
            api_key=rag_api_key or "",
            model=rag_embedding_model,
        )
        if cache_valid:
            indexed_chunks = index_payload["chunks"]
        else:
            vectors = client.embed([chunk["text"] for chunk in chunks])
            indexed_chunks = [
                {**chunk, "embedding": vector}
                for chunk, vector in zip(chunks, vectors)
            ]
            index_payload = {
                "version": 1,
                "corpus_sha256": corpus_digest,
                "embedding_model": rag_embedding_model,
                "chunks": indexed_chunks,
                "query_embeddings": {},
            }
        query_cache = index_payload.setdefault("query_embeddings", {})
        missing_keys: List[str] = []
        missing_prompts: List[str] = []
        task_query_keys: Dict[str, str] = {}
        for task_id, prompt in task_prompts.items():
            query_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            task_query_keys[task_id] = query_key
            if query_key not in query_cache:
                missing_keys.append(query_key)
                missing_prompts.append(prompt)
        if missing_prompts:
            query_vectors = client.embed(missing_prompts)
            for query_key, vector in zip(missing_keys, query_vectors):
                query_cache[query_key] = vector
        embedding_input_tokens = int(getattr(client, "input_tokens", 0) or 0)
        _atomic_json(index_path, index_payload)

        for task_id, prompt in task_prompts.items():
            budget = reference_token_budget(
                task_prompt=prompt,
                context_window=context_window,
                max_reference_tokens=DEFAULT_RAG_MAX_TOKENS,
            )
            query_vector = query_cache[task_query_keys[task_id]]
            ranked = sorted(
                (
                    (_cosine_similarity(query_vector, item["embedding"]), item)
                    for item in indexed_chunks
                ),
                key=lambda pair: (-pair[0], pair[1]["chunk_id"]),
            )
            selected: List[str] = []
            selected_ids: List[str] = []
            remaining = budget
            for score, item in ranked:
                candidate = f"{item['text']}\nRelevance: {score:.6f}"
                candidate_tokens = estimate_tokens(candidate)
                if candidate_tokens <= remaining:
                    selected.append(candidate)
                    selected_ids.append(item["chunk_id"])
                    remaining -= candidate_tokens
                elif remaining > 0 and not selected:
                    selected.append(truncate_to_token_budget(candidate, remaining))
                    selected_ids.append(item["chunk_id"])
                    remaining = 0
                if remaining <= 0:
                    break
            reference = "\n\n".join(selected)
            task_metadata[task_id] = {
                "reference_token_budget": budget,
                "reference_tokens_estimated": estimate_tokens(reference),
                "retrieved_chunk_ids": selected_ids,
            }
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
            reference_path = references_dir / f"{safe_task_id or 'task'}.txt"
            reference_path.write_text(reference, encoding="utf-8")
            reference_paths[task_id] = str(reference_path.resolve())
    else:
        corpus = append_trace_corpus(traces)
        for task_id, prompt in task_prompts.items():
            budget = reference_token_budget(
                task_prompt=prompt,
                context_window=context_window,
            )
            reference = truncate_to_token_budget(corpus, budget)
            task_metadata[task_id] = {
                "reference_token_budget": budget,
                "reference_tokens_estimated": estimate_tokens(reference),
                "corpus_truncated": len(reference) < len(corpus),
            }
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
            reference_path = references_dir / f"{safe_task_id or 'task'}.txt"
            reference_path.write_text(reference, encoding="utf-8")
            reference_paths[task_id] = str(reference_path.resolve())

    manifest = {
        "version": 1,
        "mode": "rag" if rag else "append",
        "context_window": int(context_window),
        "trace_count": len(traces),
        "corpus_sha256": corpus_digest,
        "rag_embedding_model": rag_embedding_model if rag else None,
        "rag_embedding_input_tokens": embedding_input_tokens,
        "tasks": task_metadata,
    }
    manifest_path = output_dir / "pooling_manifest.json"
    _atomic_json(manifest_path, manifest)
    return reference_paths, {**manifest, "manifest_path": str(manifest_path.resolve())}
