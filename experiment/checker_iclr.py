"""
Check accepted ICLR papers for insight guidance using Gemini or OpenRouter API.
- Preferentially reads metadata and papers from a local scraper corpus.
- Keeps OpenReview/proceedings access only as a legacy fallback when no local
  --papers-dir is supplied.
- Uses a provided insights encyclopedia (JSON mapping of name->description or plain text) as guidance.
- Sends title + abstract + insights to the API and records which insights apply.
- Outputs a summary count (guided/total) and a JSON report with per-paper results.

Usage example:
  python guided_accept_oral_checker.py \
      --api-type gemini \
      --key $API_KEY \
      --api-model gemini-3-pro-preview \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --year 2024 \
      --output guided_oral_results.json

  python guided_accept_oral_checker.py \
      --api-type openrouter \
      --key $OPENROUTER_API_KEY \
      --api-model openai/gpt-4o \
      --encyclopedia important_checkpoints/client_aime25_server_math500/encyclopedia.json \
      --year 2024 \
      --output guided_oral_results.json
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import logging
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Prefer new google.genai; fall back to deprecated google.generativeai
HAS_GENAI = False
HAS_GEMINI = False
try:
    import google.genai as genai_new  # type: ignore

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import google.generativeai as genai_old  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not (HAS_GENAI or HAS_GEMINI):
            raise ImportError(
                "Install google-genai (preferred) or google-generativeai. Example: pip install google-genai"
            )
        self.model_name = model_name
        self.backend = "new" if HAS_GENAI else "old"
        if self.backend == "new":
            self.client = genai_new.Client(api_key=api_key)
        else:
            genai_old.configure(api_key=api_key)
            self.model = genai_old.GenerativeModel(model_name)

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        """Generate text and return (text, token_info) tuple."""
        if self.backend == "new":
            from google.genai import types
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens
                )
            )
            # Try primary accessor
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
            # Fallback: attempt to stitch candidate parts
            try:
                candidates = getattr(resp, "candidates", []) or []
                parts = []
                for c in candidates:
                    content = getattr(c, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if hasattr(p, "text") and p.text:
                                parts.append(p.text)
                if parts:
                    text = "\n".join(parts).strip()
                # Try to extract token usage from usage_metadata
                if hasattr(resp, "usage_metadata"):
                    usage = resp.usage_metadata
                    output_tokens = (
                        getattr(usage, "output_token_count", 0)
                        or getattr(usage, "candidates_token_count", 0)
                        or 0
                    )
            except Exception:
                pass
            if text:
                return text, {"output_tokens": output_tokens}
            raise RuntimeError("Failed to extract text from google.genai response")
        else:
            generation_config = {"max_output_tokens": max_output_tokens}
            resp = self.model.generate_content(prompt, generation_config=generation_config)
            text = None
            output_tokens = 0
            if hasattr(resp, "text") and resp.text:
                text = resp.text.strip()
            # Try to extract token usage
            try:
                if hasattr(resp, "usage_metadata"):
                    usage = resp.usage_metadata
                    output_tokens = (
                        getattr(usage, "output_token_count", 0)
                        or getattr(usage, "candidates_token_count", 0)
                        or 0
                    )
            except Exception:
                pass
            if text:
                return text, {"output_tokens": output_tokens}
            # Fallback similar to client.py logic
            try:
                candidate = resp.candidates[0]
                if getattr(
                    candidate, "finish_reason", None
                ) == "RECITATION" and getattr(candidate, "safety_ratings", None):
                    raise RuntimeError(
                        "Gemini API blocked the response due to recitation."
                    )
                if candidate.content and candidate.content.parts:
                    parts = [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]
                    if parts:
                        text = "\n".join(parts).strip()
                        return text, {"output_tokens": output_tokens}
            except Exception:
                pass
            raise RuntimeError(
                "Failed to extract text from google.generativeai response"
            )


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-4o",
        request_timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"
        self.batch_url = "https://openrouter.ai/api/beta/batches"
        self.request_timeout = request_timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dixiyao/Federation-of-Text",
            "X-Title": "ICLR Insight Checker",
        }

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        """Generate text using OpenRouter API and return (text, token_info) tuple."""
        import requests
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "reasoning": {"enabled": False},
        }
        
        max_retries = 6
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=data,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"].strip()
                    usage = result.get("usage", {})
                    return text, {
                        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    }
                else:
                    raise RuntimeError("Failed to extract text from OpenRouter response")
                    
            except requests.exceptions.HTTPError as e:
                try:
                    error_detail = response.json()
                except Exception:
                    error_detail = response.text[:2000]
                if (
                    response.status_code == 429
                    or response.status_code >= 500
                ):
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        retry_after = response.headers.get("Retry-After")
                        if not retry_after and isinstance(error_detail, dict):
                            metadata = error_detail.get("error", {}).get(
                                "metadata", {}
                            )
                            if isinstance(metadata, dict):
                                retry_after = metadata.get("headers", {}).get(
                                    "Retry-After"
                                )
                        try:
                            requested_delay = float(retry_after)
                        except (TypeError, ValueError):
                            requested_delay = 0.0
                        delay = max(requested_delay, base_delay * (2 ** attempt))
                        delay += random.uniform(0.0, min(5.0, delay * 0.1))
                        print(
                            f"    OpenRouter HTTP {response.status_code}. Retrying in "
                            f"{delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        print(
                            f"    OpenRouter HTTP {response.status_code}. "
                            "Max retries exceeded."
                        )
                        raise RuntimeError(
                            f"OpenRouter HTTP {response.status_code} for model "
                            f"{self.model_name}: {error_detail}"
                        ) from e
                else:
                    hint = (
                        " The model may be retired or unavailable; verify it in "
                        "https://openrouter.ai/api/v1/models."
                        if response.status_code == 404
                        else ""
                    )
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status_code} for model "
                        f"{self.model_name}: {error_detail}.{hint}"
                    ) from e
            except requests.exceptions.RequestException:
                if attempt >= max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                print(
                    f"    OpenRouter connection error. Retrying in {delay:.1f} "
                    f"seconds... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
        
        raise RuntimeError(f"Failed after {max_retries} attempts")

    def generate_text_batch(
        self,
        requests_to_run: Sequence[Tuple[str, str]],
        max_output_tokens: int,
        poll_interval: float,
        state_path: str,
    ) -> Dict[str, Tuple[str, Dict]]:
        """Submit or resume an OpenRouter asynchronous text batch.

        The returned mapping is keyed by the caller-provided custom request ID.
        A small state file preserves the remote batch ID, allowing a rerun to
        resume polling rather than submitting and paying for the batch twice.
        """
        if not requests_to_run:
            return {}

        request_ids = [custom_id for custom_id, _ in requests_to_run]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("OpenRouter batch custom IDs must be unique")

        fingerprint_payload = {
            "model": self.model_name,
            "max_output_tokens": max_output_tokens,
            "requests": requests_to_run,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        batch_id = None
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Existing batch state {state_path} belongs to different "
                    "inputs. Remove or rename it before submitting a new batch."
                )
            batch_id = state.get("batch_id")
            if batch_id:
                print(f"Resuming OpenRouter batch {batch_id} from {state_path}")

        if not batch_id:
            # Keep endpoint and model before requests. OpenRouter stream-parses
            # this payload and requires this top-level key order.
            payload = {
                "endpoint": "/v1/chat/completions",
                "model": self.model_name,
                "requests": [
                    {
                        "custom_id": custom_id,
                        "body": {
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_output_tokens,
                            "reasoning": {"enabled": False},
                        },
                    }
                    for custom_id, prompt in requests_to_run
                ],
            }
            print(
                f"Submitting {len(requests_to_run)} requests as one OpenRouter "
                f"batch with model {self.model_name}...",
                flush=True,
            )
            response = requests.post(
                self.batch_url,
                headers=self._headers(),
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            batch = response.json()
            batch_id = batch.get("id")
            if not batch_id:
                raise RuntimeError(
                    f"OpenRouter batch submission returned no ID: {batch}"
                )
            os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "batch_id": batch_id,
                        "fingerprint": fingerprint,
                        "model": self.model_name,
                        "request_count": len(requests_to_run),
                    },
                    handle,
                    indent=2,
                )
            print(f"Submitted batch {batch_id}; state saved to {state_path}")

        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        last_progress = None
        while True:
            response = requests.get(
                f"{self.batch_url}/{batch_id}",
                headers=self._headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            batch = response.json()
            status = batch.get("status")
            counts = batch.get("request_counts") or {}
            progress = (
                status,
                counts.get("completed", 0),
                counts.get("failed", 0),
                counts.get("total", len(requests_to_run)),
            )
            if progress != last_progress:
                print(
                    f"Batch {batch_id}: {progress[0]} "
                    f"({progress[1]} completed, {progress[2]} failed, "
                    f"{progress[3]} total)",
                    flush=True,
                )
                last_progress = progress
            if status in terminal_statuses:
                break
            time.sleep(max(poll_interval, 1.0))

        if status != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch_id} ended with status={status}: "
                f"{batch.get('error')}"
            )

        raw_results = batch.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError(
                f"Completed OpenRouter batch {batch_id} returned no results"
            )

        parsed_results: Dict[str, Tuple[str, Dict]] = {}
        failures = []
        for item in raw_results:
            custom_id = item.get("custom_id")
            error = item.get("error")
            response_item = item.get("response") or {}
            status_code = response_item.get("status_code")
            if error or status_code != 200:
                failures.append(
                    f"{custom_id}: status={status_code}, error={error}"
                )
                continue
            body = response_item.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                failures.append(f"{custom_id}: response contained no choices")
                continue
            content = choices[0].get("message", {}).get("content")
            if not isinstance(content, str):
                failures.append(f"{custom_id}: response content was not text")
                continue
            usage = body.get("usage") or {}
            parsed_results[custom_id] = (
                content.strip(),
                {
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "batch_id": batch_id,
                },
            )

        missing = sorted(set(request_ids) - set(parsed_results))
        if failures or missing:
            preview = "; ".join(failures[:10])
            raise RuntimeError(
                f"OpenRouter batch {batch_id} had {len(missing)} unusable "
                f"request(s). {preview}"
            )
        return parsed_results


def _run_openrouter_batch_process(task):
    """Submit and poll one OpenRouter batch inside an OS process."""
    (
        part_index,
        api_key,
        model_name,
        request_timeout,
        requests_to_run,
        max_output_tokens,
        poll_interval,
        state_path,
    ) = task
    client = OpenRouterClient(
        api_key=api_key,
        model_name=model_name,
        request_timeout=request_timeout,
    )
    results = client.generate_text_batch(
        requests_to_run,
        max_output_tokens=max_output_tokens,
        poll_interval=poll_interval,
        state_path=state_path,
    )
    return part_index, os.getpid(), results


_CHECKER_JUDGE_CLIENT = None
_CHECKER_JUDGE_PROMPTS: Optional[List[str]] = None
_CHECKER_JUDGE_MAX_OUTPUT_TOKENS = 8
_CHECKER_JUDGE_MAX_PAPER_CHARS = 20000


def _initialize_checker_judge_process(
    api_key: str,
    model_name: str,
    request_timeout: float,
    encyclopedia_prompts: List[str],
    max_output_tokens: int,
    max_paper_chars: int,
) -> None:
    """Create one reusable OpenRouter judge client per OS process."""
    global _CHECKER_JUDGE_CLIENT
    global _CHECKER_JUDGE_PROMPTS
    global _CHECKER_JUDGE_MAX_OUTPUT_TOKENS
    global _CHECKER_JUDGE_MAX_PAPER_CHARS
    _CHECKER_JUDGE_CLIENT = OpenRouterClient(
        api_key=api_key,
        model_name=model_name,
        request_timeout=request_timeout,
    )
    _CHECKER_JUDGE_PROMPTS = encyclopedia_prompts
    _CHECKER_JUDGE_MAX_OUTPUT_TOKENS = max_output_tokens
    _CHECKER_JUDGE_MAX_PAPER_CHARS = max_paper_chars


def _run_openrouter_judgment_process(task):
    """Judge one paper/library pair and return its raw boolean response."""
    if _CHECKER_JUDGE_CLIENT is None or _CHECKER_JUDGE_PROMPTS is None:
        raise RuntimeError("OpenRouter judge process was not initialized")
    enc_index, paper_index, paper = task
    prompt = build_score_prompt(
        _CHECKER_JUDGE_PROMPTS[enc_index],
        paper,
        _CHECKER_JUDGE_MAX_PAPER_CHARS,
    )
    raw, token_info = _CHECKER_JUDGE_CLIENT.generate_text(
        prompt,
        max_output_tokens=_CHECKER_JUDGE_MAX_OUTPUT_TOKENS,
    )
    verdict = parse_verdict_json(raw)
    token_info = dict(token_info)
    token_info["input_chars"] = len(prompt)
    return enc_index, paper_index, raw, verdict, token_info, os.getpid()


class LocalHFClient:
    """HuggingFace local model wrapper with the same interface as API clients."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        from utils import check_cuda, load_hf_model

        self.model_name = model_name
        self.device = device or ("cuda" if check_cuda() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.model, self.tokenizer = load_hf_model(
            self.model_name,
            self.device,
            self.load_in_8bit,
        )

    def generate_text(self, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
        import torch
        from utils import _resolve_hf_context_limit

        system_prompt = (
            "You are a strict JSON classifier. Return exactly one JSON object "
            "and no other text. Do not explain your reasoning."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full_prompt = f"{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"

        model_context_limit = _resolve_hf_context_limit(self.model, self.tokenizer)
        input_max_length = min(int(model_context_limit), 65536)
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_max_length,
        ).to(self.device)
        input_token_count = int(inputs["input_ids"].shape[1])
        print(
            f"Input tokens: {input_token_count}, Max new tokens: {max_output_tokens}",
            flush=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return text, {
            "backend": "huggingface",
            "input_tokens": input_token_count,
            "output_tokens": int(output_ids.shape[0]),
            "input_truncated": bool(input_token_count >= input_max_length),
            "input_limit": int(input_max_length),
        }


def load_insights(encyclopedia_path: str) -> Tuple[List[Tuple[str, str]], str]:
    """Load insights from encyclopedia file.

    Returns:
        A list of (name, description) tuples and a formatted string for prompting.
    """
    if not os.path.exists(encyclopedia_path):
        raise FileNotFoundError(f"Encyclopedia not found at {encyclopedia_path}")

    def _parse_insight_item(item):
        if isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("insight_name")
                or item.get("skill_name")
                or item.get("title")
                or item.get("key")
                or item.get("id")
            )
            desc = (
                item.get("description")
                or item.get("desc")
                or item.get("detail")
                or item.get("text")
                or item.get("insight")
                or item.get("skill")
                or ""
            )
            if not name and isinstance(desc, str) and len(desc.strip()) > 0:
                return ("insight", desc.strip())
            if name:
                return (str(name), str(desc) if desc is not None else "")
            return None
        if isinstance(item, str):
            return ("insight", item)
        return None

    def _extract_insights_from_data(data):
        extracted = []
        if isinstance(data, dict):
            if "skills" in data and isinstance(data["skills"], list):
                for item in data["skills"]:
                    parsed = _parse_insight_item(item)
                    if parsed:
                        extracted.append(parsed)
                if extracted:
                    return extracted
            if "insights" in data:
                insights_value = data["insights"]
                if isinstance(insights_value, dict):
                    for k, v in insights_value.items():
                        extracted.append((str(k), str(v) if v is not None else ""))
                    if extracted:
                        return extracted
                if isinstance(insights_value, list):
                    for item in insights_value:
                        parsed = _parse_insight_item(item)
                        if parsed:
                            extracted.append(parsed)
                    if extracted:
                        return extracted
            if "insight" in data:
                insight_value = data["insight"]
                if isinstance(insight_value, dict):
                    for k, v in insight_value.items():
                        extracted.append((str(k), str(v) if v is not None else ""))
                    if extracted:
                        return extracted
                if isinstance(insight_value, list):
                    for item in insight_value:
                        parsed = _parse_insight_item(item)
                        if parsed:
                            extracted.append(parsed)
                    if extracted:
                        return extracted
            # Legacy or flat mapping: use string values only, ignore metadata keys.
            candidate_keys = [
                k for k, v in data.items() if isinstance(v, (str, int, float, bool))
            ]
            if candidate_keys:
                for k in candidate_keys:
                    extracted.append((str(k), str(data[k])))
                return extracted
        elif isinstance(data, list):
            for item in data:
                parsed = _parse_insight_item(item)
                if parsed:
                    extracted.append(parsed)
            return extracted
        return extracted

    insights: List[Tuple[str, str]] = []
    if encyclopedia_path.endswith(".json"):
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.strip()
            data = None
            if cleaned.startswith("{") or cleaned.startswith("["):
                import re

                json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
                if json_match:
                    candidate = json_match.group(1)
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        data = None
            if data is None:
                insights = [("encyclopedia_text", raw_text.strip())]
        if data is not None:
            extracted = _extract_insights_from_data(data)
            if extracted:
                insights = extracted
            elif isinstance(data, dict):
                insights = [(k, str(v) if v is not None else "") for k, v in data.items()]
            elif isinstance(data, list):
                insights = [item for item in (_parse_insight_item(item) for item in data) if item]
            else:
                insights = [("encyclopedia_text", raw_text.strip())]
    else:
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        insights = [("encyclopedia_text", text)]

    if not insights:
        raise ValueError("No insights found in encyclopedia")

    prompt_block = []
    for idx, (name, desc) in enumerate(insights, 1):
        prompt_block.append(f"{idx}. {name}: {desc}")
    return insights, "\n".join(prompt_block)


def find_encyclopedia_paths(encyclopedia_path: str) -> List[str]:
    """Return a list of encyclopedia JSON files for evaluation."""
    if os.path.isdir(encyclopedia_path):
        paths = sorted(
            [
                os.path.join(encyclopedia_path, fn)
                for fn in os.listdir(encyclopedia_path)
                if fn.lower().endswith(".json") and os.path.isfile(os.path.join(encyclopedia_path, fn))
            ]
        )
        if not paths:
            raise FileNotFoundError(
                f"No JSON encyclopedia files found in directory {encyclopedia_path}"
            )
        return paths
    if os.path.isfile(encyclopedia_path):
        return [encyclopedia_path]
    raise FileNotFoundError(f"Encyclopedia path not found: {encyclopedia_path}")


def _index_local_pdfs(papers_dir: Optional[str]) -> Dict[str, str]:
    """Index scraper PDFs by their stable paper ID suffix."""
    if not papers_dir or not os.path.isdir(papers_dir):
        return {}
    index: Dict[str, str] = {}
    for path in Path(papers_dir).rglob("*.pdf"):
        stem = path.stem
        # scraper.py writes <safe-title>_<paper-id>.pdf. Also support a plain
        # <paper-id>.pdf layout for manually prepared corpora.
        candidates = [stem]
        if "_" in stem:
            candidates.append(stem.rsplit("_", 1)[-1])
        for candidate in candidates:
            index.setdefault(candidate, str(path))
    return index


def _load_local_accepted_papers(
    papers_dir: Optional[str],
    year: int,
    max_papers: Optional[int],
    accept_oral: bool,
    accept_spotlight: bool,
    accept_poster: bool,
) -> List[Dict]:
    """Load accepted-paper metadata and local PDF paths from a scraper corpus."""
    if not papers_dir:
        return []
    metadata_path = os.path.join(papers_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return []
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Could not read local paper metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, list):
        raise ValueError(f"Local paper metadata must be a JSON list: {metadata_path}")

    pdf_index = _index_local_pdfs(papers_dir)
    all_tracks = accept_oral and accept_spotlight and accept_poster
    papers: List[Dict] = []

    def value(content: Dict, key: str, default=""):
        result = content.get(key, default)
        if isinstance(result, dict):
            result = result.get("value", default)
        return result

    for raw in metadata:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else {}
        paper_id = str(raw.get("forum") or raw.get("id") or "").strip()
        if not paper_id:
            continue
        venue = str(value(content, "venue", raw.get("venue", "")) or "")
        track = str(raw.get("track") or "").lower()
        track_text = f"{track} {venue}".lower()
        if "oral" in track_text:
            track = "oral"
            include = accept_oral
        elif "spotlight" in track_text:
            track = "spotlight"
            include = accept_spotlight
        elif "poster" in track_text:
            track = "poster"
            include = accept_poster
        else:
            track = "conference"
            include = all_tracks
        if not include:
            continue

        local_pdf_path = pdf_index.get(paper_id)
        papers.append(
            {
                "id": str(raw.get("id") or paper_id),
                "forum": paper_id,
                "title": str(value(content, "title", raw.get("title", "")) or ""),
                "abstract": str(value(content, "abstract", raw.get("abstract", "")) or ""),
                "venue": venue or f"ICLR {year} Conference",
                "track": track,
                "pdf_url": raw.get("pdf_url") or raw.get("pdf"),
                "local_pdf_path": local_pdf_path,
            }
        )
        if max_papers and len(papers) >= max_papers:
            break

    local_count = sum(bool(paper.get("local_pdf_path")) for paper in papers)
    print(
        f"Loaded {len(papers)} ICLR {year} papers from {metadata_path}; "
        f"{local_count} local PDFs available."
    )
    return papers


def fetch_accept_tracks(
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
    or_username: str = None,
    or_password: str = None,
    papers_dir: str = None,
) -> List[Dict]:
    """Fetch accepted papers using OpenReview client.

    Uses openreview-py to query ICLR submissions and filter by venue field.
    """
    # A supplied local corpus is already the accepted-paper set, so no track
    # flags means one overall evaluation across every paper. Keep the legacy
    # network-mode default of oral-only for backward compatibility.
    accept_any = accept_oral or accept_spotlight or accept_poster
    if papers_dir and not accept_any:
        accept_oral = accept_spotlight = accept_poster = True
    else:
        accept_oral = accept_oral or not accept_any

    local_papers = _load_local_accepted_papers(
        papers_dir,
        year,
        max_papers,
        accept_oral,
        accept_spotlight,
        accept_poster,
    )
    if local_papers:
        return local_papers
    if papers_dir:
        raise RuntimeError(
            f"No accepted papers could be loaded from local corpus {papers_dir}. "
            "Expected metadata.json and the already-downloaded paper files; "
            "OpenReview fallback is disabled when --papers-dir is supplied."
        )

    try:
        import openreview

        use_or_client = True
    except ImportError:
        use_or_client = False
        print("Warning: openreview-py not installed, falling back to requests")

    decisions: List[Dict] = []
    all_accept_tracks = accept_oral and accept_spotlight and accept_poster

    def proceedings_fallback() -> List[Dict]:
        if not all_accept_tracks:
            return []
        index_url = f"https://proceedings.iclr.cc/paper_files/paper/{year}"
        print(
            "Falling back to the official ICLR proceedings for all accepted "
            f"papers: {index_url}"
        )
        try:
            response = requests.get(index_url, timeout=120)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            pattern = re.compile(
                rf"/paper_files/paper/{year}/hash/"
                r"([0-9a-f]+)-Abstract-Conference\.html$",
                re.IGNORECASE,
            )
            papers = []
            seen_ids = set()
            for anchor in soup.find_all("a", href=True):
                match = pattern.search(anchor["href"])
                if not match:
                    continue
                paper_id = match.group(1).lower()
                if paper_id in seen_ids:
                    continue
                title = anchor.get_text(" ", strip=True)
                if not title:
                    continue
                seen_ids.add(paper_id)
                papers.append(
                    {
                        "id": paper_id,
                        "forum": paper_id,
                        "title": title,
                        "abstract": "",
                        "venue": f"ICLR {year} Conference",
                        "track": "conference",
                        "pdf_url": (
                            "https://proceedings.iclr.cc/paper_files/paper/"
                            f"{year}/file/{paper_id}-Paper-Conference.pdf"
                        ),
                    }
                )
                if max_papers and len(papers) >= max_papers:
                    break
            print(
                f"Found {len(papers)} accepted ICLR {year} papers via official "
                "proceedings. Track-level labels are unavailable in this fallback."
            )
            return papers
        except Exception as exc:
            print(f"Official ICLR proceedings fallback failed: {exc}")
            return []

    # Try OpenReview client first
    if use_or_client:
        try:
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=or_username,
                password=or_password,
            )
            print(f"Fetching ICLR {year} submissions via OpenReview client...")
            submissions = list(
                client.get_all_notes(
                    invitation=f"ICLR.cc/{year}/Conference/-/Submission",
                    details="directReplies",
                )
            )
            print(f"Retrieved {len(submissions)} submissions, filtering by venue...")

            for sub in submissions:
                content = sub.content
                # API v2 nests values
                venue = str(content.get("venue", {}).get("value", ""))

                # Check if accepted and what track
                track = None
                if "oral" in venue.lower():
                    track = "oral"
                    if not accept_oral:
                        continue
                elif "spotlight" in venue.lower():
                    track = "spotlight"
                    if not accept_spotlight:
                        continue
                elif "poster" in venue.lower():
                    track = "poster"
                    if not accept_poster:
                        continue
                else:
                    continue

                decisions.append(
                    {
                        "id": sub.id,
                        "forum": sub.forum,
                        "title": content.get("title", {}).get("value", ""),
                        "abstract": content.get("abstract", {}).get("value", ""),
                        "venue": venue,
                        "track": track,
                    }
                )
                if max_papers and len(decisions) >= max_papers:
                    break

            if decisions:
                print(
                    f"Found {len(decisions)} accepted papers via client (no bulk hydration; will fetch content on-demand)"
                )
                # Sort by track priority: oral > spotlight > poster
                track_order = {"oral": 0, "spotlight": 1, "poster": 2}
                decisions.sort(key=lambda p: track_order.get(p.get("track", ""), 999))
                return decisions
            else:
                print("No accepted papers found via client")
                return proceedings_fallback()
        except Exception as e:
            print(f"OpenReview client error: {e}")
            return proceedings_fallback()

    # Fallback: old requests-based approach
    print("Using requests-based fallback (may not work for ICLR 2024+)...")
    return proceedings_fallback()


@contextmanager
def _quiet_pdf_parser_diagnostics():
    """Hide known non-fatal warnings emitted while recovering malformed PDFs."""
    logger_names = ("pypdf", "PyPDF2", "pdfminer")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for logger, previous_level in zip(loggers, previous_levels):
            logger.setLevel(previous_level)


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract plain text from raw PDF bytes.

    Tries pypdf first (lightweight), then pdfminer.six as fallback.
    Returns empty string if neither is available or extraction fails.
    """
    # Try pypdf
    try:
        import io
        import pypdf
        with _quiet_pdf_parser_diagnostics():
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            parts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
        return "\n".join(parts)
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pypdf extraction failed: {e}")

    # Try pdfminer.six
    try:
        import io
        from pdfminer.high_level import extract_text as pdfminer_extract
        with _quiet_pdf_parser_diagnostics():
            return pdfminer_extract(io.BytesIO(pdf_bytes))
    except ImportError:
        pass
    except Exception as e:
        print(f"    Warning: pdfminer extraction failed: {e}")

    return ""


def _fetch_paper_content(
    forum_id: str,
    session: requests.Session = None,
    or_client=None,
    cache_dir: str = "data/iclr25",
    pdf_url: str = None,
    local_pdf_path: str = None,
    local_only: bool = False,
) -> str:
    """Fetch full paper content for scoring, with disk caching.

    On first fetch the extracted text is saved to
    <cache_dir>/<forum_id>.txt so subsequent runs skip the download.

    Strategy (in order):
      1. Return cached text if <cache_dir>/<forum_id>.txt exists.
      2. Extract a PDF supplied by the local scraper corpus.
      3. If available, use an authenticated OpenReview client.
      4. Otherwise, scrape the unauthenticated forum page (older years).

    ``pdf_url`` and ``local_only`` are accepted for call compatibility but no
    longer change behavior: there is no proceedings-PDF download and no
    title/abstract-only shortcut.

    Returns paper text (up to 50k chars) or empty string.
    """
    def _sanitize(s: str) -> str:
        """Replace surrogate / non-encodable characters with '?'."""
        return s.encode("utf-8", errors="replace").decode("utf-8")

    # --- Cache check ---
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{forum_id}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read()
            print(f"    Loaded from cache: {len(cached)} chars")
            return cached

    # --- Check for local PDF files ---
    local_pdf_paths = ([local_pdf_path] if local_pdf_path else []) + [
        os.path.join("data", "papers", f"{forum_id}.pdf"),
        os.path.join("data", "papers", "iclr23_top5", f"{forum_id}.pdf"),
        os.path.join("data", "papers", "iclr23_diffusion", f"{forum_id}.pdf"),
    ]
    for pdf_path in local_pdf_paths:
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
                if pdf_text.strip():
                    print(f"    Loaded from local PDF: {len(pdf_text)} chars")
                    # Save to cache for future use
                    if cache_path:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            f.write(pdf_text)
                    return pdf_text
            except Exception as e:
                print(f"    Warning: Failed to extract text from local PDF {pdf_path}: {e}")
                continue

    if or_client is not None:
        full_text_parts = []

        # --- Step 1: download and extract PDF text ---
        try:
            pdf_bytes = or_client.get_pdf(forum_id, is_reference=False)
            if pdf_bytes:
                pdf_text = _extract_text_from_pdf_bytes(pdf_bytes)
                if pdf_text.strip():
                    print(f"    PDF extracted: {len(pdf_text)} chars")
                    full_text_parts.append(pdf_text)
                else:
                    print(f"    Warning: PDF downloaded but text extraction yielded nothing")
        except Exception as e:
            print(f"    Warning: PDF download failed for {forum_id}: {e}")

        # --- Step 2: supplement with API metadata fields ---
        try:
            note = or_client.get_note(forum_id)
            content = note.content
            meta_fields = (
                "title", "abstract", "keywords", "tldr", "summary",
                "primary_area", "research_area",
            )
            meta_parts = []
            for field in meta_fields:
                val = content.get(field)
                if val is None:
                    continue
                if isinstance(val, dict):
                    val = val.get("value", "")
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                val = str(val).strip()
                if val:
                    meta_parts.append(f"[{field}] {val}")
            if meta_parts:
                full_text_parts.insert(0, "\n".join(meta_parts))
        except Exception as e:
            print(f"    Warning: API metadata fetch failed for {forum_id}: {e}")

        if full_text_parts:
            text = "\n\n".join(full_text_parts)[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

    # --- Step 4: unauthenticated HTML scrape (ICLR 2024 and older) ---
    if session is None:
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
    try:
        url = f"https://openreview.net/forum?id={forum_id}"
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        content_parts = []
        for content_div in soup.find_all(
            ["div", "section"], class_=re.compile("note-content|paper-content", re.I)
        ):
            text = content_div.get_text(separator=" ", strip=True)
            if text and len(text) > 100:
                content_parts.append(text)

        if content_parts:
            text = " ".join(content_parts)[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

        page_text = soup.get_text(separator=" ", strip=True)
        if len(page_text) > 1000:
            text = page_text[:50000]
            text = _sanitize(text)
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
            return text

        return ""
    except Exception as e:
        print(f"    Warning: Could not fetch paper content for {forum_id}: {e}")
        return ""


_CHECKER_PROCESS_SESSION = None


def _initialize_checker_process() -> None:
    """Create one reusable HTTP session inside each checker process."""
    global _CHECKER_PROCESS_SESSION
    _CHECKER_PROCESS_SESSION = requests.Session()
    _CHECKER_PROCESS_SESSION.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36"
            )
        }
    )


def _prepare_paper_content_process(task):
    """Load/extract one paper in a separate OS process."""
    global _CHECKER_PROCESS_SESSION
    if _CHECKER_PROCESS_SESSION is None:
        _initialize_checker_process()
    paper_index, forum_id, cache_dir, pdf_url, local_pdf_path, local_only = task
    content = _fetch_paper_content(
        forum_id,
        _CHECKER_PROCESS_SESSION,
        or_client=None,
        cache_dir=cache_dir,
        pdf_url=pdf_url,
        local_pdf_path=local_pdf_path,
        local_only=local_only,
    )
    return paper_index, content, os.getpid()


def call_api(client, prompt: str, max_output_tokens: int = 16384) -> Tuple[str, Dict]:
    """Call API via wrapper and return (raw_text, token_info) tuple."""
    return client.generate_text(prompt, max_output_tokens=max_output_tokens)


def parse_verdict_json(raw: str) -> Dict[str, Any]:
    """Parse a boolean checker verdict, with legacy JSON compatibility."""
    text = raw.replace("Ġ", " ").replace("Ċ", "\n").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # Current checker contract: exactly one lowercase boolean. Quoted forms
    # are tolerated for provider compatibility, but prompts request bare text.
    simple = text.strip().strip('"').strip("'").strip().lower()
    if simple == "true":
        return {"guided": True, "matched_insights": []}
    if simple == "false":
        return {"guided": False, "matched_insights": []}

    candidates = [text]
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start: idx + 1])
                    break

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        normalized = candidate
        normalized = re.sub(r"\bTrue\b", "true", normalized)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        normalized = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', normalized)
        normalized = re.sub(r"([{,]\s*)(guided|guid|matched_insights|insights)\s*:", r'\1"\2":', normalized)
        normalized = normalized.replace('"guid"', '"guided"')
        normalized = normalized.replace('"insights"', '"matched_insights"')
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                if "guided" not in parsed and "guid" in parsed:
                    parsed["guided"] = parsed.pop("guid")
                if "matched_insights" not in parsed and "insights" in parsed:
                    parsed["matched_insights"] = parsed.pop("insights")
                parsed.setdefault("guided", bool(parsed.get("matched_insights")))
                parsed.setdefault("matched_insights", [])
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc

    lowered = text.lower()
    insight_names = []
    for pat in (
        r"matched_insights\s*[:=]\s*\[([^\]]*)\]",
        r"insights\s*[:=]\s*\[([^\]]*)\]",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            insight_names = [
                item.strip().strip('"').strip("'")
                for item in m.group(1).split(",")
                if item.strip().strip('"').strip("'")
            ]
            break

    if re.search(r"\b(guided|guid)\s*[:=]\s*true\b", lowered) or insight_names:
        return {"guided": True, "matched_insights": insight_names}
    if re.search(r"\b(guided|guid)\s*[:=]\s*false\b", lowered) or re.search(r"\bno\b", lowered):
        return {"guided": False, "matched_insights": []}

    raise ValueError(f"Failed to parse verdict JSON: {last_error}; raw preview={raw[:500]}")


def build_score_prompt(
    insights_prompt: str,
    paper: Dict,
    max_paper_chars: int = 20000,
) -> str:
    """Build the strict boolean guidance-classification prompt."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    content = paper.get("content", "")
    if max_paper_chars and len(content) > max_paper_chars:
        content = content[:max_paper_chars]

    paper_text = (
        f"Title: {title}\n\nAbstract: {abstract}\n\nFull Paper Content: {content}"
    )

    return f"""
Classify whether the research paper's proposed solution is guided by or directly derived from any insight in the encyclopedia.

Output exactly one lowercase word and nothing else: true or false.

Insights:
{insights_prompt}

Research Paper:
{paper_text}

Evaluation Criteria - An insight guides the paper ONLY IF ALL of the following are true:

1. CONCRETE METHODOLOGY USAGE: The insight's methodology or approach is concretely used in the paper's methods/approach section, not just theoretically relevant or mentioned in motivation.

2. METHODS SECTION PRESENCE: The insight must be related to how the paper actually implements its solution (methods, algorithms, techniques), not just in problem statement or related work.

3. SPECIFICITY: The insight must specifically address a key challenge or component of the paper's solution, not just be generally applicable background knowledge.

Response Format:
- Return true ONLY when at least one insight passes ALL criteria above.
- Otherwise return false.
- Do not return JSON, insight names, punctuation, markdown, explanation, or reasoning.
- If unsure, return false.
"""


def score_paper(
    model: Any,
    insights_prompt: str,
    paper: Dict,
    max_output_tokens: int = 8,
    max_paper_chars: int = 20000,
) -> Tuple[Dict, Dict]:
    """Ask a model whether any encyclopedia insight guides the paper."""
    prompt = build_score_prompt(insights_prompt, paper, max_paper_chars)
    print(
        f"      Model input chars: prompt={len(prompt)} max_output_tokens={max_output_tokens}",
        flush=True,
    )
    raw, token_info = call_api(model, prompt, max_output_tokens=max_output_tokens)
    token_info = dict(token_info)
    token_info["input_chars"] = len(prompt)
    token_info["raw_response"] = raw
    print(
        f"      Model returned {len(raw)} chars; output_tokens={token_info.get('output_tokens', 0)}",
        flush=True,
    )
    return parse_verdict_json(raw), token_info


def main():
    parser = argparse.ArgumentParser(
        description="Check ICLR Accept papers (oral/spotlight/poster) for insight guidance using Gemini, OpenRouter, or a local HuggingFace model"
    )
    parser.add_argument(
        "--key",
        "--gemini-key",
        dest="api_key",
        type=str,
        default=None,
        help="API key (or set API_KEY or GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--api-model",
        "--gemini-model",
        dest="api_model",
        type=str,
        default="gemini-3-pro-preview",
        help="API model name (default: gemini-3-pro-preview for gemini, openai/gpt-4o for openrouter)",
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["gemini", "openrouter", "local"],
        default="gemini",
        help="API type to use (default: gemini)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Local HuggingFace model name/path. If provided, uses local model inference instead of API.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device for local HuggingFace model (cuda or cpu).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load local HuggingFace model with 8-bit quantization.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8,
        help="Maximum generated tokens per boolean checker call (default: 8).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each OpenRouter request (default: 300)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Submit all judge requests through OpenRouter's asynchronous Batch "
            "API. Batch inputs are text-only and may take up to 24 hours."
        ),
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=60.0,
        help="Seconds between OpenRouter batch status polls (default: 60).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Concurrent OS processes for paper text extraction and checker API "
            "calls (default: 1). With --batch, every process submits and polls "
            "an independent OpenRouter batch."
        ),
    )
    parser.add_argument(
        "--max-paper-chars",
        type=int,
        default=20000,
        help="Maximum full-paper characters included in each checker prompt (default: 20000; use 0 for no truncation).",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=None,
        help=(
            "Local accepted-paper corpus created by scraper.py. Reads "
            "metadata.json and PDFs locally before using any network fallback. "
            "For year 2025, defaults to ICLR25_PAPERS when set."
        ),
    )
    parser.add_argument(
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to an insights encyclopedia JSON file or a directory containing encyclopedia JSON files",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICLR conference year (default: 2024)",
    )
    parser.add_argument(
        "--accept-oral",
        action="store_true",
        help="Include Accept (Oral) papers",
    )
    parser.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Include Accept (Spotlight) papers",
    )
    parser.add_argument(
        "--accept-poster",
        action="store_true",
        help="Include Accept (Poster) papers",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Limit number of papers (for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="guided_oral_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--judgments-jsonl",
        type=str,
        default=None,
        help=(
            "Append-only per-paper judgment journal. Defaults to "
            "<output>.judgments.jsonl and is used to resume without paying "
            "again for completed paper/library pairs."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--or-username",
        type=str,
        default=None,
        help=(
            "OpenReview account email for legacy network mode. Not used when "
            "--papers-dir is supplied."
        ),
    )
    parser.add_argument(
        "--or-password",
        type=str,
        default=None,
        help=(
            "OpenReview password for legacy network mode. Not used when "
            "--papers-dir is supplied."
        ),
    )

    args = parser.parse_args()
    args.or_username = args.or_username or os.getenv("OPENREVIEW_USERNAME")
    args.or_password = args.or_password or os.getenv("OPENREVIEW_PASSWORD")
    if not args.papers_dir and args.year == 2025:
        args.papers_dir = os.getenv("ICLR25_PAPERS")

    use_local_model = args.api_type == "local" or bool(args.model)
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be at least 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.batch and (args.api_type != "openrouter" or use_local_model):
        raise ValueError("--batch is supported only with --api-type openrouter")

    if args.api_type == "openrouter":
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
    else:
        api_key = args.api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not use_local_model and not api_key:
        raise ValueError(
            "API key is required. Provide --key or set API_KEY (or GEMINI_API_KEY)."
        )

    if use_local_model:
        if not args.model:
            raise ValueError("Provide --model when using --api-type local.")
        model = LocalHFClient(
            model_name=args.model,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
        )
        active_model_name = args.model
    elif args.api_type == "gemini":
        model = GeminiClient(api_key=api_key, model_name=args.api_model)
        active_model_name = args.api_model
    elif args.api_type == "openrouter":
        # Use provided model or default to gpt-4o
        model_name = args.api_model if args.api_model != "gemini-3-pro-preview" else "openai/gpt-4o"
        model = OpenRouterClient(
            api_key=api_key,
            model_name=model_name,
            request_timeout=args.request_timeout,
        )
        active_model_name = model_name
    else:
        raise ValueError(f"Unsupported API type: {args.api_type}")

    papers = fetch_accept_tracks(
        args.year,
        max_papers=args.max_papers,
        accept_oral=args.accept_oral,
        accept_spotlight=args.accept_spotlight,
        accept_poster=args.accept_poster,
        or_username=args.or_username,
        or_password=args.or_password,
        papers_dir=args.papers_dir,
    )
    if not papers:
        print("No Accept papers found.")
        return

    encyclopedia_paths = find_encyclopedia_paths(args.encyclopedia)
    print(f"\nFound {len(encyclopedia_paths)} encyclopedia file(s) to evaluate.", flush=True)
    print(f"Processing {len(papers)} accepted papers as one overall set...\n", flush=True)

    # Build authenticated OpenReview client for content fetching (ICLR 2025+)
    or_client = None
    proceedings_only = bool(papers) and all(
        paper.get("local_pdf_path") or paper.get("pdf_url") for paper in papers
    )
    if proceedings_only:
        print(
            "Using official proceedings PDFs; skipping a second OpenReview login."
        )
    elif args.or_username and args.or_password:
        try:
            import openreview
            or_client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=args.or_username,
                password=args.or_password,
            )
            print("Authenticated OpenReview client ready for paper content fetching.")
        except Exception as e:
            print(f"Warning: could not build authenticated client for content fetching: {e}")

    # Pre-load all encyclopedias
    encyclopedias_data = []
    for enc_path in encyclopedia_paths:
        insights, insights_prompt = load_insights(enc_path)
        with open(enc_path, "rb") as encyclopedia_file:
            encyclopedia_sha256 = hashlib.sha256(encyclopedia_file.read()).hexdigest()
        encyclopedias_data.append({
            'path': enc_path,
            'sha256': encyclopedia_sha256,
            'insights': insights,
            'prompt': insights_prompt,
            'results': [],
            'track_stats': {
                "oral": {"total": 0, "guided": 0},
                "spotlight": {"total": 0, "guided": 0},
                "poster": {"total": 0, "guided": 0},
                "conference": {"total": 0, "guided": 0},
            }
        })

    print(
        f"\nFetching paper contents with {args.num_workers} worker(s)...",
        flush=True,
    )
    cache_dir = (
        os.path.join(args.papers_dir, "text_cache")
        if args.papers_dir
        else "data/iclr25"
    )
    if args.num_workers > 1 and or_client is not None:
        raise ValueError(
            "Multiprocess paper preparation cannot share an authenticated "
            "OpenReview client. Use --papers-dir with local PDFs (recommended), "
            "use proceedings PDF URLs, or set --num-workers 1."
        )

    preparation_tasks = [
        (
            paper_index,
            paper.get("forum") or paper.get("id"),
            cache_dir,
            paper.get("pdf_url"),
            paper.get("local_pdf_path"),
            bool(args.papers_dir),
        )
        for paper_index, paper in enumerate(papers)
    ]
    if args.num_workers == 1:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36"
                )
            }
        )
        prepared_outputs = []
        for paper_index, paper in enumerate(papers):
            paper_content = _fetch_paper_content(
                paper.get("forum") or paper.get("id"),
                session,
                or_client=or_client,
                cache_dir=cache_dir,
                pdf_url=paper.get("pdf_url"),
                local_pdf_path=paper.get("local_pdf_path"),
                local_only=bool(args.papers_dir),
            )
            prepared_outputs.append((paper_index, paper_content, os.getpid()))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_initialize_checker_process,
        ) as executor:
            prepared_outputs = list(
                executor.map(_prepare_paper_content_process, preparation_tasks)
            )

    for completed, (returned_index, paper_content, worker_pid) in enumerate(
        prepared_outputs, 1
    ):
        papers[returned_index]["content"] = paper_content
        content_status = (
            f"{len(paper_content)} characters"
            if paper_content
            else "title/abstract fallback"
        )
        print(
            f"[{completed}/{len(papers)}] Prepared "
            f"{papers[returned_index].get('title', '')[:80]} "
            f"({content_status}) in PID {worker_pid}",
            flush=True,
        )

    judgment_log_path = args.judgments_jsonl or f"{args.output}.judgments.jsonl"
    os.makedirs(os.path.dirname(judgment_log_path) or ".", exist_ok=True)
    completed_result_keys = set()

    def result_key(enc_data: Dict[str, Any], paper: Dict[str, Any]) -> str:
        paper_key = str(paper.get("forum") or paper.get("id") or "")
        return f"{active_model_name}|{enc_data['sha256']}|{paper_key}"

    def update_track_stats(
        enc_data: Dict[str, Any], track_label: str, guided: bool
    ) -> None:
        if track_label in enc_data["track_stats"]:
            enc_data["track_stats"][track_label]["total"] += 1
            if guided:
                enc_data["track_stats"][track_label]["guided"] += 1

    # Recover completed judgments before making any new paid API calls. Keep
    # the latest valid occurrence of a key if a manually concatenated journal
    # contains duplicates.
    recovered_by_key: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(judgment_log_path):
        with open(judgment_log_path, encoding="utf-8") as judgment_log:
            for line_number, line in enumerate(judgment_log, 1):
                if not line.strip():
                    continue
                try:
                    recovered = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: ignoring malformed judgment journal line "
                        f"{line_number}: {judgment_log_path}",
                        flush=True,
                    )
                    continue
                recovered_key = recovered.get("result_key")
                if recovered_key and isinstance(recovered.get("guided"), bool):
                    recovered_by_key[recovered_key] = recovered

    paper_by_key = {
        str(paper.get("forum") or paper.get("id") or ""): paper
        for paper in papers
    }
    paper_index_by_key = {
        str(paper.get("forum") or paper.get("id") or ""): paper_index
        for paper_index, paper in enumerate(papers)
    }
    encyclopedia_by_sha = {
        enc_data["sha256"]: enc_data for enc_data in encyclopedias_data
    }
    for recovered_key, recovered in recovered_by_key.items():
        if recovered.get("judge_model") != active_model_name:
            continue
        enc_data = encyclopedia_by_sha.get(recovered.get("encyclopedia_sha256"))
        paper = paper_by_key.get(str(recovered.get("paper_key") or ""))
        if enc_data is None or paper is None:
            continue
        expected_key = result_key(enc_data, paper)
        if recovered_key != expected_key:
            continue
        completed_result_keys.add(recovered_key)
        enc_data["results"].append(recovered)
        update_track_stats(
            enc_data,
            str(recovered.get("track") or ""),
            bool(recovered["guided"]),
        )

    if completed_result_keys:
        print(
            f"Recovered {len(completed_result_keys)} completed per-paper "
            f"judgments from {judgment_log_path}.",
            flush=True,
        )
    # A hard process interruption can leave a partial final JSONL line. It was
    # ignored above; terminate it before appending so every subsequent record
    # remains independently parseable.
    if os.path.isfile(judgment_log_path) and os.path.getsize(judgment_log_path) > 0:
        with open(judgment_log_path, "rb+") as interrupted_log:
            interrupted_log.seek(-1, os.SEEK_END)
            if interrupted_log.read(1) != b"\n":
                interrupted_log.write(b"\n")
    judgment_log = open(judgment_log_path, "a", encoding="utf-8")

    def record_result(
        enc_data: Dict[str, Any],
        paper: Dict[str, Any],
        verdict: Dict[str, Any],
        token_info: Dict[str, Any],
        raw_response: str,
    ) -> None:
        key = result_key(enc_data, paper)
        if key in completed_result_keys:
            return
        guided = bool(verdict.get("guided"))
        track_label = paper.get("track", "")
        update_track_stats(enc_data, track_label, guided)
        paper_key = str(paper.get("forum") or paper.get("id") or "")
        result = {
            "schema_version": 1,
            "result_key": key,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model": active_model_name,
            "setting": os.path.basename(
                os.path.dirname(os.path.abspath(enc_data["path"]))
            ),
            "encyclopedia": os.path.basename(enc_data["path"]),
            "encyclopedia_path": os.path.abspath(enc_data["path"]),
            "encyclopedia_sha256": enc_data["sha256"],
            "paper_key": paper_key,
            "paper_index": paper_index_by_key[paper_key],
            "id": paper.get("id"),
            "forum": paper.get("forum"),
            "title": paper.get("title", ""),
            "guided": guided,
            "judgment": "true" if guided else "false",
            "judgment_value": 1 if guided else 0,
            "raw_response": str(raw_response),
            # Boolean-only judging intentionally does not ask the model to
            # repeat insight names, minimizing completion-token charges.
            "matched_insights": [],
            "track": track_label,
            "venue": paper.get("venue", ""),
            "venueid": paper.get("venueid", ""),
            "input_chars": int(token_info.get("input_chars", 0) or 0),
            "input_tokens": int(token_info.get("input_tokens", 0) or 0),
            "output_tokens": int(token_info.get("output_tokens", 0) or 0),
        }
        if token_info.get("batch_id"):
            result["batch_id"] = token_info["batch_id"]
        enc_data["results"].append(result)
        completed_result_keys.add(key)
        judgment_log.write(json.dumps(result, ensure_ascii=False) + "\n")
        judgment_log.flush()

    if args.batch:
        batch_requests: List[Tuple[str, str]] = []
        request_targets: Dict[str, Tuple[int, int]] = {}
        for enc_index, enc_data in enumerate(encyclopedias_data):
            for paper_index, paper in enumerate(papers):
                if result_key(enc_data, paper) in completed_result_keys:
                    continue
                custom_id = f"enc-{enc_index:04d}-paper-{paper_index:06d}"
                prompt = build_score_prompt(
                    enc_data["prompt"], paper, args.max_paper_chars
                )
                batch_requests.append((custom_id, prompt))
                request_targets[custom_id] = (enc_index, paper_index)

        api_processes = min(args.num_workers, len(batch_requests)) if batch_requests else 0
        chunk_size = (
            (len(batch_requests) + api_processes - 1) // api_processes
            if api_processes
            else 0
        )
        batch_process_tasks = []
        offsets = range(0, len(batch_requests), chunk_size) if chunk_size else []
        for part_index, offset in enumerate(offsets):
            request_chunk = batch_requests[offset : offset + chunk_size]
            state_path = (
                f"{args.output}.batch_state.json"
                if api_processes == 1
                else f"{args.output}.batch_part_{part_index:04d}_state.json"
            )
            batch_process_tasks.append(
                (
                    part_index,
                    api_key,
                    active_model_name,
                    args.request_timeout,
                    request_chunk,
                    args.max_output_tokens,
                    args.batch_poll_interval,
                    state_path,
                )
            )

        print(
            f"\nBatch-evaluating {len(batch_requests)} paper/library pairs "
            f"with {active_model_name} across {api_processes} API process(es); "
            f"each response is capped at {args.max_output_tokens} tokens.",
            flush=True,
        )
        if not batch_requests:
            batch_process_outputs = []
        elif api_processes == 1:
            batch_process_outputs = [
                _run_openrouter_batch_process(batch_process_tasks[0])
            ]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=api_processes
            ) as executor:
                batch_process_outputs = list(
                    executor.map(
                        _run_openrouter_batch_process,
                        batch_process_tasks,
                    )
                )

        batch_results = {}
        for part_index, worker_pid, part_results in batch_process_outputs:
            overlap = set(batch_results).intersection(part_results)
            if overlap:
                raise RuntimeError(
                    f"Duplicate checker result IDs across batch part {part_index}: "
                    f"{sorted(overlap)[:5]}"
                )
            batch_results.update(part_results)
            print(
                f"API process PID {worker_pid} completed batch part "
                f"{part_index} ({len(part_results)} judgments)",
                flush=True,
            )
        for custom_id, prompt in batch_requests:
            enc_index, paper_index = request_targets[custom_id]
            raw, token_info = batch_results[custom_id]
            token_info = dict(token_info)
            token_info["input_chars"] = len(prompt)
            try:
                verdict = parse_verdict_json(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid boolean verdict for {custom_id}: {raw!r}"
                ) from exc
            record_result(
                encyclopedias_data[enc_index],
                papers[paper_index],
                verdict,
                token_info,
                raw,
            )
    else:
        pending_judgments = [
            (enc_index, paper_index, paper)
            for enc_index, enc_data in enumerate(encyclopedias_data)
            for paper_index, paper in enumerate(papers)
            if result_key(enc_data, paper) not in completed_result_keys
        ]
        if (
            args.api_type == "openrouter"
            and not use_local_model
            and args.num_workers > 1
            and pending_judgments
        ):
            worker_count = min(args.num_workers, len(pending_judgments))
            print(
                f"\nEvaluating {len(pending_judgments)} remaining judgments "
                f"with {worker_count} OpenRouter API process(es)...",
                flush=True,
            )
            failures = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_checker_judge_process,
                initargs=(
                    api_key,
                    active_model_name,
                    args.request_timeout,
                    [enc_data["prompt"] for enc_data in encyclopedias_data],
                    args.max_output_tokens,
                    args.max_paper_chars,
                ),
            ) as executor:
                future_to_target = {
                    executor.submit(_run_openrouter_judgment_process, task): (
                        task[0],
                        task[1],
                    )
                    for task in pending_judgments
                }
                for completed, future in enumerate(
                    concurrent.futures.as_completed(future_to_target), 1
                ):
                    enc_index, paper_index = future_to_target[future]
                    try:
                        (
                            returned_enc_index,
                            returned_paper_index,
                            raw,
                            verdict,
                            token_info,
                            worker_pid,
                        ) = future.result()
                    except Exception as exc:
                        failures.append((enc_index, paper_index, exc))
                        print(
                            f"[{completed}/{len(pending_judgments)}] API/model "
                            f"error for paper {paper_index}: {exc}",
                            flush=True,
                        )
                        continue
                    record_result(
                        encyclopedias_data[returned_enc_index],
                        papers[returned_paper_index],
                        verdict,
                        token_info,
                        raw,
                    )
                    print(
                        f"[{completed}/{len(pending_judgments)}] PID {worker_pid}: "
                        f"{'true' if verdict.get('guided') else 'false'} for "
                        f"paper {returned_paper_index + 1}",
                        flush=True,
                    )
            if failures:
                judgment_log.flush()
                judgment_log.close()
                preview = "; ".join(
                    f"paper {paper_index + 1}: {error}"
                    for _, paper_index, error in failures[:10]
                )
                raise RuntimeError(
                    f"{len(failures)} judgment(s) failed and were not recorded "
                    f"as false. Completed results are saved in "
                    f"{judgment_log_path}. First failures: {preview}"
                )
        else:
            print(
                f"\nEvaluating {len(pending_judgments)} remaining judgments "
                "synchronously...",
                flush=True,
            )
            for enc_index, paper_index, paper in pending_judgments:
                enc_data = encyclopedias_data[enc_index]
                print(
                    f"[{paper_index + 1}/{len(papers)}] Evaluating with encyclopedia "
                    f"{os.path.basename(enc_data['path'])} and model "
                    f"{active_model_name}...",
                    flush=True,
                )
                try:
                    verdict, token_info = score_paper(
                        model,
                        enc_data["prompt"],
                        paper,
                        max_output_tokens=args.max_output_tokens,
                        max_paper_chars=args.max_paper_chars,
                    )
                except Exception as exc:
                    print(f"    API/model error: {exc}", flush=True)
                    raise RuntimeError(
                        "Aborting instead of recording an API/model failure as an "
                        "unguided paper. The output would otherwise contain biased "
                        "false negatives."
                    ) from exc
                raw_response = str(token_info.pop("raw_response", ""))
                record_result(
                    enc_data,
                    paper,
                    verdict,
                    token_info,
                    raw_response,
                )
                print(
                    f"    Result: {'✓ GUIDED' if verdict.get('guided') else '✗ Not guided'} "
                    f"| Tokens: {token_info.get('output_tokens', 0)}",
                    flush=True,
                )
                time.sleep(max(args.sleep, 0))

    judgment_log.flush()
    judgment_log.close()

    # Build evaluations from the collected data
    evaluations = []
    expected_paper_keys = [
        str(paper.get("forum") or paper.get("id") or "") for paper in papers
    ]
    paper_order = {
        paper_key: index for index, paper_key in enumerate(expected_paper_keys)
    }
    for enc_data in encyclopedias_data:
        total = len(papers)
        observed_paper_keys = [
            str(result.get("paper_key") or "") for result in enc_data["results"]
        ]
        if (
            len(observed_paper_keys) != total
            or len(set(observed_paper_keys)) != total
            or set(observed_paper_keys) != set(expected_paper_keys)
        ):
            raise RuntimeError(
                f"Incomplete or duplicate per-paper judgment log for "
                f"{enc_data['path']}: expected {total} unique papers, received "
                f"{len(observed_paper_keys)} records. Durable partial results "
                f"remain in {judgment_log_path}."
            )
        enc_data["results"].sort(
            key=lambda result: paper_order[str(result.get("paper_key") or "")]
        )
        total_guided = sum(enc_data['track_stats'][t]["guided"] for t in enc_data['track_stats'])
        guidance_rate = total_guided / total if total else 0.0

        evaluation = {
            "judge_model": active_model_name,
            "encyclopedia": os.path.basename(enc_data['path']),
            "path": enc_data['path'],
            "encyclopedia_sha256": enc_data["sha256"],
            "judgments_jsonl": judgment_log_path,
            "metrics": {
                "total_papers": total,
                "guided_papers": total_guided,
                "guidance_rate": guidance_rate,
            },
            "results": enc_data['results'],
        }
        evaluations.append(evaluation)

        print(f"\n[{len(evaluations)}/{len(encyclopedias_data)}] {os.path.basename(enc_data['path'])}: {total_guided}/{total} papers guided ({guidance_rate*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if len(evaluations) == 1:
        output_data = {
            "judge_model": active_model_name,
            "encyclopedia": evaluations[0]["encyclopedia"],
            "path": evaluations[0]["path"],
            "encyclopedia_sha256": evaluations[0]["encyclopedia_sha256"],
            "judgments_jsonl": judgment_log_path,
            "metrics": evaluations[0]["metrics"],
            "results": evaluations[0]["results"],
        }
    else:
        output_data = {
            "judge_model": active_model_name,
            "judgments_jsonl": judgment_log_path,
            "evaluations": evaluations,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary in requested style
    if len(evaluations) > 1:
        print(f"\n{'='*80}")
        print("GUIDANCE RATE SUMMARY")
        print(f"{'='*80}")
        summary_parts = []
        for idx, eval_data in enumerate(evaluations, 1):
            metrics = eval_data["metrics"]
            overall_rate = metrics["guidance_rate"] * 100
            summary_parts.append(f"file {idx}: all:{overall_rate:.1f}%")
        print(" ".join(summary_parts))
        print(f"{'='*80}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
