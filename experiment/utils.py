"""
Shared utilities for LLM model loading and inference.
Used by client.py, client_metacognitive.py, server.py, server_text.py,
server_cod.py, server_claude_compact.py, etc.
"""

import base64
import io
import os
import json
import time
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# API-only callers should not require the optional local Hugging Face stack.
# Importing transformers can also raise an ImportError for an incompatible
# huggingface-hub version, so defer that failure until load_hf_model() is used.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:
    AutoModelForCausalLM = Any  # type: ignore[misc,assignment]
    AutoTokenizer = Any  # type: ignore[misc,assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    HAS_GEMINI = True
    HAS_GENAI_NEW = True
except ImportError:
    HAS_GEMINI = False
    HAS_GENAI_NEW = False

try:
    from openai import OpenAI as _OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Max tokens allowed by backend/model routing policy.
MAX_ALLOWED_INPUT_TOKENS = 1_048_576
# When exceeding MAX_ALLOWED_INPUT_TOKENS, truncate to first 1,000,000 tokens.
TRUNCATED_INPUT_TOKENS = 1_000_000
# Conservative fallback estimate when token counting is unavailable.
_FALLBACK_CHARS_PER_TOKEN_ESTIMATE = 1
TRUNCATED_INPUT_CHARS_FALLBACK = (
    TRUNCATED_INPUT_TOKENS * _FALLBACK_CHARS_PER_TOKEN_ESTIMATE
)

# OpenRouter reports a 1,048,576-token context window for Gemini 2.5 Flash.
# Reserve room for output, request metadata/schema, reasoning, and image
# tokens. A one-character-per-token input bound is deliberately conservative:
# it guarantees that long profiling prompts are reduced before the API call
# even when the provider tokenizer is unavailable locally.
OPENROUTER_CONTEXT_WINDOWS = {
    "google/gemini-2.5-flash": 1_048_576,
    "google/gemini-2.5-flash-lite": 1_048_576,
}
OPENROUTER_CONTEXT_SAFETY_TOKENS = 16_384


class OpenRouterInFlightBudgetError(RuntimeError):
    """OpenRouter refused admission because shared in-flight credit is full."""


def sanitize_unicode_text(value: Any) -> str:
    """Return UTF-8-safe text, replacing malformed UTF-16 surrogates.

    Some PDF parsers preserve unpaired surrogate code points. Python strings
    can hold them, but HTTP/JSON UTF-8 encoders cannot. Replacing only those
    invalid code points keeps the rest of the extracted paper text unchanged.
    """
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _openrouter_retry_details(exc: Exception) -> Tuple[Optional[int], str, str, float]:
    """Extract status, provider reason, limit_source, and Retry-After from an SDK error."""
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    body = getattr(exc, "body", None)
    error_body = body.get("error", body) if isinstance(body, dict) else {}
    metadata = error_body.get("metadata", {}) if isinstance(error_body, dict) else {}
    reason = str(metadata.get("reason", "")) if isinstance(metadata, dict) else ""
    # "in_flight_budget_exhausted" (too many concurrent requests) reports its
    # classification under "reason"; a genuinely low/exhausted account
    # balance instead reports "limit_source": "openrouter_credits" with no
    # "reason" key at all. Both are account-wide conditions where every
    # subsequent request is doomed identically until something external
    # changes (other jobs finish, or credits are added) — neither is a
    # per-problem failure worth burning through the rest of a benchmark for.
    limit_source = (
        str(metadata.get("limit_source", "")) if isinstance(metadata, dict) else ""
    )

    retry_after: Any = None
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None and isinstance(metadata, dict):
        metadata_headers = metadata.get("headers", {})
        if isinstance(metadata_headers, dict):
            retry_after = metadata_headers.get("Retry-After") or metadata_headers.get(
                "retry-after"
            )
    try:
        retry_seconds = max(0.0, float(retry_after))
    except (TypeError, ValueError):
        retry_seconds = 0.0

    # Older openai SDK releases do not consistently expose ``body`` or even a
    # numeric status attribute. Preserve provider-specific classification from
    # the exception text as a final compatibility fallback.
    error_text = str(exc)
    error_text_lower = error_text.lower()
    if status is None and any(
        marker in error_text_lower
        for marker in ("error code: 402", "http 402", "'code': 402", '"code": 402')
    ):
        status = 402
    if not reason and "in_flight_budget_exhausted" in error_text:
        reason = "in_flight_budget_exhausted"
    if not limit_source and (
        "openrouter_credits" in error_text
        or "requires more credits" in error_text_lower
        or "can only afford" in error_text_lower
    ):
        limit_source = "openrouter_credits"
    return status, reason, limit_source, retry_seconds


def is_openrouter_budget_error(exc: Exception) -> bool:
    """Return whether an exception is an account-wide OpenRouter 402."""
    if isinstance(exc, OpenRouterInFlightBudgetError):
        return True
    status, reason, limit_source, _ = _openrouter_retry_details(exc)
    return status == 402 and (
        reason == "in_flight_budget_exhausted"
        or limit_source == "openrouter_credits"
    )


def _truncate_prompt_preserving_ends(text: str, max_chars: int) -> Tuple[str, bool]:
    """Conservatively fit text while retaining instructions at both ends."""
    if len(text) <= max_chars:
        return text, False
    marker = (
        "\n\n[... middle of oversized input truncated to fit the model context ...]\n\n"
    )
    usable = max(1, max_chars - len(marker))
    head_chars = int(usable * 0.7)
    tail_chars = usable - head_chars
    return text[:head_chars] + marker + text[-tail_chars:], True

DEFAULT_API_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "openrouter": "google/gemini-2.5-flash-lite",
}
OPENROUTER_RETIRED_MODEL_REPLACEMENTS = {
    "google/gemini-2.0-flash-001": "google/gemini-2.5-flash",
}
_WARNED_MODEL_REPLACEMENTS = set()


def normalize_api_model(provider: str, model_name: Optional[str] = None) -> str:
    """Return a provider-ready model name.

    OpenRouter accepts slugs such as ``google/gemini-2.5-flash-lite``.  The
    optional ``openrouter/`` prefix used by some agent harnesses is stripped
    before sending the request to OpenRouter itself.
    """
    normalized_provider = (provider or "gemini").strip().lower()
    if normalized_provider not in DEFAULT_API_MODELS:
        raise ValueError(
            f"Unsupported API provider {provider!r}; expected one of "
            f"{sorted(DEFAULT_API_MODELS)}"
        )
    model = (model_name or DEFAULT_API_MODELS[normalized_provider]).strip()
    if normalized_provider == "openrouter" and model.startswith("openrouter/"):
        model = model[len("openrouter/") :]
    if (
        normalized_provider == "openrouter"
        and model in OPENROUTER_RETIRED_MODEL_REPLACEMENTS
    ):
        replacement = OPENROUTER_RETIRED_MODEL_REPLACEMENTS[model]
        if model not in _WARNED_MODEL_REPLACEMENTS:
            print(
                f"OpenRouter model {model} is retired/unavailable; "
                f"using active replacement {replacement}."
            )
            _WARNED_MODEL_REPLACEMENTS.add(model)
        model = replacement
    if not model:
        raise ValueError("API model name cannot be empty")
    return model


def resolve_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the explicit or environment-provided key for an API provider."""
    if api_key:
        return api_key
    normalized_provider = (provider or "gemini").strip().lower()
    env_name = {
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(normalized_provider)
    if env_name is None:
        raise ValueError(f"Unsupported API provider {provider!r}")
    return os.getenv(env_name)


# ---------------------------------------------------------------------------
# CUDA check
# ---------------------------------------------------------------------------
def check_cuda() -> bool:
    """Check if CUDA is available."""
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def _compose_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Combine optional system prompt and user prompt."""
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


class _GeminiModel:
    """Thin wrapper around google.genai.Client providing a stable interface
    for count_tokens() and generate_content() used by call_gemini()."""

    def __init__(self, client, model_name: str):
        self._client = client
        self._model_name = model_name

    def count_tokens(self, text: str):
        """Return an object with .total_tokens attribute."""
        result = self._client.models.count_tokens(
            model=self._model_name, contents=text
        )
        return result

    def generate_content(
        self,
        prompt,
        generation_config: Optional[Dict] = None,
        request_options: Optional[Dict] = None,
    ):
        """Call generate_content via new SDK."""
        from google.genai import types as _gt
        kwargs: Dict = {}
        if generation_config and generation_config.get("max_output_tokens"):
            kwargs["max_output_tokens"] = generation_config["max_output_tokens"]
        if generation_config and generation_config.get("temperature") is not None:
            kwargs["temperature"] = generation_config["temperature"]
        config = _gt.GenerateContentConfig(**kwargs) if kwargs else None
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )


def _count_gemini_tokens(gemini_model, text: str) -> Optional[int]:
    """Best-effort Gemini token counting. Returns None if unavailable."""
    try:
        token_count_result = gemini_model.count_tokens(text)
        total_tokens = getattr(token_count_result, "total_tokens", None)
        if total_tokens is None:
            return None
        return int(total_tokens)
    except Exception:
        return None


def _truncate_gemini_prompt_to_limit(gemini_model, full_prompt: str) -> Tuple[str, Optional[int], bool]:
    """Apply 1,048,576-token ceiling and truncate to 1,000,000 tokens when exceeded.

    Returns:
        (possibly_truncated_prompt, measured_input_tokens_or_none, was_truncated)
    """
    input_tokens = _count_gemini_tokens(gemini_model, full_prompt)
    if input_tokens is not None:
        if input_tokens <= MAX_ALLOWED_INPUT_TOKENS:
            return full_prompt, input_tokens, False

        # Exceeded allowed limit: truncate to first ~1,000,000 tokens.
        target_tokens = TRUNCATED_INPUT_TOKENS
        truncated_prompt = full_prompt

        for _ in range(6):
            current_tokens = _count_gemini_tokens(gemini_model, truncated_prompt)
            if current_tokens is None:
                break
            if current_tokens <= target_tokens:
                return truncated_prompt, current_tokens, True

            shrink_ratio = target_tokens / max(current_tokens, 1)
            new_char_len = max(1, int(len(truncated_prompt) * shrink_ratio))
            if new_char_len >= len(truncated_prompt):
                new_char_len = len(truncated_prompt) - 1
            truncated_prompt = truncated_prompt[:new_char_len]

        # Final conservative fallback if iterative token counting is unavailable/inconclusive.
        truncated_prompt = truncated_prompt[:TRUNCATED_INPUT_CHARS_FALLBACK]
        final_tokens = _count_gemini_tokens(gemini_model, truncated_prompt)
        return truncated_prompt, final_tokens, True

    # Fallback when token counting is unavailable.
    if len(full_prompt) > TRUNCATED_INPUT_CHARS_FALLBACK:
        return full_prompt[:TRUNCATED_INPUT_CHARS_FALLBACK], None, True

    return full_prompt, None, False


def _resolve_hf_context_limit(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> int:
    """Resolve effective HuggingFace input-token context limit for safe truncation."""
    limits = []

    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max > 0 and tokenizer_max < 10_000_000:
        limits.append(int(tokenizer_max))

    config = getattr(model, "config", None)
    if config is not None:
        for attr in (
            "max_position_embeddings",
            "max_sequence_length",
            "n_positions",
            "sliding_window",
        ):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                limits.append(int(value))

        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            for key in ("original_max_position_embeddings", "max_position_embeddings"):
                value = rope_scaling.get(key)
                if isinstance(value, int) and value > 0:
                    limits.append(int(value))

    if not limits:
        return MAX_ALLOWED_INPUT_TOKENS

    return min(limits)


# ---------------------------------------------------------------------------
# Gemini setup & call
# ---------------------------------------------------------------------------
def setup_gemini(
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-lite",
) -> "_GeminiModel":
    """Initialize Gemini API and return a model wrapper.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        model_name: Gemini model name.

    Returns:
        _GeminiModel wrapper around google.genai.Client.
    """
    if not HAS_GEMINI:
        raise ImportError(
            "google-genai is required for Gemini API. "
            "Install with: pip install google-genai"
        )
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
        )
    client = genai_new.Client(api_key=api_key)
    model = _GeminiModel(client, model_name)
    print(f"Gemini model initialized: {model_name}")
    return model


def extract_gemini_response_text(response) -> str:
    """Extract text from Gemini response with robust fallbacks."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    text_parts = [
        part.text for part in parts if hasattr(part, "text") and part.text
    ]
    return "\n".join(text_parts).strip()


def call_gemini(
    gemini_model,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """Call Gemini API with input truncation.

    Truncates input to ~1,000,000 tokens if it exceeds Gemini's limit.

    Args:
        gemini_model: Initialized GenerativeModel instance.
        prompt: User prompt text.
        system_prompt: Optional system prompt (prepended to prompt).
        max_new_tokens: Max output tokens.

    Returns:
        Tuple of (generated_text, token_info_dict).
        token_info_dict contains: output_tokens, finish_reason, backend.
    """
    try:
        full_prompt = _compose_prompt(prompt, system_prompt)

        full_prompt, input_token_count, was_truncated = _truncate_gemini_prompt_to_limit(
            gemini_model, full_prompt
        )
        if was_truncated:
            if input_token_count is not None:
                print(
                    "Warning: Input prompt exceeded allowed token limit "
                    f"({MAX_ALLOWED_INPUT_TOKENS}). Truncated to first "
                    f"~{TRUNCATED_INPUT_TOKENS} tokens; current input tokens={input_token_count}."
                )
            else:
                print(
                    "Warning: Input prompt likely exceeded allowed token limit "
                    f"({MAX_ALLOWED_INPUT_TOKENS}). Truncated to safe fallback "
                    f"length (~{TRUNCATED_INPUT_TOKENS} tokens)."
                )

        # Configure generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        if max_new_tokens:
            generation_config["max_output_tokens"] = max_new_tokens

        # Generate response
        if generation_config:
            response = gemini_model.generate_content(
                full_prompt, generation_config=generation_config
            )
        else:
            response = gemini_model.generate_content(full_prompt)

        # Handle response safely
        if not response.candidates:
            raise RuntimeError(
                "Gemini API returned no candidates. Response may have been blocked."
            )

        candidate = response.candidates[0]
        # New SDK returns finish_reason as a FinishReason enum or string;
        # normalize to a lowercase string for consistent handling.
        raw_reason = candidate.finish_reason
        raw_reason_str = str(raw_reason).upper()
        # Support both old int-based (legacy) and new name-based finish reasons.
        _INT_REASON_MAP = {2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION"}
        if isinstance(raw_reason, int):
            raw_reason_str = _INT_REASON_MAP.get(raw_reason, str(raw_reason))
        finish_reason = raw_reason_str.lower()

        token_info = {
            "backend": "gemini",
            "finish_reason": finish_reason,
            "output_tokens": 0,
            "input_tokens": input_token_count,
            "input_truncated": was_truncated,
        }

        if "MAX_TOKENS" in raw_reason_str:
            text_parts = []
            if candidate.content and candidate.content.parts:
                text_parts = [
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]
            if text_parts:
                text = "\n".join(text_parts).strip()
            else:
                # Input was so long that no output tokens were left in the
                # context window.  Return empty string with a warning rather
                # than raising, so callers can handle it gracefully.
                print(
                    "Warning: Gemini hit output token limit with no text generated "
                    "(input likely consumed the entire context window). "
                    f"Input tokens: {input_token_count}. Returning empty string."
                )
                text = ""
            token_info["output_tokens"] = len(text) // 4
            token_info["finish_reason"] = "max_tokens"
            return text, token_info
        elif "SAFETY" in raw_reason_str:
            raise RuntimeError(
                "Gemini API blocked the response due to safety filters."
            )
        elif "RECITATION" in raw_reason_str:
            raise RuntimeError(
                "Gemini API blocked the response due to recitation."
            )

        text = extract_gemini_response_text(response)
        if text:
            token_info["output_tokens"] = len(text) // 4
            return text, token_info
        raise RuntimeError("Failed to extract text from Gemini response.")

    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")


def call_openrouter(
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_format: Optional[Dict[str, Any]] = None,
    reasoning_enabled: Optional[bool] = None,
    image: Optional[Any] = None,
) -> Tuple[str, Dict]:
    """Call OpenRouter API (OpenAI-compatible endpoint).

    Args:
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var).
        model_name: Model slug e.g. "anthropic/claude-opus-4.6".
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        max_new_tokens: Max output tokens.
        temperature: Optional sampling temperature.
        response_format: Optional OpenAI-compatible JSON response constraint.
        reasoning_enabled: Explicitly enable or disable provider reasoning.
        image: Optional image for a multimodal request. Accepts a PIL-compatible
            object with ``save()``, raw image bytes, or a ``data:image`` URI.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    if not HAS_OPENAI:
        raise ImportError(
            "openai package is required for OpenRouter. "
            "Install with: pip install openai"
        )
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY env var or pass api_key."
        )

    model_name = normalize_api_model("openrouter", model_name)
    prompt = sanitize_unicode_text(prompt)
    if system_prompt is not None:
        system_prompt = sanitize_unicode_text(system_prompt)
    context_window = OPENROUTER_CONTEXT_WINDOWS.get(model_name)
    input_truncated = False
    original_prompt_chars = len(prompt)
    if context_window is not None:
        reserved_output = int(max_new_tokens or 0)
        prompt_char_budget = max(
            1,
            context_window
            - reserved_output
            - OPENROUTER_CONTEXT_SAFETY_TOKENS
            - len(system_prompt or ""),
        )
        prompt, input_truncated = _truncate_prompt_preserving_ends(
            prompt, prompt_char_budget
        )
        if input_truncated:
            print(
                "OpenRouter prompt exceeded the conservative context budget; "
                f"truncated from {original_prompt_chars} to {len(prompt)} characters "
                f"while reserving {reserved_output} output tokens."
            )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image is None:
        user_content: Any = prompt
    else:
        if isinstance(image, str):
            if not image.startswith("data:image"):
                raise ValueError(
                    "OpenRouter image strings must be data:image URIs."
                )
            image_uri = image
        elif isinstance(image, (bytes, bytearray)):
            image_uri = (
                "data:image/png;base64,"
                + base64.b64encode(bytes(image)).decode("ascii")
            )
        elif hasattr(image, "save"):
            with io.BytesIO() as image_buffer:
                image.save(image_buffer, format="PNG")
                image_uri = (
                    "data:image/png;base64,"
                    + base64.b64encode(image_buffer.getvalue()).decode("ascii")
                )
        else:
            raise TypeError(
                "OpenRouter image must be a PIL-compatible image, bytes, or data URI."
            )
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]
    messages.append({"role": "user", "content": user_content})

    kwargs = {}
    if max_new_tokens:
        kwargs["max_tokens"] = max_new_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    extra_body: Dict[str, Any] = {}
    reasoning_max_tokens: Optional[int] = None
    if reasoning_enabled is not None:
        reasoning_config: Dict[str, Any] = {"enabled": reasoning_enabled}
        if reasoning_enabled and max_new_tokens:
            try:
                configured_reasoning_tokens = max(
                    1024,
                    int(os.getenv("OPENROUTER_REASONING_MAX_TOKENS", "8192")),
                )
            except ValueError:
                configured_reasoning_tokens = 8192
            # OpenRouter requires the overall completion maximum to be larger
            # than the reasoning budget so a final answer can still be emitted.
            reasoning_max_tokens = min(
                configured_reasoning_tokens,
                max(0, int(max_new_tokens) - 1024),
            )
            if reasoning_max_tokens >= 1024:
                # max_tokens itself enables reasoning. Do not also send the
                # default-config switch; use the unambiguous budgeted form from
                # OpenRouter's unified reasoning API.
                reasoning_config = {"max_tokens": reasoning_max_tokens}
        extra_body["reasoning"] = reasoning_config
    if context_window is not None:
        # Final provider-side guard using OpenRouter's exact tokenizer. This is
        # normally a no-op after the conservative local bound, but protects
        # against image/schema/tokenizer overhead that character counting
        # cannot measure.
        extra_body["transforms"] = ["middle-out"]
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        max_attempts = max(1, int(os.getenv("OPENROUTER_MAX_ATTEMPTS", "5")))
    except ValueError:
        max_attempts = 5
    response = None
    for attempt in range(max_attempts):
        client = _OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
            break
        except Exception as exc:
            status, reason, limit_source, retry_after = _openrouter_retry_details(exc)
            in_flight_402 = (
                status == 402 and reason == "in_flight_budget_exhausted"
            )
            credits_402 = status == 402 and limit_source == "openrouter_credits"
            if in_flight_402 or credits_402:
                # Both are account-wide admission failures, not a bad request
                # for this one problem: every subsequent call will fail
                # identically until something external changes. Raise a
                # dedicated type so callers (task_benchmark_domain.py,
                # pinchbench/claweval eval loops) can stop the whole batch
                # instead of burning through every remaining item with the
                # same doomed request.
                detail = (
                    "the account's shared in-flight budget is exhausted"
                    if in_flight_402
                    else "the account does not have enough remaining credits "
                    "for the requested max_tokens"
                )
                raise OpenRouterInFlightBudgetError(
                    f"OpenRouter rejected the request because {detail}. No "
                    "automatic retry was made; completed checkpoints are "
                    "preserved. Wait for other jobs to finish, reduce "
                    "concurrent API requests, lower max_tokens, or add "
                    "credits at https://openrouter.ai/settings/credits."
                ) from exc
            retryable = status == 429 or (
                status is not None and status >= 500
            )
            if not retryable or attempt + 1 >= max_attempts:
                raise RuntimeError(f"Error calling OpenRouter API: {exc}") from exc
            if retry_after <= 0:
                retry_after = min(60.0, 2.0**attempt)
            delay = retry_after
            print(
                f"OpenRouter HTTP {status} ({reason or 'transient'}); "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_attempts})",
                flush=True,
            )
            time.sleep(delay)
        finally:
            # A fresh SDK client is created for each independent API call.
            # Closing it prevents long parallel runs from leaking sockets.
            try:
                client.close()
            except Exception:
                pass
    if response is None:
        raise RuntimeError("OpenRouter API returned no response")

    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            str(
                block.get("text", "")
                if isinstance(block, dict)
                else getattr(block, "text", "")
            )
            for block in content
            if (
                isinstance(block, dict) and block.get("text")
            ) or getattr(block, "text", None)
        )
    else:
        text = ""

    message_data = message.model_dump() if hasattr(message, "model_dump") else {}
    reasoning_parts = []
    for key in ("reasoning", "reasoning_content"):
        value = message_data.get(key, getattr(message, key, None))
        if isinstance(value, str) and value.strip():
            reasoning_parts.append(value.strip())
    details = message_data.get("reasoning_details") or []
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            value = detail.get("text") or detail.get("summary")
            if isinstance(value, str) and value.strip():
                reasoning_parts.append(value.strip())
    reasoning_text = "\n".join(reasoning_parts).strip()
    used_reasoning_fallback = False
    if not text.strip() and reasoning_text:
        # Reasoning endpoints may exhaust their generation budget before
        # emitting final content. Reflection is itself reasoning text, and a
        # complete reasoning-only extraction response is still preferable to
        # silently turning a non-empty provider response into an empty string.
        text = reasoning_text
        used_reasoning_fallback = True
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

    usage = getattr(response, "usage", None)
    completion_details = (
        getattr(usage, "completion_tokens_details", None) if usage else None
    )
    reasoning_tokens = (
        getattr(completion_details, "reasoning_tokens", 0)
        if completion_details is not None
        else getattr(usage, "reasoning_tokens", 0) if usage else 0
    )
    token_info = {
        "backend": "openrouter",
        "model": model_name,
        "finish_reason": finish_reason,
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", len(text) // 4) if usage else len(text) // 4,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "cost_usd": getattr(usage, "cost", 0.0) if usage else 0.0,
        "request_count": 1,
        "reasoning_chars": len(reasoning_text),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "used_reasoning_fallback": used_reasoning_fallback,
        "max_new_tokens": max_new_tokens,
        "reasoning_enabled": reasoning_enabled,
        "reasoning_max_tokens": reasoning_max_tokens,
        "multimodal_image": image is not None,
        "input_truncated": input_truncated,
        "original_prompt_chars": original_prompt_chars,
        "sent_prompt_chars": len(prompt),
    }
    return text, token_info


# ---------------------------------------------------------------------------
# Gemini with ThinkingConfig (new google-genai SDK)
# ---------------------------------------------------------------------------
def call_gemini_thinking(
    api_key: str,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    thinking_level: str = "high",
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict]:
    """
    Call Gemini using the new google-genai SDK with ThinkingConfig support.

    Uses:
        from google import genai
        from google.genai import types
        client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
            ),
        )

    Args:
        api_key: Gemini API key.
        model_name: Model name (e.g., "gemini-3.1-pro-preview").
        prompt: User prompt.
        system_prompt: Optional system prompt prepended to prompt.
        thinking_level: "low", "medium", or "high" (default: "high").
        max_new_tokens: Optional max output tokens.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    if not HAS_GENAI_NEW:
        raise ImportError(
            "google-genai is required for ThinkingConfig support. "
            "Install with: pip install google-genai"
        )

    full_prompt = _compose_prompt(prompt, system_prompt)

    client = genai_new.Client(api_key=api_key)

    # Only gemini-3.1-pro-preview supports ThinkingConfig; all other models
    # must not include thinking_config in the request.
    _THINKING_MODELS = {"gemini-3.1-pro-preview"}
    supports_thinking = model_name in _THINKING_MODELS

    gen_config_kwargs: Dict = {}
    if supports_thinking:
        gen_config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=thinking_level
        )
        print(f"[Gemini] Using ThinkingConfig(thinking_level={thinking_level!r}) for {model_name}")
    else:
        print(
            f"[Gemini] Model {model_name!r} does not support ThinkingConfig — "
            "calling without thinking_config"
        )
    if max_new_tokens:
        gen_config_kwargs["max_output_tokens"] = max_new_tokens

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=genai_types.GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None,
        )
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini (thinking) API: {e}")
    finally:
        # A fresh SDK client is created for each independent API call (see
        # call_openrouter's client lifecycle). Closing it prevents long
        # parallel runs from leaking sockets/file descriptors.
        try:
            client.close()
        except Exception:
            pass

    text = getattr(response, "text", None)
    if not text:
        # Fallback: walk candidates
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            text_parts = [
                p.text for p in parts
                if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
            ]
            text = "\n".join(text_parts).strip()

    if not text:
        print(
            f"Warning: Gemini (thinking) returned no text for model={model_name}, "
            f"thinking_level={thinking_level}. Returning empty string."
        )
        text = ""

    token_info = {
        "backend": "gemini_thinking",
        "thinking_level": thinking_level,
        "model": model_name,
        "output_tokens": len(text) // 4,
        "finish_reason": "stop",
    }
    usage = getattr(response, "usage_metadata", None)
    if usage:
        token_info["input_tokens"] = getattr(usage, "prompt_token_count", 0)
        token_info["output_tokens"] = getattr(usage, "candidates_token_count", len(text) // 4)
        token_info["thinking_tokens"] = getattr(usage, "thoughts_token_count", 0)

    print(f"[Gemini thinking={thinking_level}] output_tokens={token_info['output_tokens']}")
    return text, token_info


# ---------------------------------------------------------------------------
# HuggingFace model loading & call
# ---------------------------------------------------------------------------
def load_hf_model(
    model_name: str,
    device: str = "cpu",
    load_in_8bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a HuggingFace model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: "cuda" or "cpu".
        load_in_8bit: Whether to use 8-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    if _TRANSFORMERS_IMPORT_ERROR is not None or torch is None:
        detail = (
            str(_TRANSFORMERS_IMPORT_ERROR)
            if _TRANSFORMERS_IMPORT_ERROR is not None
            else "PyTorch is not installed"
        )
        raise ImportError(
            "Local Hugging Face model loading is unavailable. Install compatible "
            f"torch, transformers, and huggingface-hub packages. Original error: {detail}"
        )

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    is_lora_adapter_dir = os.path.isdir(model_name) and os.path.exists(adapter_config_path)

    tokenizer_source = model_name
    adapter_base_model: Optional[str] = None
    if is_lora_adapter_dir:
        with open(adapter_config_path, "r", encoding="utf-8") as fp:
            adapter_cfg = json.load(fp)
        adapter_base_model = adapter_cfg.get("base_model_name_or_path")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        if is_lora_adapter_dir and adapter_base_model:
            print(
                "Tokenizer files not found in adapter directory; "
                f"falling back to base model tokenizer: {adapter_base_model}"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_base_model, trust_remote_code=True
            )
        else:
            raise

    model_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if is_lora_adapter_dir:
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Loading LoRA adapter checkpoints requires peft. "
                "Install with: pip install peft"
            ) from exc

        model = AutoPeftModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!")
    return model, tokenizer


def call_hf_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[str, Dict]:
    """Call a HuggingFace model.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        model_name: Model name (used to detect DeepSeek-R1 settings).
        prompt: User prompt text.
        system_prompt: Optional system prompt (prepended to prompt).
        max_new_tokens: Max output tokens. Defaults to 32768.
        device: Device the model is on.

    Returns:
        Tuple of (generated_text, token_info_dict).
    """
    full_prompt = _compose_prompt(prompt, system_prompt)

    try:
        model_context_limit = _resolve_hf_context_limit(model, tokenizer)
        input_max_length = int(min(model_context_limit, MAX_ALLOWED_INPUT_TOKENS))
        if input_max_length > TRUNCATED_INPUT_TOKENS:
            input_max_length = TRUNCATED_INPUT_TOKENS

        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_max_length,
        ).to(device)

        input_token_count = inputs["input_ids"].shape[1]
        if input_token_count >= input_max_length:
            print(
                f"Warning: Input prompt reached model context limit "
                f"({input_max_length} tokens) and may be truncated."
            )

        if max_new_tokens is None:
            max_new_tokens = 32768

        print(f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}")

        with torch.no_grad():
            is_deepseek_r1 = "DeepSeek-R1" in model_name

            if is_deepseek_r1:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        output_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
        output_token_count = output_token_ids.shape[0]

        token_info = {
            "backend": "huggingface",
            "output_tokens": output_token_count,
            "input_tokens": int(input_token_count),
            "input_truncated": bool(input_token_count >= input_max_length),
            "input_limit": int(input_max_length),
        }
        return generated_text.strip(), token_info

    except Exception as e:
        print(f"Error calling model: {e}")
        return "", {"output_tokens": 0}
