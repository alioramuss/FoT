"""
Chain of Density (CoD) Server - Collects all insights from JSON files, then uses
the Chain of Density prompting method to generate increasingly dense summaries.

Step 1: Same as server_text.py - collect all insights from problem/paper JSON files.
Step 2: Apply Chain of Density (CoD) summarization from:
        "From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"
        (Adams et al., 2023, https://aclanthology.org/2023.newsum-1.7.pdf)
        - Iteratively generate 5 increasingly dense summaries
        - Each iteration adds 1-3 missing entities while maintaining word count
        - Final (5th) summary is the most dense and is saved as encyclopedia
Output: encyclopedia.json with {"insight_summary": <densest_summary>}
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from utils import setup_gemini, call_gemini


# ---------------------------------------------------------------------------
# Chain of Density prompt (exact prompt from Adams et al., 2023)
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_WORDS = 80


COD_PROMPT = """\
Article: {article}
You will generate increasingly concise, entity-dense summaries of the above Article.
Repeat the following 2 steps 5 times.
Step 1. Identify 1-3 informative Entities (";" delimited) from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.
A Missing Entity is: - Relevant: to the main story. - Specific: descriptive yet concise (5 words or fewer). - Novel: not in the previous summary. - Faithful: present in the Article. - Anywhere: located anywhere in the Article.
Guidelines:
- The first summary should be approximately {summary_words} words yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach approximately {summary_words} words.
- Make every word count: re-write the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
Remember, use the exact same number of words for each summary.
Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""


# ---------------------------------------------------------------------------
# Insight collection (same as server_text.py Step 1)
# ---------------------------------------------------------------------------

def collect_insight_books(input_dir: str, max_files: Optional[int] = None) -> Dict[str, str]:
    """
    Collect ALL insights from problem*.json and paper*.json files.

    Args:
        input_dir: Directory containing insight JSON files.
        max_files: Maximum number of files to process (sorted by numeric suffix).

    Returns:
        Dictionary of {insight_name: description}.
    """
    input_path = Path(input_dir)
    print(f"Searching for problem*.json and paper*.json files under {input_path}...")
    json_files = list(input_path.rglob("problem_*.json")) + list(input_path.rglob("paper_*.json"))
    json_files = [str(f) for f in json_files]

    # Sort files by numeric suffix (e.g. paper_0001.json -> 1, problem_0042.json -> 42)
    def _extract_number(filepath):
        basename = Path(filepath).stem
        match = re.search(r'_(\d+)$', basename)
        return int(match.group(1)) if match else float('inf')

    json_files.sort(key=_extract_number)

    print(f"Found {len(json_files)} problem/paper*.json files")

    # Limit to max_files if specified
    if max_files is not None and max_files < len(json_files):
        json_files = json_files[:max_files]
        print(f"Using first {max_files} files (sorted by number)")

    if not json_files:
        print("ERROR: No problem*.json or paper*.json files found!")
        return {}

    all_insights = {}
    insight_counter = 0
    files_processed = 0

    print(f"Collecting insights from {len(json_files)} files...")

    # Debug: show first few files
    if len(json_files) > 0:
        print(f"  Sample files:")
        for f in json_files[:3]:
            print(f"    - {f}")

    # Metadata keys to skip
    metadata_keys = {
        "paper_name", "problem", "problem_id", "iteration",
        "is_correct", "number_output_tokens", "loop_count",
    }

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print(f"  Warning: {json_file} is not a dict!")
                print(f"    Type: {type(data)}")
                print(f"    Content preview: {str(data)[:200]}")
                continue

            # Check if wrapped in insight_book/behavior_book
            if "insight_book" in data:
                insights_dict = data["insight_book"]
            elif "behavior_book" in data:
                insights_dict = data["behavior_book"]
            else:
                insights_dict = data

            if not isinstance(insights_dict, dict):
                print(f"  Warning: insights in {json_file} is not a dict, skipping")
                continue

            file_insight_count = 0
            for insight_name, insight_desc in insights_dict.items():
                if insight_name in metadata_keys:
                    continue

                insight_counter += 1
                file_insight_count += 1
                indexed_key = f"{insight_name}_{insight_counter:06d}"
                all_insights[indexed_key] = insight_desc

            if file_insight_count > 0:
                files_processed += 1
                if files_processed % 100 == 0:
                    print(f"  Processed {files_processed} files, collected {insight_counter} insights...")
            else:
                print(f"  Warning: No insights found in {json_file}")
                print(f"    Keys in file: {list(insights_dict.keys())[:10]}")

        except Exception as e:
            print(f"  Warning: Failed to read {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nCollected ALL {insight_counter} insights from {files_processed} files")
    print(f"Insight store contains {len(all_insights)} entries (no deduplication)")

    if files_processed == 0 and len(json_files) > 0:
        print(f"\nWARNING: Found {len(json_files)} files but processed 0!")
        print("This likely means all files have empty 'insight_book' dictionaries.")

    return all_insights


# ---------------------------------------------------------------------------
# Gemini model setup/call (shared utils)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chain of Density summarization
# ---------------------------------------------------------------------------

def format_insights_as_text(insights: Dict[str, str]) -> str:
    """Format all collected insights into a single text block."""
    lines = []
    for name, desc in insights.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 50000) -> List[str]:
    """Split text into chunks by character count."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _strip_reasoning_blocks(response_text: str) -> str:
    """Remove reasoning tags sometimes emitted in DeepSeek message content."""
    return re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _normalize_cod_payload(payload) -> list:
    """Normalize common OpenAI-compatible JSON variants to the CoD schema."""
    if isinstance(payload, dict):
        for key in ("summaries", "chain_of_density", "results", "output"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                payload = candidate
                break

    if not isinstance(payload, list):
        raise ValueError("CoD response must contain a JSON list")
    if len(payload) != 5:
        raise ValueError(
            f"CoD response must contain exactly 5 summaries; got {len(payload)}"
        )

    normalized = []
    for index, entry in enumerate(payload, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"CoD summary {index} is not a JSON object")
        keys = {
            re.sub(r"[^a-z0-9]", "", str(key).lower()): value
            for key, value in entry.items()
        }
        missing = keys.get("missingentities", "")
        summary = keys.get("densersummary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"CoD summary {index} has no non-empty Denser_Summary"
            )
        if isinstance(missing, list):
            missing = "; ".join(str(item) for item in missing)
        normalized.append(
            {
                "Missing_Entities": str(missing),
                "Denser_Summary": summary.strip(),
            }
        )
    return normalized


def extract_json_from_response(response_text: str) -> list:
    """Extract and validate CoD JSON from Gemini or DeepSeek-style output."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("CoD response is empty")

    cleaned = _strip_reasoning_blocks(response_text)
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = fenced + [cleaned]
    decoder = json.JSONDecoder()
    errors = []
    for candidate in candidates:
        for start, character in enumerate(candidate):
            if character not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[start:])
                return _normalize_cod_payload(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(str(exc))
    detail = next(
        (error for error in errors if "exactly 5 summaries" in error),
        errors[-1] if errors else "no JSON value found",
    )
    raise ValueError(f"Could not parse a valid CoD response: {detail}")


def cod_summarize(
    article_text: str,
    model,
    max_output_tokens: int = 32768,
    call_model: Optional[Callable[[str, int], tuple]] = None,
    summary_words: int = DEFAULT_SUMMARY_WORDS,
) -> tuple:
    """
    Apply Chain of Density prompting to a single article/text block.

    Returns (densest_summary, output_tokens).
    """
    if summary_words < 1:
        raise ValueError("summary_words must be positive")
    prompt = COD_PROMPT.format(
        article=article_text,
        summary_words=summary_words,
    )
    if call_model is None:
        response, token_info = call_gemini(
            model, prompt, max_new_tokens=max_output_tokens
        )
    else:
        response, token_info = call_model(prompt, max_output_tokens)
    out_tok = token_info.get("output_tokens", 0)

    try:
        cod_results = extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        preview = response[:2000] if isinstance(response, str) else str(response)
        raise RuntimeError(
            "Model returned an invalid Chain-of-Density response: "
            f"{e}. Response preview: {preview}"
        ) from e

    # Print each iteration
    for i, entry in enumerate(cod_results):
        missing = entry.get("Missing_Entities", "N/A")
        summary = entry.get("Denser_Summary", "")
        word_count = len(summary.split())
        print(f"  Iteration {i + 1}: +[{missing}] ({word_count} words)")

    # Return the densest (last) summary
    densest = cod_results[-1].get("Denser_Summary", "")
    return densest, out_tok


def cod_summarize_with_chunking(
    insights_text: str,
    model,
    chunk_size: int = 50000,
    max_output_tokens: int = 32768,
    call_model: Optional[Callable[[str, int], tuple]] = None,
    summary_words: int = DEFAULT_SUMMARY_WORDS,
) -> tuple:
    """
    Apply Chain of Density summarization with chunking for large inputs.

    1. Chunk insights into manageable pieces
    2. Run CoD on each chunk to get dense summaries
    3. If multiple chunks, consolidate with a final CoD pass

    Returns:
        (final_summary, final_output_tokens) — tokens only for the last
        CoD call that produces the final insight library.
    """
    chunks = chunk_text(insights_text, chunk_size=chunk_size)
    print(f"  Split insights into {len(chunks)} chunks (chunk_size={chunk_size})")

    if len(chunks) == 1:
        # Single chunk - directly apply CoD
        print("  Single chunk, applying Chain of Density directly...")
        return cod_summarize(
            chunks[0], model, max_output_tokens, call_model=call_model,
            summary_words=summary_words,
        )

    # Multiple chunks - CoD each chunk, then consolidate
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"\n  --- Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars) ---")
        summary, _ = cod_summarize(
            chunk, model, max_output_tokens, call_model=call_model,
            summary_words=summary_words,
        )
        chunk_summaries.append(summary)
        print(f"  Chunk {i + 1} densest summary: {len(summary.split())} words")

    # Consolidate: combine chunk summaries and run final CoD pass
    combined = "\n\n".join(
        f"[Section {i + 1}] {s}" for i, s in enumerate(chunk_summaries)
    )
    print(f"\n  --- Final consolidation ({len(combined)} chars) ---")

    # If combined is still too large, recursively chunk
    if len(combined) > chunk_size:
        print(f"  Combined summaries exceed chunk_size, recursing...")
        return cod_summarize_with_chunking(
            combined,
            model,
            chunk_size,
            max_output_tokens,
            call_model=call_model,
            summary_words=summary_words,
        )

    # Final call — this produces the library; return its token count
    return cod_summarize(
        combined, model, max_output_tokens, call_model=call_model,
        summary_words=summary_words,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_encyclopedia(summary: str, output_dir: str):
    """Save densest summary using the canonical flat insight_* protocol."""
    os.makedirs(output_dir, exist_ok=True)
    encyclopedia_path = os.path.join(output_dir, "encyclopedia.json")

    encyclopedia = {"insight_summary": summary}

    with open(encyclopedia_path, "w", encoding="utf-8") as f:
        json.dump(encyclopedia, f, indent=2, ensure_ascii=False)
    print(f"Encyclopedia saved to: {encyclopedia_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain of Density Server - Collect insights and summarize via CoD prompting (Adams et al., 2023)"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="math_output",
        help="Input directory containing insight JSON files (default: math_output)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Output directory for encyclopedia (default: math_output)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of JSON files to use, sorted by number. If not provided, all files are used.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Max characters per chunk for CoD summarization (default: 50000)",
    )
    parser.add_argument(
        "--cod-summary-words",
        type=int,
        default=DEFAULT_SUMMARY_WORDS,
        help=(
            "Target word count for every Chain-of-Density summary "
            f"(default: {DEFAULT_SUMMARY_WORDS})"
        ),
    )

    args = parser.parse_args()

    if args.cod_summary_words < 1:
        parser.error("--cod-summary-words must be positive")

    start_time = time.time()

    # Step 1: Collect all insights (same as server_text.py Step 1)
    print("=" * 80)
    print("STEP 1: Collecting Insights")
    print("=" * 80)
    insights = collect_insight_books(args.input_dir, max_files=args.max_files)

    if not insights:
        print("No insights collected. Exiting.")
        exit(1)

    # Step 2: Chain of Density summarization (replaces server_text.py Steps 2-3)
    print("\n" + "=" * 80)
    print("STEP 2: Chain of Density Summarization (Adams et al., 2023)")
    print("=" * 80)
    gemini_model = setup_gemini(api_key=args.gemini_api_key, model_name=args.gemini_model)
    insights_text = format_insights_as_text(insights)
    print(f"Total insights text: {len(insights_text)} characters")

    summary, total_output_tokens = cod_summarize_with_chunking(
        insights_text,
        model=gemini_model,
        chunk_size=args.chunk_size,
        summary_words=args.cod_summary_words,
    )

    print(f"\nDensest summary:\n{summary}")

    # Save as encyclopedia
    save_encyclopedia(summary, output_dir=args.output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("CHAIN OF DENSITY COMPLETE")
    print("=" * 80)
    print(f"Total insights collected: {len(insights)}")
    print(f"Densest summary length: {len(summary.split())} words, {len(summary)} chars")
    print(f"Insight library output tokens: {total_output_tokens}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time elapsed: {elapsed:.1f}s")
