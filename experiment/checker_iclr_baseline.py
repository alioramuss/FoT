"""
Baseline: Check ICLR Accept papers using learned skills from encyclopedia.
- Reads ICLR evaluation papers from a local scraper corpus when --papers-dir
  is supplied, without contacting OpenReview.
- Uses learned skills encyclopedia to extract skill names and check if they guide papers.
- Each skill includes: year proposed, is_iclr2023 (boolean).
- Outputs: overall percentage, percentage for pre-2023 skills, percentage for ICLR2023 skills.

Usage example:
  python checker_iclr_baseline.py \
      --gemini-key $GEMINI_API_KEY \
      --year 2024 \
      --output baseline_results.json
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Import shared functions from checker_iclr for maximum consistency
from checker_iclr import (
    GeminiClient,
    OpenRouterClient,
    _extract_text_from_pdf_bytes,
    _fetch_paper_content,
    _load_local_accepted_papers,
    _quiet_pdf_parser_diagnostics,
    call_api,
    score_paper,
)

# Reuse the repository's existing RAG context-budget helpers so retrieval
# here is bounded the same way as trace_pooling.py's retrieval path.
from trace_pooling import (
    DEFAULT_RAG_MAX_TOKENS,
    estimate_tokens,
    truncate_to_token_budget,
)

# Note: We keep fetch_accept_tracks and _hydrate_papers_from_client local
# because _hydrate_papers_from_client needs to extract keywords


def fetch_accept_tracks(
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
) -> List[Dict]:
    """Fetch accepted papers using OpenReview client.

    Uses openreview-py to query ICLR submissions and filter by venue field.
    """
    try:
        import openreview

        use_or_client = True
    except ImportError:
        use_or_client = False
        print("Warning: openreview-py not installed, falling back to requests")

    # Default to oral if nothing specified (backward compatible)
    accept_any = accept_oral or accept_spotlight or accept_poster
    accept_oral = accept_oral or not accept_any

    decisions: List[Dict] = []

    # Try OpenReview client first
    if use_or_client:
        try:
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net"
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

                decisions.append({"forum": sub.forum, "track": track})
                if max_papers and len(decisions) >= max_papers:
                    break

            if decisions:
                print(f"Found {len(decisions)} accepted papers via client")
                # Sort by track priority: oral > spotlight > poster
                track_order = {"oral": 0, "spotlight": 1, "poster": 2}
                decisions.sort(key=lambda p: track_order.get(p.get("track", ""), 999))
                # Store client for later use
                for decision in decisions:
                    decision["_client"] = client
                return decisions
            else:
                print("No accepted papers found via client")
                return []
        except Exception as e:
            print(f"OpenReview client error: {e}")
            return []

    # Fallback: old requests-based approach
    print("Using requests-based fallback (may not work for ICLR 2024+)...")
    return []


def generate_skills_from_keywords(
    model: Any,
    keywords: List[str],
) -> Tuple[List[Dict], Dict]:
    """Generate skills/insights from paper keywords using the configured model.

    For each keyword, asks the model for a corresponding skill description.

    Returns:
        Tuple of (skills_list, token_info) where skills_list contains skill dicts with name/description
    """
    if not keywords:
        return [], {"output_tokens": 0}

    skills = []
    total_tokens = 0

    for keyword in keywords:
        prompt = f"""Generate a skill corresponding to the given keyword: {keyword}.

Provide a technical description and guidelines of using this skill/insight to resolve questions.

Respond in the following JSON format:
{{
  "skill_name": "concise name of the skill",
  "description": "detailed technical description and usage guidelines"
}}"""
        try:
            response, token_info = call_api(model, prompt)
            total_tokens += token_info.get("output_tokens", 0)

            # Try to parse JSON from response
            try:
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    skill_data = json.loads(json_match.group())
                    skills.append(
                        {
                            "name": skill_data.get("skill_name", keyword),
                            "description": skill_data.get(
                                "description", response.strip()
                            ),
                        }
                    )
                else:
                    skills.append({"name": keyword, "description": response.strip()})
            except json.JSONDecodeError:
                skills.append({"name": keyword, "description": response.strip()})

            time.sleep(0.3)  # Rate limiting between keyword processing
        except Exception as e:
            print(f"    Warning: Failed to generate skill for keyword '{keyword}': {e}")
            skills.append(
                {"name": keyword, "description": f"Skill related to {keyword}"}
            )

    return skills, {"output_tokens": total_tokens}


def generate_skills_phase_from_iclr2023_keywords(
    model: Any,
    output_file: str,
) -> List[Dict]:
    """Phase 1, Mode 1: Generate skills from ICLR 2023 top25 paper keywords.

    Fetches ICLR 2023 top25 papers from OpenReview API using Blind_Submission invitation.
    Implements exponential backoff to handle rate limiting.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Generate Skills from ICLR 2023 Top25 Paper Keywords")
    print("=" * 80)

    # Fetch ICLR 2023 notable top25 papers from OpenReview API
    print("Fetching ICLR 2023 Notable Top 25% papers from OpenReview...")
    import requests

    papers = []
    api_url = "https://api.openreview.net/notes"
    offset = 0
    limit = 1000

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    while True:
        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries and not success:
            try:
                params = {
                    "invitation": "ICLR.cc/2023/Conference/-/Blind_Submission",
                    "details": "replyCount,invitation,original",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = session.get(api_url, params=params, timeout=60)
                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])

                if not notes:
                    success = True  # Reached end of results
                    break

                # Filter for notable top 25% papers
                for note in notes:
                    content = note.get("content", {})
                    venue = content.get("venue", "")

                    # Check if this is a notable top 25% paper
                    if (
                        "Notable Top 25%" in venue
                        or ("Notable" in venue and "Top 25%" in venue)
                        or "notable top 25%" in venue.lower()
                    ):
                        papers.append(note)

                offset += limit
                print(
                    f"  Searched offset {offset}, found {len(papers)} notable top 25% papers so far..."
                )
                success = True  # Successfully fetched this batch

            except requests.exceptions.HTTPError as e:
                error_code = e.response.status_code if hasattr(e, "response") else 0
                error_str = str(e)

                # Check if rate limited (429)
                if (
                    error_code == 429
                    or "429" in error_str
                    or "Too Many Requests" in error_str
                ):
                    retry_count += 1
                    if retry_count < max_retries:
                        # Exponential backoff: start at 10s, then 20s, 40s, 80s, 160s
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  HTTP Error: {error_str}")
                    break
            except Exception as e:
                error_str = str(e)

                # Check if rate limited in error message
                if (
                    "429" in error_str
                    or "RateLimitError" in error_str
                    or "Too many requests" in error_str.lower()
                ):
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited. Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  Error fetching papers: {e}")
                    break

        if not success or not notes:
            break

    if not papers:
        print("Error: No ICLR 2023 notable top 25% papers found")
        print(
            "Tip: OpenReview API rate limit is 60 requests/minute. Try again later or contact OpenReview support."
        )
        return []

    print(f"Retrieved {len(papers)} ICLR 2023 notable top 25% papers")

    # Extract all unique keywords from papers
    all_keywords = set()
    for note in papers:
        content = note.get("content", {})
        keywords_raw = content.get("keywords", {})

        if isinstance(keywords_raw, dict):
            keywords_raw = keywords_raw.get("value", [])

        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = keywords_raw
        else:
            keywords = []

        all_keywords.update(keywords)

    all_keywords = sorted(list(all_keywords))
    print(f"Extracted {len(all_keywords)} unique keywords from ICLR 2023 papers")
    print(f"Sample keywords: {all_keywords[:5]}")

    # Generate skills from keywords
    print(f"\nGenerating skills from {len(all_keywords)} keywords...")
    skills, token_info = generate_skills_from_keywords(model, all_keywords)
    print(
        f"Generated {len(skills)} skills with {token_info.get('output_tokens', 0)} tokens"
    )

    # Save skills
    skills_data = {
        "source": "ICLR 2023 Notable Top 25% Papers",
        "num_papers": len(papers),
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": token_info.get("output_tokens", 0),
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(skills_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_file}")

    return skills


def generate_skills_phase_from_iclr2024_keywords(
    model: Any,
    output_file: str,
) -> List[Dict]:
    """Phase 1, Mode 1b: Generate skills from ICLR 2024 accepted paper keywords.

    Fetches ICLR 2024 accepted papers (Oral/Spotlight/Poster) from OpenReview API.
    Implements exponential backoff to handle rate limiting.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Generate Skills from ICLR 2024 Accepted Paper Keywords")
    print("=" * 80)

    # Fetch ICLR 2024 accepted papers
    print("Fetching ICLR 2024 accepted papers from OpenReview...")
    import requests

    papers = []
    api_url = "https://api2.openreview.net/notes"
    offset = 0
    limit = 200

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    while True:
        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries and not success:
            try:
                params = {
                    "invitation": "ICLR.cc/2024/Conference/-/Submission",
                    "details": "replyCount",
                    "offset": offset,
                    "limit": limit,
                    "sort": "number:asc",
                }

                response = session.get(api_url, params=params, timeout=60)

                if response.status_code == 429:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break

                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])

                if not notes:
                    success = True
                    break

                # Filter for accepted papers (Oral/Spotlight/Poster)
                for note in notes:
                    content = note.get("content", {})
                    venue_obj = content.get("venue", {})
                    venue_val = ""
                    if isinstance(venue_obj, dict):
                        venue_val = str(venue_obj.get("value", ""))
                    elif isinstance(venue_obj, str):
                        venue_val = venue_obj
                    venue_lower = venue_val.lower()

                    # Check if accepted (oral, spotlight, or poster)
                    if (
                        "oral" in venue_lower
                        or "spotlight" in venue_lower
                        or "poster" in venue_lower
                    ):
                        papers.append(note)

                offset += limit
                if offset % 1000 == 0:
                    print(
                        f"  Searched offset {offset}, found {len(papers)} accepted papers so far..."
                    )
                success = True
                time.sleep(1.0)  # Rate limit between requests

            except requests.exceptions.HTTPError as e:
                error_code = e.response.status_code if hasattr(e, "response") else 0
                error_str = str(e)

                if error_code == 429 or "429" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited (429). Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  HTTP Error: {error_str}")
                    break
            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "RateLimitError" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(
                            f"  Rate limited. Waiting {wait_time}s before retry ({retry_count}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        print(f"  Error: Rate limited - max retries exceeded")
                        break
                else:
                    print(f"  Error fetching papers: {e}")
                    break

        if not success or not notes:
            break

    if not papers:
        print("Error: No ICLR 2024 accepted papers found")
        return []

    print(f"Retrieved {len(papers)} ICLR 2024 accepted papers")

    # Extract all unique keywords from papers
    all_keywords = set()
    for note in papers:
        content = note.get("content", {})
        keywords_raw = content.get("keywords", {})

        if isinstance(keywords_raw, dict):
            keywords_raw = keywords_raw.get("value", [])

        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, list):
            keywords = keywords_raw
        else:
            keywords = []

        all_keywords.update(keywords)

    all_keywords = sorted(list(all_keywords))
    print(f"Extracted {len(all_keywords)} unique keywords from ICLR 2024 papers")
    print(f"Sample keywords: {all_keywords[:5]}")

    # Generate skills from keywords
    print(f"\nGenerating skills from {len(all_keywords)} keywords...")
    skills, token_info = generate_skills_from_keywords(model, all_keywords)
    print(
        f"Generated {len(skills)} skills with {token_info.get('output_tokens', 0)} tokens"
    )

    # Save skills
    skills_data = {
        "source": "ICLR 2024 Accepted Papers (Oral/Spotlight/Poster)",
        "num_papers": len(papers),
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": token_info.get("output_tokens", 0),
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(skills_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_file}")

    return skills


def generate_skills_phase_from_general_knowledge(
    model: Any,
    num_skills: int,
    output_file: str,
    year: int = 2024,
) -> List[Dict]:
    """Phase 1, Mode 2: Generate x skills from general machine learning knowledge."""
    print("\n" + "=" * 80)
    print(f"PHASE 1: Generate {num_skills} Skills from General ML Knowledge")
    print("=" * 80)

    skills: List[Dict] = []
    seen_names = set()
    total_output_tokens = 0
    batch_size = 25
    max_requests = math.ceil(num_skills / batch_size) + 4

    for request_index in range(1, max_requests + 1):
        remaining = num_skills - len(skills)
        if remaining <= 0:
            break
        requested_now = min(batch_size, remaining)
        existing_names = ", ".join(skill["name"] for skill in skills)
        avoid_section = (
            f"\nDo not repeat any of these previously generated skill names: {existing_names}"
            if existing_names
            else ""
        )
        prompt = f"""Generate exactly {requested_now} distinct, important, and fundamental skills/techniques in machine learning.

For each skill, provide:
1. A concise skill name
2. A detailed technical description and usage guidelines
{avoid_section}

Respond with exactly {requested_now} objects in one JSON array:
[
  {{
    "skill_name": "skill name",
    "description": "technical description and guidelines"
  }},
  ...
]
Return JSON only."""

        print(
            f"Requesting general-knowledge skill batch {request_index}: "
            f"{requested_now} new skills ({len(skills)}/{num_skills} complete)..."
        )
        response, token_info = call_api(model, prompt, max_output_tokens=8192)
        total_output_tokens += int(token_info.get("output_tokens", 0) or 0)
        parsed = _parse_generated_skills(response)
        added = _append_unique_skills(skills, seen_names, parsed, remaining)
        print(f"  Accepted {added} distinct valid skills from this batch")

    _require_nonempty_skills(skills, num_skills, "general-knowledge")
    print(f"Generated {len(skills)} skills with {total_output_tokens} tokens")

    # Save skills
    output_filename = output_file or f"general_baseline_skills_{num_skills}_year{year}.json"
    skills_result = {
        "source": f"General ML Knowledge ({num_skills} skills requested)",
        "provider": (
            "openrouter" if isinstance(model, OpenRouterClient) else "gemini"
        ),
        "generation_model": getattr(model, "model_name", None),
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": total_output_tokens,
    }
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(skills_result, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_filename}")

    return skills


def _parse_generated_skills(response_text: str) -> List[Dict]:
    """Parse and normalize a JSON-array skill response."""
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not match:
        print(f"Warning: no JSON array found in response: {response_text[:200]}")
        return []
    try:
        raw_skills = json.loads(match.group())
    except json.JSONDecodeError as exc:
        print(f"Warning: failed to parse skill JSON array: {exc}")
        return []
    if not isinstance(raw_skills, list):
        return []

    normalized = []
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("skill_name") or item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name and description:
            normalized.append({"name": name, "description": description})
    return normalized


def _append_unique_skills(
    destination: List[Dict],
    seen_names: set,
    candidates: List[Dict],
    limit: int,
) -> int:
    """Append up to ``limit`` case-insensitively unique, valid skills."""
    added = 0
    for skill in candidates:
        canonical_name = skill["name"].casefold()
        if canonical_name in seen_names:
            continue
        seen_names.add(canonical_name)
        destination.append(skill)
        added += 1
        if added >= limit:
            break
    return added


def _require_exact_skill_count(
    skills: List[Dict], expected: int, source_name: str
) -> None:
    if len(skills) != expected:
        raise RuntimeError(
            f"{source_name} generation produced {len(skills)} distinct valid skills; "
            f"exactly {expected} were required. No partial library will be written."
        )


def _require_nonempty_skills(
    skills: List[Dict], expected: int, source_name: str
) -> None:
    """Accept a short library, but never an empty one.

    General-knowledge generation asks the model for distinct fundamental ML
    skills with no source corpus to draw from, so it saturates and starts
    returning only duplicates well before an arbitrary target. Falling short
    is expected there and does not invalidate the library, unlike the
    paper-grounded/RAG modes where a short result means papers were missed.
    """
    if not skills:
        raise RuntimeError(
            f"{source_name} generation produced no distinct valid skills; "
            f"{expected} were requested. No empty library will be written."
        )
    if len(skills) < expected:
        print(
            f"Warning: {source_name} generation produced {len(skills)} distinct "
            f"valid skills out of {expected} requested. The model saturated on "
            "duplicates; writing the smaller library as-is."
        )


def _load_rag_metadata_fallbacks(papers_dir: Path) -> Dict[str, str]:
    """Map scraper paper IDs to local metadata text for damaged PDFs."""
    metadata_path = papers_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Warning: could not load RAG metadata fallback {metadata_path}: {exc}")
        return {}
    if not isinstance(metadata, list):
        print(f"Warning: RAG metadata fallback is not a list: {metadata_path}")
        return {}

    def field(raw: Dict, content: Dict, name: str) -> str:
        value = content.get(name, raw.get(name, ""))
        if isinstance(value, dict):
            value = value.get("value", "")
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        return str(value or "").strip()

    fallbacks: Dict[str, str] = {}
    for raw in metadata:
        if not isinstance(raw, dict):
            continue
        paper_ids = {
            str(raw.get(name) or "").strip()
            for name in ("forum", "id")
            if str(raw.get(name) or "").strip()
        }
        if not paper_ids:
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else {}
        sections = []
        for name in (
            "title",
            "abstract",
            "keywords",
            "tldr",
            "summary",
            "primary_area",
            "research_area",
        ):
            value = field(raw, content, name)
            if value:
                sections.append(f"[{name}] {value}")
        if sections:
            fallback_text = "\n".join(sections)
            for paper_id in paper_ids:
                fallbacks[paper_id] = fallback_text
    return fallbacks


def _drop_unencodable_text(value: Any) -> str:
    """Drop code points that UTF-8 cannot encode, keeping everything else.

    PDF extraction can emit unpaired UTF-16 surrogates (e.g. '\\ud835' from
    mathematical alphanumeric symbols). Python strings hold them, but the
    embeddings request's UTF-8 JSON encoder raises UnicodeEncodeError.
    Unlike utils.sanitize_unicode_text(), which substitutes '?', this removes
    the offending characters so no placeholder noise enters embedded text.
    """
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def _use_progress_bar() -> bool:
    """True only when a live tqdm bar makes sense.

    A tqdm bar redraws itself with a carriage return, which an interactive
    terminal overwrites in place but a redirected SLURM log file cannot: every
    refresh lands as another line, so a long phase buries the log in hundreds
    of near-identical bars. Callers fall back to sparse periodic progress
    lines instead.

    isatty() alone turned out to be too weak a test in practice, so a run that
    carries a Slurm job id is treated as batch regardless of what the stream
    reports. Set FOT_PROGRESS_BAR=1 to force a bar anyway (e.g. an interactive
    srun --pty session), or FOT_PROGRESS_BAR=0 to suppress one everywhere.
    """
    override = os.environ.get("FOT_PROGRESS_BAR", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if tqdm is None:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _progress_iter(iterable, *, total: int, desc: str, unit: str):
    """Wrap an iterable in a tqdm bar when a live bar is appropriate."""
    if not _use_progress_bar():
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        mininterval=1.0,
        dynamic_ncols=True,
    )


class OpenRouterPaperRAG:
    """Persistent local vector index using OpenRouter's embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        index_path: Path,
        chunk_chars: int = 2400,
        overlap_chars: int = 320,
        embedding_batch_size: int = 64,
        metadata_fallbacks: Optional[Dict[str, str]] = None,
        metadata_path: Optional[Path] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenRouter RAG requires the openai package: pip install openai"
            ) from exc
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for RAG generation")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key
        )
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.chunk_chars = max(400, int(chunk_chars))
        self.overlap_chars = min(
            max(0, int(overlap_chars)), self.chunk_chars // 2
        )
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.metadata_fallbacks = metadata_fallbacks or {}
        self.metadata_path = metadata_path
        self.chunks: List[Dict[str, Any]] = []
        self.embedding_input_tokens = 0
        self.reused = False
        self.metadata_fallback_files: List[str] = []
        self.skipped_files: List[str] = []
        self.skipped_chunks: List[int] = []
        self.last_retrieval_stats: Dict[str, int] = {}

    def _metadata_fallback(self, path: Path) -> str:
        candidates = [path.stem]
        if "_" in path.stem:
            candidates.append(path.stem.rsplit("_", 1)[-1])
        for candidate in candidates:
            text = self.metadata_fallbacks.get(candidate)
            if text:
                return text
        return ""

    def _read_paper(self, path: Path) -> Tuple[str, bool]:
        if path.suffix.lower() == ".pdf":
            pdf_bytes = path.read_bytes()
            text = _extract_text_from_pdf_bytes(pdf_bytes).strip()
            if text:
                return text, False
            # task_paper_insight_reading.py supports the older PyPDF2 package;
            # keep the RAG corpus compatible with the same environments.
            try:
                import io
                import PyPDF2

                with _quiet_pdf_parser_diagnostics():
                    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    ).strip()
                if text:
                    return text, False
            except Exception:
                pass
            fallback = self._metadata_fallback(path)
            return fallback, bool(fallback)
        return path.read_text(encoding="utf-8", errors="replace").strip(), False

    def _chunk_papers(self, paper_files: Sequence[Path]) -> List[Dict[str, str]]:
        chunks: List[Dict[str, str]] = []
        self.metadata_fallback_files = []
        self.skipped_files = []
        total_files = len(paper_files)
        # PDF text extraction over a few thousand papers takes many minutes
        # with no network traffic to observe, so show a progress bar rather
        # than looking hung.
        extraction_start = time.time()
        print(
            f"  [1/2] Extracting text from {total_files} paper file(s) "
            "before embedding..."
        )
        progress_every = max(1, total_files // 20)
        file_iter = _progress_iter(
            paper_files,
            total=total_files,
            desc="  [1/2] extracting papers",
            unit="paper",
        )
        show_periodic_progress = not _use_progress_bar()
        for file_index, paper_path in enumerate(file_iter, 1):
            if show_periodic_progress and (
                file_index % progress_every == 0 or file_index == total_files
            ):
                elapsed = time.time() - extraction_start
                rate = file_index / elapsed if elapsed > 0 else 0.0
                remaining = (total_files - file_index) / rate if rate > 0 else 0.0
                print(
                    f"    extracted {file_index}/{total_files} papers "
                    f"({file_index / total_files * 100:.0f}%), "
                    f"{len(chunks)} chunks so far, "
                    f"{elapsed / 60:.1f}m elapsed, ~{remaining / 60:.1f}m left",
                    flush=True,
                )
            # A filename can itself hold unpaired surrogates (undecodable
            # bytes reach Python via surrogateescape). Those would raise
            # UnicodeEncodeError when printed, or when the finished index is
            # written as UTF-8 — losing an hours-long build at the last step.
            paper_label = _drop_unencodable_text(str(paper_path))
            try:
                text, used_metadata = self._read_paper(paper_path)
            except Exception as exc:
                self.skipped_files.append(paper_label)
                print(f"  Warning: could not read {paper_label}: {exc}")
                continue
            if not text:
                self.skipped_files.append(paper_label)
                print(f"  Warning: skipping file with no extractable text: {paper_label}")
                continue
            if used_metadata:
                self.metadata_fallback_files.append(paper_label)
                print(
                    "  Warning: PDF text extraction failed; indexing its local "
                    f"metadata title/abstract instead: {paper_label}"
                )
            start = 0
            part = 1
            while start < len(text):
                end = min(len(text), start + self.chunk_chars)
                # PDF extraction can preserve unpaired UTF-16 surrogates (e.g.
                # '\ud835' from mathematical alphanumeric symbols). Python
                # strings hold them, but the embeddings request's UTF-8 JSON
                # encoder raises UnicodeEncodeError, which previously killed
                # the whole index build after an hour of extraction. Sanitize
                # at chunk creation so the persisted index is clean too.
                content = _drop_unencodable_text(text[start:end]).strip()
                if content:
                    chunks.append(
                        {
                            "source": paper_label,
                            "part": str(part),
                            "text": content,
                        }
                    )
                if end >= len(text):
                    break
                start = end - self.overlap_chars
                part += 1
        if not chunks:
            raise RuntimeError("No readable paper text was available for RAG indexing")
        print(
            f"RAG corpus recovery: {len(self.metadata_fallback_files)} metadata "
            f"fallback(s), {len(self.skipped_files)} skipped file(s)."
        )
        return chunks

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        """Embed one already-sanitized batch, returning ordered vectors."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=list(batch),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError(
                "OpenRouter returned "
                f"{len(ordered)} embeddings for {len(batch)} inputs"
            )
        usage = getattr(response, "usage", None)
        self.embedding_input_tokens += int(
            getattr(usage, "prompt_tokens", 0) if usage else 0
        )
        return [list(item.embedding) for item in ordered]

    def _embed(
        self, texts: Sequence[str], progress: bool = False
    ) -> List[Optional[List[float]]]:
        """Embed texts in batches, skipping any that the API cannot accept.

        Returns one entry per input text, index-aligned, with ``None`` where
        embedding failed. Callers must drop those positions rather than
        zipping blindly, or chunks would receive other chunks' embeddings.

        A single unusable chunk (e.g. text the UTF-8 JSON encoder rejects)
        previously aborted an hour-long index build. Now the offending batch
        is retried item-by-item, only the genuinely bad items are skipped,
        and embedding continues.

        Progress is opt-in because retrieve() embeds a single query per
        generation batch and would otherwise flood the log.
        """
        vectors: List[Optional[List[float]]] = []
        offsets = list(range(0, len(texts), self.embedding_batch_size))
        total_batches = max(1, len(offsets))
        # Roughly 20 progress lines total, so a redirected log stays readable
        # no matter how many chunks the corpus produced.
        embed_progress_every = max(1, total_batches // 20)
        embed_start = time.time()
        offset_iter = (
            _progress_iter(
                offsets,
                total=total_batches,
                desc="  [2/2] embedding chunks",
                unit="batch",
            )
            if progress
            else offsets
        )
        for batch_index, offset in enumerate(offset_iter, 1):
            # Sanitize here as well as at chunk creation: this also covers
            # retrieval queries and any index written before the chunk-level
            # fix existed.
            batch = [
                _drop_unencodable_text(text)
                for text in texts[offset : offset + self.embedding_batch_size]
            ]
            try:
                vectors.extend(self._embed_batch(batch))
            except Exception as exc:
                # Fall back to one request per item so a single unusable text
                # cannot discard its whole batch.
                print(
                    f"  Warning: embedding batch {batch_index}/{total_batches} "
                    f"failed ({type(exc).__name__}: {exc}); retrying its "
                    f"{len(batch)} item(s) individually",
                    flush=True,
                )
                for item_index, text in enumerate(batch):
                    try:
                        vectors.extend(self._embed_batch([text]))
                    except Exception as item_exc:
                        self.skipped_chunks.append(offset + item_index)
                        vectors.append(None)
                        print(
                            f"    Skipping chunk {offset + item_index} — "
                            f"cannot embed ({type(item_exc).__name__}: "
                            f"{item_exc})",
                            flush=True,
                        )
            if (
                progress
                and not _use_progress_bar()
                and (
                    batch_index % embed_progress_every == 0
                    or batch_index == total_batches
                )
            ):
                elapsed = time.time() - embed_start
                rate = batch_index / elapsed if elapsed > 0 else 0.0
                remaining = (total_batches - batch_index) / rate if rate > 0 else 0.0
                print(
                    f"    embedded {len(vectors)}/{len(texts)} chunks "
                    f"({len(vectors) / max(1, len(texts)) * 100:.0f}%), "
                    f"batch {batch_index}/{total_batches}, "
                    f"{self.embedding_input_tokens} input tokens, "
                    f"{elapsed / 60:.1f}m elapsed, ~{remaining / 60:.1f}m left",
                    flush=True,
                )
        if len(vectors) != len(texts):
            # Index alignment is a correctness requirement, not a nicety: the
            # caller pairs these positionally with its chunk list.
            raise RuntimeError(
                f"Embedding result count {len(vectors)} does not match "
                f"{len(texts)} inputs; refusing to return misaligned vectors"
            )
        return vectors

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @staticmethod
    def _corpus_digest(paper_files: Sequence[Path]) -> str:
        digest = hashlib.sha256()
        for path in paper_files:
            stat = path.stat()
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        return digest.hexdigest()

    def _index_digest(self, paper_files: Sequence[Path]) -> str:
        digest = hashlib.sha256(self._corpus_digest(paper_files).encode("ascii"))
        if self.metadata_path and self.metadata_path.is_file():
            stat = self.metadata_path.stat()
            digest.update(str(self.metadata_path.resolve()).encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        return digest.hexdigest()

    def build_or_reuse(self, paper_files: Sequence[Path]) -> None:
        legacy_corpus_sha256 = self._corpus_digest(paper_files)
        corpus_sha256 = self._index_digest(paper_files)
        if self.index_path.is_file():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                current_index = (
                    payload.get("version") == 2
                    and payload.get("corpus_sha256") == corpus_sha256
                )
                legacy_complete_index = (
                    payload.get("version") == 1
                    and payload.get("corpus_sha256") == legacy_corpus_sha256
                )
                if (
                    (current_index or legacy_complete_index)
                    and payload.get("embedding_model") == self.embedding_model
                    and isinstance(payload.get("chunks"), list)
                    and payload["chunks"]
                ):
                    self.chunks = payload["chunks"]
                    self.metadata_fallback_files = list(
                        payload.get("metadata_fallback_files") or []
                    )
                    self.skipped_files = list(payload.get("skipped_files") or [])
                    self.reused = True
                    print(
                        f"Reusing OpenRouter RAG index with {len(self.chunks)} chunks: "
                        f"{self.index_path}"
                    )
                    if legacy_complete_index:
                        print(
                            "  Reused compatible version-1 index; it was written "
                            "only after every corpus file was readable."
                        )
                    return
            except (OSError, ValueError, TypeError):
                pass

        build_start = time.time()
        print(
            "Building RAG index (no reusable index found for this corpus + "
            "embedding model)"
        )
        raw_chunks = self._chunk_papers(paper_files)
        print(
            f"  [2/2] Embedding {len(raw_chunks)} full-paper chunks with OpenRouter "
            f"model {self.embedding_model} "
            f"(batch size {self.embedding_batch_size})..."
        )
        vectors = self._embed(
            [chunk["text"] for chunk in raw_chunks], progress=True
        )
        # Keep only chunks that embedded successfully. Pairing positionally
        # is safe because _embed guarantees one entry per input, with None
        # for skipped items.
        self.chunks = [
            {**chunk, "embedding": vector}
            for chunk, vector in zip(raw_chunks, vectors)
            if vector is not None
        ]
        if not self.chunks:
            raise RuntimeError(
                f"No chunk could be embedded ({len(raw_chunks)} attempted); "
                "the RAG index would be empty."
            )
        if self.skipped_chunks:
            print(
                f"  Skipped {len(self.skipped_chunks)} unembeddable chunk(s) "
                f"of {len(raw_chunks)}; indexing the remaining "
                f"{len(self.chunks)}."
            )
        payload = {
            "version": 2,
            "corpus_sha256": corpus_sha256,
            "embedding_model": self.embedding_model,
            "embedding_input_tokens": self.embedding_input_tokens,
            "metadata_fallback_files": self.metadata_fallback_files,
            "skipped_files": self.skipped_files,
            "skipped_chunks": self.skipped_chunks,
            "chunks": self.chunks,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name(f".{self.index_path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.index_path)
        print(
            f"RAG index built in {(time.time() - build_start) / 60:.1f}m: "
            f"{len(self.chunks)} chunks, {self.embedding_input_tokens} embedding "
            f"input tokens, saved to {self.index_path}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int,
        max_context_tokens: int = DEFAULT_RAG_MAX_TOKENS,
    ) -> str:
        """Return the highest-similarity chunks within a token budget.

        ``top_k`` bounds how many chunks are considered; ``max_context_tokens``
        bounds the assembled context itself (default 4096, matching
        trace_pooling.DEFAULT_RAG_MAX_TOKENS). Chunks are added in
        similarity order until the budget is reached, so a large chunk_chars
        setting cannot silently blow up the generation prompt.
        """
        if not self.chunks:
            raise RuntimeError("RAG index has not been built")
        query_vector = self._embed([query])[0]
        if query_vector is None:
            # Unlike a corpus chunk, a query cannot be skipped: without it
            # there is nothing to rank against.
            raise RuntimeError(
                "Could not embed the retrieval query; see the preceding "
                "embedding warning for the cause."
            )
        ranked = sorted(
            (
                (self._cosine(query_vector, chunk["embedding"]), chunk)
                for chunk in self.chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )[: max(1, int(top_k))]

        budget = max(0, int(max_context_tokens))
        selected: List[str] = []
        remaining = budget
        for score, item in ranked:
            block = (
                f"[Source: {item['source']} | part {item['part']} | "
                f"similarity={score:.4f}]\n{item['text']}"
            )
            block_tokens = estimate_tokens(block)
            if block_tokens <= remaining:
                selected.append(block)
                remaining -= block_tokens
            elif remaining > 0 and not selected:
                # Always return something for the top hit, even when a single
                # chunk exceeds the whole budget.
                selected.append(truncate_to_token_budget(block, remaining))
                remaining = 0
            if remaining <= 0:
                break
        self.last_retrieval_stats = {
            "considered_chunks": len(ranked),
            "used_chunks": len(selected),
            "token_budget": budget,
            "tokens_used": budget - remaining,
        }
        return "\n\n".join(selected)


def generate_skills_phase_from_rag(
    model: Any,
    api_key: str,
    papers_dir: str,
    num_skills: int,
    output_file: str,
    year: int = 2024,
    model_name: str = "google/gemini-2.5-flash",
    embedding_model: str = "openai/text-embedding-3-small",
    index_path: Optional[str] = None,
    top_k: int = 24,
    max_context_tokens: int = DEFAULT_RAG_MAX_TOKENS,
) -> List[Dict]:
    """Generate a 200-skill baseline using local RAG and OpenRouter only."""
    print("\n" + "=" * 80)
    print(f"PHASE 1: Generate {num_skills} Skills using OpenRouter RAG")
    print("=" * 80)

    papers_path = Path(papers_dir)
    if not papers_path.exists():
        raise ValueError(f"Papers directory does not exist: {papers_dir}")
    paper_files = sorted(
        path
        for path in papers_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if not paper_files:
        raise ValueError(f"No PDF, TXT, or Markdown papers found in {papers_dir}")
    print(f"Found {len(paper_files)} full-paper files in {papers_dir}")

    output_filename = output_file or f"rag_baseline_skills_{num_skills}_year{year}.json"
    resolved_index_path = Path(index_path) if index_path else Path(
        f"{output_filename}.openrouter_rag_index.json"
    )
    rag = OpenRouterPaperRAG(
        api_key=api_key,
        embedding_model=embedding_model,
        index_path=resolved_index_path,
        metadata_fallbacks=_load_rag_metadata_fallbacks(papers_path),
        metadata_path=papers_path / "metadata.json",
    )
    print("\n" + "-" * 80)
    print("STAGE 1/2: Build (or reuse) the embedding-based RAG index")
    print("-" * 80)
    rag.build_or_reuse(paper_files)

    print("\n" + "-" * 80)
    print(
        f"STAGE 2/2: Retrieve from {len(rag.chunks)} indexed chunks "
        f"(top_k={top_k}, context budget {max_context_tokens} tokens) and "
        f"generate the {num_skills}-skill library"
    )
    print("-" * 80)

    focus_areas = [
        "model architectures and representation learning",
        "optimization, training stability, and efficiency",
        "generative modeling and probabilistic methods",
        "reinforcement learning, planning, and agents",
        "robustness, uncertainty, safety, and alignment",
        "multimodal, vision, language, audio, and embodied learning",
        "graphs, geometry, causality, and structured prediction",
        "evaluation, data quality, generalization, and interpretability",
    ]
    skills: List[Dict] = []
    seen_names = set()
    total_output_tokens = 0
    batch_size = 25
    max_requests = math.ceil(num_skills / batch_size) + 4

    for request_index in range(1, max_requests + 1):
        remaining = num_skills - len(skills)
        if remaining <= 0:
            break
        requested_now = min(batch_size, remaining)
        focus = focus_areas[(request_index - 1) % len(focus_areas)]
        query = (
            f"ICLR {year} concrete research methodologies, reusable techniques, "
            f"and empirical lessons about {focus}"
        )
        print(
            f"OpenRouter RAG batch {request_index}: retrieving top-{top_k} chunks "
            f"for focus '{focus}'...",
            flush=True,
        )
        retrieve_start = time.time()
        try:
            retrieved_context = rag.retrieve(
                query, top_k=top_k, max_context_tokens=max_context_tokens
            )
        except Exception as exc:
            # The index cost hours to build and the remaining focus areas are
            # independent, so a failed query embedding skips this batch rather
            # than discarding the run. max_requests carries spare batches.
            print(
                f"  Warning: retrieval failed for focus '{focus}' "
                f"({type(exc).__name__}: {exc}); skipping this batch",
                flush=True,
            )
            continue
        stats = rag.last_retrieval_stats
        print(
            f"  retrieved {stats.get('used_chunks', 0)}/"
            f"{stats.get('considered_chunks', 0)} chunks, "
            f"~{stats.get('tokens_used', 0)}/{max_context_tokens} tokens "
            f"({len(retrieved_context)} chars) in "
            f"{time.time() - retrieve_start:.1f}s",
            flush=True,
        )
        existing_names = ", ".join(skill["name"] for skill in skills)
        avoid_section = (
            f"\nDo not repeat these previously generated names: {existing_names}"
            if existing_names
            else ""
        )
        prompt = f"""Use only the retrieved full-paper evidence below to generate exactly {requested_now} distinct, concrete research skills or methodological insights from ICLR {year}.

Current focus: {focus}

Retrieved evidence:
{retrieved_context}

For each skill, provide a concise name and an actionable technical description explaining what it is, when to use it, and how to apply it.{avoid_section}

Return exactly {requested_now} objects as one JSON array and no other text:
[
  {{"skill_name": "skill name", "description": "technical description and usage guidelines"}}
]"""
        print(
            f"  generating {requested_now} new skills "
            f"({len(skills)}/{num_skills} complete)...",
            flush=True,
        )
        try:
            response_text, token_info = call_api(
                model, prompt, max_output_tokens=8192
            )
        except Exception as exc:
            print(
                f"  Warning: generation call failed for focus '{focus}' "
                f"({type(exc).__name__}: {exc}); skipping this batch",
                flush=True,
            )
            continue
        total_output_tokens += int(token_info.get("output_tokens", 0) or 0)
        parsed = _parse_generated_skills(response_text)
        added = _append_unique_skills(skills, seen_names, parsed, remaining)
        print(f"  Accepted {added} distinct valid skills from this batch")

    _require_nonempty_skills(skills, num_skills, "OpenRouter RAG")
    skills_result = {
        "source": (
            f"OpenRouter RAG over {len(paper_files)} full ICLR {year} papers "
            f"({num_skills} skills)"
        ),
        "provider": "openrouter",
        "generation_model": model_name,
        "embedding_model": embedding_model,
        "rag_index": str(resolved_index_path),
        "rag_index_reused": rag.reused,
        "num_papers": len(paper_files),
        "num_chunks": len(rag.chunks),
        "metadata_fallback_files": rag.metadata_fallback_files,
        "skipped_files": rag.skipped_files,
        "num_skills": len(skills),
        "skills": skills,
        "generation_tokens": total_output_tokens,
        "embedding_input_tokens": rag.embedding_input_tokens,
    }
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as handle:
        json.dump(skills_result, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(skills)} skills to {output_filename}")
    return skills


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase baseline: Generate skill sets, then check if ICLR papers are guided by them"
    )

    # Phase control
    parser.add_argument(
        "--phase",
        type=str,
        choices=["generate", "check", "both"],
        default="both",
        help="Which phase to run: 'generate' (Phase 1), 'check' (Phase 2), or 'both' (default)",
    )

    # Phase 1: Generate skills
    parser.add_argument(
        "--generate-mode",
        type=str,
        choices=["iclr2023_keywords", "iclr2024_keywords", "general_knowledge", "rag"],
        default="iclr2023_keywords",
        help="Mode for Phase 1 skill generation (default: iclr2023_keywords)",
    )
    parser.add_argument(
        "--num-skills",
        type=int,
        default=50,
        help="Number of skills to generate in 'general_knowledge' or 'rag' mode (default: 50)",
    )
    parser.add_argument(
        "--rag-papers-dir",
        type=str,
        help="Directory containing PDF/TXT/MD papers for RAG mode",
    )
    parser.add_argument(
        "--rag-embedding-model",
        type=str,
        default="openai/text-embedding-3-small",
        help="OpenRouter embedding model for the local RAG index",
    )
    parser.add_argument(
        "--rag-index",
        type=str,
        default=None,
        help="Optional persistent OpenRouter RAG index path",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=24,
        help="Number of full-paper chunks retrieved per generation batch",
    )
    parser.add_argument(
        "--rag-max-context-tokens",
        type=int,
        default=DEFAULT_RAG_MAX_TOKENS,
        help=(
            "Token budget for the retrieved evidence in each generation "
            f"prompt (default: {DEFAULT_RAG_MAX_TOKENS}). Chunks are added in "
            "similarity order until the budget is reached."
        ),
    )
    parser.add_argument(
        "--skills-output",
        type=str,
        help="Output file for generated skills (Phase 1)",
    )

    # Phase 2: Check papers
    parser.add_argument(
        "--skills-file",
        type=str,
        help="Path to skills file from Phase 1 (required for Phase 2 if not running Phase 1)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="ICLR conference year for Phase 2 (default: 2024)",
    )
    parser.add_argument(
        "--accept-oral",
        action="store_true",
        help="Include Accept (Oral) papers in Phase 2",
    )
    parser.add_argument(
        "--accept-spotlight",
        action="store_true",
        help="Include Accept (Spotlight) papers in Phase 2",
    )
    parser.add_argument(
        "--accept-poster",
        action="store_true",
        help="Include Accept (Poster) papers in Phase 2",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Max papers to process in Phase 2 (default: all)",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default=None,
        help=(
            "Local accepted-paper corpus created by scraper.py for Phase 2. "
            "When set, metadata and PDFs are read locally and OpenReview is "
            "never contacted. For year 2025, defaults to ICLR25_PAPERS."
        ),
    )
    parser.add_argument(
        "--check-output",
        type=str,
        required=False,
        help="Output JSON file for Phase 2 results (default: baseline_check_results_{year}.json)",
    )

    # Common arguments
    parser.add_argument(
        "--api-provider",
        choices=["openrouter", "gemini"],
        default="openrouter",
        help="Generation/check provider (default: openrouter)",
    )
    parser.add_argument(
        "--api-key",
        "--gemini-key",
        dest="api_key",
        type=str,
        default=None,
        help="Provider API key; defaults to OPENROUTER_API_KEY or GEMINI_API_KEY",
    )
    parser.add_argument(
        "--api-model",
        "--gemini-model",
        dest="api_model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Generation/check model (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between model calls (default: 0.5)",
    )

    args = parser.parse_args()

    retired_openrouter_models = {
        "google/gemini-2.0-flash-001": "google/gemini-2.5-flash",
    }
    if args.api_provider == "openrouter" and args.api_model in retired_openrouter_models:
        replacement = retired_openrouter_models[args.api_model]
        print(
            f"OpenRouter model {args.api_model} is retired/unavailable; "
            f"using active replacement {replacement}."
        )
        args.api_model = replacement

    if args.num_skills < 1:
        parser.error("--num-skills must be at least 1")
    if args.rag_top_k < 1:
        parser.error("--rag-top-k must be at least 1")
    if args.rag_max_context_tokens < 1:
        parser.error("--rag-max-context-tokens must be at least 1")
    if args.generate_mode == "rag" and args.api_provider != "openrouter":
        parser.error("RAG generation is OpenRouter-only; use --api-provider openrouter")
    if not args.papers_dir and args.year == 2025:
        args.papers_dir = os.getenv("ICLR25_PAPERS")

    api_key = args.api_key or os.getenv(
        "OPENROUTER_API_KEY" if args.api_provider == "openrouter" else "GEMINI_API_KEY"
    )
    if not api_key:
        raise ValueError(
            f"{args.api_provider} API key is required. Provide --api-key or set "
            + (
                "OPENROUTER_API_KEY."
                if args.api_provider == "openrouter"
                else "GEMINI_API_KEY."
            )
        )

    if args.api_provider == "openrouter":
        model = OpenRouterClient(api_key=api_key, model_name=args.api_model)
    else:
        model = GeminiClient(api_key=api_key, model_name=args.api_model)

    # ========== PHASE 1: Generate Skills ==========
    skills = []
    skills_file = args.skills_file

    if args.phase in ["generate", "both"]:
        if args.generate_mode == "iclr2023_keywords":
            skills_output = args.skills_output or "iclr2023_top25_baseline_skills.json"
            skills = generate_skills_phase_from_iclr2023_keywords(model, skills_output)
            skills_file = skills_output
        elif args.generate_mode == "iclr2024_keywords":
            skills_output = (
                args.skills_output or "iclr2024_accepted_baseline_skills.json"
            )
            skills = generate_skills_phase_from_iclr2024_keywords(model, skills_output)
            skills_file = skills_output
        elif args.generate_mode == "rag":
            # RAG mode requires papers directory
            if not args.rag_papers_dir:
                raise ValueError(
                    "RAG mode requires --rag-papers-dir to specify the papers directory"
                )
            skills_output = (
                args.skills_output or f"rag_baseline_skills_{args.num_skills}_year{args.year}.json"
            )
            skills = generate_skills_phase_from_rag(
                model,
                api_key,
                args.rag_papers_dir,
                args.num_skills,
                skills_output,
                year=args.year,
                model_name=args.api_model,
                embedding_model=args.rag_embedding_model,
                index_path=args.rag_index,
                top_k=args.rag_top_k,
                max_context_tokens=args.rag_max_context_tokens,
            )
            skills_file = skills_output
        else:  # general_knowledge
            skills_output = (
                args.skills_output or f"general_baseline_skills_{args.num_skills}_year{args.year}.json"
            )
            skills = generate_skills_phase_from_general_knowledge(
                model, args.num_skills, skills_output, year=args.year
            )
            skills_file = skills_output

    # ========== PHASE 2: Check Papers ==========
    if args.phase in ["check", "both"]:
        if not skills and not skills_file:
            raise ValueError(
                "Phase 2 requires either Phase 1 to run or --skills-file to be specified"
            )

        # Load skills if not already generated
        if not skills and skills_file:
            print(f"\nLoading skills from {skills_file}...")
            try:
                with open(skills_file, "r") as f:
                    skills_data = json.load(f)
                    skills = skills_data.get("skills", [])
                    print(f"Loaded {len(skills)} skills")
            except FileNotFoundError:
                print(f"Error: Skills file not found: {skills_file}")
                return

        check_papers_phase(
            model,
            skills,
            args.year,
            args.max_papers,
            args.accept_oral,
            args.accept_spotlight,
            args.accept_poster,
            args.check_output,
            args.sleep,
            args.papers_dir,
        )


def check_papers_phase(
    model: Any,
    skills: List[Dict],
    year: int,
    max_papers: int = None,
    accept_oral: bool = True,
    accept_spotlight: bool = False,
    accept_poster: bool = False,
    output_file: str = None,
    sleep_duration: float = 0.5,
    papers_dir: Optional[str] = None,
):
    """Phase 2: Check if ICLR papers are guided by the skill set."""
    print("\n" + "=" * 80)
    print(f"PHASE 2: Check ICLR {year} Papers Against Skill Set")
    print("=" * 80)

    # Set defaults
    if not accept_oral and not accept_spotlight and not accept_poster:
        accept_oral = True

    if not output_file:
        output_file = f"baseline_check_results_{year}.json"

    if papers_dir:
        papers = _load_local_accepted_papers(
            papers_dir,
            year,
            max_papers,
            accept_oral,
            accept_spotlight,
            accept_poster,
        )
        if not papers:
            raise RuntimeError(
                f"No accepted papers found in local corpus {papers_dir}. "
                "Expected metadata.json plus downloaded PDFs."
            )
    else:
        papers = fetch_accept_tracks(
            year,
            max_papers=max_papers,
            accept_oral=accept_oral,
            accept_spotlight=accept_spotlight,
            accept_poster=accept_poster,
        )
    if not papers:
        print(f"No Accept papers found for ICLR {year}")
        return

    print(f"\nProcessing {len(papers)} papers (sorted: oral → spotlight → poster)...\n")

    results = []
    all_matched_skills = set()
    track_stats = {
        "oral": {"total": 0, "guided": 0},
        "spotlight": {"total": 0, "guided": 0},
        "poster": {"total": 0, "guided": 0},
    }

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    )

    # Format skills for evaluation
    skills_list = skills if isinstance(skills, list) else skills.get("skills", [])
    skills_text = "\n".join(
        [
            f"{i+1}. {s.get('name', f'Skill {i+1}')}: {s.get('description', '')}"
            for i, s in enumerate(skills_list)
        ]
    )

    for idx, paper in enumerate(papers, 1):
        track_label = paper.get("track", "")
        forum_id = paper.get("forum") or paper.get("id")

        print(f"[{idx}/{len(papers)}] Processing ({track_label}): Paper {forum_id}")

        # Fetch full paper content
        print("  Fetching paper content...")
        cache_dir = os.path.join(papers_dir, "text_cache") if papers_dir else "data/iclr25"
        paper_content = _fetch_paper_content(
            forum_id,
            session,
            cache_dir=cache_dir,
            pdf_url=paper.get("pdf_url"),
            local_pdf_path=paper.get("local_pdf_path"),
            local_only=bool(papers_dir),
        )
        paper["content"] = paper_content
        if paper_content:
            print(f"  Retrieved {len(paper_content)} characters")
        else:
            print("  No full content available, using title/abstract only")

        if not papers_dir:
            time.sleep(1)  # Rate limit network fetches only

        # Evaluate if paper is guided by skill set
        print("  Evaluating guidance with configured model...")
        guided = False
        matched_insights = []
        total_tokens = 0

        try:
            # Modify paper object to include the skills as insights
            paper_with_skills = paper.copy()
            verdict, token_info = score_paper(model, skills_text, paper_with_skills)
            total_tokens += token_info.get("output_tokens", 0)
            guided = bool(verdict.get("guided"))
            matched_insights = verdict.get("matched_insights") or []

            # Track matched skills
            for insight in matched_insights:
                all_matched_skills.add(insight)

            print(
                f"  Result: {'✓ GUIDED' if guided else '✗ Not guided'} | Matched: {len(matched_insights)} | Tokens: {total_tokens}"
            )
        except Exception as exc:
            print(f"  Model error during evaluation: {exc}")

        # Update statistics
        if track_label in track_stats:
            track_stats[track_label]["total"] += 1
            if guided:
                track_stats[track_label]["guided"] += 1

        results.append(
            {
                "id": paper.get("id"),
                "forum": paper.get("forum"),
                "title": paper.get("title", ""),
                "track": paper.get("track", ""),
                "venue": paper.get("venue", ""),
                "venueid": paper.get("venueid", ""),
                "guided": guided,
                "matched_skills": matched_insights,
                "output_tokens": total_tokens,
            }
        )

        time.sleep(max(sleep_duration, 0))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print overall and per-track statistics
    total = len(papers)
    total_guided = sum(track_stats[t]["guided"] for t in track_stats)

    print(f"\n{'='*80}")
    print(f"PHASE 2 RESULTS: ICLR {year} Paper Guidance Analysis")
    print(f"{'='*80}")

    print(f"\nSkill set size: {len(skills_list)}")
    print(f"Unique skills matched: {len(all_matched_skills)}")
    print(
        f"\nOverall: {total_guided}/{total} papers guided ({total_guided/total*100:.1f}%)\n"
    )

    for track in ["oral", "spotlight", "poster"]:
        stats = track_stats[track]
        if stats["total"] > 0:
            pct = stats["guided"] / stats["total"] * 100
            print(
                f"  {track.capitalize():10s}: {stats['guided']}/{stats['total']} guided ({pct:.1f}%)"
            )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
