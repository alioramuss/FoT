#!/usr/bin/env bash
# =============================================================================
# FoT demo experiments (research code, synced from Federation-of-Text).
#
#   1. Multi-domain collaboration  : HLE -> AIME26 / GPQA Diamond / LiveCodeBench v6
#   2. Real-world daily tasks      : PinchBench V1 -> PinchBench V2\V1 + ClawEval
#   3. Research insight discovery  : ICLR 2025 papers -> ICLR 2026 papers
#
# This file is a menu of commands, not a pipeline: copy the block you need.
# Run every command from this `experiment/` directory.
#
# Credentials are read from the environment; never paste keys into this file.
#   export OPENROUTER_API_KEY=...   # all model, judge, and embedding calls
#   export HF_TOKEN=...             # gated datasets (cais/hle)
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"

LOG_ROOT="${LOG_ROOT:-logs}"
MODEL="${MODEL:-google/gemini-2.5-flash-lite}"   # client + server model (paper default)
ITERS="${ITERS:-5}"                              # FoT rounds
WORKERS="${WORKERS:-16}"                         # parallel agents per round

# Optional: check the OpenRouter budget before a long sweep.
# curl -fsS https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" | jq .data

# =============================================================================
# 0. Setup
# =============================================================================
# python -m pip install -r requirements.txt
# bash setup_claweval_ds.sh          # ClawEval + local sandbox tools (expects ~/.venv)
# npm install -g openclaw            # only for --agent-backend openclaw

# =============================================================================
# 1. Multi-domain collaboration (paper Section 4.2, Table 3)
# =============================================================================
# Iteration 0 evaluates HLE plus the three held-out benchmarks with no library.
# Each later iteration trains on multimodal HLE (clients upload reasoning
# traces, the server rebuilds the library) and evaluates the held-out sets with
# only that round's library: <out>/encyclopedia_all_iter_NN.json.
# Requires HF_TOKEN with accepted access to cais/hle.
MD_OUT="$LOG_ROOT/hle_to_aime26_gpqa_lcbv6_gemini25flashlite_${ITERS}iter"
python task_benchmark_domain.py \
  --datasets hle \
  --eval-datasets aime26 gpqa_diamond livecodebench_v6 \
  --num-iterations "$ITERS" \
  --mode text \
  --use-api --api-provider openrouter \
  --model "$MODEL" --api-model "$MODEL" \
  --thinking \
  --num-workers "$WORKERS" \
  --output-dir "$MD_OUT"

# Isolated agents (no library) on the held-out benchmarks.
python task_benchmark_domain.py \
  --datasets aime26 gpqa_diamond livecodebench_v6 \
  --eval-only \
  --mode text \
  --use-api --api-provider openrouter \
  --model "$MODEL" --api-model "$MODEL" \
  --thinking \
  --num-workers "$WORKERS" \
  --output-dir "$LOG_ROOT/heldout_isolated_gemini25flashlite"

# Plug another local-reasoning strategy into FoT aggregation
# (metacognitive | hyperagents | trt | evolveprompt | ace).
# python task_benchmark_domain.py --datasets hle \
#   --eval-datasets aime26 gpqa_diamond livecodebench_v6 --num-iterations "$ITERS" \
#   --mode text --use-api --api-provider openrouter --model "$MODEL" --api-model "$MODEL" \
#   --client metacognitive --num-workers "$WORKERS" \
#   --output-dir "$LOG_ROOT/hle_to_heldout_metacognitive"

# =============================================================================
# 2. Real-world daily tasks with OpenClaw (paper Section 4.1, Table 2)
# =============================================================================
# Clients run the 23 PinchBench V1 tasks; the shared library is then reused on
# held-out PinchBench tasks (V2 \ V1) and on ClawEval general tasks.
# --agent-backend direct uses the repository's OpenAI-compatible tool loop;
# use --agent-backend openclaw to drive the OpenClaw CLI instead.
JUDGE="${JUDGE:-deepseek/deepseek-v4-flash}"
PB_COMMON=(--agent-backend direct --use-api --api-provider openrouter
           --model "$MODEL" --api-model "$MODEL"
           --judge "$JUDGE" --judge-provider openrouter
           --num-workers "$WORKERS" --timeout-multiplier 1)
CE_COMMON=(--use-api --api-provider openrouter
           --model "$MODEL" --api-model "$MODEL"
           --judge "$JUDGE" --judge-provider openrouter
           --tag general --trials 1 --parallel "$WORKERS" --timeout 300)
FOT_RUN="$LOG_ROOT/pinchbench_v1_fot_gemini25flashlite_${ITERS}iter"

# 2a. FoT: ITERS rounds of solve -> reflect -> trace -> server aggregation on V1.
#     Round N writes $FOT_RUN/iter_NN/encyclopedia.json (the insight library).
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" \
  --V1-only --iterations "$ITERS" \
  --output-dir "$FOT_RUN"

# 2b. Held-out PinchBench (V2 \ V1) and ClawEval with the final FoT library.
FOT_LIBRARY="$FOT_RUN/iter_$(printf '%02d' "$ITERS")/encyclopedia.json"
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" \
  --exclude-V1 --eval-only --iterations 1 \
  --encyclopedia "$FOT_LIBRARY" \
  --output-dir "$LOG_ROOT/pinchbench_nonv1_fot_eval_gemini25flashlite"
python task_openclaw_claweval.py "${CE_COMMON[@]}" \
  --eval-only --iterations 1 \
  --encyclopedia "$FOT_LIBRARY" \
  --output-dir "$LOG_ROOT/claweval_fot_eval_gemini25flashlite"

# 2c. Isolated single-agent baselines (no library, graded once).
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --eval-only --iterations 1 \
  --output-dir "$LOG_ROOT/baseline_pinchbench_v1_gemini25flashlite"
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --exclude-V1 --eval-only --iterations 1 \
  --output-dir "$LOG_ROOT/baseline_pinchbench_nonv1_gemini25flashlite"
python task_openclaw_claweval.py "${CE_COMMON[@]}" --eval-only --iterations 1 \
  --output-dir "$LOG_ROOT/baseline_claweval_gemini25flashlite"
#     + Curated OpenClaw skills: add --openclaw-skill to the three commands above.

# 2d. Aggregation ablations (Table 2, "FoT with Various Global Aggregation Strategies").
#     Isolated: no server; each client only sees its own previous-round traces.
ISO_RUN="$LOG_ROOT/pinchbench_v1_isolated_gemini25flashlite_${ITERS}iter"
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --isolated \
  --iterations "$ITERS" --output-dir "$ISO_RUN"
#     FoT aggregation on each agent: one server per client, no cross-agent sharing.
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --individual \
  --iterations "$ITERS" --output-dir "$LOG_ROOT/pinchbench_v1_individual_gemini25flashlite_${ITERS}iter"
#     Pooling / RAG over the isolated run's traces, round-matched (RAG budget: 4,096 tokens).
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --exclude-V1 --iterations "$ITERS" \
  --pooling --pooling-dir "$ISO_RUN" --pooling-context-window 1048576 \
  --output-dir "$LOG_ROOT/pinchbench_nonv1_pooling_gemini25flashlite_${ITERS}iter"
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --exclude-V1 --iterations "$ITERS" \
  --pooling --pooling-dir "$ISO_RUN" --rag --rag-embedding-model openai/text-embedding-3-small \
  --output-dir "$LOG_ROOT/pinchbench_nonv1_rag_gemini25flashlite_${ITERS}iter"
python task_openclaw_claweval.py "${CE_COMMON[@]}" --iterations "$ITERS" \
  --pooling --pooling-dir "$ISO_RUN" \
  --output-dir "$LOG_ROOT/claweval_pooling_gemini25flashlite_${ITERS}iter"
python task_openclaw_claweval.py "${CE_COMMON[@]}" --iterations "$ITERS" \
  --pooling --pooling-dir "$ISO_RUN" --rag --rag-embedding-model openai/text-embedding-3-small \
  --output-dir "$LOG_ROOT/claweval_rag_gemini25flashlite_${ITERS}iter"
#     Context compaction / Chain-of-Density servers replace server_text.py only.
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --iterations "$ITERS" --compact \
  --output-dir "$LOG_ROOT/pinchbench_v1_compact_gemini25flashlite_${ITERS}iter"
python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --iterations "$ITERS" --cod \
  --output-dir "$LOG_ROOT/pinchbench_v1_cod_gemini25flashlite_${ITERS}iter"

# 2e. Ablations from Section 6 (on the FoT run above).
#     Participation rate: aggregate only the first N clients of each round.
# python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --exclude-V1 --iterations "$ITERS" \
#   --participate 12 --pooling-dir "$LOG_ROOT/pinchbench_v1_individual_gemini25flashlite_${ITERS}iter" \
#   --output-dir "$LOG_ROOT/pinchbench_nonv1_participate12"
#     Local reasoning rounds before each upload (1 = no warm-up).
# python task_openclaw_pinchbench.py "${PB_COMMON[@]}" --V1-only --iterations 5 \
#   --local-reflect-round 3 --output-dir "$LOG_ROOT/pinchbench_v1_local3_global5"

# =============================================================================
# 3. Research insight discovery (paper Section 4.3, Table 4)
# =============================================================================
# Agents read ICLR 2025 papers (25 papers per agent read) and FoT curates a
# 200-insight library; a judge then decides, paper by paper, whether each
# ICLR 2026 paper's core method is guided by the library (Prompt 6).
PAPERS_ROOT="${PAPERS_ROOT:-data/papers}"
ICLR25_PAPERS="$PAPERS_ROOT/iclr2025_all_accepted"
ICLR26_PAPERS="$PAPERS_ROOT/iclr2026_all_accepted"
ICLR_OUT="$LOG_ROOT/iclr2025_to_2026_200"
GENERATION_MODEL="${GENERATION_MODEL:-google/gemini-2.5-flash}"
# Table 4 cross-validates with Gemini 3.1 Pro, GPT-5.6 Sol, and Claude Opus 5;
# set JUDGE_MODEL to the OpenRouter slug of the judge you want.
JUDGE_MODEL="${JUDGE_MODEL:-deepseek/deepseek-v3.2-exp}"

# 3a. Download accepted papers (writes PDFs + metadata.json).
python scraper.py --year 2025 --accept-oral --accept-spotlight --accept-poster --output-dir "$ICLR25_PAPERS"
python scraper.py --year 2026 --accept-oral --accept-spotlight --accept-poster --output-dir "$ICLR26_PAPERS"

# 3b. FoT library: paper reading (Prompt 11) -> reflection -> traces -> aggregation.
python task_paper_insight_reading.py \
  --papers-dir "$ICLR25_PAPERS" --file-pattern '*.pdf' \
  --agent-read-num 25 --num-insights 200 --num-workers "$WORKERS" \
  --use-api --api-provider openrouter --model "$GENERATION_MODEL" \
  --output-dir "$ICLR_OUT/fot"

# 3c. Baseline libraries with the same budget: direct generation and RAG.
python checker_iclr_baseline.py --phase generate --generate-mode general_knowledge \
  --num-skills 200 --year 2025 \
  --api-provider openrouter --api-model "$GENERATION_MODEL" \
  --skills-output "$ICLR_OUT/general/general_knowledge_200.json"
python checker_iclr_baseline.py --phase generate --generate-mode rag \
  --rag-papers-dir "$ICLR25_PAPERS" --rag-embedding-model openai/text-embedding-3-small \
  --num-skills 200 --year 2025 \
  --api-provider openrouter --api-model "$GENERATION_MODEL" \
  --skills-output "$ICLR_OUT/rag/rag_iclr2025_all_200.json"

# 3d. Judge every ICLR 2026 paper against each library (resumable judgment journal).
for setting in fot general rag; do
  case "$setting" in
    fot)     library="$ICLR_OUT/fot/paper_encyclopedia.json" ;;
    general) library="$ICLR_OUT/general/general_knowledge_200.json" ;;
    rag)     library="$ICLR_OUT/rag/rag_iclr2025_all_200.json" ;;
  esac
  judge_tag="$(echo "$JUDGE_MODEL" | tr '/.' '__')"
  python checker_iclr.py \
    --api-type openrouter --api-model "$JUDGE_MODEL" \
    --num-workers "$WORKERS" --sleep 0 \
    --encyclopedia "$library" \
    --papers-dir "$ICLR26_PAPERS" --year 2026 \
    --judgments-jsonl "$ICLR_OUT/$setting/check_${judge_tag}_judgments.jsonl" \
    --output "$ICLR_OUT/$setting/check_${judge_tag}.json"
done

# 3e. Judge-judge / human-judge agreement plots (Cohen's kappa, macro-F1).
# python plot_pairwise_agreement.py --output-dir figures/agreement --no-show
