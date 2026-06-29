# Redrob Candidate Ranker — Production Architecture

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role:** Senior AI Engineer — Founding Team (Redrob AI)  
**Version:** 1.1  
**Status:** Implementation blueprint (hybrid rules + ML)

This document describes the system implemented in this repository, focusing on the actual ranking pipeline and the performance-critical Stage 3 command.

---

## 1. Objectives

Build a **working ranker** that:

1. Reads the full candidate pool (`candidates.jsonl`, 100,000 records).
2. Applies rules-based integrity and availability filtering.
3. Computes a multi-stage ranking score and writes the top 100 candidate CSV.
4. Executes the Stage 3 ranking command in **≤ 5 minutes** on **CPU only**.
5. Avoids common dataset traps and bad signal candidates through explicit integrity checks.
6. Is reproducible via a documented `rank.py` command.

---

## 2. Built Implementation

### 2.1 Two-stage ranking architecture

1. **Stage 1 — fast filter + score:**
   - stream candidates from `candidates.jsonl`
   - apply integrity checks, availability filters, and lightweight fit scoring
   - keep the top 500 candidates by preliminary score
2. **Stage 2 — deep re-ranking:**
   - build rich profile text for the top candidates
   - compute on-the-fly semantic similarity using cached embeddings and `SentenceTransformer`
   - recompute fit and final scores
   - sort top candidates and assign monotonic ranks/CSV rows

### 2.2 Stage 3 runtime behavior

The implemented Stage 3 command is:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This command is designed to finish under 5 minutes on CPU-only hardware in the built system, with offline precomputation separated into `scripts/precompute.py`.

## 2.3 Built repository contents

- `rank.py`: main Stage 3 entrypoint
- `scripts/precompute.py`: offline semantic embedding preparation
- `src/pipeline/ranker.py`: ranking orchestration, scoring, and CSV output
- `src/integrity/`: candidate validity and honeypot detection
- `src/availability/`: availability and recruiter-match filters
- `src/fit/`: feature combination and scoring logic
- `src/semantic/`: similarity loader and embedding support
- `src/reasoning/`: reasoning justification generation
- `validate_submission.py`: local CSV format validator
- `submission_metadata.yaml`: portal metadata mirror
- `sample_candidates.jsonl`: sample pool for sandbox and tests

## 3. Data Flow (Ranking Step)

- Parse CLI args (`--candidates`, `--out`, optional `--config`)
- Load `config/jd_requirements.yaml`
- Load precomputed artifacts if present (`artifacts/` embeddings, index)
- Stream candidates from `candidates.jsonl`; avoid loading full 100K in memory
- For each candidate:
  - compute `INTEGRITY`
  - skip if below exclusion threshold
  - compute `FIT`
  - compute `AVAILABILITY`
- Keep top 500 candidates by preliminary score
- Recompute semantic fit for the top candidates
- Sort, tie-break by `candidate_id`, and write top 100 CSV
