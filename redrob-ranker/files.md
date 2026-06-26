# Project Files — Purpose & Rationale

This document explains **why each file exists** in `redrob-ranker/`.  
Build and validation follow **phase order** (1 → 5 now; 6+ for ML later).

---

## Root

| File | Purpose |
|------|---------|
| `architecture.md` | (parent folder) System design, constraints, ML roadmap |
| `files.md` | This file — map of every module and how to validate each phase |
| `requirements.txt` | Pinned Python dependencies for reproducible Stage 3 runs |
| `rank.py` | **Submission entry point** — `reproduce_command` CLI: reads candidates, writes CSV |

---

## `config/`

| File | Purpose |
|------|---------|
| `jd_requirements.yaml` | **Phase 1** — Encodes JD intent: must-have skills, disqualifier titles, consulting firms, weights, thresholds. Single source of truth; no magic numbers in code. |

**Validate Phase 1:**
```bash
pytest tests/phase1/test_jd_config.py -v
```

---

## `src/config_loader/`

| File | Purpose |
|------|---------|
| `jd_config.py` | Loads YAML into `JDConfig` dataclass; expands skill synonyms for matching |
| `__init__.py` | Package exports |

**Why separate:** Config parsing isolated from scoring — easy to tune weights without touching logic.

---

## `src/io/`

| File | Purpose |
|------|---------|
| `candidates.py` | Reads `candidates.jsonl` (streaming) and `sample_candidates.json` (array) |
| `__init__.py` | Package exports |

**Why separate:** I/O format changes (`.jsonl` vs `.gz`) stay in one place.

---

## `src/text/`

| File | Purpose |
|------|---------|
| `profile_text.py` | Builds canonical lowercase text from headline, career, skills — used by FIT and (later) embeddings |
| `__init__.py` | Package exports |

**Why separate:** Shared text normalization for rules now and semantic layer in Phase 6.

---

## `src/integrity/`

| File | Purpose |
|------|---------|
| `checker.py` | **Phase 2** — INTEGRITY score (0–1): honeypots, skill fraud, timeline issues, keyword stuffer patterns. **Always rule-based — never ML.** |
| `__init__.py` | Package exports |

**Why separate:** Trap/honeypot defense is critical for Stage 3 (>10% honeypots in top 100 = DQ). Isolated for audit and testing.

**Validate Phase 2:**
```bash
pytest tests/phase2/test_integrity.py -v
```

---

## `src/features/`

| File | Purpose |
|------|---------|
| `extractor.py` | **Phase 3** — Structured features: title/career, trusted skills, experience band, location, lexical semantic placeholder |
| `__init__.py` | Package exports |

**Why separate:** Feature engineering feeds both rule-based FIT and future LightGBM LTR (Phase 6b).

**Validate Phase 3:**
```bash
pytest tests/phase3/test_fit.py -v
```

---

## `src/fit/`

| File | Purpose |
|------|---------|
| `scorer.py` | **Phase 3** — Weighted FIT score from features per `jd_requirements.yaml` weights |
| `__init__.py` | Package exports |

**Why separate:** FIT logic can switch between `rules`, `rules+embeddings`, and `ltr` modes without changing integrity or availability.

---

## `src/availability/`

| File | Purpose |
|------|---------|
| `multiplier.py` | **Phase 4** — AVAILABILITY multiplier (0.75–1.10) from `redrob_signals` |
| `__init__.py` | Package exports |

**Why separate:** Behavioral signals are a distinct JD requirement (“perfect on paper but inactive”).

**Validate Phase 4:**
```bash
pytest tests/phase4/test_availability.py -v
```

---

## `src/scoring/`

| File | Purpose |
|------|---------|
| `combiner.py` | **Phase 5** — `FINAL = FIT × AVAILABILITY × INTEGRITY`; returns `CandidateScore` dataclass |
| `__init__.py` | Package exports |

**Why separate:** Single place for the scoring formula used by pipeline and tests.

---

## `src/reasoning/`

| File | Purpose |
|------|---------|
| `generator.py` | **Phase 7** — Fact-based 1–2 sentence reasoning (no LLM); Stage 4 safe |
| `__init__.py` | Package exports |

**Why separate:** Reasoning templates evolve independently from ranking math.

---

## `src/pipeline/`

| File | Purpose |
|------|---------|
| `ranker.py` | **Phase 5** — Orchestrates load → score → top-100 heap → monotonic scores → CSV |
| `__init__.py` | Package exports |

**Why separate:** End-to-end workflow; `rank.py` stays thin.

**Validate Phase 5:**
```bash
pytest tests/phase5/test_pipeline.py -v
```

---

## `scripts/`

| File | Purpose |
|------|---------|
| `eval_sample.py` | **Sample evaluation** — prints top-K from `sample_candidates.json` for manual review. Does **not** produce spec CSV (needs 100 rows). |
| `precompute.py` | *(Phase 6 — not yet)* Offline embeddings |
| `train_ltr.py` | *(Phase 6b — not yet)* LightGBM training |

**Validate on sample (manual):**
```bash
python scripts/eval_sample.py --top-k 15
```

---

## `tests/`

| Path | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures: `config`, `sample_candidates`, synthetic `honeypot_candidate` |
| `phase1/test_jd_config.py` | Config loads; disqualifier titles present |
| `phase2/test_integrity.py` | Honeypot fixture penalized; clean samples pass |
| `phase3/test_fit.py` | Negative titles score low; FIT in [0,1] |
| `phase4/test_availability.py` | Multiplier bounds; stale profile penalized |
| `phase5/test_pipeline.py` | Tie-break, monotonic scores, reasoning not empty |

**Run all tests:**
```bash
cd redrob-ranker
pip install -r requirements.txt
pytest tests/ -v
```

---

## Full pool validation (after sample passes)

```bash
cd redrob-ranker
python rank.py --candidates ../candidates.jsonl --out ../team_xxx.csv
python ../validate_submission.py ../team_xxx.csv
```

Requires **≥100 candidates** in pool (100K in production file).

---

## Planned additions (Phase 6+)

| Path | Purpose |
|------|---------|
| `artifacts/` | Precomputed embeddings, LTR model (gitignored or committed) |
| `src/semantic/embeddings.py` | Load `.npy` embeddings; cosine vs JD |
| `src/fit/ml_fit.py` | LightGBM inference path |
| `sandbox/app.py` | Streamlit demo for portal sandbox link |
| `submission_metadata.yaml` | Portal metadata mirror |
| `Dockerfile` | Stage 3 reproduction container |

---

## Scoring flow (quick reference)

```
candidate.json
    → integrity/checker.py     → exclude if < 0.30
    → features/extractor.py    → title, skills, experience, semantic, location
    → fit/scorer.py            → weighted FIT
    → availability/multiplier  → AVAILABILITY
    → scoring/combiner.py      → FINAL = FIT × AVAIL × INTEGRITY
    → pipeline/ranker.py       → top 100, monotonic scores
    → reasoning/generator.py   → reasoning column
    → submission.csv
```

---

*Updated: Phases 1–5 implemented (rules baseline). Phase 6 ML layer pending.*
