# Redrob Candidate Ranker — Production Architecture

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role:** Senior AI Engineer — Founding Team (Redrob AI)  
**Version:** 1.1  
**Status:** Implementation blueprint (hybrid rules + ML)

This document defines the architecture we will implement. It is derived from `README.docx`, `submission_spec.docx`, `submission_metadata_template.yaml`, `candidate_schema.json`, `redrob_signals_doc.docx`, and `validate_submission.py`.

---

## 1. Objectives

Build a **production-grade proof-of-concept ranker** that:

1. Reads the full candidate pool (`candidates.jsonl`, 100,000 records).
2. Ranks the **top 100** candidates for the released job description.
3. Outputs a spec-compliant CSV: `candidate_id`, `rank`, `score`, `reasoning`.
4. Runs the **ranking step** in **≤ 5 minutes** on **CPU only**, **16 GB RAM**, **no network**.
5. Avoids dataset traps: keyword stuffers, plain-language gems mis-ranked, behavioral ghosts, and honeypots.
6. Is **reproducible** via a single documented command for Stage 3 review.
7. Ships with GitHub repo, README, sandbox demo, and `submission_metadata.yaml`.

---

## 2. Spec Constraints (Non-Negotiable)

### 2.1 Ranking step (enforced at Stage 3)

| Constraint | Limit |
|------------|-------|
| Wall-clock runtime | ≤ 5 minutes |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU during ranking |
| Network | Off — no external API calls (OpenAI, Anthropic, Cohere, Gemini, etc.) |
| Disk (intermediate) | ≤ 5 GB |

**What must finish in ≤ 5 min:** the `reproduce_command` — typically:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**What may exceed 5 min:** offline pre-computation (embeddings, indexes). Must be documented in README and `submission_metadata.yaml` (`pre_computation_required: true`).

### 2.2 Output format

| Rule | Requirement |
|------|-------------|
| Filename | Registered participant ID, e.g. `team_xxx.csv` |
| Encoding | UTF-8 |
| Header (row 1) | `candidate_id,rank,score,reasoning` (exact order) |
| Data rows | Exactly 100 (rows 2–101) |
| `candidate_id` | `CAND_XXXXXXX` — must exist in `candidates.jsonl` |
| `rank` | Integers 1–100, each used exactly once |
| `score` | Float, **non-increasing** by rank |
| Ties | Allowed; break ties by `candidate_id` ascending |
| `reasoning` | Optional but strongly recommended (1–2 sentences) |

Validate locally before every upload:

```bash
python validate_submission.py team_xxx.csv
```

### 2.3 Submission package (portal + repo)

| Part | Contents |
|------|----------|
| CSV | Top-100 ranking |
| Portal metadata | Team, contacts, GitHub, sandbox link, AI tools, compute summary |
| GitHub repo | Code, README, deps, `submission_metadata.yaml`, optional artifacts |
| Sandbox | Hosted demo on ≤100 candidates, ≤5 min CPU (HuggingFace Spaces, Streamlit, Docker, etc.) |

### 2.4 Disqualification triggers

| Trigger | Stage | Threshold |
|---------|-------|-----------|
| Format violation | 1 | Any spec breach |
| Cannot reproduce ranking step | 3 | >5 min / >16 GB / GPU / network |
| Honeypot rate in top 100 | 3 | > 10% |
| Bad reasoning / fake repo / no defense | 4–5 | Manual + interview |

### 2.5 Scoring metrics (Stage 2 — hidden ground truth)

```
composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**Design implication:** optimize heavily for **top-10 precision** (50% of score).

---

## 3. Architectural Principles

1. **Hybrid, not ML-only:** rules for traps and disqualifiers; ML for ranking quality among clean candidates.
2. **Two-phase execution:** heavy work offline; ranking step is load + score + write.
3. **Three scoring parameters:** FIT × AVAILABILITY × INTEGRITY — simple, tunable, explainable.
4. **Title + career over skills:** JD explicitly warns keyword matching is a trap.
5. **Behavioral as multiplier:** perfect profile + dead signals = not hireable.
6. **Deterministic output:** same input → same CSV (reproducibility for Stage 3).
7. **No LLM at inference:** semantic fit via precomputed embeddings + optional LTR; no hosted APIs.
8. **Test small, validate big:** develop on `sample_candidates.json`, gate on full `candidates.jsonl`.

---

## 3.5 Rules vs ML — What We Need

### Short answer

| Approach | Good enough to submit? | Competitive for NDCG@10? | Recommendation |
|----------|------------------------|--------------------------|----------------|
| **Rules only** (no ML) | Yes — passes format, honeypot, Stage 3 | Moderate — misses semantic gems, hand-tuned weights | Baseline v1 |
| **Rules + embeddings** | Yes | Good — catches plain-language fits | **Minimum target** |
| **Rules + embeddings + LTR** | Yes | Best — learns feature mix, aligns with ranking metrics | **Final target** |
| **ML only** (no integrity rules) | Risky — honeypots in top 100 | Poor — fails Stage 3 filter | **Do not use** |

**We do not need ML to pass the spec.** We **do** want ML to win on ranking quality (50% of score is NDCG@10).

### What must stay rule-based (never replace with ML)

| Layer | Why rules, not ML |
|-------|-------------------|
| **INTEGRITY** | Honeypots have deterministic impossible facts; ~80 traps → >10% in top 100 = DQ |
| **JD disqualifiers** | Consulting-only, wrong title, research-only — explicit JD logic |
| **AVAILABILITY** | Interpretable multiplier from `redrob_signals`; easy to defend in interview |
| **Reasoning** | Stage 4 checks facts in profile — template from rules, not LLM |

### What ML improves

| Layer | ML component | Gain |
|-------|--------------|------|
| **FIT — semantic** | Precomputed `bge-small-en-v1.5` embeddings | Finds candidates who describe real ranking/retrieval work without buzzwords |
| **FIT — ranking** | LightGBM LambdaMART (optional) | Better top-10 ordering than hand-picked weights; NDCG-aligned objective |
| **FIT — hybrid** | BM25 + embedding cosine | Robust lexical + semantic match |

### Final system choice

```
BEST = Rules (INTEGRITY + disqualifiers + AVAILABILITY)
     + Feature engineering (title, career, trusted skills, location)
     + ML (precomputed embeddings + optional LightGBM LTR for FIT)
```

**Build order:** rules baseline first → add embeddings → add LTR if baseline top-10 looks weak on sample review.

---

## 4. System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE A — OFFLINE (may exceed 5 min; documented in submission_metadata)     │
│                                                                              │
│  job_description.docx  ──►  config/jd_requirements.yaml                      │
│  candidates.jsonl      ──►  scripts/precompute.py                            │
│  sample labels (manual) ──►  scripts/train_ltr.py   [optional Phase 6b]     │
│                                    │                                         │
│                                    ▼                                         │
│                         artifacts/                                           │
│                           ├── jd_embedding.npy                               │
│                           ├── candidate_embeddings.npy                     │
│                           ├── candidate_id_index.json                        │
│                           ├── feature_scaler.pkl         [if LTR]            │
│                           └── ltr_model.lgb              [if LTR]            │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE B — RANKING STEP (must be ≤ 5 min, CPU, no network)                   │
│                                                                              │
│  rank.py --candidates candidates.jsonl --out submission.csv                  │
│                                                                              │
│  ┌────────┐  ┌──────────────┐  ┌─────────────────────┐  ┌──────────────┐   │
│  │ Load   │─►│ RULES        │─►│ ML-ENHANCED FIT     │─►│ Top 100      │   │
│  │ config │  │ INTEGRITY    │  │ features + embed +  │  │ × AVAIL      │   │
│  │ + arti │  │ disqualifiers│  │ LTR (or rule weights)│  │ + reasoning  │   │
│  └────────┘  └──────────────┘  └─────────────────────┘  └──────────────┘   │
│                     │                      │                                 │
│                     │ exclude traps        │ FINAL = FIT × AVAIL × INT       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Scoring Model — Three Parameters

Final score per candidate:

```
FINAL = FIT × AVAILABILITY × INTEGRITY
```

Sort by `FINAL` descending. Tie-break: `candidate_id` ascending. Take top 100. Assign ranks 1–100 with monotonically non-increasing scores.

### 5.1 FIT (0.0 – 1.0) — Role relevance

*"Should we hire this person for this JD?"*

| Sub-component | Weight | Signals |
|---------------|--------|---------|
| Title + career trajectory | 35% | `current_title`, `career_history` titles/descriptions, product vs consulting |
| Trusted skills | 25% | Must-have skill match × credibility (proficiency, endorsements, `duration_months`) |
| Experience band | 15% | Soft target 5–9 years (`profile.years_of_experience`) |
| Semantic match | 15% | Precomputed embedding cosine vs JD (+ optional BM25) |
| Location + logistics | 10% | India, Pune/Noida/Delhi NCR, `willing_to_relocate`, `notice_period_days`, work mode |

**FIT computation modes** (switch via `config/jd_requirements.yaml` → `fit_mode`):

| Mode | When | How FIT is computed |
|------|------|---------------------|
| `rules` | Phase 3–5 baseline | Weighted sum of sub-components (hand-tuned weights) |
| `rules+embeddings` | Phase 6 default | Rules + embedding cosine in semantic slot |
| `ltr` | Phase 6b final | LightGBM predicts FIT from feature vector (embeddings as features) |

**JD hard disqualifiers (cap FIT or apply penalty):**

- Consulting-only career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) with no product-company experience.
- Pure research background with no production deployment evidence.
- LangChain-only recent AI (<12 months) without pre-LLM ML production depth.
- CV / speech / robotics primary expertise without NLP/IR in career narrative.
- Title mismatch: e.g. Marketing Manager, HR Manager, Content Writer with AI skill stuffing.

**JD must-haves (positive signals):**

- Production embeddings / retrieval systems.
- Vector DB or hybrid search operations experience.
- Strong Python, production code quality signals in career text.
- Ranking evaluation experience (NDCG, MAP, A/B testing).
- Nice-to-have: LoRA/PEFT, learning-to-rank, HR-tech, distributed systems.

**Skill trust formula (per skill):**

```
trust = proficiency_weight × log1p(endorsements) × min(1, duration_months / 12)

penalties:
  - expert + duration_months == 0  → trust = 0
  - expert + endorsements == 0     → heavy discount
  - skill count > 12 with avg trust < threshold → stuffer flag
```

### 5.2 AVAILABILITY (0.75 – 1.10) — Hireability now

*"Can we actually reach and hire them?"*

Multiplier on FIT from `redrob_signals`:

| Signal | Effect |
|--------|--------|
| `last_active_date` stale (>180 days) | Strong down-weight |
| `recruiter_response_rate` < 0.20 | Down-weight |
| `open_to_work_flag` == false | Down-weight |
| `notice_period_days` > 30 | Mild penalty (JD: bar gets higher) |
| `notice_period_days` ≤ 30 | Mild boost |
| `saved_by_recruiters_30d` > 0 | Boost |
| `profile_views_received_30d` | Mild boost |
| `skill_assessment_scores` (JD-relevant) | Boost if high |
| `interview_completion_rate` low | Down-weight |
| `verified_email` + `verified_phone` | Mild boost |

Clamp result to `[0.75, 1.10]` so availability modulates but does not dominate FIT.

### 5.3 INTEGRITY (0.0 – 1.0) — Profile believability

*"Is this profile real or a trap/honeypot?"*

| Check | Detection logic |
|-------|-----------------|
| Timeline impossibility | Tenure at company exceeds plausible window; total experience inconsistent with career dates |
| Skill fraud | `expert` proficiency + `duration_months == 0` |
| Skill inflation | ≥8 expert skills with low endorsements and low duration |
| Education fraud | `end_year < start_year`; degree timeline vs experience |
| Title–narrative contradiction | e.g. "10 years NLP" in summary but `years_of_experience < 6` |
| Keyword stuffer pattern | High AI skill density + low title relevance + weak career ML evidence |

**Application:**

- `INTEGRITY < 0.30` → exclude from ranking pool (honeypot / trap).
- `0.30 ≤ INTEGRITY < 0.60` → heavy score penalty.
- `INTEGRITY ≥ 0.60` → pass through normally.

**Target:** honeypot rate in top 100 **< 3%** (spec disqualifies at >10%).

### 5.4 ML Enhancement Layer (inside FIT — not a replacement)

ML only affects **FIT** among candidates that pass INTEGRITY. It does not replace INTEGRITY or AVAILABILITY.

#### 5.4.1 Embedding model (required for competitive quality)

| Item | Choice |
|------|--------|
| Model | `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB) |
| Library | `sentence-transformers` (offline only in `precompute.py`) |
| Input text | JD canonical text; per-candidate canonical profile text (see §9.2) |
| Ranking-time cost | Load `.npy` (~150 MB) + matrix dot product: **< 60 s** for 100K |

```python
semantic_score = cosine(jd_embedding, candidate_embedding)  # mapped to [0, 1]
```

#### 5.4.2 Learning-to-rank — LightGBM (optional, recommended for final submit)

| Item | Choice |
|------|--------|
| Model | `lightgbm.LGBMRanker` with `objective='lambdarank'` |
| Training data | 200–500 manually labeled candidates from `sample_candidates.json` + random sample |
| Labels | Tier 0–5 proxy (0 = honeypot/disqualified, 5 = ideal Senior AI Engineer fit) |
| Features (~25 dims) | See table below |
| Training | Offline in `scripts/train_ltr.py`; minutes on CPU |
| Inference | `model.predict(X)` on 100K rows: **< 30 s** |

**LTR feature vector (per candidate):**

| Feature group | Examples |
|---------------|----------|
| Title / career | `title_relevance`, `ml_in_career`, `is_consulting_only`, `product_company_flag` |
| Skills | `trusted_must_have_score`, `stuffer_score`, `expert_skill_count` |
| Experience / location | `experience_fit`, `location_fit`, `notice_penalty` |
| Semantic | `embedding_cosine`, `bm25_score` |
| Behavioral (as features, not multiplier) | `response_rate`, `days_since_active`, `open_to_work` |
| Integrity | `integrity_score` (also used as gate — duplicated for LTR context) |

**FIT with LTR:**

```python
if fit_mode == "ltr" and ltr_model exists:
    fit = sigmoid(ltr_model.predict(feature_vector))
else:
    fit = weighted_sum(title, skills, experience, semantic, location)
```

#### 5.4.3 What we explicitly do NOT use

| Approach | Reason |
|----------|--------|
| Hosted LLM per candidate | Network banned; too slow |
| GPU inference | Banned during ranking |
| GNN at ranking time | Too slow unless precomputed; defer unless time permits |
| ML for honeypot detection | Rules are more reliable and explainable |
| End-to-end neural ranker on raw JSON | Opaque, slow, hard to defend at Stage 5 |

#### 5.4.4 Graph-based features (optional, Phase 6c — only if time permits)

Build offline skill co-occurrence graph with `networkx`; precompute per-candidate features:

- `jd_skill_coverage_ratio` — fraction of JD must-have skills reachable via skill graph
- `skill_graph_centrality` — centrality of candidate's skills in AI/ML subgraph

Store in `artifacts/graph_features.npz`. Use as extra LTR inputs. **Not required for v1.**

---

## 6. Module Design

```
redrob-ranker/
├── architecture.md                 # This document
├── README.md                       # Setup, reproduce_command, methodology
├── requirements.txt                # Pinned dependencies
├── submission_metadata.yaml        # Portal metadata mirror
├── Dockerfile                      # Stage 3 reproduction
├── config/
│   └── jd_requirements.yaml        # JD must-haves, disqualifiers, synonyms, weights
├── artifacts/                      # Precomputed (committed or built via precompute.py)
│   ├── candidate_embeddings.npy
│   ├── candidate_id_index.json
│   ├── jd_embedding.npy
│   ├── ltr_model.lgb               # optional — LightGBM ranker
│   ├── feature_scaler.pkl          # optional — StandardScaler for LTR features
│   └── graph_features.npz          # optional — skill graph features
├── data/                           # gitignored — not committed
│   └── candidates.jsonl
├── src/
│   ├── __init__.py
│   ├── io.py                       # JSONL streaming loader
│   ├── jd.py                       # Load jd_requirements.yaml
│   ├── text.py                     # Canonical profile text builder
│   ├── features.py                 # Structured feature extraction
│   ├── fit.py                      # FIT sub-scorers
│   ├── availability.py             # AVAILABILITY multiplier
│   ├── integrity.py                # INTEGRITY / honeypot / trap rules
│   ├── semantic.py                 # Embedding lookup + optional BM25
│   ├── ml_features.py              # Feature vector builder for LTR
│   ├── ml_fit.py                   # Rules FIT vs LTR FIT (fit_mode switch)
│   ├── scorer.py                   # FINAL = FIT × AVAIL × INTEGRITY
│   ├── reasoning.py                # Fact-based 1–2 sentence reasoning
│   └── pipeline.py                 # End-to-end orchestration
├── scripts/
│   ├── precompute.py               # Offline embeddings (Phase 6)
│   └── train_ltr.py                # Offline LTR training (Phase 6b, optional)
├── data/
│   └── labels/                     # Manual tier labels for LTR training
│       └── train_labels.csv        # candidate_id, tier (0-5)
├── rank.py                         # CLI entry point (reproduce_command)
├── validate_submission.py          # From challenge bundle
├── tests/
│   ├── fixtures/
│   │   └── sample_candidates.json
│   ├── test_integrity.py
│   ├── test_fit.py
│   ├── test_scorer.py
│   ├── test_reasoning.py
│   └── test_format.py
└── sandbox/
    └── app.py                      # Streamlit demo (≤100 candidates)
```

---

## 7. Data Flow (Ranking Step)

```
1. Parse CLI args (--candidates, --out, optional --config, --artifacts)
2. Load config/jd_requirements.yaml
3. Load artifacts/ if present (embeddings, id index)
4. Stream candidates.jsonl (one pass, no full 100K dict in memory if avoidable)
5. For each candidate:
     a. integrity = compute_integrity(candidate)
     b. if integrity < EXCLUDE_THRESHOLD: skip
     c. fit = compute_fit(candidate, jd_config, artifacts)
     d. availability = compute_availability(candidate.redrob_signals)
     e. final = fit * availability * integrity
     f. push (candidate_id, final) into min-heap or partial sort buffer
6. Select top 100 by final score
7. Tie-break: candidate_id ascending
8. Assign ranks 1–100; map scores monotonically non-increasing
9. Generate reasoning per row (from profile facts + JD gaps)
10. Write UTF-8 CSV
11. Log timing + memory summary
```

### 7.1 Performance strategy

| Stage | Technique | Target |
|-------|-----------|--------|
| Load JSONL | `orjson` + streaming | < 90 s |
| Integrity | Cheap rules first — skip bad profiles early | < 30 s |
| FIT (full) | Vectorized numpy where possible | < 90 s |
| Semantic | Precomputed embeddings OR BM25 on coarse top-K | < 60 s |
| Sort top 100 | `heapq.nlargest` or partial sort | < 5 s |
| Reasoning | Template + fact extraction for 100 rows | < 10 s |
| **Total** | | **< 5 min** |

**Coarse-to-fine (if needed):** compute cheap FIT proxy on all 100K, fully score top 5,000 only.

---

## 8. JD Understanding Layer

**Source:** `job_description.docx` → curated `config/jd_requirements.yaml`

```yaml
role:
  title_keywords: ["AI Engineer", "ML Engineer", "Search Engineer", "Applied Scientist", ...]
  exclude_titles: ["Marketing Manager", "HR Manager", "Content Writer", ...]

experience:
  ideal_min: 5
  ideal_max: 9
  hard_max: 15

location:
  preferred_cities: ["Pune", "Noida", "Delhi", "Gurgaon", "Mumbai", "Hyderabad", "Bangalore"]
  preferred_country: "India"

must_have_skills:
  - embeddings
  - vector search
  - retrieval
  - python
  - ranking evaluation
  - ndcg
  # + synonyms map

nice_to_have_skills:
  - lora
  - peft
  - learning to rank
  - xgboost

consulting_firms:
  - TCS
  - Infosys
  - Wipro
  - Accenture
  - Cognizant
  - Capgemini
  - Mindtree
  - HCL

production_signals:
  - "shipped"
  - "production"
  - "deployed"
  - "A/B"
  - "offline benchmark"
  - "users"

weights:
  title_career: 0.35
  skills: 0.25
  experience: 0.15
  semantic: 0.15
  location: 0.10

thresholds:
  integrity_exclude: 0.30
  integrity_penalty: 0.60
  skill_stuffer_expert_count: 8
```

This file is the **single source of truth** for JD intent. Tuned on sample set, frozen before full run.

---

## 9. Candidate Profile Representation

Per candidate, build:

### 9.1 Structured record

From `candidate_schema.json` fields: `profile`, `career_history`, `education`, `skills`, `redrob_signals`, optional `certifications`, `languages`.

### 9.2 Canonical text (for semantic matching)

```
{headline} | {current_title} @ {current_company}
{summary}
{career_history descriptions joined}
{skill names joined}
```

### 9.3 Derived features

| Feature | Source |
|---------|--------|
| `is_consulting_only` | All companies in consulting list, no product co |
| `title_relevance` | Match against `role.title_keywords` / `exclude_titles` |
| `ml_in_career` | NLP/IR/retrieval keywords in career descriptions |
| `trusted_skill_score` | Must-have skills × trust formula |
| `experience_fit` | Gaussian or trapezoid around 5–9 years |
| `location_fit` | India + preferred city + relocate flag |
| `stuffer_score` | High skills / low title+career alignment |

---

## 10. Reasoning Generation

Spec Stage 4 checks: specific facts, JD connection, honest concerns, no hallucination, variation, rank consistency.

**Approach:** rule-based fact assembly — **no LLM at runtime**.

Template structure:

```
"{current_title} with {years} yrs; {top_matching_skills}; {location/notice note}; {concern if any}."
```

Rules:

- Only cite fields present in candidate JSON.
- Rank ≤ 20: confident tone; rank ≥ 80: note gaps explicitly.
- Vary sentence structure by rank bucket (top / mid / tail).
- Mention one JD-relevant concern when applicable (notice >30d, consulting background, weak retrieval evidence).

---

## 11. Testing & Validation Strategy

### Phase 1 — Unit tests (fixtures)

**Input:** `sample_candidates.json` (50 candidates) + synthetic honeypot fixtures in `tests/fixtures/`.

| Test | Asserts |
|------|---------|
| `test_integrity.py` | Honeypot patterns detected; clean profiles pass |
| `test_fit.py` | Marketing Manager + AI skills scores below ML Engineer |
| `test_scorer.py` | FINAL ordering, tie-break by candidate_id |
| `test_reasoning.py` | No hallucinated skills; non-empty; varied |
| `test_format.py` | Output passes `validate_submission.py` |

Run:

```bash
pytest tests/ -v
```

### Phase 2 — Sample integration (manual + script)

**Input:** `sample_candidates.json` (treat as mini pool).

```bash
python rank.py --candidates sample_candidates.json --out sample_out.csv
python validate_submission.py sample_out.csv
```

**Manual review checklist (top 20 from 50):**

- [ ] No Marketing/HR/Content titles in top 5
- [ ] Experience mostly 4–10 years in top 10
- [ ] No obvious honeypot patterns in top 10
- [ ] Scores decrease monotonically
- [ ] Reasoning cites real profile facts

### Phase 3 — Full pool validation

**Input:** `candidates.jsonl` (100,000 records).

```bash
/usr/bin/time -v python rank.py --candidates candidates.jsonl --out team_xxx.csv
python validate_submission.py team_xxx.csv
```

**Automated gates:**

| Gate | Pass criteria |
|------|---------------|
| Runtime | Wall time ≤ 300 s |
| Memory | Peak RSS ≤ 16 GB |
| Format | `validate_submission.py` exits 0 |
| Score spread | `score[rank=1] - score[rank=100] > 0.20` |
| Title sanity | < 2 excluded titles in top 20 |
| Integrity audit | Manual spot-check 20 rows in top 100 for timeline/skill fraud |

**Ablation tests (before final submit):**

| Ablation | Expected effect if removed |
|----------|---------------------------|
| No INTEGRITY | Honeypots rise in top 100 — bad |
| No AVAILABILITY | Inactive profiles rise — bad |
| No title/career | Keyword stuffers rise — bad |
| No embeddings | Plain-language gems drop — bad |
| No LTR (rules only) | Top-10 ordering may be suboptimal vs NDCG@10 |

### Phase 4 — Pre-upload checklist

- [ ] `validate_submission.py team_xxx.csv` passes
- [ ] `rank.py` completes < 5 min on full pool (CPU, no network)
- [ ] README documents `reproduce_command`
- [ ] `submission_metadata.yaml` filled and accurate
- [ ] Sandbox demo works on ≤100 candidates
- [ ] `pre_computation_required` set correctly in metadata
- [ ] AI tools declared honestly
- [ ] Git history shows real iteration (not single dump)

---

## 12. Evaluation Stage Mapping

| Our artifact | Spec stage | Purpose |
|--------------|------------|---------|
| `validate_submission.py` | Stage 1 | Format gate |
| Top-100 CSV quality | Stage 2 | NDCG@10, NDCG@50, MAP, P@10 |
| `rank.py` + Dockerfile | Stage 3 | Reproduce ≤5 min; honeypot rate |
| Reasoning + README + git | Stage 4 | Manual review |
| Architecture knowledge | Stage 5 | Interview defense |

---

## 13. Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Python | 3.11 | Per metadata template |
| JSON | `orjson` | Fast 100K parse |
| Numerics | `numpy` | Vectorized scoring |
| Fuzzy match | `rapidfuzz` | Skill/title normalization |
| Semantic (offline) | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | **Required** for competitive FIT |
| LTR (offline) | `lightgbm` | **Recommended** final FIT mode |
| BM25 (optional) | `rank-bm25` or sklearn `TfidfVectorizer` | Extra LTR feature / fallback |
| Graph (optional) | `networkx` | Offline skill graph features only |
| Config | `pyyaml` | JD requirements |
| CLI | `argparse` | `rank.py` |
| Tests | `pytest` | Unit + integration |
| Sandbox | `streamlit` | HuggingFace Spaces demo |
| Container | `Docker` | Stage 3 reproduction |

**Excluded at ranking time:** hosted LLMs, GPU, network calls, LangChain runtime dependency.

---

## 14. Implementation Phases

| Phase | Deliverable | ML? | Validation |
|-------|-------------|-----|------------|
| **1** | `config/jd_requirements.yaml` | No | Manual score 10 sample candidates |
| **2** | `src/integrity.py` + tests | No | Honeypot fixtures pass |
| **3** | `src/fit.py`, `src/features.py` + tests | No | `fit_mode=rules`; stuffer < gem on samples |
| **4** | `src/availability.py` + tests | No | Dead profiles down-ranked |
| **5** | `src/scorer.py`, `src/pipeline.py` | No | Sample CSV passes `validate_submission.py` |
| **6** | `scripts/precompute.py`, `src/semantic.py` | **Yes** | Embeddings load <30 s; semantic improves plain-language gems |
| **6b** | `scripts/train_ltr.py`, `src/ml_fit.py`, labels | **Yes** | `fit_mode=ltr`; top-10 better on manual review vs Phase 5 |
| **6c** | Skill graph features (optional) | Optional | Marginal gain; skip if behind schedule |
| **7** | `src/reasoning.py` | No | 10-row reasoning audit passes |
| **8** | `rank.py` CLI | Uses artifacts | Full 100K < 5 min |
| **9** | README, metadata, sandbox, Docker | — | Submission-ready |

**Gate between Phase 5 and 6:** Run rules-only on `sample_candidates.json`. If top-10 manual review is clean, add embeddings. If ordering among good candidates is weak, add LTR (6b).

---

## 15. Success Criteria

### Technical

- `rank.py` on `candidates.jsonl` finishes in **< 5 minutes** on 16 GB CPU.
- `validate_submission.py` passes with zero errors.
- Honeypot rate in top 100 **< 5%** on manual audit (target < 3%).

### Ranking quality (proxy — no public labels)

- Top 10: ML/AI/search engineers at product companies, India or willing to relocate.
- Top 10: no obvious keyword-stuffed wrong-title profiles.
- Top 10: recently active, reasonable recruiter response rates.

### Submission readiness

- Single `reproduce_command` documented and tested.
- Sandbox demo functional.
- Reasoning is specific, honest, varied — ready for Stage 4.

---

## 16. Architecture Decisions Log

| Decision | Rationale |
|----------|-----------|
| **Hybrid rules + ML** | Rules pass honeypot/Stage 3; ML wins NDCG@10 |
| ML not required for spec | Rules-only can submit; embeddings + LTR for competitiveness |
| INTEGRITY always rule-based | Honeypots are deterministic; ML would miss edge cases |
| FIT uses ML (embeddings + LTR) | Semantic gems + learned top-10 ordering |
| AVAILABILITY always rule-based | Interpretable; matches signals doc |
| Title + career > skills | Explicit hackathon guidance in JD |
| `bge-small-en-v1.5` precomputed | Best CPU cost/quality for semantic slot |
| LightGBM LambdaMART optional | NDCG-aligned; fast inference on 100K |
| No GNN at ranking time | Precompute only if pursued; not in critical path |
| No LLM at inference | Compute constraint + Stage 3 enforcement |
| Sample-first testing | Fast iteration before 100K run |
| Deterministic tie-break | Spec requires `candidate_id` ascending |
| Rule-based reasoning | No network; passes Stage 4 hallucination checks |

---

## 17. References

| Document | Purpose |
|----------|---------|
| `README.docx` | Bundle overview, getting started |
| `submission_spec.docx` | Format, constraints, evaluation pipeline |
| `submission_metadata_template.yaml` | Portal + repo metadata |
| `candidate_schema.json` | Candidate field definitions |
| `redrob_signals_doc.docx` | 23 behavioral signals |
| `job_description.docx` | Ranking target role |
| `validate_submission.py` | Local format validator |
| `sample_submission.csv` | CSV structure reference |

---

*Next step: Phase 1 — implement `config/jd_requirements.yaml` and `src/integrity.py` (rules only). ML enters at Phase 6 after rules baseline passes sample review.*
