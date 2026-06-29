# Redrob Candidate Discovery & Ranking System

This repository contains the intelligent candidate discovery, scoring, and ranking system implemented for the **Redrob Hackathon Challenge**. It describes the actual pipeline we built, not the generic rulebook.

The system uses a high-performance two-stage ranking pipeline to discover the best fit candidates from 100,000 profiles, with the main Stage 3 ranking command engineered to complete in under 5 minutes on CPU-only hardware.

---

## 🚀 How to Reproduce Rankings

The execution is split into an offline precomputation step and the runtime ranking step.

### Step 1: Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### Step 2: Offline Pre-computation
Prepare semantic embeddings for job and candidate profile text using the local `SentenceTransformer` model.
```bash
python scripts/precompute.py --candidates ./candidates.jsonl
```

### Step 3: Run the Ranking Pipeline
This command executes the built Stage 3 ranking pipeline and writes the ranked CSV output.
```bash
python rank.py --candidates ./candidates.jsonl --out ./final_submission.csv
```

This runtime ranking step is implemented to meet the required Stage 3 timing on CPU-only hardware.

---

## 🛠️ Architecture & Methodology

The repo implements a **Two-Stage Ranking** pipeline:

1. **Stage 1:** fast candidate scoring and filtering using integrity checks, availability signals, and light feature scoring.
2. **Stage 2:** semantic re-ranking of the top candidates with rich profile text and on-the-fly similarity scoring.

This design keeps the Stage 3 command efficient while still using deep semantic signal among the strongest candidates.

---

## 🤖 Running the Sandbox App

The interactive sandbox application allows you to upload a small sample file and generate a ranked CSV file.

### Local Streamlit
To test the sandbox locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure

* `rank.py`: CLI entry point to execute the ranking pipeline.
* `app.py`: Streamlit Sandbox web interface.
* `requirements.txt`: Python dependencies list.
* `config/jd_requirements.yaml`: Job Description requirements, weights, and configurations.
* `scripts/precompute.py`: Offline semantic precomputation script.
* `src/`: Core pipeline source code:
  * `src/integrity/`: Fraud/Honeypot detection check logic.
  * `src/availability/`: Recruiter response rate, notice period, and active date filters.
  * `src/fit/`: Feature scorer combination.
  * `src/features/`: Feature extractors (IT consulting firm filters, YOE scoring, location scoring).
  * `src/semantic/`: Loading and on-the-fly similarity calculation.
  * `src/reasoning/`: Fact-based reasoning generator templates.
