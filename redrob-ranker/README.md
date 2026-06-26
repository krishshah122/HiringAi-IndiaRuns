# Redrob Candidate Discovery & Ranking System

This repository contains the intelligent candidate discovery, scoring, and ranking system developed for the **Redrob Hackathon Challenge** (Senior AI Engineer position). 

The system implements a high-performance, two-stage semantic matching and rule-based filter pipeline to discover the best fit candidates from a pool of 100,000 candidates while staying strictly within the 5-minute CPU constraint in offline sandboxed environments.

---

## 🚀 How to Reproduce Rankings

For **Stage 3 code reproduction**, the execution steps are split into an offline pre-computation step and a real-time ranking step.

### Step 1: Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### Step 2: Offline Pre-computation
This step generates dense semantic embeddings for the job description and candidate profiles using a local, cached `all-MiniLM-L6-v2` SentenceTransformer model. 
*(Note: As per the spec, this offline step may exceed the 5-minute window during setup).*

> Large dataset files such as `candidates.jsonl` should be kept out of the Git repo and excluded via `.gitignore`.

```bash
python scripts/precompute.py --candidates ./candidates.jsonl
```

### Step 3: Run the Ranking Pipeline
This single command runs the entire pipeline (Fast Rules filter, availability check, integrity check, and Stage 2 semantic re-ranking) and outputs the final top 100 candidates.
*(Note: This step executes in **~37 seconds** on CPU, easily satisfying the $\le 5$ minutes sandbox constraint).*
```bash
python rank.py --candidates ./candidates.jsonl --out ./final_submission.csv
```

---

## 🛠️ Architecture & Methodology (Two-Stage Ranking)

To satisfy the compute/memory constraints of the sandbox without compromising on deep semantic retrieval (career descriptions and project history), the ranker uses a **Two-Stage Retriever-Ranker** architecture:

1. **Stage 1 (Retrieval):** 
   Scans the 100,000 candidates quickly using rules (experience range, honeypot/fraud checker, location fit, IT consulting filters) and fast pre-computed embeddings. It narrows down the pool to the **top 500 candidates** in under 15 seconds.
2. **Stage 2 (Deep Re-ranking):** 
   For only these top 500 candidates, the system builds rich profile text on-the-fly (including company names, industries, and the first 300 characters of each job description). It encodes them using `SentenceTransformer` on-the-fly in **<1.5 seconds**, performing a deep semantic match of project accomplishments against the JD requirements.

---

## 🤖 Running the Sandbox App Locally

To test the Sandbox App (which allows uploader files of $\le 100$ candidates and instant CSV downloads), you can run it locally using Streamlit:
```bash
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
