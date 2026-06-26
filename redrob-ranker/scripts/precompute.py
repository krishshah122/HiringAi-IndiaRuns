#!/usr/bin/env python3
"""Offline precomputation of semantic embeddings for Phase 6."""

import argparse
import json
import os
import sys
from pathlib import Path

# Limit CPU threads to prevent thrashing and context switching overhead
num_threads = str(min(4, os.cpu_count() or 4))
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import torch
torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))

import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure src is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader.jd_config import load_jd_config
from src.io.candidates import iter_candidates, load_candidates
from src.text.profile_text import build_profile_text


def build_jd_text(config) -> str:
    parts = []
    parts.extend(config.role.get("positive_title_keywords", []))
    parts.extend(config.must_have_skills)
    parts.extend(config.nice_to_have_skills)
    parts.extend(config.ml_career_signals)
    parts.extend(config.production_signals)
    return " ".join(parts).lower()


def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--candidates", required=True, help="Path to candidates file")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory for embeddings")
    parser.add_argument("--config", default=None, help="Path to jd_requirements.yaml")
    args = parser.parse_args()

    out_dir = Path(ROOT) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JD config...")
    config = load_jd_config(args.config)
    jd_text = build_jd_text(config)

    print("Loading candidates...")
    cand_path = Path(args.candidates)
    if cand_path.suffix == ".json":
        candidates = load_candidates(cand_path)
    else:
        candidates = list(iter_candidates(cand_path))

    print(f"Loaded {len(candidates)} candidates.")

    print("Loading SentenceTransformer all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding JD...")
    jd_emb = model.encode(jd_text, normalize_embeddings=True)

    print("Encoding Candidates...")
    texts = []
    cand_ids = []
    
    # Optional: We could stream and encode in batches if 100K doesn't fit in RAM easily,
    # but for BAAI/bge-small-en-v1.5 it's small enough. We will do batching to be safe and show progress.
    for i, c in enumerate(candidates):
        cid = c.get("candidate_id")
        if not cid:
            continue
        text = build_profile_text(c)
        texts.append(text)
        cand_ids.append(cid)
        if (i + 1) % 10000 == 0:
            print(f"Processed text for {i + 1} candidates...")

    index_file = out_dir / "candidate_id_index.json"
    emb_file = out_dir / "candidate_embeddings.npy"

    id_index = {}
    existing_emb = None
    
    # Save JD emb upfront
    np.save(out_dir / "jd_embedding.npy", jd_emb)
    
    if index_file.exists() and emb_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                id_index = json.load(f)
            existing_emb = np.load(emb_file)
            print(f"Found checkpoint! Resuming from candidate {len(id_index)}...")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            id_index = {}
            existing_emb = None

    start_idx = len(id_index)
    if start_idx < len(texts):
        remaining_texts = texts[start_idx:]
        remaining_ids = cand_ids[start_idx:]
        
        checkpoint_size = 5000
        total_chunks = (len(remaining_texts) - 1) // checkpoint_size + 1
        
        for chunk_idx in range(total_chunks):
            start = chunk_idx * checkpoint_size
            end = start + checkpoint_size
            chunk = remaining_texts[start:end]
            chunk_cids = remaining_ids[start:end]
            
            print(f"\nEncoding chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} candidates)...")
            chunk_emb = model.encode(chunk, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            
            if existing_emb is not None:
                existing_emb = np.concatenate([existing_emb, chunk_emb], axis=0)
            else:
                existing_emb = chunk_emb
                
            for cid in chunk_cids:
                id_index[cid] = len(id_index)
                
            print(f"Saving checkpoint (total processed: {len(id_index)})...")
            np.save(emb_file, existing_emb)
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(id_index, f)

    print("Precomputation complete.")


if __name__ == "__main__":
    main()
