import json
from pathlib import Path
import sys

# Setup paths
ROOT = Path(__file__).resolve().parent / "redrob-ranker"
sys.path.insert(0, str(ROOT))

from src.config_loader.jd_config import load_jd_config
from src.scoring.combiner import score_candidate
from src.features.extractor import extract_features
from src.semantic.loader import SemanticLoader

candidates_path = Path(__file__).resolve().parent / "candidates.jsonl"

def main():
    # Properly initialize SemanticLoader
    artifacts_dir = ROOT.parent / "artifacts"
    SemanticLoader.initialize(artifacts_dir)
    
    config = load_jd_config(ROOT / "config" / "jd_requirements.yaml")
    
    target_ids = {"CAND_0093193", "CAND_0065878"}
    found = {}
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in target_ids:
                found[cid] = cand
                if len(found) == len(target_ids):
                    break
                    
    for cid in target_ids:
        cand = found.get(cid)
        if not cand:
            print(f"Candidate {cid} not found!")
            continue
            
        print(f"==================================================")
        print(f"Inspection for {cid} (WITH SemanticLoader)")
        print(f"==================================================")
        prof = cand.get("profile", {})
        
        # Compute scores
        sc = score_candidate(cand, config)
        features = extract_features(cand, config)
        
        print(f"\n--- Scores ---")
        print(f"Final Score: {sc.final_score:.4f}")
        print(f"Fit Score: {sc.fit:.4f}")
        print(f"Availability Multiplier: {sc.availability:.4f}")
        print(f"Integrity Score: {sc.integrity:.4f}")
        
        print(f"\n--- Feature Fit Breakdown ---")
        for k, v in features.items():
            print(f"- {k}: {v:.4f}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
