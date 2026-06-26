import json
from collections import Counter
from pathlib import Path

# Paths
candidates_path = Path(__file__).resolve().parent / "candidates.jsonl"

# We can import config and compute_integrity
import sys
ROOT = Path(__file__).resolve().parent / "redrob-ranker"
sys.path.insert(0, str(ROOT))

from src.config_loader.jd_config import load_jd_config
from src.integrity.checker import compute_integrity, should_exclude

def main():
    config = load_jd_config(ROOT / "config" / "jd_requirements.yaml")
    
    total = 0
    excluded = 0
    issue_counts = Counter()
    
    # Read the first 10,000 candidates to get a representative sample quickly
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            cand = json.loads(line)
            integrity, issues = compute_integrity(cand, config)
            is_ex = should_exclude(integrity, config)
            if is_ex:
                excluded += 1
                for issue in issues:
                    issue_counts[issue] += 1
            if total >= 10000:
                break
                
    print(f"Total analyzed: {total}")
    print(f"Total excluded: {excluded} ({excluded/total:.1%})")
    print("\nExclusion issues breakdown:")
    for issue, count in issue_counts.most_common():
        print(f"- {issue}: {count} ({count/excluded:.1%})")

if __name__ == "__main__":
    main()
