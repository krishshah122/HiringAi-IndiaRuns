import sys
import os
from pathlib import Path
import pytest

# Correct absolute path to redrob-ranker
ranker_path = Path(__file__).resolve().parent / "redrob-ranker"
sys.path.insert(0, str(ranker_path))
os.chdir(str(ranker_path))

if __name__ == "__main__":
    sys.exit(pytest.main(["tests", "-v"]))
