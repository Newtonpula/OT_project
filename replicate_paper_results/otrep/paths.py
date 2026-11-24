"""Repository paths. Importing this module also makes the repository's `snprep` package importable."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / 'replicate_paper_results' / 'outputs'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
