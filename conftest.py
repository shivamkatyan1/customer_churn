import sys
from pathlib import Path

# Make the project root importable when running `pytest` from any directory.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
