from pathlib import Path

# PRESIDENTIAL_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = PRESIDENTIAL_DIR / "data"
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent  # .../src/
ROOT_DIR = SOURCE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
