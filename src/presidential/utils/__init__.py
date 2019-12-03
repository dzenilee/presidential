from pathlib import Path

# PRESIDENTIAL_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = PRESIDENTIAL_DIR / "data"
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent  # .../src/
ROOT_DIR = SOURCE_DIR.parent
DATA_DIR = ROOT_DIR / "data"


with open(DATA_DIR / "2019-2020/moderator_names.txt", "r") as f:
    MODERATOR_NAMES = [line.strip() for line in f.readlines()]

with open(DATA_DIR / "2019-2020/candidate_names.txt", "r") as f:
    CANDIDATE_NAMES = [line.strip() for line in f.readlines()]
