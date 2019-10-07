import pandas as pd
from presidential import PRESIDENTIAL_DIR

DATA_DIR = PRESIDENTIAL_DIR / "data"

pathlist = DATA_DIR.glob("*.txt")

MODERATORS = [
    "Cooper",
    "Bash",
    "Lemon",
    "Lopez",
    "Raddatz",
    "Cordes",
    "Cooney",
    "Dickerson",
    "Obradovich",
    "Blitzer",
    "Louis",
    "Holt",
    "Mitchell",
    "Ifill",
    "Woodruff",
    "Ramos",
    "Salinas",
    "Tumulty" "Todd",
    "Maddow",
    "Cuomo",
]
CANDIDATES = ["Chafee", "Clinton", "O'Malley", "Sanders", "Webb"]


def get_data(transcript_dir=DATA_DIR):
    pathlist = transcript_dir.glob("*.txt")
    segments = []
    labels = []
    debate = []
    for path in pathlist:
        with open(path, "r") as f:
            for line in f:
                if not line:
                    continue
                for name in set(MODERATORS + CANDIDATES):
                    if name.upper() in line:
                        label = name.lower()
                segments.append(line.replace(f"{label.upper()}: ", ""))
                labels.append(label.lower())
                debate.append(path.stem)
    return segments, labels, debate


segments, labels, debate = get_data(DATA_DIR)
debate_df = pd.DataFrame({"segment": segments, "speaker": labels, "debate": debate})
