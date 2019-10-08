import pandas as pd
from presidential.utils import DATA_DIR

pathlist = DATA_DIR.glob("*.txt")

DEP_MODERATORS = [
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


OTHER = ["Unknown", "Unidentified"]
REP_MODERATORS = ["Muir", "Raddatz", "Kelly", "Wallace", "Baier"]
DEM_CANDIDATES = ["Chafee", "Clinton", "O'Malley", "Sanders", "Webb"]
REP_CANDIDATES = [
    "Bush",
    "Carson",
    "Christie",
    "Cruz",
    "Kasich",
    "Rubio",
    "Trump",
    "Huckabee",
    "Walker",
    "Paul",
]


def get_data(transcript_dir=DATA_DIR):
    pathlist = transcript_dir.glob("*.txt")
    segments = []
    labels = []
    debate = []
    for path in pathlist:
        print(path)
        with open(path, "r") as f:
            for line in f:
                if not line:
                    continue
                for name in set(DEP_MODERATORS + DEM_CANDIDATES + OTHER):
                    if name.upper() in line:
                        label = name.lower()
                for name in set(REP_MODERATORS + REP_CANDIDATES + OTHER):
                    if name.upper() in line:
                        label = name.lower()
                segments.append(line.replace(f"{label.upper()}: ", ""))
                labels.append(label.lower())
                debate.append(path.stem)
    return segments, labels, debate


segments, labels, debate = get_data(DATA_DIR)


def clean_up_segments(segments):
    return [s.strip() for s in segments]


debate_df = pd.DataFrame(
    {"segment": clean_up_segments(segments), "speaker": labels, "debate": debate}
)
