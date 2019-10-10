import pandas as pd
from presidential.utils import DATA_DIR

pathlist = DATA_DIR.glob("*.txt")

DEP_MODERATORS = [
    "Bash",
    "Cooper",
    "Cuomo",
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
    "Ramos",
    "Salinas",
    "Todd",
    "Tumulty",
    "Maddow",
    "Woodruff",
]
REP_MODERATORS = [
    "Arrarás",
    "Baier",
    "Bartiromo",
    "Bash",
    "Blitzer",
    "Cavuto",
    "Dickerson",
    "Dinan",
    "Garrett",
    "Hewitt",
    "Kelly",
    "Muir",
    "Raddatz",
    "Tapper",
    "Wallace",
    "Strassel",
]
DEM_CANDIDATES = ["Chafee", "Clinton", "O'Malley", "Sanders", "Webb"]
REP_CANDIDATES = [
    "Bush",
    "Carson",
    "Christie",
    "Cruz",
    "Fiorina",
    "Huckabee",
    "Kasich",
    "Paul",
    "Rubio",
    "Trump",
    "Walker",
]
OTHER = ["Unknown", "Unidentified", "Unidentifiable"]
ALL_SPEAKERS = set(
    DEP_MODERATORS + DEM_CANDIDATES + REP_MODERATORS + REP_CANDIDATES + OTHER
)


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
                for name in ALL_SPEAKERS:
                    if name.upper() in line:
                        label = name.lower()
                for name in ALL_SPEAKERS:
                    if name.upper() in line:
                        label = name.lower()
                segments.append(line.replace(f"{label.upper()}: ", ""))
                labels.append(label.lower())
                debate.append(path.stem)
    return segments, labels, debate


def clean_up_segments(segments):
    segments = [s.replace("[crosstalk]", "") for s in segments]
    segments = [s.replace("[applause]", "").strip() for s in segments]
    return segments


segments, labels, debate = get_data(DATA_DIR)

df = pd.DataFrame(
    {"segment": clean_up_segments(segments), "speaker": labels, "debate": debate}
)
debate_df = df.dropna(subset=["segment"])
