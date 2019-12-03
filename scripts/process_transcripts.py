import numpy as np
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

from presidential.utils import DATA_DIR, CANDIDATE_NAMES, MODERATOR_NAMES


url_2019_06_27 = "https://www.washingtonpost.com/politics/2019/06/27/transcript-night-one-first-democratic-debate-annotated/"
url_2019_06_28 = "https://www.washingtonpost.com/politics/2019/06/28/transcript-night-first-democratic-debate/"
url_2019_07_31 = "https://www.washingtonpost.com/politics/2019/07/31/transcript-first-night-second-democratic-debate/"
url_2019_08_01 = "https://www.washingtonpost.com/politics/2019/08/01/transcript-night-second-democratic-debate/"
url_2019_09_12 = "https://www.washingtonpost.com/politics/2019/09/13/transcript-third-democratic-debate/"
url_2019_10_15 = "https://www.washingtonpost.com/politics/2019/10/15/october-democratic-debate-transcript/"
url_2019_11_20 = "https://www.washingtonpost.com/politics/2019/10/15/october-democratic-debate-transcript/"

DEBATE_METADATA = [
    {"date": "2019-06-27", "url": url_2019_06_27, "location": "Miami-FL"},
    {"date": "2019-06-28", "url": url_2019_06_28, "location": "Miami-FL"},
    {"date": "2019-07-31", "url": url_2019_07_31, "location": "Detroit-MI"},
    {"date": "2019-08-01", "url": url_2019_08_01, "location": "Detroit-MI"},
    {"date": "2019-09-12", "url": url_2019_09_12, "location": "Houston-TX"},
    {"date": "2019-10-15", "url": url_2019_10_15, "location": "Westerville-OH"},
    {"date": "2019-11-20", "url": url_2019_11_20, "location": "Atlanta-GA"},
]


def get_p_tag_blocks(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.findAll("p")
    blocks = [p.text for p in paragraphs if len(p) > 0]  # don't include empty lines
    return blocks


def _get_speaker_name(block):
    pattern = r"^.{2,}:"  # match at least two upper case letters from
    # beginning of line
    match = re.match(pattern, str(block))  # match up to and including ":"
    if match:
        name = match.group(0).replace(":", "")
        name_split = name.split()
        if len(name_split) == 1:
            return name
        elif len(name_split) > 1:
            speaker_last_name = [
                word
                for word in name_split
                if word.upper() in CANDIDATE_NAMES + MODERATOR_NAMES
            ]
            if speaker_last_name:
                return speaker_last_name[0].upper()


def assign_speaker_tags(blocks):
    blocks = [t for t in blocks if len(t.split()) > 0]
    name_segment_blocks = []
    for i, block in enumerate(blocks):
        speaker_name = _get_speaker_name(block)
        if speaker_name:
            segment = block.split(": ", 1)[1]  # text after first instance of ": "
            name_segment_blocks.append((speaker_name, segment))
        else:
            if i == 0:  # escape indexing error
                name_segment_blocks.append((None, block))
            else:
                last_block = name_segment_blocks[-1]
                prev_name = last_block[0]
                name_segment_blocks.append((prev_name, block))
    return name_segment_blocks


def create_dataframe(debate_info):
    blocks = get_p_tag_blocks(debate_info["url"])
    name_segment_blocks = assign_speaker_tags(blocks)
    names, segments = list(zip(*name_segment_blocks))
    debate_id = np.full(len(names), debate_info["location"])
    df = pd.DataFrame({"speaker": names, "segment": segments, "debate": debate_id})
    return df


if __name__ == "__main__":
    from datetime import date

    dataframes = []
    for debate in DEBATE_METADATA:
        print(f"Creating dataframe for {debate['date']}")
        dataframes.append(create_dataframe(debate))

    # Combine all dataframes.
    df0 = pd.concat(dataframes, ignore_index=True)

    # Write the master dataframe to a file.
    filename = f"processed_transcripts_{date.today()}.csv"
    df0.to_csv(DATA_DIR / filename, index=False)
    n_rows = df0.shape[0]
    n_columns = df0.shape[1]
    print(
        f"Wrote a dataframe containing {n_rows} rows and {n_columns} "
        f"columns to {filename}."
    )
