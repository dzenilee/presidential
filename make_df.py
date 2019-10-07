import spacy
import pandas as pd

from presidential import PRESIDENTIAL_DIR
from presidential.preprocess import debate_df
from presidential.features import Featurizer

DATA_DIR = PRESIDENTIAL_DIR / "data"
nlp = spacy.load("en_core_web_md")
f = Featurizer()
df = debate_df

col_dict = {}

# type: pandas.core.series.Series
mtld = df.segment.apply(f.get_mtld)
n_difficult_words = df.segment.apply(f.get_difficult_words_count)
n_opponent_mentions = df.segment.apply(f.get_n_opponent_mentions)
n_words_before_main_verb = df.segment.apply(f.get_n_words_before_main_verb)
readability = df.segment.apply(f.get_readability_score)

# dictionaries
pronoun_counts_dict = df.segment.apply(f.count_pronouns)
sent_length_dict = df.segment.apply(f.get_sent_length_stats)

# convert to df
pronoun_df = pd.DataFrame(pronoun_counts_dict)
sent_length_df = pd.DataFrame(sent_length_dict)

n_1sg = [d.get("1sg") for d in pronoun_df.segment]
n_1pl = [d.get("1pl") for d in pronoun_df.segment]
n_2 = [d.get("2") for d in pronoun_df.segment]
n_3 = [d.get("3") for d in pronoun_df.segment]

# add new series to col_dict
col_dict["mtld"] = mtld
col_dict["n_difficult_words"] = n_difficult_words
col_dict["n_opponent_mentions"] = n_opponent_mentions
col_dict["n_words_before_main_verb"] = n_words_before_main_verb
col_dict["readability"] = readability
col_dict["person_1sg"] = n_1sg
col_dict["person_1pl"] = n_1pl
col_dict["person_2"] = n_2
col_dict["person_3"] = n_3

# add new columns
for k, v in col_dict.items():
    df[k] = v

df.head()


if __name__ == "__main__":
    print(df.columns)
    df.to_csv(DATA_DIR / "new_df.csv")
