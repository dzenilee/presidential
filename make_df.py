import spacy
from presidential.preprocess import debate_df
from presidential.features import Featurizer


nlp = spacy.load("en_core_web_md")

f = Featurizer()
segments = debate_df.segment

# list_of_feature_dicts = []
# for d in segments:
#     feature_dict = featurizer.featurize(nlp(d))
#     list_of_feature_dicts.append(feature_dict)

df = debate_df


# add more columns
df["segment_length"] = df.segment.str.len()
df["mtld"] = df.segment




col_dict = {}

n_opponent_mentions = segments.apply(f.get_n_opponent_mentions)
pronoun_counts = segments.apply(f.count_pronouns)


n_1sg = segments.map(pronoun_counts["1sg"])


col_dict.update({"n_opponent_mentions": n_opponent_mentions})


for k, v in col_dict.items():
    df[k] = v
