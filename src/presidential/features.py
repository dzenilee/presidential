import numpy as np
import pandas as pd
import re
import spacy
from textstat import textstat
from lexicalrichness import LexicalRichness

from presidential.utils import CANDIDATE_NAMES

nlp = spacy.load("en_core_web_md")


FIRST_SINGULAR = ["i", "my", "mine", "myself"]
FIRST_PLURAL = ["we", "our", "ours", "ourselves"]
SECOND_SINGULAR = ["you", "your", "yours", "yourself"]
SECOND_PLURAL = ["you", "your", "yours", "yourselves"]
THIRD_SINGULAR_MASCULINE = ["he", "his", "him", "himself"]
THIRD_SINGULAR_FEMININE = ["she", "her", "hers", "herself"]
THIRD_PLURAL = ["they", "their", "theirs", "themselves"]
FIRST_SECOND_PERSON = set(
    FIRST_SINGULAR + SECOND_PLURAL + SECOND_SINGULAR + SECOND_PLURAL
)
THIRD_PERSON = set(THIRD_SINGULAR_FEMININE + THIRD_SINGULAR_MASCULINE + THIRD_PLURAL)


class SegmentFeaturizer:
    def __init__(self):
        pass

    @staticmethod
    def count_pronouns(segment):
        segment = segment.lower().split()
        counter = {"1sg": 0, "1pl": 0, "2": 0, "3": 0}
        for pronoun in FIRST_SINGULAR:
            counter["1sg"] += segment.count(pronoun)
        for pronoun in FIRST_PLURAL:
            counter["1pl"] += segment.count(pronoun)
        for pronoun in set(SECOND_PLURAL + SECOND_SINGULAR):
            counter["2"] += segment.count(pronoun)
        for pronoun in THIRD_PERSON:
            counter["3"] += segment.count(pronoun)
        return counter

    @staticmethod
    def get_difficult_words_count(segment):
        return textstat.difficult_words(segment)

    @staticmethod
    def get_mtld(segment):
        # measure of textual lexical diversity (McCarthy 2005, McCarthy and
        # Jarvis 2010)
        if re.search("[a-zA-Z]", segment) is None:  # e.g., "2", "223)."
            return 0
        else:
            return LexicalRichness(segment).mtld(threshold=0.72)

    @staticmethod
    def get_n_applauses(segment):
        return segment.count("APPLAUSE")

    @staticmethod
    def get_n_crosstalks(segment):
        return segment.count("CROSSTALK")

    @staticmethod
    def get_n_opponent_mentions(segment):
        return sum(segment.count(c.title()) for c in CANDIDATE_NAMES)

    @staticmethod
    def get_trump_mentions(segment):
        return segment.count("Trump")

    @staticmethod
    def get_n_words_before_main_verb(segment):
        numbers = [0]
        doc = nlp(segment)
        for sent in doc.sents:
            main = [t for t in sent if t.dep_ == "ROOT"][0]
            if main.pos_ == "VERB":
                dist_to_init = main.i - sent[0].i
                numbers.append(dist_to_init)
        return np.mean(numbers)

    @staticmethod
    def get_readability_score(segment):
        scores = []
        scores.append(textstat.automated_readability_index(segment))
        scores.append(textstat.coleman_liau_index(segment))
        scores.append(textstat.flesch_kincaid_grade(segment))
        scores.append(textstat.gunning_fog(segment))
        return np.mean(scores)

    @staticmethod
    def get_sent_length_stats(segment):
        word_count_per_sent = []
        doc = nlp(segment)
        for sent in doc.sents:
            word_count_per_sent.append(len(sent.text.split()))
        sent_length_dict = {}
        sent_length_dict["n_words"] = len(segment.split())
        sent_length_dict["n_sents"] = len(list(doc.sents))
        sent_length_dict["mean_sent_length"] = np.mean(word_count_per_sent)
        sent_length_dict["std_sent_length"] = np.std(word_count_per_sent)
        return sent_length_dict

    @staticmethod
    def get_spacy_vector(segment):
        # vector = 0
        doc = nlp(segment)
        # for tok in doc:
        #     vector += tok.vector
        return doc.vector

    def featurize(self, series):
        f = SegmentFeaturizer()
        feature_dict = {}

        # Create new series to add to df0. Each series corresponds to a feature.
        n_applauses = series.apply(f.get_n_applauses)
        n_crosstalks = series.apply(f.get_n_crosstalks)
        n_difficult_words = series.apply(f.get_difficult_words_count)
        n_opponent_mentions = series.apply(f.get_n_opponent_mentions)
        n_trump_mentions = series.apply(f.get_trump_mentions)
        mtld = series.apply(f.get_mtld)
        n_words_before_main_verb = series.apply(f.get_n_words_before_main_verb)
        readability = series.apply(f.get_readability_score)
        segment_length = series.str.len()

        # Some of the features are originally in dictionary format. Create a column
        # for each integer value extracted.
        pronoun_counts_dict = series.apply(f.count_pronouns)
        sent_length_dict = series.apply(f.get_sent_length_stats)
        pronoun_df = pd.DataFrame(pronoun_counts_dict)
        sent_length_df = pd.DataFrame(sent_length_dict)

        n_1sg = [d.get("1sg") for d in pronoun_df.segment]
        n_1pl = [d.get("1pl") for d in pronoun_df.segment]
        n_2 = [d.get("2") for d in pronoun_df.segment]
        n_3 = [d.get("3") for d in pronoun_df.segment]

        n_words = [d.get("n_words") for d in sent_length_df.segment]
        n_sents = [d.get("n_sents") for d in sent_length_df.segment]
        mean_sent_length = [d.get("mean_sent_length") for d in sent_length_df.segment]
        std_sent_length = [d.get("std_sent_length") for d in sent_length_df.segment]

        # Add new series to feature_dict.
        # -> lexical features
        feature_dict["n_applauses"] = n_applauses
        feature_dict["n_crosstalks"] = n_crosstalks
        feature_dict["n_opponent_mentions"] = n_opponent_mentions
        feature_dict["n_trump_mentions"] = n_trump_mentions

        # -> semantic and syntactic features
        feature_dict["lexical_diversity"] = mtld
        feature_dict["n_difficult_words"] = n_difficult_words
        feature_dict["n_words_before_main_verb"] = n_words_before_main_verb
        feature_dict["readability"] = readability
        feature_dict["person_1sg"] = n_1sg
        feature_dict["person_1pl"] = n_1pl
        feature_dict["person_2"] = n_2
        feature_dict["person_3"] = n_3

        # -> length features
        feature_dict["n_words"] = n_words
        feature_dict["n_sents"] = n_sents
        feature_dict["mean_sent_length"] = mean_sent_length
        feature_dict["std_sent_length"] = std_sent_length
        feature_dict["segment_length"] = segment_length

        # Add manual features.
        # pronoun_count_dict = self.count_pronouns(segment)
        # sent_length_stats = self.get_sent_length_stats(segment)
        #
        # feature_dict = {
        #     # lexical and syntactic features
        #     "lexical_diversity": self.get_mtld(segment),
        #     "n_difficult_words": self.get_difficult_words_count(segment),
        #     "n_words_before_main_verb": self.get_n_words_before_main_verb(segment),
        #     "n_opponent_mentions": self.get_n_opponent_mentions(segment),
        #     "n_trump_mentions": self.get_trump_mentions(segment),
        #     # pronoun features
        #     "person_1sg": pronoun_count_dict["1sg"],
        #     "person_1pl": pronoun_count_dict["1pl"],
        #     "person_2": pronoun_count_dict["2"],
        #     "person_3": pronoun_count_dict["3"],
        #     "readability_score": self.get_readability_score(segment),
        #     # length features
        #     "segment_length": len(segment),
        #     "n_sents": sent_length_stats["n_sents"],
        #     "n_words": sent_length_stats["n_words"],
        #     "mean_sent_length": sent_length_stats["mean_sent_length"],
        #     "std_sent_length": sent_length_stats["std_sent_length"],
        # }

        return feature_dict
