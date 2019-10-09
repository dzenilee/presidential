import numpy as np
import re
import spacy
import textstat

from lexicalrichness import LexicalRichness

from presidential.preprocess import debate_df, DEM_CANDIDATES, REP_CANDIDATES
CANDIDATES = set(DEM_CANDIDATES + REP_CANDIDATES)


nlp = spacy.load("en_core_web_md")

segments = debate_df.segment

df = debate_df

first_singular = ["i", "my", "mine", "myself"]
first_plural = ["we", "our", "ours", "ourselves"]
second_singular = ["you", "your", "yours", "yourself"]
second_plural = ["you", "your", "yours", "yourselves"]
third_singular_masculine = ["he", "his", "him", "himself"]
third_singular_feminine = ["she", "her", "hers", "herself"]
third_plural = ["they", "their", "theirs", "themselves"]
FIRST_SECOND_PERSON = set(
    first_singular + first_plural + second_singular + second_plural
)
THIRD_PERSON = set(third_singular_feminine + third_singular_masculine + third_plural)


class Featurizer:
    def __init__(self):
        pass

    @staticmethod
    def count_pronouns(text):
        text = text.lower()
        counter = {"1sg": 0, "1pl": 0, "2": 0, "3": 0}
        for pronoun in first_singular:
            tokens = re.findall(rf"\b{pronoun}", text)
            counter["1sg"] += len(tokens)
        for pronoun in first_plural:
            tokens = re.findall(rf"\b{pronoun}", text)
            counter["1pl"] += len(tokens)
        for pronoun in set(second_plural + second_singular):
            tokens = re.findall(rf"\b{pronoun}", text)
            counter["2"] += len(tokens)
        for pronoun in THIRD_PERSON:
            tokens = re.findall(rf"\b{pronoun}", text)
            counter["3"] += len(tokens)
        return counter

    @staticmethod
    def get_difficult_words_count(text):
        return textstat.difficult_words(text)

    @staticmethod
    def get_mtld(text):
        # measure of textual lexical diversity (McCarthy 2005, McCarthy and
        # Jarvis 2010)
        if re.search("[a-zA-Z]", text) is None:  # e.g., "2", "223)."
            return 0
        else:
            return LexicalRichness(text).mtld(threshold=0.72)

    @staticmethod
    def get_n_opponent_mentions(text):
        return sum(text.count(c) for c in CANDIDATES)

    @staticmethod
    def get_n_words_before_main_verb(text):
        total = 0
        doc = nlp(text)
        for sent in doc.sents:
            main = [t for t in sent if t.dep_ == "ROOT"][0]
            if main.pos_ == "VERB":
                dist_to_init = main.i - sent[0].i
                total += dist_to_init
        return total

    @staticmethod
    def get_readability_score(text):
        text = text
        scores = []
        scores.append(textstat.automated_readability_index(text))
        scores.append(textstat.coleman_liau_index(text))
        scores.append(textstat.flesch_kincaid_grade(text))
        scores.append(textstat.gunning_fog(text))
        return np.mean(scores)

    @staticmethod
    def get_sent_length_stats(text):
        word_count_per_sent = []
        doc = nlp(text)
        for sent in doc.sents:
            word_count_per_sent.append(len(sent.text.split()))
        sent_length_dict = {}
        sent_length_dict["n_words"] = len(text.split())
        sent_length_dict["n_sents"] = len(list(doc.sents))
        sent_length_dict["mean_sent_length"] = np.mean(word_count_per_sent)
        sent_length_dict["std_sent_length"] = np.std(word_count_per_sent)
        return sent_length_dict

    @staticmethod
    def get_spacy_vector(text):
        # vector = 0
        doc = nlp(text)
        # for tok in doc:
        #     vector += tok.vector
        return doc.vector

    # def featurize(self, text):
    #     pronoun_count_dict = self.count_pronouns(text)
    #     sent_length_stats = self.get_sent_length_stats(text)
    #     vector = self.get_spacy_vector(text)
    #
    #     # Add manual features.
    #     feature_dict = {
    #         "mtld": self.get_mtld(text),
    #         "n_difficult_words": self.get_difficult_words_count(text),
    #         "n_words_before_main_verb": self.get_n_words_before_main_verb(text),
    #         "person_1sg": pronoun_count_dict["1sg"],
    #         "person_1pl": pronoun_count_dict["1pl"],
    #         "person_2": pronoun_count_dict["2"],
    #         "person_3": pronoun_count_dict["3"],
    #         "readability_score": self.get_readability_score(text),
    #         "sent_length": sent_length_stats["length"],
    #         "sent_length_average": sent_length_stats["avg"],
    #         "sent_length_sd": sent_length_stats["sd"],
    #     }
    #
    #     # add vector embeddings
    #     # for i, dim in enumerate(vector):
    #     #     feature_dict.update({f"dim_{str(i)}": dim})
    #     return feature_dict
