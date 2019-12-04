import numpy as np
import re
import spacy
from textstat import textstat
from lexicalrichness import LexicalRichness

from presidential.utils import CANDIDATE_NAMES


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
        self.nlp = spacy.load("en_core_web_md")
        self.neg_words = [
            "neither",
            "never",
            "no",
            "nobody",
            "none",
            "nothing",
            "nowhere",
            "hardly",
            "seldom",
        ]
        self.future_words = ["tomorrow", "future", "futures"]

    @staticmethod
    def count_pronouns(doc):
        segment = doc.text.lower().split()
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
    def get_mtld(doc):
        segment = doc.text
        # measure of textual lexical diversity (McCarthy 2005, McCarthy and
        # Jarvis 2010)
        if re.search("[a-zA-Z]", segment) is None:  # e.g., "2", "223)."
            return 0
        else:
            return LexicalRichness(segment).mtld(threshold=0.72)

    def get_n_negative_words(self, doc):
        neg_deps = [t for t in doc if t.dep_ == "neg"]
        other_neg_words = [t for t in doc if t.lower_ in self.neg_words]
        return len(neg_deps) + len(other_neg_words)

    def get_n_future_oriented_words(self, doc):
        will_aux = [
            t
            for t in doc
            if t.tag_ == "MD" and t.lower_ in {"will", "wo", "shall", "sha"}
        ]
        going_to = [t for t in doc if t.dep_ == "xcomp" and t.head.lemma_ == "go"]
        other_future_words = [t for t in doc if t.lower_ in self.future_words]
        return len(will_aux) + len(going_to) + len(other_future_words)

    @staticmethod
    def get_n_word_mentions(doc):
        segment = doc.text
        word_mentions_dict = {
            "n_applauses": segment.count("APPLAUSE"),
            "n_crosstalks": segment.count("CROSSTALK"),
            "n_trump_mentions": segment.count("Trump"),
            "n_other_candidate_mentions": sum(
                [segment.count(c.title()) for c in CANDIDATE_NAMES]
            ),
        }
        return word_mentions_dict

    @staticmethod
    def get_n_words_before_main_verb(doc):
        numbers = [0]
        for sent in doc.sents:
            main = [t for t in sent if t.dep_ == "ROOT"][0]
            if main.pos_ == "VERB":
                dist_to_init = main.i - sent[0].i
                numbers.append(dist_to_init)
        return np.mean(numbers)

    @staticmethod
    def get_n_complexes_clauses(doc):
        embedded_elements_count = []
        for sent in doc.sents:
            n_embedded = len(
                [t for t in sent if t.dep_ in {"ccomp", "xcomp", "advcl", "dative"}]
            )
            embedded_elements_count.append(n_embedded)
        return np.mean(embedded_elements_count)

    def get_readability_scores(self, doc):
        segment = doc.text
        readability_dict = {
            "automated_readability_index": textstat.automated_readability_index(
                segment
            ),
            "coleman_liau_index": textstat.coleman_liau_index(segment),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(
                segment
            ),
            "difficult_words": textstat.difficult_words(segment),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(segment),
            "flesch_reading_ease": textstat.flesch_reading_ease(segment),
            "gunning_fog": textstat.gunning_fog(segment),
            "linsear_write_formula": textstat.linsear_write_formula(segment),
            "smog_index": textstat.smog_index(segment),
            "text_standard": self._convert_text_standard_to_integer(
                textstat.text_standard(segment)
            ),
        }
        return readability_dict

    @staticmethod
    def _convert_text_standard_to_integer(text_standard):
        return np.mean([int(d) for d in re.findall(r"-?\d+", text_standard)])

    @staticmethod
    def get_sent_length_stats(doc):
        token_counts = [len(sent) for sent in doc.sents]
        sent_length_dict = {
            "n_tokens": len(doc),
            "n_sents": len(list(doc.sents)),
            "mean_sent_length": np.mean(token_counts),
            "std_sent_length": np.std(token_counts),
        }
        return sent_length_dict

    # @staticmethod
    # def get_spacy_vector(doc):
    #     # vector = 0
    #     doc = nlp(doc)
    #     # for tok in doc:
    #     #     vector += tok.vector
    #     return doc.vector

    def featurize(self, segments):
        feature_dicts = []
        docs = self.nlp.pipe(segments)
        for doc in docs:
            # Add manual features.
            pronoun_count_dict = self.count_pronouns(doc)
            sent_length_stats = self.get_sent_length_stats(doc)
            word_mentions_dict = self.get_n_word_mentions(doc)
            readability_dict = self.get_readability_scores(doc)

            feature_dict = {
                # interaction features
                "n_applauses": word_mentions_dict["n_applauses"],
                "n_crosstalks": word_mentions_dict["n_crosstalks"],
                # lexical features
                "n_other_candidate_mentions": word_mentions_dict[
                    "n_other_candidate_mentions"
                ],
                "n_trump_mentions": word_mentions_dict["n_trump_mentions"],
                "lexical_diversity": self.get_mtld(doc),
                "n_negative_words": self.get_n_negative_words(doc),
                "n_future_oriented_words": self.get_n_future_oriented_words(doc),
                # pronoun features
                "person_1sg": pronoun_count_dict["1sg"],
                "person_1pl": pronoun_count_dict["1pl"],
                "delta_1st_person": pronoun_count_dict["1sg"]
                - pronoun_count_dict["1pl"],
                "person_2": pronoun_count_dict["2"],
                "person_3": pronoun_count_dict["3"],
                # syntactic complexity
                "n_words_before_main_verb": self.get_n_words_before_main_verb(doc),
                "n_complexes_clauses": self.get_n_complexes_clauses(doc),
                # readability
                "n_difficult_words": readability_dict["difficult_words"],
                "automated_readability_index": readability_dict[
                    "automated_readability_index"
                ],
                "coleman_liau_index": readability_dict["coleman_liau_index"],
                "dale_chall_readability": readability_dict[
                    "dale_chall_readability_score"
                ],
                "flesch_kincaid_grade": readability_dict["flesch_kincaid_grade"],
                "flesch_reading_ease": readability_dict["flesch_reading_ease"],
                "gunning_fog": readability_dict["gunning_fog"],
                "linsear_write_formula": readability_dict["linsear_write_formula"],
                "smog_index": readability_dict["smog_index"],
                "text_standard": readability_dict["text_standard"],
                # length features
                "segment_length": len(doc),
                "n_sents": sent_length_stats["n_sents"],
                "n_tokens": sent_length_stats["n_tokens"],
                "mean_sent_length": sent_length_stats["mean_sent_length"],
                "std_sent_length": sent_length_stats["std_sent_length"],
            }
            feature_dicts.append(feature_dict)

        return feature_dicts
