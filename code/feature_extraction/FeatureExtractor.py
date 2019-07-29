from functools import reduce

import pandas as pd
import tensorflow as tf
from data_pipeline.data_utils import sentences_to_sparse_tensor as to_sparse
import numpy as np
import nltk
from definitions import ROOT_DIR
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
Inspired by https://aclweb.org/anthology/W17-0908
"""


class FeatureExtractor:

    def __init__(self, context):
        """

        :param context: First 4 sentences of the story. Array of strings.
        """
        self.context = context

    @staticmethod
    def generate_feature_records_train_set(tf_session,
                                           for_methods=("pronoun_contrast", "n_grams_overlap", "sentiment_analysis")):
        FeatureExtractor._generate_feature_records_set(
            tf_session,
            for_methods,
            "train_stories",
            ending_key="sentence5",
            sentence_key_suffix="sentence"
        )

    @staticmethod
    def generate_feature_records_eval_set(tf_session,
                                          for_methods=("pronoun_contrast", "n_grams_overlap", "sentiment_analysis")):
        FeatureExtractor._generate_feature_records_set(
            tf_session,
            for_methods,
            "eval_stories",
            ending_key="RandomFifthSentenceQuiz1",
            sentence_key_suffix="InputSentence"
        )
        FeatureExtractor._generate_feature_records_set(
            tf_session,
            for_methods,
            "eval_stories",
            ending_key="RandomFifthSentenceQuiz2",
            sentence_key_suffix="InputSentence"
        )

    def pronoun_contrast(self, ending):
        def get_pronouns(strings):
            return list(map(
                lambda word_and_tag: word_and_tag[0],
                filter(
                    lambda word_and_tag: "PRP" in word_and_tag[1],
                    nltk.pos_tag(nltk.word_tokenize(strings))
                )
            ))
        story_pronouns = get_pronouns(self._merged_story())
        ending_pronouns = get_pronouns(ending)
        ending_pronouns_matches = 0
        for story_pronoun in story_pronouns:
            if story_pronoun in ending_pronouns:
                ending_pronouns_matches += 1
        return tf.constant(ending_pronouns_matches, shape=[1])

    def n_grams_overlap(self, ending, ngram_range=range(1, 4), character_count=True):
        def preprocess(s):
            return s.strip().replace(".", "").lower().split(" ")
        ending = preprocess(ending)
        story = preprocess(self._merged_story())
        ending_grams = list(list(nltk.ngrams(ending, n)) for n in ngram_range)
        story_ngrams = list(nltk.ngrams(story, n) for n in ngram_range)

        # Checks for n-gram matchings.
        ending_ngram_overlaps = 0
        for n in range(len(ngram_range)):
            for story_gram in story_ngrams[n]:
                if story_gram in ending_grams[n]:
                    ending_ngram_overlaps += sum(len(w) for w in story_gram) if character_count else 1
        return tf.constant(ending_ngram_overlaps, shape=[1])

    def _merged_story(self):
        return reduce(lambda sen1, sen2: sen1 + " " + sen2, self.context)

    @staticmethod
    def _generate_feature_records_set(tf_session, for_methods, filename, ending_key, sentence_key_suffix):
        """
        Generates .tfrecords files containing the extracted features for all the endings in the file at
        the given filepath.
        :param for_methods: Feature extraction methods.
        :param ending_key: Name of a column containing endings.
        """
        data = pd.read_csv(f"{ROOT_DIR}/data/{filename}.csv", delimiter=",")
        features = {method: [] for method in for_methods}
        for ind, sentences in data.iterrows():
            story = list(sentences[f"{sentence_key_suffix}{i}"] for i in range(1, 5))
            ending = sentences[ending_key]
            fe = FeatureExtractor(story)
            for method in for_methods:
                if method == "pronoun_contrast":
                    features[method].append(fe.pronoun_contrast(ending))
                elif method == "n_grams_overlap":
                    features[method].append(fe.n_grams_overlap(ending))
                elif method == "sentiment_analysis":
                    features[method].append(fe.sentiment_analysis(ending, None))
                else:
                    raise NotImplementedError("Feature extraction method not implemented.")

        for method in for_methods:
            feature_values = features[method]
            # Evens out tensors' dimensions by padding with 0s
            with tf.python_io.TFRecordWriter(f'{ROOT_DIR}/data/features/{method}_{filename}_{ending_key.lower()}.tfrecords') as writer:
                # Writes a feature to a .tfrecords file
                def write_data(feature_val):
                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        "extracted_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=feature_val))
                    }))
                    writer.write(tf_example.SerializeToString())
                    return tf.constant([1])

                i = 0
                for feature_value in feature_values:
                    write_data(feature_value)
                    i += 1
                    print(f"{len(feature_values) - i} remaining")

    def sentiment_analysis(self, ending1, ending2):
        analyzer = SentimentIntensityAnalyzer()
        score = []
        for stc in self.context + [ending1, ending2]:
            vs = analyzer.polarity_scores(stc)
            score.append(list(vs.values()))
        score = np.array(score)

        return score


def save_all_features():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            FeatureExtractor.generate_feature_records_train_set(sess)


def test_ngrams():
    with tf.Graph().as_default():
        story = ["My man took a hat.", "He gave me the hat."]
        ending1 = "I took the hat."
        ending2 = "I tame tigers with hat."
        sess = tf.Session()
        with sess.as_default():
            fe = FeatureExtractor(story)
            ending1_ngram_overlaps = fe.n_grams_overlap(ending1)
            ending2_ngram_overlaps = fe.n_grams_overlap(ending2)
            init = tf.global_variables_initializer()
            sess.run(init)
            print(ending1_ngram_overlaps.eval())
            print(ending2_ngram_overlaps.eval())


def test_pronoun_contrast():
    with tf.Graph().as_default():
        story = ["The man saw a boat.", "He bought it."]
        ending1 = "He then proceeded to sail the boat"
        ending2 = "She then proceeded to sail the boat"
        sess = tf.Session()
        with sess.as_default() as default_sess:
            fe = FeatureExtractor(story)
            mis1 = fe.pronoun_contrast(ending1)
            mis2 = fe.pronoun_contrast(ending2)
            print(mis1.eval())
            print(mis2.eval())
