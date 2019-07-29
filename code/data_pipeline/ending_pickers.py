from definitions import ROOT_DIR
import tensorflow as tf
import pandas as pd
import numpy as np


FLAGS = tf.flags.FLAGS


class Picker:

    def pick(self, context):
        raise NotImplementedError()


class RandomPicker(Picker):

    # Initialize with a random dictionary of sentences
    def __init__(self, dictionary, length):
        self.dictionary = dictionary
        self.length = length

    # Pick a random sample sentence
    def pick(self, context):
        # picks = []
        rand_index = tf.random.uniform([1], 0, self.length, dtype=tf.int32)
        return tf.gather(self.dictionary, rand_index)


class BackPicker(Picker):

    # Pick a random sample sentence
    def pick(self, context):
        rand_index = tf.random.uniform([1], 0, FLAGS.num_context_sentences, dtype=tf.int32)
        return tf.gather(context, rand_index).numpy()[0].decode("utf-8")


class PlainRandomPicker(Picker):

    def __init__(self, *args, **kwargs):
        super(PlainRandomPicker, self).__init__(*args, **kwargs)
        csv_path = f"{ROOT_DIR}/data/train_stories.csv"
        self.dataset = pd.read_csv(csv_path, delimiter=",", usecols=["sentence5"], dtype=str, encoding='utf-8').values.flatten()

    def pick(self, N=1):
        return np.random.choice(self.dataset)
