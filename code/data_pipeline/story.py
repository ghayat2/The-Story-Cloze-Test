import tensorflow as tf
import numpy as np


class Story:

    def __init__(self, context, ending1, ending2=None, features_ending_1=None, features_ending_2=None, labels=None):
        self.context = context
        self.ending1 = ending1
        self.ending2 = ending2
        self.features_ending_1 = features_ending_1
        self.features_ending_2 = features_ending_2
        self.labels = labels

    def set_labels(self, new_labels):
        self.labels = tf.constant(value=new_labels, dtype=tf.int32, name="labels")

    @property
    def real_ending(self):
        return tf.cond(
            tf.equal(self.labels[0], 1),
            true_fn=lambda: self.ending1,
            false_fn=lambda: self.ending2
        )

    @property
    def fake_ending(self):
        return tf.cond(
            tf.equal(self.labels[0], 0),
            true_fn=lambda: self.ending1,
            false_fn=lambda: self.ending2
        )

    def randomize_labels(self):
        if self.features_ending_1 is not None or self.features_ending_2 is not None:
            raise AssertionError("Features already computed. Cannot randomize labels.")
        if self.ending1 is None or self.ending2 is None:
            raise AssertionError("One ending missing. Cannot randomize labels")

        def real_ending_second():
            return tf.constant(value=(0, 1), dtype=tf.int32), self.fake_ending, self.real_ending

        def real_ending_first():
            return tf.constant(value=(1, 0), dtype=tf.int32), self.real_ending, self.fake_ending

        new_labels, new_ending1, new_ending2 = tf.cond(
            tf.less(tf.random_uniform(shape=[1])[0], 0.5),
            real_ending_first,
            real_ending_second
        )
        self.labels = new_labels
        self.ending1 = new_ending1
        self.ending2 = new_ending2
        return self
