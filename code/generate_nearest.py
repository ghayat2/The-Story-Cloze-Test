# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:33:31 2019

@author: Gabriel Hayat
"""

import numpy as np
import functools
import tensorflow as tf
from data_pipeline import data_utils as d

CONTEXT_LENGTH = 4

class BackPicker:

    # Pick a random sample sentence
    def pick(self, context):
        return np.random.sample(context)
    

def augment_data(context, endings,
                 BackPicker = None): # Augment the data

    if BackPicker is not None:
        randomSentence = BackPicker.pick(context)
        endings = [endings[0], randomSentence]

    return context, endings

def get_data_iterator(sentences,
                        augment_fn=functools.partial(augment_data),
                        threads=5,
                        batch_size=1,
                        repeat_train_dataset=5):

    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices(sentences) \
        .map(d.split_sentences, num_parallel_calls=threads) \
        .map(augment_fn, num_parallel_calls=threads) \
        .batch(batch_size, drop_remainder=True) \
        .repeat(repeat_train_dataset)

    return dataset
