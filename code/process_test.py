# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import Counter
import codecs
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import itertools
import pickle

""" Selecting file to encode based on parameters fed to the program"""
if len(sys.argv) < 2:
    filename = "test-stories.csv"
else:
    filename = sys.argv[1]

is_with_labels = filename != "test-stories.csv"

""" Selecting tokenizer based on json config"""
if len(sys.argv) < 3:
    tokenizer_json = None  # If nothing is specified, generate a new vocab!
else:
    tokenizer_json = sys.argv[2]

if tokenizer_json == None:
    print("You probably  did not intend to generate a new vocabulary for a file other than test_stories file. Exiting")
    print("Did you mean to specify: processed/sentences.train_vocab.npy ?")
    exit(1)

# generates an array of the lines present in the file
texts = []
with codecs.open("data/" + filename, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        if is_with_labels:
            texts.append(values[1:])
        else:
            texts.append(values)

MAX_NB_WORDS = 20000  # cause pad
MAX_SEQUENCE_LENGTH = 30

flattened = [item for sublist in texts for item in sublist]

if tokenizer_json is None:
    tokenizer = Tokenizer(MAX_NB_WORDS, oov_token='<unk>', filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(flattened)  # gen new
else:
    print(f"Loading existing tokenizer with {tokenizer_json}")
    with open(tokenizer_json, 'rb') as handle:
        tokenizer = pickle.load(handle)

word_index = tokenizer.word_index  # the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index))
sequences = []
labels = []
for story in texts:
    if is_with_labels:
        label = story[-1]
        story = story[:-1]
        labels.append(label)

    sequences.append(
        # tokenizer.texts_to_sequences(story)

        pad_sequences(
            tokenizer.texts_to_sequences(story), maxlen=MAX_SEQUENCE_LENGTH, padding='post'
        )
    )

# data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
word_index["<pad>"] = 0

# labels = np.expand_dims(np.expand_dims(labels, 0), 1)
data = np.array(sequences, dtype=int)
# data = np.concatenate([data, labels], axis=1)
np.save("data/processed/" + filename + "_vocab", dict(itertools.islice(word_index.items(), MAX_NB_WORDS)))
np.save("data/processed/" + filename, data)
if is_with_labels:
    np.save("data/processed/" + filename + "_labels", labels)

with open('data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
