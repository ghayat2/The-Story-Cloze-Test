from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os


BASE_DIR = 'C:/Users/gianc/Desktop/PhD/Progetti/vae/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'#'train_micro.csv'
GLOVE_EMBEDDING = BASE_DIR + 'glove.6B.50d.txt'
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 15
MAX_NB_WORDS = 12000
EMBEDDING_DIM = 50

texts = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts.append(values[3])
        texts.append(values[4])
print('Found %s texts in train.csv' % len(texts))