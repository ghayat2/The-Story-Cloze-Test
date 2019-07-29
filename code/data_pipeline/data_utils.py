# Data utilities
from gensim import models
from gensim.scripts.glove2word2vec import glove2word2vec

import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
CONTEXT_LENGTH = 4


def make_symbol_story(array, vocabLookup):
    return [make_symbols(s, vocabLookup) for s in array.tolist()]


def make_symbols(array, vocabLookup):
    """
    Convert array of integers into a sentence based on the dic argument
    """
    conv = list(vocabLookup[x] for x in array)
    filtered = filter(lambda x: x != '<pad>', conv)  # For readability
    return list(filtered)


def endings(sentences):
    return [split_sentences(sentence)[1][0] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH, :], sentences[CONTEXT_LENGTH:, :]


def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    """
          session        Tensorflow session object
          vocab          A dictionary mapping token strings to vocabulary IDs
          emb            Embedding tensor of shape vocabulary_size x dim_embedding
          path           Path to embedding file
          dim_embedding  Dimensionality of the external embedding.
        """


    if FLAGS.embeddings == "w2v":
        print("Loading w2v")
        model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    elif FLAGS.embeddings == "w2v_google":
        print("Loading w2v_google")
        model = models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)  
    elif FLAGS.embeddings == "glove":
        print("Loading glove")
        glove2word2vec(glove_input_file="data/glove.42B.300d.txt", word2vec_output_file="data/gensim_glove_vectors.txt")
        model = models.KeyedVectors.load_word2vec_format("data/gensim_glove_vectors.txt", binary=False)
    else:
        print("Unknown embeddings...")
        exit(1)

    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.item().items():
        if tok in model.vocab and tok != "<pad>":
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            if tok == "<pad>":
                external_embedding[idx] = np.zeros(dim_embedding)
            else:
                external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set


def sentences_to_sparse_tensor(sentences):
    separated_sentences = sentences.split(".")
    dim1 = len(separated_sentences)
    dim2 = 0
    indices = []
    values = []
    for i in range(len(separated_sentences)):
        words = separated_sentences[i].split()
        dim2 = max(dim2, len(words))
        for j in range(len(words)):
            indices.append((i, j))
            values.append(words[j])
    return tf.SparseTensor(values=values, indices=indices, dense_shape=(dim1, dim2))
