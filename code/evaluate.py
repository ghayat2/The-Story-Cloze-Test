import datetime
import functools
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from definitions import ROOT_DIR
import pandas as pd
from data_pipeline.generate_combined import create_story

from models.bidirectional_lstm import BiDirectional_LSTM

""" Selecting the adequate experiment and checkpoint file to evaluate based on arguments fed to the program """

LSTM_SIZE = 1000
CONTEXT_LENGTH = 4
NB_ENDINGS = 2
NB_SENTENCES = CONTEXT_LENGTH + NB_ENDINGS

if len(sys.argv) < 2:
    raise AssertionError("Please specify the checkpoint folder name")
CHECKPOINT_FILE = sys.argv[1]

"""Flags representing constants of our project """

# Data loading parameters
tf.flags.DEFINE_string("group_number", "19", "Our group number")
tf.flags.DEFINE_string("data_sentences_vocab_path", f"{ROOT_DIR}/data/processed/train_stories.csv_vocab.npy",
                       "Path to vocabulary file.")
# Test parameters
tf.flags.DEFINE_bool("predict", True, "If predicting labels for the test-stories.csv file or assessing the performance"
                                      "instead, using test_for_report-stories_labels.csv")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_bool("use_validation_set", False, "Calculating average accuracy on validation set")
tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{CHECKPOINT_FILE}/checkpoints/",
                       "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Model Parameters
tf.flags.DEFINE_integer("rnn_cell_size", LSTM_SIZE, "LSTM Size (default: 1000)")
tf.flags.DEFINE_string("rnn_cell", "LSTM", "Type of rnn cell")
tf.flags.DEFINE_boolean("use_skip_thoughts", True, "True if skip thoughts embeddings should be used")
tf.flags.DEFINE_integer("sentence_len", 30, "Length of sentence")
tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_string("attention", None,
                       'Attention type (add ~ Bahdanau, mult ~ Luong, None). Only for Roemmele ''models.')
tf.flags.DEFINE_integer("attention_size", 1000, "Attention size.")
tf.flags.DEFINE_bool("used_features", True, "If features were used during training.")
tf.flags.DEFINE_bool("use_pronoun_contrast", True, "Whether the pronoun contrast feature vector should be added to the"
                                                   " networks' input.")
tf.flags.DEFINE_bool("use_n_grams_overlap", True, "Whether the n grams overlap feature vector should be added to the "
                                                  "network's input.")
tf.flags.DEFINE_bool("use_sentiment_analysis", True, "Whether to use the sentiment intensity analysis (4 dimensional "
                                                     "vectors)")
tf.flags.DEFINE_integer("num_sentences_train", 5, "Number of sentences in training set (default: 5)")
tf.flags.DEFINE_integer("sentence_length", 30, "Sentence length (default: 30)")
tf.flags.DEFINE_integer("word_embedding_dimension", 100, "Word embedding dimension size (default: 100)")
tf.flags.DEFINE_integer("num_context_sentences", 4, "Number of context sentences")
tf.flags.DEFINE_integer("classes", 2, "Number of output classes")
tf.flags.DEFINE_integer("num_eval_sentences", 2, "Number of eval sentences")

tf.flags.DEFINE_integer("sentence_embedding_length", 4800, "Length of the sentence embeddings")

tf.flags.DEFINE_integer("num_neg_random", 3, "Number of negative random endings")
tf.flags.DEFINE_integer("num_neg_back", 2, "Number of negative back endings")
tf.flags.DEFINE_integer("ratio_neg_random", 4, "Ratio of negative random endings")
tf.flags.DEFINE_integer("ratio_neg_back", 2, "Ratio of negative back endings")

tf.flags.DEFINE_float("dropout_rate", 0.0, "Dropout rate")

tf.flags.DEFINE_string("path_embeddings", "data/wordembeddings-dim100.word2vec", "Path to the word2vec embeddings")
tf.flags.DEFINE_string("embeddings", "w2v", "embedding types. Options are: w2v, w2v_google, glove")

tf.flags.DEFINE_integer("hidden_layer_size", 100, "Size of hidden layer")
tf.flags.DEFINE_integer("rnn_num", 2, "Number of RNNs")
tf.flags.DEFINE_integer("feature_integration_layer_output_size", 100, "Number of outputs from the dense layer after the"
                                                                      " RNN cell that includes the features")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

""" Processing of testing data"""

# Load data
if FLAGS.predict:
    labels = None
    if FLAGS.use_skip_thoughts:
        filepath = f"{ROOT_DIR}/data/test-stories.csv"
    else:
        filepath = f"{ROOT_DIR}/data/processed/test-stories.csv.npy"
else:
    if FLAGS.use_skip_thoughts:
        if FLAGS.use_validation_set:
            filepath = f"{ROOT_DIR}/data/eval_stories.csv"
            labels = np.array(pd.read_csv(filepath, sep=',', usecols=["AnswerRightEnding"]).values).flatten()
        else:
            filepath = f"{ROOT_DIR}/data/test_for_report-stories_labels.csv"
            labels = np.array(pd.read_csv(filepath_or_buffer=filepath, sep=',', usecols=["AnswerRightEnding"]).values).flatten()
    else:
        if FLAGS.use_validation_set:
            filepath = f"{ROOT_DIR}/data/processed/eval_stories.csv.npy"
            labels = np.load(f"{ROOT_DIR}/data/processed/eval_stories.csv_labels.npy").astype(dtype=np.int32)
        else:
            filepath = f"{ROOT_DIR}/data/processed/test_for_report-stories_labels.csv.npy"
            labels = np.load(f"{ROOT_DIR}/data/processed/test_for_report-stories_labels.csv_labels.npy").astype(
                dtype=np.int32)
    labels -= 1


EMBEDDING_SIZE = 4800 if FLAGS.use_skip_thoughts else FLAGS.word_embedding_dimension
FEATURES_SIZE = 22 if FLAGS.used_features else 0

if FLAGS.use_skip_thoughts:
    x_test = pd.read_csv(filepath_or_buffer=filepath, sep=',',
                         usecols=["InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4",
                                  "RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2"]).values
else:
    x_test = np.load(filepath).astype(dtype=np.int32)

vocab = np.load(FLAGS.data_sentences_vocab_path, allow_pickle=True)  # vocab contains [symbol: id]
vocabLookup = dict((v, k) for k, v in vocab.item().items())  # flip our vocab dict so we can easy lookup [id: symbol]
vocabLookup[0] = '<pad>'

""" Evaluating model"""

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    # Placeholders for stories and labels
    if FLAGS.use_skip_thoughts:
        output_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]
    else:
        output_types = [tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32]
    sentence_length = FLAGS.sentence_embedding_length if FLAGS.use_skip_thoughts else FLAGS.sentence_length
    shapes = (
        [FLAGS.batch_size, 1, sentence_length * FLAGS.num_context_sentences] if FLAGS.use_skip_thoughts
        else [FLAGS.batch_size, FLAGS.num_context_sentences, sentence_length],
        [FLAGS.batch_size, 1, sentence_length],
        [FLAGS.batch_size, 1, sentence_length],
        [FLAGS.batch_size, 1, FEATURES_SIZE],
        [FLAGS.batch_size, 1, FEATURES_SIZE],
        [FLAGS.batch_size, 1, 2]
    )
    next_story = list(tf.placeholder(output_types[i], shape=shapes[i]) for i in range(len(output_types)))

    # Generate batches for one epoch
    def get_batch():
        num_batches_per_epoch = int((len(x_test) - 1) / FLAGS.batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * FLAGS.batch_size
            end_index = min((batch_num + 1) * FLAGS.batch_size, len(x_test))

            if labels is None:
                # dummy constant if we're predicting since we don't have the labels
                y = tuple(1 for _ in range(end_index - start_index))
            else:
                y = labels[start_index:end_index]
            yield tuple(x_test[start_index:end_index][0]) + tuple(y)


    # Creates the dataset
    if FLAGS.use_skip_thoughts:
        types = tuple(tf.string for _ in range(6)) + tuple([tf.int32])
    else:
        types = tuple(tf.int32 for _ in range(7))
    dataset = tf.data.Dataset.from_generator(get_batch, output_types=types) \
        .map(functools.partial(create_story, **{
            "use_skip_thoughts": bool(FLAGS.use_skip_thoughts),
            "vocabLookup": vocabLookup,
            "vocab": vocab
        })).batch(FLAGS.batch_size)
    #
    # create the iterator
    iterator = dataset.make_initializable_iterator()  # create the iterator
    next_batch = list(iterator.get_next())
    for i in range(len(shapes)):
        next_batch[i] = tf.reshape(next_batch[i], shape=shapes[i])
        next_batch[i].set_shape(shapes[i])
    network_input = {
        "context": next_batch[0],
        "ending1": next_batch[1],
        "ending2": next_batch[2],
        "features1": next_batch[3],
        "features2": next_batch[4]
    }

    next_batch_endings_y = tf.argmax(next_batch[5], axis=2, output_type=tf.int32)

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Creating the model
        network = BiDirectional_LSTM(sess, vocab, network_input, FEATURES_SIZE, FLAGS.attention,
                                     FLAGS.attention_size)
        eval_predictions, _, _ = network.build_model()
        if FLAGS.use_skip_thoughts:
            next_batch_endings_y = next_batch_endings_y[0]

        # Restore the variables without loading the meta graph!
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        # Collect the predictions here
        results = []

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(eval_predictions, next_batch_endings_y), dtype=tf.float32
            )
        )

        sess.run(iterator.initializer)

        handle = tf.placeholder(tf.string, shape=[])
        test_handle = sess.run(iterator.string_handle())

        it = 0
        while True:
            try:
                a = datetime.datetime.now()
                if FLAGS.predict:
                    keyword = "prediction:"
                    fetches = [eval_predictions]
                else:
                    keyword = "accuracy"
                    fetches = [accuracy]
                res = sess.run(fetches, feed_dict={
                    handle: test_handle
                })
                if FLAGS.use_skip_thoughts:
                    pred = eval_predictions.eval()[0]
                    actual = next_batch_endings_y.eval()[0]
                    res = int(pred == actual)
                else:
                    res = res[0][0] if FLAGS.predict else res[0]
                results.append(res)
                it += 1
                print(f"Iteration {it} - {datetime.datetime.now() - a} - {keyword}: {res}")
            except tf.errors.OutOfRangeError:
                break

if FLAGS.predict:
    with open(f"group{FLAGS.group_number}_predictions_{CHECKPOINT_FILE}.csv", 'w') as f:
        for i in range(len(results)):
            f.write(str(results[i]+1) + "\n")
else:
    # Only printing out the average accuracy
    avg = np.average(results)
    print(f"Avg accuracy: {avg}")
    with open(f"group{FLAGS.group_number}_accuracy_{CHECKPOINT_FILE}", "w") as f:
        f.write(str(avg))
