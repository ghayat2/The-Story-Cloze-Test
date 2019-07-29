import functools
import tensorflow as tf
from data_pipeline.operations import augment_data, create_story, get_features, compute_sentence_embeddings, \
    split_sentences
from definitions import ROOT_DIR

FLAGS = tf.flags.FLAGS


def get_data_iterator(sentences,
                      story_creation_fn,
                      threads=5,
                      batch_size=1,
                      repeat_train_dataset=5):
    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices(sentences[:, :5, :]) \
        .map(lambda t: tf.unstack(t)) \
        .map(story_creation_fn, num_parallel_calls=threads) \
        .repeat(repeat_train_dataset) \
        .shuffle(buffer_size=FLAGS.train_shuffle_buffer_size) \
        .batch(batch_size, drop_remainder=True)

    return dataset


def get_skip_thoughts_data_iterator(story_creation_fn=create_story, threads=5, batch_size=1, repeat_train_dataset=5):
    csv_path = f"{ROOT_DIR}/data/train_stories.csv"

    train_stories = tf.data.experimental.CsvDataset(
        filenames=csv_path,
        record_defaults=[tf.string for _ in range(5)],
        select_cols=[2, 3, 4, 5, 6],
        field_delim=",",
        use_quote_delim=True,
        header=True
    )

    return train_stories \
        .map(story_creation_fn)\
        .repeat(repeat_train_dataset) \
        .shuffle(FLAGS.train_shuffle_buffer_size) \
        .batch(batch_size, drop_remainder=True)


def transform_labels_onehot(sentences, labels, threads=5):
    one_hot = tf.one_hot(labels, FLAGS.classes, dtype=tf.int32).map(split_sentences, num_parallel_calls=threads)
    return sentences, one_hot


def get_eval_iterator(stories,
                      labels,
                      story_creation_fn,
                      threads=5,
                      batch_size=1,
                      repeat_eval_dataset=5):
    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices((stories, labels)) \
        .map(lambda story, label: tf.unstack(story) + [label])\
        .map(functools.partial(story_creation_fn, use_skip_thoughts=False))\
        .shuffle(buffer_size=FLAGS.test_shuffle_buffer_size) \
        .repeat(repeat_eval_dataset) \
        .batch(batch_size, drop_remainder=True)
    return dataset


def get_skip_thoughts_eval_iterator(threads=5, batch_size=1, repeat_eval_dataset=5):
    from definitions import ROOT_DIR

    csv_path = f"{ROOT_DIR}/data/eval_stories.csv"

    train_stories = tf.data.experimental.CsvDataset(
        filenames=csv_path,
        record_defaults=[tf.string for _ in range(6)] + [tf.int32],
        select_cols=[1, 2, 3, 4, 5, 6, 7],
        field_delim=",",
        use_quote_delim=True,
        header=True
    )

    # Zips the embeddings with the labels
    return train_stories\
        .map(create_story)\
        .shuffle(buffer_size=FLAGS.test_shuffle_buffer_size) \
        .repeat(repeat_eval_dataset) \
        .batch(batch_size, drop_remainder=True)
