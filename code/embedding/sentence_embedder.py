import sys
from math import ceil

import pandas
import tensorflow as tf
import embedding.skipthoughts as skipthoughts
import datetime
from definitions import ROOT_DIR

import numpy as np


# Uses implementation from: https://github.com/ryankiros/skip-thoughts
class SkipThoughtsEmbedder:

    def __init__(self, *args, **kwargs):
        super(SkipThoughtsEmbedder, self).__init__(*args, **kwargs)
        model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)

    def encode(self, data_to_encode, batch_size=1):
        return self.encoder.encode(data_to_encode, batch_size=batch_size, verbose=False)

    def generate_embedded_training_set(self, training_set_path, save_file_path):
        """
        Generates skip-thoughts embeddings for the training dataset.
        :param training_set_path: Path to the training dataset.
        :param save_file_path: Path to save the results to.
        """
        self._generate_embedded_set(training_set_path, save_file_path, 5, 2)

    def generate_embedded_eval_set(self, testing_set_path, save_file_path):
        """
        Generates skip-thoughts embeddings for the evaluation dataset.
        :param testing_set_path: Path to the testing dataset.
        :param save_file_path: Path to save the results to.
        """
        self._generate_embedded_set(testing_set_path, save_file_path, 6, 1)

    @staticmethod
    def similarity(vec1, vec2):
        """Cosine similarity."""
        vec1 = vec1.reshape((4800))
        vec2 = vec2.reshape((4800))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def get_eval_tf_dataset(eval_dataset_filepath=ROOT_DIR + "/data/processed/eval_stories_skip_thoughts.tfrecords",
                            embedding_size=4800):
        """

        :param eval_dataset_filepath: Path to a .tfrecords file of the evaluation dataset.
        :param embedding_size: Dimension of each sentence embedding (4800 is for bi skip-thoughts).
        :return: A Tensorflow Dataset object.
        """
        features = SkipThoughtsEmbedder.get_dataset_features(
            SkipThoughtsEmbedder.get_eval_dataset_features_str(),
            embedding_size
        )
        return SkipThoughtsEmbedder._get_tf_dataset(eval_dataset_filepath, features)

    @staticmethod
    def get_train_tf_dataset(train_dataset_filepath=ROOT_DIR + "/data/processed/train_stories_skip_thoughts.tfrecords",
                             embedding_size=4800):
        features = SkipThoughtsEmbedder.get_dataset_features(
            SkipThoughtsEmbedder.get_train_dataset_features_str(),
            embedding_size
        )
        return SkipThoughtsEmbedder._get_tf_dataset(train_dataset_filepath, features)

    @staticmethod
    def _get_tf_dataset(dataset_filepath, features):
        def extract_fn(data_record):
            return tf.parse_single_example(data_record, features)
        eval_set = tf.data.TFRecordDataset(dataset_filepath)
        return eval_set.map(extract_fn)

    @staticmethod
    def get_train_dataset_features_str():
        return list("sentence" + str(i) for i in range(1, 6))

    @staticmethod
    def get_eval_dataset_features_str():
        return "sentence1", "sentence2", "sentence3", "sentence4", "ending1", "ending2"

    @staticmethod
    def get_dataset_features(feature_names, embedding_size):
        def get_type():
            return tf.FixedLenFeature(shape=embedding_size, dtype=tf.float32)
        return dict(((feature_name, get_type()) for feature_name in feature_names))

    @staticmethod
    def npy_to_tfrecords(numpy_dataset, save_filepath, feature_names):
        """
        Converts a 3-dimensional numpy array into a .tfrecords file.
        :param numpy_dataset: A numpy array object.
        :param save_filepath: Path to save the .tfrecords file to.
        :param feature_names: Name of the features to create. See for example `get_eval_dataset_features_str()`
        """
        # write records to a tfrecords file
        writer = tf.python_io.TFRecordWriter(save_filepath)

        # Loop through all the features you want to write
        for record in numpy_dataset:
            # Feature contains a map of string to feature proto objects
            feature = {}

            def get_vector_type(X):
                return tf.train.Feature(float_list=tf.train.FloatList(value=X))

            i = 0
            for feature_name in feature_names:
                feature[feature_name] = get_vector_type(record[i])
                i += 1

            # Construct the Example proto object
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize the example to a string
            serialized = example.SerializeToString()

            # write the serialized object to the disk
            writer.write(serialized)
        writer.close()

    def _generate_embedded_set(self, set_path, save_file_path, nb_encodings_per_story, start_ind):
        dataset = pandas.io.parsers.read_csv(set_path).values
        embeddings = list()
        nbr_stories = dataset.shape[0]
        sys.stdout.write("Starting to encode " + str(nbr_stories) + " stories\n")
        batch_size = 1000
        for i in range(int(ceil(nbr_stories / batch_size))):
            a = datetime.datetime.now()
            ubound = min(batch_size, nbr_stories - i * batch_size)
            print(
                f"Encoding sentences {i * batch_size} to {i * batch_size + ubound - 1}. {nbr_stories - i * batch_size} remaining.")
            to_encode = np.array(
                dataset[i * batch_size:i * batch_size + ubound, start_ind:start_ind + nb_encodings_per_story])
            to_encode = to_encode.flatten()
            encodings = self.encoder.encode(
                to_encode,
                batch_size=ubound,
                verbose=False
            )
            encodings = encodings.reshape((ubound, nb_encodings_per_story, -1))
            for encoding in encodings:
                embeddings.append(encoding)
            b = datetime.datetime.now()
            sys.stdout.write(f"Time elapsed: {b - a}\n")
        np.save(save_file_path, np.array(embeddings))


def example_encode():
    embedder = SkipThoughtsEmbedder()
    s1 = embedder.encode(["My name is not what you think"])
    s2 = embedder.encode(["My username is different than what you think"])
    s4 = embedder.encode(["Beach or horses, give or take, life is full of extremes."])
    s3 = embedder.encode(["That is a totally unrelated sentence"])
    print("Similarity between s1 and s2: {}".format(embedder.similarity(s1, s2)))
    print("Similarity between s1 and s3: {}".format(embedder.similarity(s1, s3)))
    print("Similarity between s1 and s4: {}".format(embedder.similarity(s1, s4)))


# train = np.load(ROOT_DIR + "/data/processed/train_stories_skip_thoughts.npy")
# SentenceEmbedder.npy_to_tfrecords(
#     train,
#     ROOT_DIR + "/data/processed/train_stories_skip_thoughts.tfrecords",
#     SentenceEmbedder.get_train_dataset_features_str()
# )
#
# with tf.Session() as sess:
#     ds = SentenceEmbedder.get_train_tf_dataset()
#     iterator = ds.make_one_shot_iterator()
#     el = sess.run(iterator.get_next())
#     print(el)
