import tensorflow as tf

from definitions import ROOT_DIR


class FeatureReader:
    """
    Loads save .tfrecords containing features.
    """

    def load_n_grams_overlap_train(self):
        return self._load_n_grams_overlap("train_stories", "sentence5")

    def load_n_grams_overlap_eval(self):
        return (self._load_n_grams_overlap("eval_stories", f"randomfifthsentencequiz{i}") for i in (1, 2))

    def load_pronoun_contrast_train(self):
        return self._load_pronoun_contrast("train_stories", "sentence5")

    def load_pronoun_contrast_eval(self):
        return (self._load_pronoun_contrast("eval_stories", f"randomfifthsentencequiz{i}") for i in (1, 2))

    def _load_n_grams_overlap(self, ds_type, ending_key):
        return self._load_features(f"{ROOT_DIR}/data/features/n_grams_overlap_{ds_type}_{ending_key}.tfrecords")

    def _load_pronoun_contrast(self, ds_type, ending_key):
        return self._load_features(f"{ROOT_DIR}/data/features/pronoun_contrast_{ds_type}_{ending_key}.tfrecords")

    def _load_features(self, features_filepath):
        def extract_fn(data_record):
            return tf.parse_single_example(data_record, self._get_feature_name())
        eval_set = tf.data.TFRecordDataset(features_filepath)
        return eval_set.map(extract_fn)

    @staticmethod
    def _get_feature_name():
        return {
            "extracted_feature": tf.VarLenFeature(dtype=tf.int64)
        }


def test_loading_features():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            (ds, _) = FeatureReader().load_pronoun_contrast_eval()

            it = ds.make_one_shot_iterator()

            while True:
                try:
                    print(it.get_next()["extracted_feature"].eval())
                except tf.errors.OutOfRangeError:
                    break


test_loading_features()
