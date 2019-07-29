import pandas

from definitions import ROOT_DIR
from embedding.sentence_embedder import SkipThoughtsEmbedder as SentenceEmbedder
from generation.ending_generator import EndingGenerator
import numpy as np


class NearGeneration(EndingGenerator):

    def __init__(self,
                 sentence_embeddings,
                 embeddings_hashable=False,
                 distance_function=lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                 *args, **kwargs):
        """
        :param sentence_embeddings: A vector of sentence embeddings of any length.
        :param embeddings_hashable: If the sentence embeddings are hashable python objects. If not, they'll be transformed to
        tuples on the fly.
        """
        super(NearGeneration, self).__init__(*args, **kwargs)
        self.sentence_embeddings = sentence_embeddings
        self.distances = {}
        self.embeddings_hashable = embeddings_hashable
        self.encoder = None
        self.dist_function = distance_function

    def generate_ending(self,
                        correct_ending,
                        optimal_endings_distance=0.78409,
                        is_encoded=True,
                        is_hashable=False):
        """
        :param correct_ending: A correct story ending.
        :param dist_function: A distance function to apply to the embeddings.
        :param optimal_endings_distance: The ideal distance between :correct_ending and the returned generated ending.
        :param is_encoded: If correct_ending is already an embedded vector. If this is false, then skip thoughts
        is used to encode it.
        :param is_hashable: If correct_ending is hashable
        :return: An ending from self.sentence_embeddings that is as close as possible to the given optima distance.
        """
        # Embed the ending if it isn't already
        if not is_encoded:
            correct_ending = self._get_encoder().encode(correct_ending)[0]
        # Ensures the ending is a hashable object
        if not is_encoded or not is_hashable:
            correct_ending = tuple(correct_ending)
        # Checks if we have already calculated some distances for this ending
        if not(correct_ending in self.distances):
            self.distances[correct_ending] = {}
        # Retrieves the distances
        ending_distances = self.distances[correct_ending]
        # Starts calculating or simply retrieving the distances if they've already been computed
        closest_to_optimal = None
        best_dist_from_optimal = None
        for sentence_embedding in self.sentence_embeddings:
            if not self.embeddings_hashable:
                sentence_embedding = tuple(sentence_embedding)
            if sentence_embedding != correct_ending:
                if sentence_embedding not in ending_distances:
                    # Computes distance from correct ending to current sentence embedding
                    ending_distances[sentence_embedding] = self.dist_function(correct_ending, sentence_embedding)
                # l1 norm between optimal distance and the actual distance for this sentence embedding
                distance_to_optimal = abs(optimal_endings_distance - ending_distances[sentence_embedding])
                # Takes the vector having the closest to optimal distance
                if (closest_to_optimal is None) or (best_dist_from_optimal is None) or (distance_to_optimal < best_dist_from_optimal):
                    closest_to_optimal = sentence_embedding
                    best_dist_from_optimal = distance_to_optimal
        return closest_to_optimal

    def get_evaluation_set_avg_distance(self):
        eval_set = pandas.read_csv(ROOT_DIR + '/data/eval_stories.csv', header=0)
        avg_distance = 0.0
        set_size = eval_set.shape[0]
        for i, sentences in eval_set.iterrows():
            dist = self.dist_function(*self._get_encoder().encode([sentences["RandomFifthSentenceQuiz1"],
                                                                    sentences["RandomFifthSentenceQuiz2"]]))
            avg_distance += dist / set_size
        return avg_distance

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = SentenceEmbedder()
        return self.encoder


def test_distances():
    sentences = [
        "Tyler has released a new album called Igor.",
        "I've been listening to it all the time ever since.",
        "I like strawberries.",
        "Pizzas sometimes come with pineapple."
    ]
    embedder = SentenceEmbedder()
    embedded_sentences = embedder.encode(sentences)
    generator = NearGeneration(
        embedded_sentences
    )
    false_ending = generator.generate_ending(embedded_sentences[0])
    for i in range(len(embedded_sentences)):
        if abs(sum(embedded_sentences[i]) - sum(false_ending)) < 1e-2:
            print(sentences[i])


def test_avg_distance():
    ng = NearGeneration(sentence_embeddings=None)
    print(ng.get_evaluation_set_avg_distance())  # prints 0.7840946715410734


def test_eval_dataset():
    eval_dataset = np.load(ROOT_DIR + "/data/processed/eval_stories_skip_thoughts.npy")[:, 4, :]
    sentences = pandas.read_csv(ROOT_DIR+"/data/eval_stories.csv")["RandomFifthSentenceQuiz1"]
    print(sentences.iloc[480])
    ng = NearGeneration(sentence_embeddings=eval_dataset)
    new_ending = ng.generate_ending(eval_dataset[480], is_encoded=True)
    for i in range(len(eval_dataset)):
        if abs(sum(eval_dataset[i]) - sum(new_ending)) < 1e-2:
            print(sentences.iloc[i])
