"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""

from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(
    vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int
) -> List[str]:
    word_len = np.random.poisson(xi)
    # generate topic distribution from a Dirichlet distribution with parameter alpha randomly
    theta = np.random.dirichlet(alpha)
    # number of topics is the number of rows in beta
    num_topics = beta.shape[0]
    words = []
    for _ in range(word_len):
        # get topic randomly based on the dirichlet distribution
        topic = np.random.choice(num_topics, p=theta)
        word_index = np.random.choice(len(vocabulary), p=beta[topic])
        words.append(vocabulary[word_index])
    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass",
        "pike",
        "deep",
        "tuba",
        "horn",
        "catapult",
    ]
    beta = np.array(
        [
            [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
            [0.3, 0.0, 0.2, 0.3, 0.2, 0.0],
        ]
    )
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [lda_gen(vocabulary, alpha, beta, xi) for _ in range(100)]
    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())
    # get inferred beta from trained LDA
    inferred_beta = model.get_topics()
    for i, topic_dist in enumerate(inferred_beta):
        print(f"Topic {i}:")
        for word in vocabulary:
            word_id = dictionary.token2id[word]
            prob = topic_dist[word_id]
            print(f"{word}: {prob:.4f}")


if __name__ == "__main__":
    test()
