"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt

import time

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


def get_optimal_probs(vocabulary, tokens) -> np.ndarray:
    """Compute the optimal unigram probabilities based on token frequencies."""
    token_counts = {v: 0 for v in vocabulary}
    for token in tokens:
        if token in vocabulary:
            token_counts[token] += 1
        else:
            token_counts[None] += 1
    total_count = len(tokens)
    optimal_probs = np.array(
        [token_counts[token] / total_count for token in vocabulary]
    )
    return optimal_probs


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 500
    learning_rate = 0.01

    # start timer
    start_time = time.time()

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_values.append(loss.item())

    # End the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time
    print(f"Training time: {time_taken:.4f} seconds")

    # display results
    predicted_log_probs = torch.nn.LogSoftmax(0)(model.s).detach().numpy().flatten()
    predicted_probs = np.exp(predicted_log_probs)

    # compute optimal probabilities
    optimal_probs = get_optimal_probs(vocabulary, tokens)
    optimal_probs = torch.tensor(optimal_probs, dtype=torch.float32)

    # Transpose x
    x_T = x.T

    # Compute the minimum loss using PyTorch's log and sum
    min_loss = -torch.sum(x_T * torch.log(optimal_probs))

    # visualize final token probabilities
    indices = np.arange(len(vocabulary))
    bar_width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(
        indices - 0.5 * bar_width,
        predicted_probs,
        bar_width,
        label="Predicted Probabilities",
        color="b",
    )
    plt.bar(
        indices + 0.5 * bar_width,
        optimal_probs,
        bar_width,
        label="Optimal Probabilities",
        color="r",
    )

    vocabulary_labels = [
        str(token) if token is not None else "None" for token in vocabulary
    ]
    plt.xticks(ticks=np.arange(len(vocabulary)), labels=vocabulary_labels, rotation=90)
    plt.ylabel("Probability")
    plt.title("Final Token Probabilities")
    plt.legend()
    plt.show()

    # visualize min loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss", color="blue")
    plt.axhline(y=min_loss, color="red", linestyle="--", label="Minimum Possible Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()
