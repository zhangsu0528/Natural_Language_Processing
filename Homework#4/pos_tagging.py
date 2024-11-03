import nltk
from nltk.corpus import brown
import numpy as np
from viterbi import viterbi

nltk.download("brown")
nltk.download("universal_tagset")

# Load the first 10,000 tagged sentences using the 'universal' tagset
tagged_sentence = brown.tagged_sents(tagset="universal")[:10000]

# create mappings of states (POS taggings) to index and words to index
state_to_index = {}
# word to index would add words as "UNK" that were not in the corpus
word_to_index = {"UNK": 0}
for sentence in tagged_sentence:
    for word, tag in sentence:
        if tag not in state_to_index:
            state_to_index[tag] = len(state_to_index)
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

num_states = len(state_to_index)
num_words = len(word_to_index)

# Initialize matrices with add-1 smoothing
Transition = np.ones((num_states, num_states))
Observation = np.ones((num_states, num_words))
initial_probs = np.ones(num_states)
emission_counts = {tag: {} for tag in state_to_index}

# Count occurences and fill out the transition and observation matrix
for sentence in tagged_sentence:
    prev_tag = None
    for word, tag in sentence:
        if prev_tag is not None:
            Transition[state_to_index[prev_tag], state_to_index[tag]] += 1
        Observation[state_to_index[tag], word_to_index.get(word, 0)] += 1
        word_idx = word_to_index.get(word, 0)
        Observation[state_to_index[tag], word_idx] += 1
        # Update emission counts for the current word and tag
        if word not in emission_counts[tag]:
            emission_counts[tag][word] = 1
        else:
            emission_counts[tag][word] += 1
        prev_tag = tag

    # Count occurence of first word's state in every sentence
    first_state = sentence[0][1]
    initial_probs[state_to_index[first_state]] += 1

# Calculate probabilities of each entry in matrix
Transition = Transition / Transition.sum(axis=1, keepdims=True)
Observation = Observation / Observation.sum(axis=1, keepdims=True)
initial_probs = initial_probs / initial_probs.sum()

# Create reverse mappings for easier referencing
idx_to_state = {v: k for k, v in state_to_index.items()}
idx_to_obs = {v: k for k, v in word_to_index.items()}


# Define function to implement viterbi algorithms and tag a sentence with the best path
def tag_sentence(sentence, A, B, pi, obs_to_idx, idx_to_state):
    # create specific observation sequence ints for input sentence
    observation = [obs_to_idx.get(word.lower(), 0) for word in sentence]
    state_sequence, probability = viterbi(observation, pi, A, B)
    tags = [idx_to_state[idx] for idx in state_sequence]
    return list(tags), probability


if __name__ == "__main__":
    test_sentence = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
    # below is to generate true tags in the test sentence
    for tag, words in emission_counts.items():
        print(f"Emission counts for tag '{tag}': {words}")
    for i, sent in enumerate(test_sentence, start=10150):
        words, true_tags = zip(*sent)
        # below is predicted sentence with simulated tags by viterbi algorithm
        predicted_sentence, probability = tag_sentence(
            words, Transition, Observation, initial_probs, word_to_index, idx_to_state
        )
        print(f"\nSentence {i}:")
        print("Words:", " ".join(words))
        print("True tags:", true_tags)
        print("Predicted tags:", predicted_sentence)
        print(
            "Accuracy:",
            sum(t == p for t, p in zip(true_tags, predicted_sentence)) / len(true_tags),
        )
