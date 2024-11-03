import csv


# create unigram probabilities dictionary from the unigrams.csv
def unigram_model():
    unigram_counts = {}
    total_count = 0
    with open("unigrams.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            char, count = row
            unigram_counts[char] = int(count)
            total_count += int(count)

    return unigram_counts


# create bigram probabilities dictionary from the bigrams.csv
def bigram_model():
    bigram_counts = {}
    total_counts = 0
    with open("bigrams.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            char, count = row
            bigram_counts[char] = int(count)
            total_counts += int(count)

    return bigram_counts


# create error probabilities dictionary from csv
def error_model(filename):
    error_counts = {}
    total_count = 0
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            char1, char2, count = row
            error_counts[char1 + char2] = int(count)
            total_count += int(count)

    return error_counts


# get_edits function primarily used to generate all the possible candidates
def get_edits(original: str, characters: list[str]) -> list[tuple[str, str]]:
    edits = []

    # generate deletions
    for idx, char in enumerate(original):
        previous_char = original[idx - 1] if idx > 0 else "#"
        edits.append((f"d:{previous_char}{char}", original[:idx] + original[idx + 1 :]))

    # generate substitutions
    for idx, old_char in enumerate(original):
        for new_char in characters:
            edits.append(
                (
                    f"s:{old_char}{new_char}",
                    original[:idx] + new_char + original[idx + 1 :],
                )
            )

    # generate additions
    for idx, char in enumerate("#" + original):
        for new_char in characters:
            edits.append(
                (
                    f"a:{char}{new_char}",
                    original[:idx] + new_char + original[idx:],
                )
            )

    return edits


# create p(word) dictionary from count_1w.txt
def prob_words():
    word_counts = {}
    total_count = 0
    with open("count_1w.txt", "r") as file:
        for line in file:
            word, count = line.split()
            word_counts[word] = int(count)
            total_count += int(count)
    word_probs = {}
    for word, count in word_counts.items():
        word_probs[word] = count / total_count
    return word_probs


# calculate p(x|w)*p(w) for observed word
def error_probs(observed_word):
    error_probs_dict = {}
    delete_counts = error_model("deletions.csv")
    add_counts = error_model("additions.csv")
    sub_counts = error_model("substitutions.csv")
    bigram_counts = bigram_model()
    unigram_counts = unigram_model()
    all_letters = [chr(i) for i in range(ord("a"), ord("z") + 1)]
    # generate all the possible correct candidates
    # with combinations of all possible letters and three types of errors
    correct_candidates = get_edits(observed_word, all_letters)
    for char, corrected_word in correct_candidates:
        if corrected_word in prob_words().keys():
            p_word = prob_words().get(corrected_word)
            gen_type, error_str = char.split(":")
            # the corrected candidate is after deletion, meaning the error type should be addition instead
            if gen_type == "d" and error_str in add_counts.keys():
                # calculate p(x|w)
                # when the prefix did not exist
                # we assume a very low probability to p(x|w)
                if error_str[0] == "#":
                    p_x_w = 0.0000000001
                else:
                    p_x_w = add_counts.get(error_str) / unigram_counts.get(error_str[0])
            #  the corrected candidate is after addition, meaning the error type should be deletion instead
            elif gen_type == "a" and error_str in delete_counts.keys():
                # calculate p(x|w)
                # when the prefix did not exist
                # we assume a very low probability to p(x|w)
                if error_str[0] == "#":
                    p_x_w = 0.0000000001
                else:
                    p_x_w = delete_counts.get(error_str) / bigram_counts.get(error_str)
            elif gen_type == "s" and error_str[::-1] in sub_counts.keys():
                reverse_str = error_str[::-1]
                p_x_w = sub_counts.get(reverse_str) / unigram_counts.get(reverse_str[0])
            # calculate p(x|w) * p(w)
            p_error = p_x_w * p_word
            error_probs_dict[corrected_word] = p_error
    # print(error_probs_dict)
    return error_probs_dict


# define correct function to find the maximum probability in the dictionary
def correct(original):
    error_dict = error_probs(original)
    max_key = max(error_dict, key=error_dict.get)
    return max_key


if __name__ == "__main__":
    print("------Below are scenarios that worked well in this model------")
    print(correct("acress"))
    print(correct("helo"))
    print(correct("acommodate"))
    print(correct("lovr"))
    print("------Below are scenarios that did not work well in this model------")
    print(correct("survelliance"))
    print(correct("collaegue"))
    print(correct("dilaogue"))
