import collections
import math
import random


# averaging the embeddings between 2 words
# return the averaged embeddings
def average_two_embeddings_vectors(a, b):
    avg_embeddings = []
    i = 0
    for embed in a:
        z = (embed + b[i]) / 2.0
        avg_embeddings.append(z)
        i += 1

    return avg_embeddings


# helper func; updates tokens and embeddings with the new combined tokens and averaged embeddings
# return the updated tokens string and embeddings vector
def update_tok_and_embed(tokens, embeddings, index, embed2_index, averaged_embeddings):
    # update tokens
    if embed2_index > index:
        tokens[index] = tokens[index] + " " + tokens[embed2_index]
    else:
        tokens[index] = tokens[embed2_index] + " " + tokens[index]

    # update embeddings
    embeddings[index] = averaged_embeddings

    # delete old tokens and embeddings
    del tokens[embed2_index]
    del embeddings[embed2_index]

    return tokens, embeddings


# helper func
def preprocessing_helper(tokens, embeddings, e, word_list):
    index = 0
    avg_embeddings = []

    index = tokens.index(e)
    first, last = False, False
    if (index - 1) == -1:
        first = True
    if (index + 1) == len(tokens):
        last = True

    embed1 = embeddings[index]
    embed2 = []
    embed2_index = 0

    # the words following these type of words usually have some relation syntactically and semantically
    if word_list == "determiners" or word_list == "prepositions" or word_list == "helping_verbs":
        if last:  # check if element is the last element
            return tokens, embeddings
        embed2_index = index + 1
        embed2 = embeddings[embed2_index]

    else:  # the words before
        if first:  # check if first element
            return tokens, embeddings
        embed2_index = index - 1
        embed2 = embeddings[embed2_index]

    averaged_embeddings = average_two_embeddings_vectors(embed1, embed2)
    return update_tok_and_embed(tokens, embeddings, index, embed2_index, averaged_embeddings)


# common tokens that might fit well with other tokens based on syntactic rules of english
# therefore, standardize with these before running the default algorithm
# return updated tokens and embeddings
def syntactic_rules_for_preprocessing(tokens, embeddings, std_length):
    # not comprehensive but a start.
    determiners = ["a", "an", "the", "some", "each", "all"]
    punctuations = [",", ".", "!", "?", "...", ";", "-", "~"]
    prepositions = ["to", "for", "in", "on", "of"]
    helping_verbs = ["have", "has", "is", "was", "were", "will"]

    if len(tokens) > std_length:
        for e in tokens:
            if len(tokens) == std_length:
                return tokens, embeddings

            if e in determiners:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "determiners")
                continue
            if e in punctuations:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "punctuations")
                continue
            if e in prepositions:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "prepositions")
                continue
            if e in helping_verbs:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "helping_verbs")
                continue

    return tokens, embeddings


# takes in tokens list and corresponding embeddings
# shortens the list until the specified length(default 10)
# shortens by averaging the embedding vectors and combining the corresponding tokens
# combined tokens separated by a space even if it's punctuation. e.g. 'end' + '.' -> "end ."
# returns the standardized tokens and embeddings lists
# implementation: averaging some words that might go together first (e.g. "the cat", "to her")
# then, just randomly select adjacent tokens and average those embedding vectors
def standardize_by_averaging(tokens, embeddings, std_length=10):
    flag = True

    while len(tokens) > std_length:
        # error catcher
        if len(tokens) == 1:
            print("error: nothing to decrease/average\nCan't be less than 1 token")
            return tokens, embeddings

        # attempt to standardize with some regards to syntactical knowledge first
        if flag:
            flag = False
            tokens, embeddings = syntactic_rules_for_preprocessing(tokens, embeddings, std_length)
            continue

        length = len(tokens)
        index = random.randint(1, length-1)  # uses randomizer so to vary the averaging place

        embed1 = embeddings[index]
        embed2 = embeddings[index - 1]

        averaged_embeddings = average_two_embeddings_vectors(embed1, embed2)
        token, embeddings = update_tok_and_embed(tokens, embeddings, index, index-1, averaged_embeddings)

    return tokens, embeddings


def main():
    # fill
    tokens = ["this", "is", "a", "sentence", "that", "is", "over", "ten", "embeddings",
              "long", "and", "that", "there", "are", "punctuations", "."]
    embeddings = [[1.2, 3.34], [2.3, 3.5], [5.6, 6.6], [5.1, 2.3], [2.3, 4.4], [3.3, 5.8], [8.8, 7.7], [1.1, 2.3],
                  [9.9, 1.2], [2.1, 2.1], [1.0, 1.0], [1.1, 3.4], [1.2, 3.2], [3.4, 4.0], [1.1, 2.3], [1.1, 1.1]]

    # for testing purposes
    print("before tokens: ", tokens)  # before standardizing
    print("before embeddings: ", embeddings)

    tokens, embeddings = standardize_by_averaging(tokens, embeddings)
    print("after tokens: ", tokens)  # after standardizing
    print("after embeddings: ", embeddings)
    return


if __name__ == "__main__":
    # execute only if run as a script
    main()
