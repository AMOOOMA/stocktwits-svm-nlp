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
def preprocessing_helper(tokens, embeddings, e, combine_with):
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
    if combine_with == "after":
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
    combined_after_set = {"a", "an", "the", "some", "each", "all", "to", "for", "in", "on", "of", "about", "with",
                          "from", "at", "have", "has", "is", "are", "was", "were", "be", "been", "being", "should",
                          "would", "will", "do", "don't", "did", "no", "not", "my", "his", "her", "your", "their",
                          "our", "its", "whose", "go", "going", "went", "come", "came", "coming"}

    combined_before_set = {"him", "her", "them", "us", ",", ".", "!", "?", "...", ";", "-", "~"}

    if len(tokens) > std_length:
        for e in tokens:
            # average embeddings with the token that follows the current token
            if e in combined_after_set:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "after")
                if len(tokens) == std_length:
                    break
                continue
            # avg embedding with the token that precedes the current token
            elif e in combined_before_set:
                tokens, embeddings = preprocessing_helper(tokens, embeddings, e, "before")
                if len(tokens) == std_length:
                    break
                continue

    return tokens, embeddings


# takes in tokens list and corresponding embeddings
# shortens the list until the specified length(default 10)
# shortens by averaging the embedding vectors and combining the corresponding tokens
# combined tokens separated by a space even if it's punctuation. e.g. 'end' + '.' -> "end ."
# returns the standardized tokens and embeddings lists
# implementation: averaging some words that might go together first (e.g. "the cat", "to her")
# then, just randomly select tokens and their adjacent token and average those embedding vectors
def standardize_by_averaging(tokens, embeddings, std_length=10):
    flag = True

    # so as to not change the original lists
    tokens = tokens.copy()
    embeddings = embeddings.copy()

    while len(tokens) > std_length:
        # attempt to standardize with some regards to syntactical knowledge first
        if flag:
            flag = False
            tokens, embeddings = syntactic_rules_for_preprocessing(tokens, embeddings, std_length)
            continue

        length = len(tokens)
        index = random.randint(1, length - 1)  # uses randomizer so to vary the averaging place

        embed1 = embeddings[index]
        embed2 = embeddings[index - 1]

        averaged_embeddings = average_two_embeddings_vectors(embed1, embed2)
        token, embeddings = update_tok_and_embed(tokens, embeddings, index, index - 1, averaged_embeddings)

    return tokens, embeddings


def standardize_by_duplicating(tokens, embeddings, std_length=10):
    token_copy, embeddings_copy = tokens[:], embeddings[:]

    while len(tokens) < std_length:
        # duplicate the whole message once
        tokens += token_copy
        embeddings += embeddings_copy

    return standardize_by_averaging(tokens, embeddings, std_length)


def main():
    # fill
    long_tokens = ["this", "is", "a", "sentence", "that", "is", "over", "ten", "embeddings",
                   "long", "and", "that", "there", "are", "punctuations", ".",
                   "this", "is", "a", "sentence", "that", "is", "over", "ten", "embeddings",
                   "long", "and", "that", "there", "are", "punctuations", "."]

    long_tokens2 = [".", ".", "gonna", "be", "a", "long", "in", "order", "for", "the",
                    "testing", "of", "the", "code", ".", "there", "will", "be", "some", "weird",
                    "tokens", "hello", "this", "spellings", "to", "see", "how", "that's", "this", "will", "be", "the"]

    long_embeddings = [[1.2, 3.34], [2.3, 3.5], [5.6, 6.6], [5.1, 2.3], [2.3, 4.4], [3.3, 5.8], [8.8, 7.7], [1.1, 2.3],
                       [9.9, 1.2], [2.1, 2.1], [1.0, 1.0], [1.1, 3.4], [1.2, 3.2], [3.4, 4.0], [1.1, 2.3], [1.1, 1.1],
                       [1.2, 3.34], [2.3, 3.5], [5.6, 6.6], [5.1, 2.3], [2.3, 4.4], [3.3, 5.8], [8.8, 7.7], [1.1, 2.3],
                       [9.9, 1.2], [2.1, 2.1], [1.0, 1.0], [1.1, 3.4], [1.2, 3.2], [3.4, 4.0], [1.1, 2.3], [1.1, 1.1]]

    # for testing purposes
    print("test standardize_by_averaging")
    print("before; tokens:\n", long_tokens)  # before standardizing
    print("before; embeddings:\n", long_embeddings, "\n\n")

    tokens, embeddings = standardize_by_averaging(long_tokens, long_embeddings)
    print("after; tokens:\n", tokens)  # after standardizing
    print("after; embeddings:\n", embeddings, "\n\n")

    # test standardize_by_averaging #2, uses the same embeddings as test #1
    print("test standardize_by_averaging#2")
    print("before; tokens:\n", long_tokens2)  # before standardizing
    print("before; embeddings:\n", long_embeddings, "\n\n")

    tokens, embeddings = standardize_by_averaging(long_tokens2, long_embeddings)
    print("after; tokens:\n", tokens)  # after standardizing
    print("after; embeddings:\n", embeddings, "\n\n")

    # standardize by duplicating
    short_tokens = ["This", "is", "looking", "Bullish"]
    short_embeddings = [[1.2, 3.34], [2.3, 3.5], [5.6, 6.6], [5.1, 2.3]]

    # for testing purposes
    print("test standardize_by_duplicating")
    print("before; tokens:\n", short_tokens)  # before standardizing
    print("before embeddings:\n", short_embeddings, "\n\n")

    tokens, embeddings = standardize_by_duplicating(short_tokens, short_embeddings)
    print("after; tokens:\n", tokens)  # after standardizing
    print("after; embeddings:\n", embeddings, "\n\n")

    return


if __name__ == "__main__":
    # execute only if run as a script
    main()
