from tokenizer import Tokenizer

from nltk.stem import SnowballStemmer

from enum import Enum


class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"


def tokenize(message):
    """
    # return (cash_tags : [string], tokens : [string])
    #   tuple containing cash_tags and tokens
    """
    custom_tokenizer = Tokenizer()
    return custom_tokenizer.tokenize(message)


def process_tokens(tokens):
    """
    # Process the tokens with
    # custom funcs/library
    # Only stemmer for our project
    """
    stemmer = SnowballStemmer("english")
    tokens = map(lambda x: stemmer.stem(x), tokens)
    return list(tokens)


def find_index(token, low, high, features):  # binary search to find element index in list
    """
    # Binary search helper method
    # helps finding token index
    # with O(log n) time whereas
    # np.where() will need O(n)
    """
    if high >= low:
        mid = int((high + low) / 2)

        if features[mid] == token:
            return mid
        elif features[mid] > token:
            return find_index(token, low, mid - 1, features)
        else:
            return find_index(token, mid + 1, high, features)

    return -1


def PCA_reduce_dimensionality(X):
    """
    # Perform a PCA analysis
    # on the dataset X and
    # returns new X with reduced
    # dimensionality for faster
    # training time
    """
    return X
