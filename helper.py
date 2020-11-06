import csv

from tokenizer import Tokenizer

from nltk.stem import SnowballStemmer

from sklearn import decomposition

from enum import Enum

from graphs import mk_pca
# from bert_word_embedding import BertWordEmbedding


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


def PCA_reduce_dimensionality(X, n_components = 2, quiet = True):
    """
    # Perform a PCA analysis
    # on the dataset X and
    # returns new X with reduced
    # dimensionality for faster
    # training time
    """
    
    pca = decomposition.PCA()
    pca.fit(X)
    
    if not quiet:
        mk_pca(pca.explained_variance_[:n_components])
    
    pca.n_components = n_components
    X = pca.transform(X)
    return X

# def generate_embedding_for_path(path):
#     """
#     # Helper for generate word embeddings
#     # for the whole dataset, note this
#     # process will take a significant
#     # amount of time, 7k messages ~ 8 hours
#     """
#     with open(path, 'r', encoding='utf-8') as csv_file:
#         reader = csv.reader(csv_file, delimiter=',')
#         data = {
#             Label.NEG_LABEL.value: [],
#             Label.POS_LABEL.value: [],
#         }
#
#         embedding = BertWordEmbedding()
#
#         index = 0
#         prev_written = -1
#         with open("./stocktwits_embeddings.csv", "w", newline="", encoding='utf-8') as f:
#             writer = csv.writer(f)
#             for label, message in reader:
#                 if index > prev_written:
#                     data[label].append(message)
#                     tokens, embeddings = embedding.get_message_embedding(message)
#                     writer.writerow([label, embeddings])
#                     print("Message embeddings written, index: ", index)
#                 index += 1
#
#         for label, message in reader:
#             data[label].append(message)
#
#         print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
#         print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))
