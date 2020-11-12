from support.tokenizer import Tokenizer

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

########################################################################################################################
#  Function archived, used when needed.
#
# import csv
# from bert_word_embedding import BertWordEmbedding
#
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
#
########################################################################################################################
# Used for generating standardize data, archived for backup
#
# def generate_standardize_data():
#     path = "../stocktwits_embeddings.csv"
#     reader = pd.read_csv(path, header=None)
#     data = {
#         Label.NEG_LABEL.value: [],
#         Label.POS_LABEL.value: [],
#     }
#
#     for index, row in reader.iterrows():
#         # if str(row[1]) != "nan":
#             data[row[0]].append(row[1])  # eval(str(message)) to convert data from string to list form
#
#     print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
#     print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))
#
#     token_data = {
#         Label.NEG_LABEL.value: [],
#         Label.POS_LABEL.value: [],
#     }
#
#     vocab = list(tokenizer.vocab.keys())
#
#     with open("./data/stocktwits_labelled_train.csv", 'r', encoding='utf-8') as csv_file:
#         reader = csv.reader(csv_file, delimiter=',')
#
#         for label, msg in reader:
#             _, bert_tokens = tokenize(msg)
#             if len(bert_tokens) == 0:
#                 token_data[label].append(None)
#                 continue
#
#             encoded = tokenizer.encode_plus(  # encoded the pre tokenized message
#                 text=bert_tokens,
#                 add_special_tokens=True,
#                 is_split_into_words=True
#             )
#             token_data[label].append(list(map(lambda x: vocab[x], encoded['input_ids'])))
#
#     print("The NEG class' messages count: ", len(token_data[Label.NEG_LABEL.value]))
#     print("The POS class' messages count: ", len(token_data[Label.POS_LABEL.value]))
#
#     with open("./data/stocktwits_labelled_train_bert_standardized.csv", 'w', newline='', encoding='utf-8') as csv_file:
#         csv_file.truncate()  # clear the original file and rewrite
#         writer = csv.writer(csv_file)
#
#         with open("./data/bert_standardized_tokens.csv", 'w', newline='', encoding='utf-8') as token_csv_file:
#             token_csv_file.truncate()
#             token_writer = csv.writer(token_csv_file)
#
#             for label in data.keys():
#                 for i in range(len(data[label])):
#                     if token_data[label][i] is None or str(data[label][i]) == "nan":
#                         continue
#                     tokens = token_data[label][i][1:-1]
#                     embeddings = eval(str(data[label][i]))[1:-1]
#                     if len(tokens) == len(embeddings):
#                         tokens, embeddings = standardize_by_duplicating(tokens, embeddings, 10)
#                         print(tokens, i)
#                         writer.writerow([label, embeddings])
#                         token_writer.writerow([label, tokens])
#                     else:
#                         print(len(tokens), len(embeddings))
#                         print("Something is wrong!!!!\n")
