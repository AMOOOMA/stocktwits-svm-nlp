import csv

from collections import Counter

import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

from helper import tokenize
from helper import process_tokens
from helper import find_index
from helper import Label
from bert_word_embedding import BertWordEmbedding


class Trainer:

    def __init__(self, data):
        self.data = data
        self.cash_tags = []


def main():
    path = "stocktwits_labelled_train.csv"
    with open(path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = {
            Label.NEG_LABEL.value: [],
            Label.POS_LABEL.value: [],
        }

        embedding = BertWordEmbedding()

        index = 0
        prev_written = -1
        with open("./stocktwits_embeddings.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            for label, message in reader:
                if index > prev_written:
                    data[label].append(message)
                    tokens, embeddings = embedding.get_message_embedding(message)
                    writer.writerow([label, embeddings])
                    print("Message embeddings written, index: ", index)
                index += 1

        for label, message in reader:
            data[label].append(message)

        print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
        print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))

        trainer = Trainer(data)


if __name__ == "__main__":
    # execute only if run as a script
    main()
