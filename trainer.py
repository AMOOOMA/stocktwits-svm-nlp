import csv
import os
from collections import Counter

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer

from nltk.stem import SnowballStemmer

from enum import Enum


class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"


def tokenize(message):
    tokens = []
    return tokens


def process_tokens(tokens):
    tokens = map(lambda x: x, tokens)
    return tokens


class Trainer:

    def __init__(self, data):
        self.data = data

    def _generate_feature_vector(self):
        return self.data

    def _train(self):
        return self.data

    def get_score(self):
        return self.data


def main():
    path = "./stocktwits_labelled.csv"
    with open(path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = {
            Label.NEG_LABEL.value: [],
            Label.POS_LABEL.value: [],
        }

        for label, msg in reader:
            data[label].append(msg)

        print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
        print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))



if __name__ == "__main__":
    # execute only if run as a script
    main()