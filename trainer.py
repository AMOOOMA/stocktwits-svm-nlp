import csv
import os
from collections import Counter

from tokenizer import Tokenizer

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
    custom_tokenizer = Tokenizer()
    return custom_tokenizer.tokenize(message)  # returns a tuple ([cash_tag], [tokens])


def process_tokens(tokens):
    tokens = map(lambda x: x, tokens)
    return tokens


class Trainer:

    def __init__(self, data):
        # common variables
        self.data = data
        self.cash_tags = []
        self.y = []

        # variables for baseline algo
        self.bow = Counter()
        self.bow_features = []
        self.log_reg_X = []

        # variables for SVM
        self.features = []
        self.X = []

    def _fill_bow(self):  # fill the bow dictionary with self.data
        for label in self.data:
            for message in self.data[label]:
                cash_tags, tokens = tokenize(message)
                self.cash_tags.extend(x for x in cash_tags if x not in self.cash_tags)  # add new cash_tags
                for token in tokens:  # no need to process_tokens here for log reg
                    self.bow[token] = self.bow[token] + 1 if token in self.bow else 1

    def _generate_bow_feature_vector(self):  # use DictVectorizer to feature vector for log reg
        vec = DictVectorizer()
        vec.fit_transform(self.bow)
        self.bow_features = vec.get_feature_names()

    def _generate_bow_X(self):
        return self.data

    def _bow_train(self):
        return self.data

    def get_bow_score(self):
        self._fill_bow()
        self._generate_bow_feature_vector()
        print(self.cash_tags)
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

        trainer = Trainer(data)
        trainer.get_bow_score()


if __name__ == "__main__":
    # execute only if run as a script
    main()
