import csv

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
    """
    return (cash_tags : [string], tokens : [string])
        tuple containing cash_tags and tokens
    """
    custom_tokenizer = Tokenizer()
    return custom_tokenizer.tokenize(message)


def process_tokens(tokens):
    stemmer = SnowballStemmer("english")
    tokens = map(lambda x: stemmer.stem(x), tokens)
    return list(tokens)


def find_index(token, low, high, features):  # binary search to find element index in list
    if high >= low:
        mid = int((high + low) / 2)

        if features[mid] == token:
            return mid
        elif features[mid] > token:
            return find_index(token, low, mid - 1, features)
        else:
            return find_index(token, mid + 1, high, features)

    return -1


class Trainer:

    def __init__(self, data):
        # common variables
        self.data = data
        self.cash_tags = []

        # variables for baseline algo
        self.bow = Counter()
        self.bow_features = []
        self.log_reg_X = []
        self.log_reg_y = []

        # variables for SVM
        self.features = []
        self.X = []
        self.y = []

    def _fill_bow(self):  # fill the bow dictionary with self.data
        for label in self.data:
            for message in self.data[label]:
                cash_tags, tokens = tokenize(message)
                self.cash_tags.extend(x for x in cash_tags if x not in self.cash_tags)  # add new cash_tags
                for token in process_tokens(tokens):
                    self.bow[token] = self.bow[token] + 1 if token in self.bow else 1

    def _generate_bow_feature_vector(self):  # use DictVectorizer to feature vector for log reg
        vec = DictVectorizer()  # To be decided
        vec.fit_transform(self.bow)
        self.bow_features = vec.get_feature_names()
        self.bow_features.sort()  # prepare for binary search

    def _generate_bow_X(self):
        # key = (NEG_LABEL, POS_LABEL)
        for key in self.data:
            for message in self.data[key]:
                tokens = process_tokens(tokenize(message)[1])
                line_vector = np.zeros(len(self.bow_features) + 1)

                for token in tokens:
                    idx = find_index(token, 0, len(self.bow_features) - 1, self.bow_features)
                    line_vector[idx] += 1

                line_vector[len(line_vector) - 1] = len(tokens)
                self.log_reg_X.append(line_vector)

    def _bow_train(self):
        # Format into numpy data structure
        self.log_reg_y = np.concatenate(([Label.NEG_LABEL.value] * len(self.data[Label.NEG_LABEL.value]),
                                         [Label.POS_LABEL.value] * len(self.data[Label.POS_LABEL.value])))
        self.log_reg_X = np.array(self.log_reg_X)

        # Creates model and cross validation sets
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits()
        model = LogisticRegression(solver='sag', random_state=0, n_jobs=-1, verbose=10)
        model_score = []

        for train_index, test_index in kf.split(self.log_reg_X, self.log_reg_y):
            X_train, X_test = self.log_reg_X[train_index], self.log_reg_X[test_index]
            y_train, y_test = self.log_reg_y[train_index], self.log_reg_y[test_index]
            model.fit(X_train, y_train)
            model_score.append(model.score(X_test, y_test))

        return model, model_score

    def get_bow_score(self):
        self._fill_bow()
        self._generate_bow_feature_vector()
        self._generate_bow_X()
        model, scores = self._bow_train()
        return sum(scores) / len(scores)


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
        print(trainer.get_bow_score())


if __name__ == "__main__":
    # execute only if run as a script
    main()
