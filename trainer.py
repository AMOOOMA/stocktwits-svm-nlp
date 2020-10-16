import csv
from collections import Counter

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer

from nltk.stem import SnowballStemmer


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
    with open('CS 490a hw2 final dataset - Sheet1.csv', encoding="utf8") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        class_dict = {
            "pos": [],
            "neg": [],
            "neu": [],
        }

        for row in reader:
            if row[0] != "Data entries":
                class_dict[row[1]].append(row[0])

if __name__ == "__main__":
    # execute only if run as a script
    main()