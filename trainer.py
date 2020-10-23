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


class Trainer:

    def __init__(self, data):
        self.data = data
        self.cash_tags = []


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


if __name__ == "__main__":
    # execute only if run as a script
    main()
