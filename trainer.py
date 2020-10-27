import csv
import sys

from collections import Counter

import numpy as np
import pandas as pd

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
    path = "../stocktwits_embeddings.csv"
    reader = pd.read_csv(path, header=None)
    data = {
        Label.NEG_LABEL.value: [],
        Label.POS_LABEL.value: [],
    }

    for index, row in reader.iterrows():
        data[row[0]].append(row[1])

    print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
    print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))

    trainer = Trainer(data)


if __name__ == "__main__":
    # execute only if run as a script
    main()
