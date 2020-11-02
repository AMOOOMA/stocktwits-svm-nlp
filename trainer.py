import csv
import sys

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import svm

from joblib import dump, load

from helper import tokenize
from helper import process_tokens
from helper import find_index
from helper import Label


class Trainer:

    def __init__(self, data):
        self.data = data
        self.X = []
        self.y = []

    def _generate_dataset(self):
        for label in self.data:
            for vector in self.data[label]:
                self.X.append(vector)
                self.y.append(label)

    def _SVM_train(self, kernel):
        # make sure the format is np.array and make copy
        X = np.array(self.X).copy()
        y = np.array(self.y).copy()

        X = preprocessing.scale(X)

        # Creates model and cross validation sets
        model = svm.SVC(kernel=kernel, cache_size=4000, max_iter=5000, verbose=True)
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits()
        model_score = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            model_score.append(model.score(X_test, y_test))

        # save the model locally
        dump(model, f'{kernel}.joblib')
        return model_score

    def print_kernels_score(self):
        self._generate_dataset()

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            kernel_score = self._SVM_train(kernel)
            print(f"Kernel {kernel} scores: {sum(kernel_score) / len(kernel_score)}", kernel_score)


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
