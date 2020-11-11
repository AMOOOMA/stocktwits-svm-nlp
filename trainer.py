import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import svm

from joblib import dump, load

from helper import Label


class Trainer:

    def __init__(self, data):
        self.data = data
        self.X = []
        self.y = []

    def _generate_dataset(self):
        """
        # generate X and y using
        # self.data, format in
        # {label} : [{message vector}]
        """
        for label in self.data:
            for vector in self.data[label]:
                self.X.append(vector)
                self.y.append(label)

    def _SVM_train(self, kernel):
        """
        # Train a SVM model with kernel
        # After training finished, dumps the
        # model locally as {kernel name}.joblib
        # and prints out the accuracy stats
        """
        # make sure the format is np.array and make copy
        X = np.array(self.X).copy()
        y = np.array(self.y).copy()

        X = preprocessing.scale(X)

        # Creates model and cross validation sets
        model = svm.SVC(kernel=kernel, cache_size=4000, max_iter=10000, verbose=True)
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits()
        model_score = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            model_score.append(model.score(X_test, y_test))

        # save the model locally
        path = f'./pretrained_model/{kernel}.joblib'
        if os.path.exists(path):
            os.remove(path)
        dump(model, path)

        return model_score

    def print_kernels_score(self):
        """
        # train four different SVM models
        # with different kernels to compare
        # and see which one is best for
        # this particular dataset/task
        """
        self._generate_dataset()

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            kernel_score = self._SVM_train(kernel)
            print(f"Kernel {kernel} scores: {sum(kernel_score) / len(kernel_score)}", kernel_score)


def main():
    path = "./data/stocktwits_labelled_train_bert_cls.csv"
    reader = pd.read_csv(path, header=None)
    data = {
        Label.NEG_LABEL.value: [],
        Label.POS_LABEL.value: [],
    }

    for index, row in reader.iterrows():
        data[row[0]].append(eval(str(row[1])))  # eval(str(message)) to convert data from string to list form

    print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
    print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))

    trainer = Trainer(data)
    trainer.print_kernels_score()
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
