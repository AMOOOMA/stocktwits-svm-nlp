import csv

import numpy as np

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn import svm

from support.helper import tokenize
from support.helper import process_tokens
from support.helper import find_index
from support.helper import Label


class BaselineTrainer:

    def __init__(self, data):
        self.data = data
        self.cash_tags = []
        self.bow = Counter()
        self.bow_features = []
        self.bow_X = []
        self.bow_y = []

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

    def _generate_bow_dataset(self):
        # key = (NEG_LABEL, POS_LABEL)
        for key in self.data:
            for message in self.data[key]:
                tokens = process_tokens(tokenize(message)[1])
                line_vector = np.zeros(len(self.bow_features) + 1)

                for token in tokens:
                    idx = find_index(token, 0, len(self.bow_features) - 1, self.bow_features)
                    line_vector[idx] += 1

                line_vector[len(line_vector) - 1] = len(tokens)
                self.bow_X.append(line_vector)

        self.bow_y = np.concatenate(([Label.NEG_LABEL.value] * len(self.data[Label.NEG_LABEL.value]),
                                     [Label.POS_LABEL.value] * len(self.data[Label.POS_LABEL.value])))

        # Format into numpy data structure just in case
        self.bow_X = np.array(self.bow_X)
        self.bow_y = np.array(self.bow_y)

    def _bow_train(self, model):
        # Make copies because of reuse
        X = self.bow_X.copy()
        y = self.bow_y.copy()

        X = preprocessing.scale(X)

        # Creates model and cross validation sets
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits()
        model_accuracy = []
        model_f1 = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            model_accuracy.append(model.score(X_test, y_test))
            model_f1.append(f1_score(y_test, model.predict(X_test), pos_label='Bullish'))

        return model_accuracy, model_f1

    def print_bow_score(self):
        self._fill_bow()
        self._generate_bow_feature_vector()
        self._generate_bow_dataset()

        # Print stats about BOW and dataset
        print("Total messages count: ", len(self.bow_y))
        print("Vocab size: ", len(self.bow_features) - 1)
        print("Tokens size: ", sum(self.bow.values()))
        print("Unique cash tags: ", len(self.cash_tags))

        log_reg_model = LogisticRegression(solver='sag', random_state=0, max_iter=100,
                                           n_jobs=-1, verbose=1, class_weight={Label.NEG_LABEL.value: 2, Label.POS_LABEL.value: 1})
        log_reg_accuracy, log_reg_f1 = self._bow_train(log_reg_model)
        print("Log Reg accuracy: ", sum(log_reg_accuracy) / len(log_reg_accuracy))
        print(log_reg_accuracy)
        print("Log Reg f1 score: ", sum(log_reg_f1) / len(log_reg_f1))
        print(log_reg_f1)

        naive_bayes_model = GaussianNB()
        naive_bayes_accuracy, naive_bayes_f1 = self._bow_train(naive_bayes_model)
        print("Naive Bayes accuracy: ", sum(naive_bayes_accuracy) / len(naive_bayes_accuracy))
        print(naive_bayes_accuracy)
        print("Naive Bayes f1 score: ", sum(naive_bayes_f1) / len(naive_bayes_f1))
        print(naive_bayes_f1)

        # svm_model = svm.SVC(kernel='rbf', cache_size=4000, max_iter=5000, verbose=True)  # performs similar to LR, disabled for speed
        # svm_accuracy, svm_precision = self._bow_train(svm_model)
        # print("SVM accuracy: ", sum(svm_accuracy) / len(svm_accuracy))
        # print(svm_accuracy)
        # print("SVM precision: ", sum(svm_precision) / len(svm_precision))
        # print(svm_precision)


def main():
    path = "data/stocktwits_labelled_train.csv"
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

        trainer = BaselineTrainer(data)
        trainer.print_bow_score()


if __name__ == "__main__":
    # execute only if run as a script
    main()
