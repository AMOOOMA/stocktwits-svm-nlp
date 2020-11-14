import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import svm
from sklearn import decomposition
from sklearn.metrics import precision_score

from joblib import dump, load

from support.helper import Label
from bert_word_embedding import BertWordEmbedding


def predict_prob_with_pretrain():
    embedding = BertWordEmbedding()
    test_neg = "down down down down stock downs down"
    test_pos = "This is looking pretty good."

    _, neg = embedding.get_message_embedding(test_neg)
    _, pos = embedding.get_message_embedding(test_pos)
    X = [np.sum(np.array(neg[1:-1]), axis=0), np.sum(np.array(pos[1:-1]), axis=0)]

    model = load("./pretrained_model/rbf.joblib")
    print(model.predict(X))


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
        model = svm.SVC(kernel=kernel, cache_size=4000, class_weight='balanced', max_iter=10000, verbose=True)
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits()
        model_accuracy = []
        model_precision = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            model_accuracy.append(model.score(X_test, y_test))
            model_precision.append(precision_score(y_test, model.predict(X_test), pos_label='Bullish'))

        # # save the model locally
        # path = f'./pretrained_model/{kernel}.joblib'
        # dump(model, path)

        print(len(list(filter(lambda x: x == 'Bullish', model.predict(self.X)))))

        return model_accuracy, model_precision

    def print_kernels_score(self):
        """
        # train four different SVM models
        # with different kernels to compare
        # and see which one is best for
        # this particular dataset/task
        """
        # self._generate_dataset()

        kernels = ['rbf']
        for kernel in kernels:
            kernel_accuracy, kernel_precision = self._SVM_train(kernel)
            print(f"Kernel {kernel} accuracy: {sum(kernel_accuracy) / len(kernel_accuracy)}", kernel_accuracy)
            print(f"Kernel {kernel} precision: {sum(kernel_precision) / len(kernel_precision)}", kernel_precision)

    def grid_search_kernel_params(self, kernel):
        self._generate_dataset()

        parameters = {'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100]}
        model = svm.SVC(kernel=kernel, class_weight='balanced', max_iter=10000, cache_size=4000)
        clf = GridSearchCV(model, parameters, n_jobs=-1, verbose=10)
        clf.n_splits_ = 5
        clf.fit(self.X, self.y)

        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.cv_results_)

    def pca(self, n_components):  # replaced later with pca in helper.py
        self._generate_dataset()

        clf = decomposition.PCA(n_components=n_components)
        self.X = clf.fit_transform(self.X)

        print(sum(clf.explained_variance_ratio_[:n_components]))


def main():
    path = "./data/stocktwits_labelled_train_bert_average.csv"
    reader = pd.read_csv(path, header=None)
    data = {
        Label.NEG_LABEL.value: [],
        Label.POS_LABEL.value: [],
    }

    for index, row in reader.iterrows():
        if str(row[1]) != "nan":
            data[row[0]].append(eval(str(row[1])))  # eval(str(message)) to convert data from string to list form

    print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
    print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))

    test_list = [100, 200, 300]
    for test in test_list:
        trainer = Trainer(data)
        trainer.pca(test)
        trainer.print_kernels_score()


def run_predict():
    predict_prob_with_pretrain()
    

if __name__ == "__main__":
    # execute only if run as a script
    main()
    # run_predict()
