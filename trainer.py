import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import svm
from sklearn import decomposition
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from joblib import dump, load

from support.helper import Label
from bert_word_embedding import BertWordEmbedding
from data_scraper import parse_messages_json
import requests


def predict_prob_with_pretrain(messages):
    embedding = BertWordEmbedding()

    X = []

    for message in messages:
        _, message_embeddings = embedding.get_message_embedding(message)
        X.append(list(np.sum(np.array(message_embeddings[1:-1]), axis=0)))

    model = load("./pretrained_model/rbf.joblib")
    print(messages)
    print(model.predict(np.array(X)))


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
        model = svm.SVC(kernel=kernel, cache_size=4000,
                        class_weight={Label.NEG_LABEL.value: 2, Label.POS_LABEL.value: 1}, max_iter=10000, verbose=True)
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

        # save the model locally
        # path = f'./pretrained_model/{kernel}.joblib'
        # dump(model, path)

        print(len(
            list(filter(lambda x: x == 'Bullish', model.predict(self.X)))))  # double check the prediction label ratio

        return model_accuracy, model_f1

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
            kernel_accuracy, kernel_f1 = self._SVM_train(kernel)
            print(f"Kernel {kernel} accuracy: {sum(kernel_accuracy) / len(kernel_accuracy)}", kernel_accuracy)
            print(f"Kernel {kernel} f1 score: {sum(kernel_f1) / len(kernel_f1)}", kernel_f1)

    def grid_search_kernel_params(self, kernel):
        self._generate_dataset()

        parameters = {'C': [1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1, 10, 100]}
        model = svm.SVC(kernel=kernel, class_weight={Label.NEG_LABEL.value: 2, Label.POS_LABEL.value: 1},
                        max_iter=25000, cache_size=4000)
        f1_scorer = make_scorer(f1_score, pos_label=Label.POS_LABEL.value)
        clf = GridSearchCV(model, parameters, n_jobs=-1, verbose=10, scoring=f1_scorer)
        clf.n_splits_ = 5
        clf.fit(self.X, self.y)

        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.cv_results_)

    def pca(self, n_components):  # replaced later with pca in helper.py
        self._generate_dataset()

        clf = decomposition.PCA(n_components=n_components)
        self.X = clf.fit_transform(self.X)


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

    trainer = Trainer(data)
    trainer.pca(200)
    trainer.print_kernels_score()


def predict_real_data(symbol):
    api_url = "https://api.stocktwits.com/api/2/streams/symbol/"  # + {id}.json
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'referer': 'https://google.com',
    }

    messages = []

    response = requests.get(api_url + f"{symbol}.json", headers)
    if response.status_code == 200:
        print("GET messages/posts success")
        messages = messages + parse_messages_json(response.content)
    else:
        print(f"GET request from messages failed: {response.status_code} {response.reason}")

    predict_prob_with_pretrain(map(lambda x: list(x.items())[0][1], messages))


if __name__ == "__main__":
    # execute only if run as a script
    main()

    # test_messages = ["going short",
    #                  "falling out of bed",
    #                  "No we’re not Green we are red",
    #                  "This stock is going down, and I’ll tell you why.",
    #                  "Not too bad, we will see what happens"]
    # predict_prob_with_pretrain(test_messages)

    # predict_real_data()
