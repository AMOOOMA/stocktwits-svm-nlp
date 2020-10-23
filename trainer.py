
# imports Value
import csv
# dictionary
from collections import Counter
# tokenize Messages
from tokenizer import Tokenizer
# Creates data structures for models
import numpy as np
#M odels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
#
from nltk.stem import SnowballStemmer
# Labels
from enum import Enum


class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"


def tokenize(message):
    """
    Parameters
    ----------
    message : string
        twit message

    Returns
    -------
    (cash_tag : [string], token : [string])
        tuple containing cash tag and tokens

    """
    custom_tokenizer = Tokenizer()
    return custom_tokenizer.tokenize(message)


def process_tokens(tokens):
    stemmer = SnowballStemmer("english")
    tokens = map(lambda x: stemmer.stem(x), tokens)
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
                for token in process_tokens(tokens):
                    self.bow[token] = self.bow[token] + 1 if token in self.bow else 1

    #_generate_bow_feature_vector or _generate_bow_X can be deleted, they are a bit redudant 
    def _generate_bow_feature_vector(self):  # use DictVectorizer to feature vector for log reg
        vec = DictVectorizer()
        vec.fit_transform(self.bow)
        self.bow_features = vec.get_feature_names()

    def _generate_bow_X(self):
        self._fill_bow()
        self._generate_bow_feature_vector()
        stemmer = SnowballStemmer("english")
        
        # key = (NEG_LABEL, POS_LABEL)
        for key in self.data:
            for message in self.data[key]:
                tokens = tokenize(message)[1]
                line_vector = np.zeros(len(self.bow_features) + 1)
                
                for token in tokens:
                    token = stemmer.stem(token) #
                    idx = np.where(np.array(self.bow_features) == token)[0] 
                    '''
                    E:\Programs\Anaconda\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
                        STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
            
                    Increase the number of iterations (max_iter) or scale the data as shown in: https://scikit-learn.org/stable/modules/preprocessing.html
                    Please also refer to the documentation for alternative solver options: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
                      n_iter_i = _check_optimize_result(
                    '''
                    line_vector[idx] += 1
                    
                line_vector[len(line_vector) - 1] = len(tokens)
                self.X.append(line_vector)
                
    def _bow_train(self):
        # Format into numpy data structure
        y = np.concatenate(([Label.NEG_LABEL.value] * len(self.data[Label.NEG_LABEL.value]),
                            [Label.POS_LABEL.value] * len(self.data[Label.POS_LABEL.value])))
        self.X = np.array(self.X)
        
        # Creates model
        model = LogisticRegression(random_state=0).fit(self.X,y);
        return model

    def get_bow_score(self):
        self._fill_bow()
        self._generate_bow_feature_vector()
        # print(len(self.bow))
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
        trainer._generate_bow_X()
        trainer._bow_train()
        


if __name__ == "__main__":
    # execute only if run as a script
    main()
