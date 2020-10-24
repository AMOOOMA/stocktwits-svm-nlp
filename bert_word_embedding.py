import torch
from transformers import BertTokenizer, BertModel

import logging

import matplotlib.pyplot as plt

from helper import tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertWordEmbedding:

    def __init__(self):
        self.tokens = ""

    def get_message_embedding(self, message):
        cash_tags, self.tokens = tokenize(message)  # use our custom tokenizer
        encoded = tokenizer.encode_plus(
            text=self.tokens,
            add_special_tokens=True,
            is_split_into_words=True
        )  # encoded the pre tokenized message
        print(encoded.tokens())

        return self.tokens


def main():
    # for testing propose
    test_message = "$AAPL, $JD Hello this is a @someone https://github.com/AMOOOMA/stocktwits-svm-nlp test message $AMZN"
    embedding = BertWordEmbedding()
    embedding.get_message_embedding(test_message)


if __name__ == "__main__":
    # execute only if run as a script
    main()