import torch
from transformers import BertTokenizer, BertModel

import logging

import matplotlib.pyplot as plt

from helper import tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertWordEmbedding:

    def __init__(self):
        self.tokens = []

    def _update_tokens(self, token_ids):
        self.tokens = []  # clear list first
        vocab = list(tokenizer.vocab.keys())
        for token_id in token_ids:
            self.tokens.append(vocab[token_id])

    def get_message_embedding(self, message):
        _, self.tokens = tokenize(message)  # use our custom tokenizer
        encoded = tokenizer.encode_plus(  # encoded the pre tokenized message
            text=self.tokens,
            add_special_tokens=True,
            is_split_into_words=True
        )
        self._update_tokens(encoded['input_ids'])  # update with BERT compatible tokens list
        print(self.tokens)

        input_ids_tensor = torch.tensor([encoded['input_ids']])
        attention_mask_tensors = torch.tensor([encoded['attention_mask']])

        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        model.eval()

        with torch.no_grad():

            outputs = model(input_ids_tensor, attention_mask_tensors)
            hidden_states = outputs[2]

            print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
            layer_i = 0

            print("Number of batches:", len(hidden_states[layer_i]))
            batch_i = 0

            print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
            token_i = 0

            print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

        return self.tokens


def main():
    # for testing propose
    test_message = "$AAPL, $JD Hello this is a @someone https://github.com/AMOOOMA/stocktwits-svm-nlp test message $AMZN"
    embedding = BertWordEmbedding()
    embedding.get_message_embedding(test_message)


if __name__ == "__main__":
    # execute only if run as a script
    main()
