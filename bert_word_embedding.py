import numpy as np
import torch
from transformers import BertTokenizer, BertModel

from helper import tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertWordEmbedding:

    def __init__(self):
        self.tokens = []
        self.embeddings = []

    def _update_tokens(self, token_ids):
        """
        # Update self.tokens with BERT
        # model compatible tokens
        """
        self.tokens = []  # clear list first
        vocab = list(tokenizer.vocab.keys())
        for token_id in token_ids:
            self.tokens.append(vocab[token_id])

    def get_message_embedding(self, message):
        """
        # Use the BERT pretrained model
        # to generate word embeddings,
        # using the last four layers in BERT
        # returns a tuple containing
        # the BERT style tokens and embeddings
        """
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
            batch_index = 1

            for token_index in range(len(self.tokens)):
                layers = []
                for layer_index in range(-4, 0):  # only use the last four layers
                    layers.append(hidden_states[layer_index][batch_index][token_index])

                self.embeddings.append(map(lambda x: x / 4, np.sum(layers, 0)))  # add the avg of the last four layers

        return self.tokens, self.embeddings


def main():
    # for testing propose
    test_message = "$AAPL, $JD Hello this is a @someone https://github.com/AMOOOMA/stocktwits-svm-nlp test message $AMZN"
    embedding = BertWordEmbedding()
    embedding.get_message_embedding(test_message)


if __name__ == "__main__":
    # execute only if run as a script
    main()
