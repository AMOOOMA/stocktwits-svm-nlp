import numpy as np
import torch

from transformers import BertTokenizer, BertModel

from scipy.spatial.distance import cosine

from support.helper import tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertWordEmbedding:

    def __init__(self):
        self.tokens = []
        self.embeddings = []
        print(torch.cuda.is_available())

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
        # Reset the global variables
        self.tokens = []
        self.embeddings = []

        _, self.tokens = tokenize(message)  # use our custom tokenizer

        if len(self.tokens) == 0:
            return None, None

        encoded = tokenizer.encode_plus(  # encoded the pre tokenized message
            text=self.tokens,
            add_special_tokens=True,
            is_split_into_words=True
        )
        self._update_tokens(encoded['input_ids'])  # update with BERT compatible tokens list
        print(self.tokens)

        input_ids_tensor = torch.tensor([encoded['input_ids']]).to('cuda:0')
        attention_mask_tensors = torch.tensor([encoded['attention_mask']]).to('cuda:0')

        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to('cuda:0')
        model.eval()

        with torch.no_grad():

            outputs = model(input_ids_tensor, attention_mask_tensors)
            hidden_states = outputs[2]
            batch_index = 0

            for token_index in range(len(self.tokens)):
                layers = []
                for layer_index in range(-4, 0):  # only use the last four layers
                    layers.append(hidden_states[layer_index][batch_index][token_index].tolist())

                self.embeddings.append(list(map(lambda x: x / 4, np.sum(layers, axis=0))))  # add the avg of the last four layers

        return self.tokens, self.embeddings


def main():
    # for testing propose, credits to https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    embedding = BertWordEmbedding()
    context_test_message = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    tokens, embeddings = embedding.get_message_embedding(context_test_message)
    print('First 5 vector values for each instance of "bank".')
    print('')
    print("bank vault   ", str(embeddings[6][:5]))
    print("bank robber  ", str(embeddings[10][:5]))
    print("river bank   ", str(embeddings[19][:5]))

    # Calculate the cosine similarity between the word bank
    # in "bank robber" vs "river bank" (different meanings).
    diff_bank = 1 - cosine(embeddings[10], embeddings[19])

    # Calculate the cosine similarity between the word bank
    # in "bank robber" vs "bank vault" (same meaning).
    same_bank = 1 - cosine(embeddings[10], embeddings[6])

    print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)
    print('Vector similarity for *different* meanings:  %.2f' % diff_bank)


if __name__ == "__main__":
    # execute only if run as a script
    main()
