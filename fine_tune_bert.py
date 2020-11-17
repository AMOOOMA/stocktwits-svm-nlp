import torch
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig

from support.helper import tokenize
from support.helper import Label

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_transform(messages, labels):
    input_ids = []
    attention_masks = []

    for message in messages:
        _, tokens = tokenize(message)  # use our custom tokenizer

        encoded = tokenizer.encode_plus(  # encoded the pre tokenized message
            text=tokens,
            max_length=200,
            padding=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


class StocktwitsBERT:
    def __init__(self, data):
        self.data = data
        self.train_dataset = []
        self.val_dataset = []

    def generate_dataset(self):
        messages = []
        labels = []

        for label in self.data:
            for message in messages:
                messages.append(message)
                labels.append(label)

        input_ids, attention_masks, labels = tokenize_transform(messages, labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))


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

    bert = StocktwitsBERT(data)


if __name__ == "__main__":
    main()
