import torch
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from support.helper import tokenize
from support.helper import Label

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
batch_size = 32
epochs = 4
seed_val = 42


def tokenize_transform(messages, labels):
    input_ids = []
    attention_masks = []
    new_labels = []

    for i in range(len(messages)):
        _, tokens = tokenize(messages[i])  # use our custom tokenizer

        if len(tokens) == 0:
            continue

        encoded = tokenizer.encode_plus(  # encoded the pre tokenized message
            text=tokens,
            max_length=128,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            is_split_into_words=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        new_labels.append(labels[i])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    new_labels = torch.tensor(new_labels)

    return input_ids, attention_masks, new_labels


def flat_accuracy(prediction, labels):
    prediction_flat = np.argmax(prediction, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(prediction_flat == labels_flat) / len(labels_flat)


class StocktwitsBERT:
    def __init__(self, data):
        self.data = data
        self.train_data = None
        self.val_data = None

    def _generate_dataset(self):
        messages = []
        labels = []

        for label in self.data:
            for message in self.data[label]:
                messages.append(message)
                labels.append(1 if label == Label.POS_LABEL.value else 0)

        input_ids, attention_masks, labels = tokenize_transform(messages, labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        self.train_data = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )

        self.val_data = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )

    def train(self):
        self._generate_dataset()

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        model.cuda()
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(self.train_data) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # TO DO


def main():
    path = "./data/stocktwits_labelled_train.csv"
    reader = pd.read_csv(path, header=None)
    data = {
        Label.NEG_LABEL.value: [],
        Label.POS_LABEL.value: [],
    }

    for index, row in reader.iterrows():
        if str(row[1]) != "nan":
            data[row[0]].append(row[1])

    print("The NEG class' messages count: ", len(data[Label.NEG_LABEL.value]))
    print("The POS class' messages count: ", len(data[Label.POS_LABEL.value]))

    bert = StocktwitsBERT(data)
    bert.train()


if __name__ == "__main__":
    main()
