import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from support.helper import tokenize
from support.helper import Label

from sklearn.metrics import f1_score

import time
import datetime

from data_scraper import parse_messages_json
import requests

# References and work credits: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
output_dir = './pretrained_model/fine_tune_bert/'
batch_size = 32
epochs = 4
seed_val = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


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


def flat_accuracy(predictions, labels):
    prediction_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(prediction_flat == labels_flat) / len(labels_flat), f1_score(prediction_flat, labels_flat)  # use f1 score


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


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

        training_stats = []

        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            model.train()

            for step, batch in enumerate(self.train_data):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_data), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()

                loss, logits = model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)

                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(self.train_data)
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()
            total_eval_accuracy = 0
            total_eval_f1 = 0
            total_eval_loss = 0

            for batch in self.val_data:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    (loss, logits) = model(b_input_ids,
                                           token_type_ids=None,
                                           attention_mask=b_input_mask,
                                           labels=b_labels)

                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                accuracy, f1 = flat_accuracy(logits, label_ids)
                total_eval_accuracy += accuracy
                total_eval_f1 += f1

            avg_val_accuracy = total_eval_accuracy / len(self.val_data)
            avg_val_f1 = total_eval_f1 / len(self.val_data)
            print("  Accuracy: {0:f}".format(avg_val_accuracy))
            print("  F1 score: {0:f}".format(avg_val_f1))
            avg_val_loss = total_eval_loss / len(self.val_data)
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #
        # print("Saving model to %s" % output_dir)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(output_dir)


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

    messages = list(map(lambda x: list(x.items())[0][1], messages))
    input_ids, attention_masks, labels = tokenize_transform(messages, len(messages) * [1])
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    test_data = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    model = BertForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    predictions = []

    for batch in test_data:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = np.argmax(predictions[0], axis=1).flatten()
    index = 0
    for message in messages:
        _, tokens = tokenize(message)
        if len(tokens) == 0:  # skip messages with no content
            continue

        print(('Bullish' if predictions[index] == 1 else 'Bearish', message))
        index += 1


if __name__ == "__main__":
    main()
    predict_real_data('AAPL')
