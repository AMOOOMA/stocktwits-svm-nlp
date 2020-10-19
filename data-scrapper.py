import requests
import json
import csv
import sys
import os
from enum import Enum


class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"


def parse_messages_json(json_data):
    """
    # return the list of messages from the json data
    # should parse the json file into format like below
    # {label} : {messages body}
    # example: "label : this is a message"
    # some docs: https://docs.python.org/3/library/json.html
    # Note: for data with no Label, put "NO_LABEL"
    """
    dict_data = json.loads(json_data)
    messages_list = []
    for e in dict_data["messages"]:
        if e["entities"]["sentiment"] is None:
            messages_list.append({Label.NO_LABEL: e["body"]})
        else:
            messages_list.append({e["entities"]["sentiment"]["basic"]: e["body"]})

    return messages_list


def make_dict_from_list(messages_list):
    """
    # return a dict of the string list
    # the string list will follow the format of parse_json()
    # the dict should be like {label} : {array of messages}
    """
    label_dict = {}
    NO_LABEL_list = []
    bullish_list = []
    bearish_list = []

    for item in messages_list:
        label, message = list(item.items())[0]
        if label == Label.NO_LABEL:
            NO_LABEL_list.append(message)
        elif label == Label.NEG_LABEL.value:
            bearish_list.append(message)
        elif label == Label.POS_LABEL.value:
            bullish_list.append(message)

    label_dict[Label.NO_LABEL.value] = NO_LABEL_list
    label_dict[Label.NEG_LABEL.value] = bearish_list
    label_dict[Label.POS_LABEL.value] = bullish_list

    return label_dict


class DataScrapper:

    def __init__(self, auth_username, auth_passwd):
        self.auth_username = auth_username
        self.auth_passwd = auth_passwd
        self.data = {
            Label.NO_LABEL.value: [],
            Label.NEG_LABEL.value: [],
            Label.POS_LABEL.value: [],
        }

    @staticmethod
    def _get_auth_token():  # can be used to add auth stuff but ignore for now
        """
        # returns the auth token, -1 when failed
        # Some helpful articles and docs:
        # https://api.stocktwits.com/developers/docs/api#oauth-token-docs
        """
        return None

    @staticmethod
    def _get_trending_symbols():
        """
        # returns a list of symbols, in string formats
        # make a get request and get the list of trending symbols
        # some docs: https://realpython.com/python-requests/#the-get-request
        # https://api.stocktwits.com/developers/docs/api#trending-symbols-docs
        """
        api_url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
            'referer': 'https://google.com',
        }

        response = requests.get(api_url, headers)
        if response.status_code == 200:
            print("GET trending symbols success.")
            json_data = json.loads(response.content)
            symbols_list = []
            for symbol in json_data['symbols']:
                symbols_list.append(symbol['symbol'])  # only parse in the stock trading symbol

            return symbols_list
        else:
            print(f"GET list of trending symbols request failed: {response.status_code} {response.reason}")

        return None

    @staticmethod
    def _get_messages_from_symbol(symbols):
        """
        # return a list of posts, in string format {sentiment label} : {body}
        # should removes everything except body and the sentiment label
        # some docs: https://realpython.com/python-requests/#the-get-request
        # https://api.stocktwits.com/developers/docs/api#streams-symbol-docs
        """
        api_url = "https://api.stocktwits.com/api/2/streams/symbol/"  # + {id}.json
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
            'referer': 'https://google.com',
        }

        messages = []
        if symbols is not None:
            for symbol in symbols:
                response = requests.get(api_url + f"{symbol}.json", headers)
                if response.status_code == 200:
                    print("GET messages/posts success")
                    messages = messages + parse_messages_json(response.content)
                else:
                    print(f"GET request from messages failed: {response.status_code} {response.reason}")

        return messages  # return empty list if symbols is None

    def _read_from_csv(self, path):
        """
        # read the csv file from local machine
        # then populate the self.data variable
        # the csv file will follow the format {label} , {message}
        # https://realpython.com/python-csv/
        """
        try:
            os.path.exists(path)
        except OSError:
            print("file does not exist")

        with open(path, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')

            for label, msg in reader:
                self.data[label].append(msg)

    def _write_to_csv(self, path):
        """"
        # write the data in self to a local csv file
        # the csv file should follow the format {label} , {message}
        # https://realpython.com/python-csv/
        """
        try:
            os.path.exists(path)
        except OSError:
            print("file does not exist")

        with open(path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_file.truncate()  # clear the original file and rewrite
            writer = csv.writer(csv_file)

            for label, messages in self.data.items():
                for msg in messages:
                    writer.writerow([label, msg])

    def _populate_data(self, message_list):
        """
        # populate the data var in self
        # the data (message_list) should be a dictionary of entries
        # the dictionary's format is {label} : {array of messages}
        # Note: this will not add duplicate messages
        """
        cleaned_messages = self._data_cleaning(message_list)
        for label in cleaned_messages.keys():
            # extend the messages list with new messages
            self.data[label].extend(cleaned_messages[label])

    def _data_cleaning(self, message_list):
        """
        # clean the data in message_list
        # remove duplicated messages
        # returns the filtered message_list
        """
        for label in message_list.keys():
            # filter out duplicated messages
            message_list[label] = filter(lambda message: message not in self.data[label], message_list[label])

        return message_list

    def run(self):
        path = "./stocktwits.csv"  # default storage file
        self._read_from_csv(path)
        symbols = self._get_trending_symbols()
        messages = self._get_messages_from_symbol(symbols)
        self._populate_data(make_dict_from_list(messages))
        self._write_to_csv(path)


def main():
    auth_username = "cs490a"  # default username
    auth_passwd = "fZ?T-----9pE3"  # default passwd, not provided for now
    args = sys.argv
    if len(args) > 1:
        auth_username = args[1]
        auth_passwd = args[2]

    scrapper = DataScrapper(auth_username, auth_passwd)
    scrapper.run()


if __name__ == "__main__":
    # execute only if run as a script
    main()
