import twokenize


class Tokenizer:

    def __init__(self):
        self.message = []
        self.cash_tags = []

    def tokenize(self, message):
        self.message = message
        self._clean_at_symbols()
        self._replace_emojis()
        self._extract_cash_tags()
        return twokenize(self.message)  # calls the Tweetmotif's tokenizer (https://github.com/brendano/tweetmotif)

    def _clean_at_symbols(self):
        self.message = ""

    def _extract_cash_tags(self):
        self.cash_tags = []

    def _replace_emojis(self):
        self.message = ""


def main():
    return


if __name__ == "__main__":
    # execute only if run as a script
    main()
