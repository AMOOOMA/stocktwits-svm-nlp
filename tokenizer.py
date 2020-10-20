import twokenize


class Tokenizer:

    def __init__(self):
        self.message = ""
        self.cash_tags = []

    def tokenize(self, message):
        """
        # Tokenize cleans the @ symbols and
        # extract the cash tag in the message
        # Then the message is pass into
        # tweetmotif's tokenizer for processing
        # returns a tuple: ([cash_tags], [tokens])
        """
        self.message = message
        self._clean_at_symbols()
        self._extract_cash_tags()
        return self.cash_tags, twokenize(
            self.message)  # calls the Tweetmotif's tokenizer (https://github.com/brendano/tweetmotif)

    def _clean_at_symbols(self):
        """
        # Should remove all @token in message
        # replace the message in self with
        # the new cleaned string
        """
        self.message = ""

    def _extract_cash_tags(self):
        """
        # Extracts the $company in message
        # and update the list of cash_tags
        # Note, the message in self should
        # be updated with the cleaned string
        """
        self.cash_tags = []
        self.message = ""


def main():
    # for testing propose
    tokenizer = Tokenizer()
    cash_tags, tokens = tokenizer.tokenize("$AAPL, $JD Hello this is a @someone test message $AMZN")
    print(cash_tags, tokens)


if __name__ == "__main__":
    # execute only if run as a script
    main()
