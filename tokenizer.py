import twokenize


class Tokenizer:

    def __init__(self):
        self.message = []
        self.cash_tags = []

    def tokenize(self, message):
        """
        # First the message is pass into
        # tweetmotif's tokenizer for processing
        # https://github.com/brendano/tweetmotif
        # Then tokenize cleans the @ symbols and
        # extract the cash tag in the message
        # returns a tuple: ([cash_tags], [tokens])
        """
        self.message = twokenize.tokenize(message)  # calls the Tweetmotif's tokenizer
        self._extract_cash_tags()
        self._clean_at_symbols()
        self._replace_urls()
        return self.cash_tags, self.message

    def _clean_at_symbols(self):
        """
        # Should remove all @token in tokens
        # replace the message in self with
        # the cleaned list of tokens
        """
        for tok in self.message:
            if tok[0] == "@":
                self.message.remove(tok)

    def _replace_urls(self):
        """
        # Should remove all urls in tokens
        # replace the message in self with
        # the cleaned list of tokens
        """
        for token in self.message:
            if len(token) > 3 and token[:4] == "http":
                self.message.remove(token)

    def _extract_cash_tags(self):
        """
        # Extracts the $company in tokens
        # and update the list of cash_tags
        # Note, the tokens in self should
        # be updated with new tokens list
        """

        def has_numbers(string):
            return any(char.isdigit() for char in string)

        def has_alpha(string):
            return any(char.isalpha() for char in string)

        for tok in self.message[:]:  # use a copy because we will modify the original list
            # if the token begins with '$' and contains no numbers and has alphabet, it's most likely a company
            if tok[0] == '$' and not has_numbers(tok[1:]) and has_alpha(tok[1:]):
                # to avoid repeats, append only if token is not in self.cash_tag
                if tok not in self.cash_tags:
                    self.cash_tags.append(tok)
                self.message.remove(tok)


def main():
    # for testing propose
    tokenizer = Tokenizer()
    test_message = "$AAPL, $JD Hello this is a @someone https://github.com/AMOOOMA/stocktwits-svm-nlp test message $AMZN"
    cash_tags, tokens = tokenizer.tokenize(test_message)
    print(cash_tags, tokens)


if __name__ == "__main__":
    # execute only if run as a script
    main()
