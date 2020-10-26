# Stocktwits messaages sentiment analysis

### Model Introduction
TO DO

### Model Performance
TO DO

### Project usage
To train and predict messages from stocktwits or just general texts messages from anywhere, you can import the Trainer class inside the trainer.py and feed in data with the constructor. The dataset should follow the same format as our stocktwits_labelled.csv file: {Label.value: String}, {message: String}. To provide a better understanding of the labels we used, here is the Enum class of this project:
```Python
class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"
```

If you want to just train the model yourself and see the tracing messages, feel free to run command `python trainer.py`.

### Credits and references
The tokenizer we used is from project [tweetmotif](https://github.com/brendano/tweetmotif).

Brendan O'Connor, Michel Krieger, and David Ahn. TweetMotif: Exploratory Search and Topic Summarization for Twitter. ICWSM-2010.
