# Stocktwits messaages sentiment analysis

### Model Introduction
This project uses Support Vector Machine and BERT word embeddings to make sentiment prediction on messages from stockttwits, one of the biggest financial forum in the world.

The main ideas behind our model can be divided into several steps:
1. Encode the messages using the BERT pretrained model "bert-base-uncased".
2. Standardize the dataset by using tricks to fit and flatten word embeddings into a standard length vector.
3. Perform Principal Component Analysis on the whole dataset to reduce the matrix dimensionality.
4. Some additional feature engineering for the flatten standardized vector.
5. Train the SVM model with the best performing kernel in hand.

For more detailed information, checkout our [project report](https://github.com/AMOOOMA/stocktwits-svm-nlp).

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
