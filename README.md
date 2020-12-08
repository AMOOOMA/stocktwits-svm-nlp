# Stocktwits messaages sentiment analysis

### Model Introduction
This project uses Support Vector Machine and BERT word embeddings to make sentiment prediction on messages from stockttwits, one of the biggest financial forum in the world.

The main ideas behind our model can be divided into several steps:
1. Encode the messages using the BERT pretrained model "bert-base-uncased".
2. Standardize the dataset by using tricks to fit and flatten word embeddings into a standard length vector.
3. Perform Principal Component Analysis on the whole dataset to reduce the matrix dimensionality.
4. Some additional feature engineering for the flatten standardized vector.
5. Train the SVM model with the best performing kernel in hand.

For more detailed information, checkout our [project report](https://1drv.ms/b/s!AoUFx9vU0-DZg7pIJKXLbRMAPjRkVA?e=96eJhP).

### Model Performance
PCA + SVM with RBF kernels with standardization data:
```
Kernel rbf accuracy: 0.7249672157176377 [0.7340797760671799, 0.7303921568627451, 0.7240896358543417, 0.7156862745098039, 0.7205882352941176]
Kernel rbf f1 score: 0.7994398211427404 [0.8067141403865716, 0.8056537102473499, 0.7971163748712666, 0.7932790224032586, 0.794435857805255]
```

PCA + SVM with RBF kernel with average sum (frozen input) data:
```
Kernel rbf accuracy: 0.7553809014186373 [0.7477288609364081, 0.7540181691125087, 0.7596086652690426, 0.7651991614255765, 0.7503496503496504]
Kernel rbf f1 score: 0.821851831591843 [0.8140133951571354, 0.8216818642350557, 0.8271356783919598, 0.8297872340425532, 0.8166409861325115]
```

### Project usage
To train and predict messages from stocktwits or just general texts messages from anywhere, you can import the Trainer class inside the trainer.py and feed in data with the constructor. The dataset should follow the same format as our stocktwits_labelled.csv file: {Label.value: String}, {message: String}. To provide a better understanding of the labels we used, here is the Enum class of this project:
```Python
class Label(Enum):
    NO_LABEL = "NO_LABEL"
    NEG_LABEL = "Bearish"
    POS_LABEL = "Bullish"
```

If you want to just train the model yourself and see the tracing messages, feel free to run command `python trainer.py`.

#### Tokenizer
If you are interested in the tokenizer, it's available in the support folder with details comments explaining its usage. Its dependency includes the two files `emoticons.py` and `twokenize.py` from tweetmotif.

#### Pretrained model
To use our pretrained model, download the saved model file in the pretrained model folder and load the model using this code for SVM:
```Python
from joblib import load

path = "./pretrained_model/rbf.joblib"
model = load(path)
```

This code for loading the fine tune BERT model:
```Python
from transformers import BertForSequenceClassification

path = "./pretrained_model/fine_tune_bert/"
model = BertForSequenceClassification.from_pretrained(path)
```

### Credits
The tokenizer we used is built upon the work from project [tweetmotif](https://github.com/brendano/tweetmotif).

Brendan O'Connor, Michel Krieger, and David Ahn. TweetMotif: Exploratory Search and Topic Summarization for Twitter. ICWSM-2010.

### Reference papers

Quanzhi Li and Sameena Shah. Learning stock market sentiment lexicon and sentiment-oriented word vector from stocktwits. In Roger Levy and Lucia Specia, editors, Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), Vancouver, Canada, August 3-4, 2017, pages 301{310. Association for Computational Linguistics, 2017.

Anshul Mittal. Stock prediction using twitter sentiment analysis. 2011.
