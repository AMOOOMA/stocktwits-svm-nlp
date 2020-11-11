# Stocktwits messaages sentiment analysis
WARNING: This project is still in development!

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
Training using [CLS] sentence wise vector:
```
Kernel sigmoid scores: 0.6771027155932816 [0.6582809224318659, 0.6848357791754018, 0.6855345911949685, 0.686932215234102, 0.66993006993007]
Kernel rbf scores: 0.7341354522486597 [0.726764500349406, 0.7365478686233403, 0.7442348008385744, 0.7239692522711391, 0.7391608391608392]
Kernel poly scores: 0.6997467661618605 [0.7030048916841369, 0.7030048916841369, 0.6932215234102026, 0.7113906359189378, 0.6881118881118881]
Kernel linear scores: 0.592256967351307 [0.5751222921034241, 0.6079664570230608, 0.5716282320055905, 0.6079664570230608, 0.5986013986013986]
```

Training using word embeddings sum average:
```
Kernel sigmoid scores: 0.6736104147424903 [0.6617749825296995, 0.6771488469601677, 0.660377358490566, 0.6862334032145353, 0.6825174825174826]
Kernel rbf scores: 0.7532850517756178 [0.7344514325646401, 0.7540181691125087, 0.7603074772886094, 0.7631027253668763, 0.7545454545454545]
Kernel poly scores: 0.7065979582960715 [0.7141858839972047, 0.7071977638015374, 0.7044025157232704, 0.6988120195667366, 0.7083916083916084]
Kernel linear scores: 0.6161605410662014 [0.6401118099231307, 0.5807127882599581, 0.6093640810621943, 0.6219426974143956, 0.6286713286713287]
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

### Credits
The tokenizer we used is from project [tweetmotif](https://github.com/brendano/tweetmotif).

Brendan O'Connor, Michel Krieger, and David Ahn. TweetMotif: Exploratory Search and Topic Summarization for Twitter. ICWSM-2010.

### Reference papers

Quanzhi Li and Sameena Shah. Learning stock market sentiment lexicon and sentiment-oriented word vector from stocktwits. In Roger Levy and Lucia Specia, editors, Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), Vancouver, Canada, August 3-4, 2017, pages 301{310. Association for Computational Linguistics, 2017.

Anshul Mittal. Stock prediction using twitter sentiment analysis. 2011.
