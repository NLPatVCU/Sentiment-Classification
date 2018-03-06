# NBSentimentClassifier
A sentiment classification script for drug reviews using NLTK Naive Bayes Classifier, Decision Tree Classifier, and Keras Neural Network.

## Requirements
* Only tested on Python 3.6
* Requires the following Python packages:
  * string
  * NLTK
  * math
  * csv
  * argparse
  * random
  * collections
  * keras
  * numpy
  * gensim
  * scikit

## File Descriptions
The main files are the python scripts "NBSentiment.py", "DTSentiment.py", and "KerasSentiment.py".  Command line options for running these files are listed below.  The .ipynb files are Jupyter Notebooks with hard coded verions of the classification code, as well as the webscraping code used to obtain the drug reviews and ratings. The other CSV and text files are input and output files.

## NBSentiment.py, DTSentiment.py, KerasSentiment.py
These scripts train and test either a Naive Bayes Classifier (NB), a Decision Tree classifier (DT), or a Neural Network to classify drug reviews.  It also can accept a text file with additional reviews to classify.  The command line options that apply to all scripts are listed below.

**Options**

  **-i**  Required. Input CSV file that includes training and testing data. Must be in the format of "review text","5", where the second entry is the rating.  See the "citalopram_effectiveness.csv" file for an example.  The program divides this data up into 3/4 used for training, and 1/4 used for testing to calculate the accuracy.
  
  **-s**  Required. Stopwords text file with a list of stopwords to remove before training the classifier or predicting sentiment class.  See the "stopwords_long.txt" file for an example.
  
  **-c**  Optional, default = None. Input text file with one review per line that needs classification. Use this option to predict semtiment class on reviews that do not yet have a rating, or to polarize neutral reviews.  See the "neutral.txt" file for an example format.
  
  **-d**  Optional, default = None. Input CSV file in the same format as the -i option.  This file contains additional ratings to classify and calculate accuracy.  This option is meant to analyze ratings from a different domain than the one being trained on.
  
  **-p**  Optional, default = ['4','5']. A list of ratings that count as positive ratings for training the classifier.  These must be strings, and must match the ratings in the input files.
  
  **-n** Optional, default = ['1','2']. A list of ratings that count as negative ratings for training the classifier.  These must be strings, and must match the ratings in the input files.
  
  **-z**  Optional, default = 1.  The number of time to repeat training the classifier to get an average accuracy when choosing different training sets of data.
  
  **-m** Required for Neural Network, not used for other NB or DT, defines the Word2Vec model to use.
**Example Usage**

 >> python NBSentiment.py -i citalopram_effectivness.csv -s stopwords_long -c neutral.txt -d gilenya_effectivness.csv -p ['4','5'] -n ['1','2'] -z 10
 >>
 >> python DTSentiment.py -i citalopram_effectivness.csv -s stopwords_long -c neutral.txt -d gilenya_effectivness.csv -p ['4','5'] -n ['1','2'] -z 10
>>
 >> python KerasSentiment.py -i citalopram_effectivness.csv -s stopwords_long -c neutral.txt -d gilenya_effectivness.csv -p ['4','5'] -n ['1','2'] -z 10 -m GoogleNews-vectors-negative300.bin
