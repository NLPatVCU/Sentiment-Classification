from keras.models import Sequential
from keras.layers import Dense, Dropout
from gensim.models import word2vec

import argparse
import numpy as np
import csv
import string
import math
import random

np.random.seed(123)

def format_sentence(sent, stopwords=None):
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    # remove stopwords
    if stopwords is not None:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word not in stopwords and word in w2v_model.wv.vocab and len(w2v_model[word].tolist()) != 0:
                filtered_words.append(w2v_model[word].tolist())
    else:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word in w2v_model.wv.vocab and len(w2v_model[word].tolist()) != 0:
                filtered_words.append(w2v_model[word].tolist())

    # comment = [item for sublist in filtered_words for item in sublist]

    # Sum all of the word vectors
    comment = [0.0 for i in range(w2vsize)]
    for word in filtered_words:
        for i in range(w2vsize):
            comment[i] += word[i]

    return comment

## Import csv file
def importCSV(filename):
    my_list = []
    with open(filename) as commentfile:
        reader = csv.DictReader(commentfile)
        for row in reader:
            my_list.append({'comment': row['comment'], 'rating': row['rating']})

    ## Parse and convert positive and negative examples.
    pos_words = []
    neg_words = []
    for c in my_list:
        tmp_com = c['comment']
        tmp_rating = c['rating']

        # remove stop words
        with open(args.s) as raw:
            stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()

            if tmp_rating in args.n:
                neg_words.append((format_sentence(tmp_com, stopwords), 0))
            if tmp_rating in args.p:
                pos_words.append((format_sentence(tmp_com, stopwords), 1))

    neg_words.extend(neg_words)
    neg_words.extend(neg_words[:100])
    print("Total Negative Instances:" + str(len(neg_words)) + "\nTotal Positive Instances:" + str(len(pos_words)))
    return neg_words, pos_words

## Parse input arguments
parser = argparse.ArgumentParser(description='Train a DNN Sentiment Classifier')
parser.add_argument('-i', metavar='inputfile', type=str,
                    help='path to the input csv file for training and testing.', required=True)
parser.add_argument('-c', metavar='toclassify', type=str, help='path to file with entries to classify.',
                        required=False, default=None)
parser.add_argument('-s', metavar='stopwords', type=str, help='path to stopwords file', required=True)
parser.add_argument('-p', metavar='posratings', type=str, help='a list of positive ratings as strings',
                    required=False, default=['4', '5'])
parser.add_argument('-n', metavar='negratings', type=str, help='a list of negative ratings as strings',
                    required=False, default=['1', '2'])
parser.add_argument('-z', metavar='iterations', type=str,
                    help='the number of times to repeat the classifier training', required=False, default=1)
parser.add_argument('-d', metavar='domain', type=str, help='a file with text from a different domain.',
                        required=False, default=None)

args = parser.parse_args()

# Build word2vec model
print("Building word2vec model on {}".format(args.i))
w2vsize = 2
sentences = word2vec.Text8Corpus(args.i)
w2v_model = word2vec.Word2Vec(sentences, size=w2vsize, min_count=1, workers=4)
neg_list,pos_list = importCSV(args.i)

### create training and test sets
## set the cutoffs
negcutoff = math.floor(len(neg_list) * 3 / 4)
poscutoff = math.floor(len(pos_list) * 3 / 4)

top10list = []
avgAccuracy = 0
for z in range(int(args.z)):
    # train = neg_list[:negcutoff] + pos_list[:poscutoff]
    # test = neg_list[negcutoff:] + pos_list[poscutoff:]
    neg_idx_train = sorted(random.sample(range(len(neg_list)), negcutoff))
    neg_train = [neg_list[i] for i in neg_idx_train]

    neg_idx_test = set(range(len(neg_list))) - set(neg_idx_train)
    neg_test = [neg_list[i] for i in neg_idx_test]

    pos_idx_train = sorted(random.sample(range(len(pos_list)), poscutoff))
    pos_train = [pos_list[i] for i in pos_idx_train]

    pos_idx_test = set(range(len(pos_list))) - set(pos_idx_train)
    pos_test = [pos_list[i] for i in pos_idx_test]

    train = neg_train + pos_train
    test = neg_test + pos_test
    print('Training on %d instances, testing on %d instances' % (len(train), len(test)))

    train_data = [x[0] for x in train]
    train_labels = [x[1] for x in train]
    test_data = [x[0] for x in test]
    test_labels = [x[1] for x in test]

model = Sequential()
model.add(Dense(w2vsize,input_shape=(w2vsize,),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(w2vsize*2,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print("Train_data: {}".format(train_data[0]))

model.fit(train_data,train_labels,epochs=100,batch_size=10)

scores = model.evaluate(test_data,test_labels)
print("\n{}: {}".format(model.metrics_names[1],scores[1]*100))

### Import the file needing classification.
if args.c is not None:
    with open(args.c) as file:
        toclass = file.readlines()
    print("Predicting...")
    predict_this = []
    sentences = []
    for sent in toclass:
        predict_this.append(format_sentence(sent))
        sentences.append(sent)
    answers = []
    predictions = model.predict(predict_this)
    for i in range(len(predictions)):
        answers.append([predictions[i],sentences[i]])
        print("{} :: {}".format(predictions[i],sentences[i]))
