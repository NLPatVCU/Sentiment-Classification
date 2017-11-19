## Deep neural network for classifying calendar-intervals and periods in tensorflow.  Takes a .csv file where the
# first line is the number of instances, number of features, and then a list of class labels.  All subsequent lines
# are the instance features with the last character as the class  Based on the tensorflow DNN tutorial found here:
# https://www.tensorflow.org/get_started/estimator
#---------------------------------------------------------------------------------------------------------
# Date: 9/19/17
#
# Programmer Name: Luke Maffey

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys
import string
import math
import random
import nltk
import itertools

from gensim.models import word2vec
import numpy as np
import tensorflow as tf



## Classifies the samples passed with the classifier passed
# @param classifier A tensorflow classifier
# @param samples A numpy array of features
# @return A numpy array of predicted classes
def classify(classifier, samples):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": samples},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=predict_input_fn)
    return predictions


def format_sentence(sent, stopwords=None):
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    # remove stopwords
    if stopwords is not None:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word not in stopwords and word in model.wv.vocab:
                filtered_words.append(model[word].tolist())
    else:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word in model.wv.vocab:
                filtered_words.append(model[word].tolist())

    return filtered_words


if __name__ == "__main__":

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
    model = word2vec.Word2Vec(sentences, size=w2vsize, min_count=1, workers=4)

    ## Import csv file
    my_list = []
    with open(args.i) as commentfile:
        reader = csv.DictReader(commentfile)
        for row in reader:
            my_list.append({'comment': row['comment'], 'rating': row['rating']})

    ## Parse and convert positive and negative examples.
    pos_list = []
    neg_list = []
    for c in my_list:
        tmp_com = c['comment']
        tmp_rating = c['rating']

        # remove stop words
        with open(args.s) as raw:
            stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()

            if tmp_rating in args.n:
                neg_list.append((format_sentence(tmp_com, stopwords), 0))
            if tmp_rating in args.p:
                pos_list.append((format_sentence(tmp_com, stopwords), 1))

    print("Total Negative Instances:" + str(len(neg_list)) + "\nTotal Positive Instances:" + str(len(pos_list)))

    ### create training and test sets
    ## set the cutoffs
    negcutoff = math.floor(len(neg_list) * 3 / 4)
    poscutoff = math.floor(len(pos_list) * 3 / 4)

    top10list = []
    avgAccuracy = 0
    for z in range(int(args.z)):
        #train = neg_list[:negcutoff] + pos_list[:poscutoff]
        #test = neg_list[negcutoff:] + pos_list[poscutoff:]
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
        test_data = [x[0] for x in test]
        length = max(len(sorted(train_data, key=len, reverse=True)[0]),len(sorted(test_data, key=len, reverse=True)[0]))

        train_data_dense = [xi + [[0.0 for i in range(w2vsize)]] * (length - len(xi)) for xi in train_data]
        train_targets = [x[1] for x in train]

        test_data_dense = [xi + [[0.0 for i in range(w2vsize)]] * (length - len(xi)) for xi in test_data]
        # print(test_data_dense)
        test_targets = [x[1] for x in test]


        def load_training_set(TRAINING_DATA, TRAINING_TARGET):
            training_set = tf.data.Dataset.from_tensor_slices((TRAINING_DATA,TRAINING_TARGET))
            return training_set


        def load_test_set(TEST_DATA, TEST_TARGET):
            test_set = tf.data.Dataset.from_tensor_slices((TEST_DATA,TEST_TARGET))
            return test_set


        def input_fn():
            training_set = load_training_set(train_data_dense, train_targets)
            training_set = training_set.batch(1)
            iterator = training_set.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return {'x': features}, labels

        def test_input_fn():
            test_set = load_test_set(test_data_dense, test_targets)
            test_set = test_set.batch(1)
            iterator = test_set.make_one_shot_iterator()
            test_features, labels = iterator.get_next()
            return {'x': test_features}, labels


        # def predict_input_fn():
        #     return {'x': np.array(predict_this)}


        def model_fn():
            feature_columns = [tf.feature_column.numeric_column("x", shape=[length,w2vsize], dtype=tf.float32)]
            return tf.estimator.DNNClassifier([2, 4, 2], feature_columns, "tmp/sentiment_model", n_classes=2)

        classifier = model_fn().train(input_fn, steps=1000)

        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

        if args.c is not None:
            with open(args.c) as file:
                toclass = file.readlines()
            predict_this = []
            for sent in toclass:
                predict_data = format_sentence(sent)
                predict_data_dense = predict_data + [[0.0 for i in range(w2vsize)]] * (length - len(predict_data))
                predict_this = predict_data_dense
                predict_input_fn = predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": np.array(predict_this)},
                    num_epochs=1,
                    shuffle=False)
                y = classifier.predict(input_fn=predict_input_fn)
                print(y)
                predictions = list(p["predictions"] for p in itertools.islice(y, 1))
                print("Predictions: {}".format(str(predictions)))

                # print(predict_this)
                # predictions = list(classifier.predict(predict_input_fn()))
                # predicted_classes = [p["classes"] for p in predictions]
                # print("Predictions: {} :: {}".format(predictions,sent))



