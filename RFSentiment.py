### RFSentiment
### Author: Luke Maffey
### 16 July 2018
### This program takes in two csv files containing text to be classified as positive or negative sentiment.
### The first file contains data for training and testing, the second contains data to be classified.
### NOT WORKING, DOESN"T LIKE FEATURES OF DIFFERING LENGTHS

import string
from sklearn.ensemble import RandomForestClassifier
import nltk.classify.util
import nltk
import math
import csv
import argparse
import random
from collections import Counter


## Formats a string for input in the NB Classifier by converting all to lowercase and removing al punctuation.
# @author Amy Olex
# @param sent The string to be formatted.
# @param stopwords A list of stopwords to be removed. Default is None.
# @return A dictionary of each word as the key and True as the value.
def format_sentence(sent, stopwords=None):
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    #remove stopwords
    if stopwords is not None:
        com_list = sent.split()
        filtered_words = []
        for word in com_list:
            if word not in stopwords:
                filtered_words.append(word)
        sent = ' '.join(filtered_words)
    
    return({word: True for word in nltk.word_tokenize(sent)})

#####
## End Function
#####




if __name__ == "__main__":
    
    ## Parse input arguments
    parser = argparse.ArgumentParser(description='Train a RF Sentiment Classifier')
    parser.add_argument('-i', metavar='inputfile', type=str, help='path to the input csv file for training and testing.', required=True)
    parser.add_argument('-c', metavar='toclassify', type=str, help='path to file with entries to classify.', required=False, default=None)
    parser.add_argument('-s', metavar='stopwords', type=str, help='path to stopwords file', required=True)
    parser.add_argument('-p', metavar='posratings', type=str, help='a list of positive ratings as strings', required=False, default=['4','5'])
    parser.add_argument('-n', metavar='negratings', type=str, help='a list of negative ratings as strings', required=False, default=['1','2'])   
    parser.add_argument('-z', metavar='iterations', type=str, help='the number of times to repeat the classifier training', required=False, default=1)   
    parser.add_argument('-d', metavar='domain', type=str, help='a file with text from a different domain.', required=False, default = None)   
    
    args = parser.parse_args()
    
    
    
    ## Import csv file
    my_list = []
    with open(args.i) as commentfile:
        reader = csv.DictReader(commentfile)
        for row in reader:
            my_list.append({'comment': row['comment'], 'rating': row['rating']})
    
    ## Parse and convert positive and negative examples.
    pos_list=[]
    neg_list=[]
    for c in my_list:
        tmp_com = c['comment']
        tmp_rating = c['rating']

        #remove stop words
        with open(args.s) as raw:
            stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()
 
            if tmp_rating in args.n:
                neg_list.append((format_sentence(tmp_com, stopwords), 'neg'))
            if tmp_rating in args.p:
                pos_list.append((format_sentence(tmp_com, stopwords), 'pos'))

    
    print("Total Negative Instances:"+str(len(neg_list))+"\nTotal Positive Instances:"+str(len(pos_list)))
    
    ### create training and test sets
    ## set the cutoffs
    negcutoff = math.floor(len(neg_list)*3/4)
    poscutoff = math.floor(len(pos_list)*3/4)

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
        train_data_keys = [list(x.keys()) for x in train_data]
        print(train_data_keys)
        train_labels = [x[1] for x in train]
        test_data = [x[0] for x in test]
        test_data_keys = [key for key in test_data]
        test_labels = [x[1] for x in test]
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(train_data_keys, train_labels)
        accuracy = classifier.score(test_data_keys,test_labels)
        avgAccuracy = avgAccuracy + accuracy
        print('Classifier accuracy:', accuracy)
        print(classifier.feature_importances_)
        ### Import the file needing classification.
        if args.c is not None:
            with open(args.c) as file:
                toclass = file.readlines()
    
            for sent in toclass:
                print(classifier.predict(format_sentence(sent))+" :: "+sent)
    
    ### Count the occurences of each word that appeared in the top 10 over the 20 runs.
    print("Average Accuracy: "+ str(avgAccuracy/int(args.z)))
    my_counts = Counter(top10list)
    print(my_counts)
    
    
    if args.d is not None:
        domain_list = []
        with open(args.d) as domainfile:
            reader = csv.DictReader(domainfile)
            for row in reader:
                domain_list.append({'comment': row['comment'], 'rating': row['rating']})
        print(str(len(domain_list)))
        
        d_list = []
        for c in range(len(domain_list)):
            tmp_c = domain_list[c]['comment']
            tmp_r = domain_list[c]['rating']

            if tmp_r in args.n:
                d_list.append((format_sentence(tmp_c, stopwords), 'neg'))
            if tmp_r in args.p:
                d_list.append((format_sentence(tmp_c, stopwords), 'pos'))
        
        #classifier2 = NaiveBayesClassifier.train(domain_list)
        # domain_accuracy = nltk.classify.util.accuracy(classifier, d_list)
        # print('Classifier domain shift accuracy:', domain_accuracy)
        
        
           
            
            
            
            
            