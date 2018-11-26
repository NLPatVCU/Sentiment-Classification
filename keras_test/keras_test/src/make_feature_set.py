'''
This script accepts one argument from the command line. The file
must be in a CSV format, for example: 'citalopram_effectiveness.csv'

'''

import math
import random
import csv
import string
import ast
import sys


def format_sentence(sent, stopwords=None):
    filtered_words = []
    # convert to lowercase
    sent = sent.translate(str.maketrans("", "", string.punctuation)).lower()
    #remove stopwords
    if stopwords is not None:
        com_list = sent.split()
        for word in com_list:
            if word not in stopwords:
                filtered_words.append(word)
        # sent = ' '.join(filtered_words)

    #return({word: True for word in nltk.word_tokenize(sent)})
    return({word: True for word in filtered_words})

my_list = []
with open(sys.argv[1]) as file:
    reader = csv.DictReader(file)
    for row in reader:
        my_list.append({'comment': row['comment'], 'rating': row['rating']})

pos_list=[]
neg_list=[]
neu_list=[]
for c in my_list:
    tmp_com = c['comment']
    tmp_rating = c['rating']

    #remove stop words
    with open('./stopwords_long') as raw:
        stopwords = raw.read().translate(str.maketrans("", "", string.punctuation)).splitlines()

        if tmp_com != '':
            if tmp_rating in ['1','2']:
                neg_list.append((format_sentence(tmp_com, stopwords), 'neg'))
            elif tmp_rating in ['4','5']:
                pos_list.append((format_sentence(tmp_com, stopwords), 'pos'))
            else:
                neu_list.append(tmp_com)

print("Neg:"+str(len(neg_list))+"\nPos:"+str(len(pos_list))+"\nNeutral:"+str(len(neu_list)))
pos_list[0]

negcutoff = math.floor(len(neg_list)*3/4)
poscutoff = math.floor(len(pos_list)*3/4)

train = neg_list[:negcutoff] + pos_list[:poscutoff]
test = neg_list[negcutoff:] + pos_list[poscutoff:]
print('train on %d instances, test on %d instances' % (len(train), len(test)))
print('negcutoff %d instances, poscutoff %d instances' % (negcutoff, poscutoff))

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

wtrain = csv.writer(open ('gilenya_train.csv', 'w'), delimiter=',', lineterminator='\n')
for label in train:
    wtrain.writerows([label])

wtest = csv.writer(open ('gilenya_test.csv', 'w'), delimiter=',', lineterminator='\n')
for label in test:
    wtest.writerows([label])


print('train on %d instances, test on %d instances' % (len(train), len(test)))
print('neg_idx_train %d instances, pos_idx_train %d instances' % (len(neg_idx_train), len(pos_idx_train)))
