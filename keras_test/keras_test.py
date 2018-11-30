'''
Written by Nathan West
VIP Nanoinformatics - Sentiment Classification using Neural Network
10/08/18

This file builds a neural network to analyze reviews of
of clinical drugs, such as citalopram. The
'''

import ast
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
import sklearn.preprocessing as process
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn import svm


def modify_data(dataArr):
    hold = list()
    for each in dataArr:
        hold.append(ast.literal_eval(each))
    return hold


# to inverse transform it is expecting a 2D-Array because that is what it recieved initially.
def createFeatureVector(v, original):
    newList = list()
    hold = np.ones((2, original[0].size))
    transformer = v.inverse_transform(hold)
    for i in transformer[0].keys():
        newList.append(i)
    feature_vector = np.array(newList)
    return feature_vector


# accepts one object to be transformed to binary feature vector manually. returned in the form of numpy array
def BinaryFeatureVector(word_vector, x):
    hold = list()
    for each in x:
        local = []
        for i in range(len(word_vector)):
            if word_vector[i] in each:
                local.append(1)
            else:
                local.append(0)
        hold.append(local)
    bin_vec = np.array(hold)
    return bin_vec


training_file = open('citalopram_train.csv')
test_file = open('citalopram_test.csv')
csv_logger = CSVLogger('log.csv', append=True, separator=',')

df = pd.read_csv(training_file, names=['data', 'target'])

data = np.array(df['data'])
target = np.array(df['target'])

dv = DictVectorizer(sparse=False)
data = modify_data(data)
train_data = dv.fit_transform(data)
# size_correction = createFeatureVector(dv, train_data)

enc = process.LabelEncoder()
y = enc.fit_transform(target)

count = 0
tot_tn = 0
tot_fn = 0
tot_tp = 0
tot_fp = 0

model_scores = []
class_weights = {0: 3, 1: 1}
in_dim = len(train_data[0])

# testing over and under sampling methods


# train a neural network with 10-fold cross validation
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
kfold = KFold(n_splits=10, shuffle=True, random_state=None)
clf = svm.SVC(gamma='scale')
bbc = BalancedBaggingClassifier(base_estimator=clf, random_state=20, sampling_strategy='not majority')

for train, test in skfold.split(train_data, y):
    model = Sequential()  # create a linear stack of layers with an activation function rectified linear unit (relu)
    model.add(Dense(20, input_dim=in_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dropout(0.6))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    count += 1

    # build/compile the actual neural network
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # using SMOTE to assist with imbalanced data set before training

    # clf.fit(train_data[train], y[train])
    bbc.fit(train_data[train], y[train])

    # tried using random sampler to fix imbalanced data set
    # X_resampled, y_resampled = rus.fit_resample(train_data[train], y[train])

    '''
    train the model
        - KFold, no SMOTE : 75.35
        - KFold, SMOTE : 74.43
        - StratifedKFold, no SMOTE : 74.33
        - StratifedKFold, SMOTE :
        - KFold, ROS : 70.87
        - StratifiedKFold, ROS :
        - StratifiedKFold, BalancedBaggingClassifier : 76.55
    '''

    model.fit(
        train_data[train], y[train], epochs=50,
        batch_size=20, class_weight=class_weights,
        verbose=0, callbacks=[csv_logger])

    # grab model score - includes [error rate, accuracy]
    raw_score = model.evaluate(train_data[test], y[test], verbose=0)
    print("[err, acc] of fold {} : {}".format(count, raw_score))
    predictions = model.predict(train_data[test], batch_size=None, verbose=0, steps=None)

    scores = list()
    for i in range(len(predictions)):
        temp = predictions[i][0]
        temp = round(temp, ndigits=None)
        temp = int(temp)
        scores.append(temp)

    tn, fp, fn, tp = confusion_matrix(y[test], scores).ravel()
    tot_tn += tn
    tot_fp += fp
    tot_fn += fn
    tot_tp += tp

    model_scores.append(raw_score[1]*100)

print("Average accuracy (train set) - %.2f%%\n" % (np.mean(model_scores)))
print("Confusion Matrix:")
print("          Predicted pos     Predicted neg")
print("Actual pos    {} (TP)        {} (FN)".format(tot_tp, tot_fn))
print("Actual neg     {} (FP)        {} (TN)".format(tot_fp, tot_tn))
