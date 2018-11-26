import pandas
import sklearn
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_breast_cancer
#percision, recall, f measure, confusion matrix

def dataChanger(dataArr):
    hold = list()
    for each in dataArr:
        hold.append(ast.literal_eval(each))
    return hold

file = 'citalopram_train.csv'
test = 'test.csv'
tfile = 'citalopram_test.csv'
out = 'randF.csv'
df = pandas.read_csv(file, names = ["data","target"])

data = np.array(df["data"])
target = np.array(df["target"])

data = dataChanger(data)

v = DictVectorizer()
enc = preprocessing.LabelEncoder()

x = v.fit_transform(data)
y = enc.fit_transform(target)

print(x)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)
print("Accuracy of training the data set {:.3f}".format(forest.score(x_train, y_train)))
print("Accuracy of of the test subset {:.3f}".format(forest.score(x_test,y_test)))
#run against test set
df2 = pandas.read_csv(tfile, names = ["data","target"])
test_features = np.array(df2["data"])
test_target = np.array(df2["target"])
test_features = dataChanger(test_features)

xt = v.fit_transform(test_features)
yt = enc.fit_transform(target)
#
# predictions = forest.predict(xt)
#
# print(predictions)
