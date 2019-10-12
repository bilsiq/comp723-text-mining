import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def feature_select(data):
    columns = data.columns.values
    X,Y = data[columns[0:len(columns)-1]], data["email_label"]
    model = LogisticRegression()
    rfe = RFE(model, 30)
    fit = rfe.fit(X, Y)
    cols = []
    x = 0
    for col in columns:
        try:
            if fit.support_[x]:
                cols.append(col)
            x+=1
        except:
            print(x)
    return cols

model = GaussianNB()
data = pd.read_csv("test_1.csv")
columns = data.columns.values
for i in columns:
    if data[i].values.sum() < 20:
        del data[i]
test = pd.read_csv("test_1.csv")

rfe = SelectKBest(score_func=chi2, k=100)

def get_accuracy(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    result = (true_negative+true_positive)/(false_negative+false_positive+true_negative+true_positive)
    return result


def get_precision(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    try:
        result = true_positive / (true_positive + false_positive)
    except:
        result = 0.404
    return result


def get_recall(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    try:
        result = true_positive / (true_positive + false_negative)
    except:
        result = 0.404
    return result

new_features = feature_select(data)
print(new_features)
X=data[new_features]
y=data['email_label']
model.fit(X,y)
for i in new_features[0:len(new_features)-1]:
    try :
        x = test[i]
    except KeyError:
        test[i] = np.zeros(len(test))

y_pred=model.predict(test[new_features])
cm = confusion_matrix(data["email_label"], y_pred)
print(cm)

print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))