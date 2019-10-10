import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np


model = GaussianNB()
data = pd.read_csv("vectorized_data_set.csv")
columns = data.columns.values
test = pd.read_csv("test.csv")


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
model.fit(data[columns[0:len(columns)-1]], data["email_label"])
column = test.columns.values
for i in columns[0:len(columns)-1]:
    try :
        x = test[i]
    except KeyError:
        test[i] = np.zeros(len(test))

y_pred=model.predict(test)
cm = confusion_matrix(data["email_label"], y_pred)
print(cm)

print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))