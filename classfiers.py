import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np


model = GaussianNB()
data = pd.read_csv("training_vectorized_data_set.csv")
columns = data.columns.values
test = pd.read_csv("test.csv")

rfe = SelectKBest(score_func=chi2, k=25)

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
foot=rfe.fit(data[columns[0:len(columns)-1]], data["email_label"])
cols=foot.transform(data[columns[0:len(columns)-1]])
mask= rfe.get_support(indices=True)
new_features = [] # The list of your K best features
for bool, feature in zip(mask, columns):
    if bool:
        new_features.append(feature)
print(new_features)
X=data[new_features]
y=data['email_label']
model.fit(X,y)
print(cols)
column = test.columns.values
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