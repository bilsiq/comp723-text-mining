from sklearn.preprocessing import StandardScaler
import pandas as pd
from nltk import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from EmailCollector import EmailCollector
from TextFilter import TextCleaner


def clean_text(text):
    tc = TextCleaner(text, PorterStemmer())
    return tc.remove_stop_words().remove_punctuation().stem().tokenize()


e = EmailCollector("enron", [1, 3])
e2 = EmailCollector("enron", [2])
data_1 = e.get_email_data("./data")
count_vect = CountVectorizer(analyzer=clean_text)
X = count_vect.fit_transform(data_1["emailContent"])
X_data_frame = pd.DataFrame(X.toarray())
X_data_frame.columns = count_vect.get_feature_names()
X_data_frame['email_label'] = data_1["class"]
data = X_data_frame
columns = data.columns.values
data_2 = e2.get_email_data("./data")
count_vect = CountVectorizer(analyzer=clean_text)
X_2 = count_vect.fit_transform(data_2["emailContent"])
X_2_data_frame = pd.DataFrame(X_2.toarray())
X_2_data_frame.columns = count_vect.get_feature_names()
X_2_data_frame['email_label'] = data_2["class"]
test = X_2_data_frame

X_train = data[columns[0:len(columns)-1]]
y_train = data["email_label"]
X_test = test[columns[0:len(columns)-1]]
y_test = test["email_label"]
scaler = StandardScaler()
scaler.fit(X_train)


StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier



model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

model.fit(X_train,y_train)


predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix


print(confusion_matrix(y_test,predictions))

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

X=data[columns]
y=data['email_label']
model.fit(X,y)
for i in columns[0:len(columns)-1]:
    try :
        x = test[i]
    except KeyError:
        test[i] = np.zeros(len(test))



print("Naive bayes :-")
y_pred=model.predict(test[columns])
cm = confusion_matrix(test["email_label"], y_pred)
print(cm)

print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))

