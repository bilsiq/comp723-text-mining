# Appendix

The code is seprated to different files and classes. Each class has a specific functionality.

## EmailCollector.py

This class is responsible for reading all the emails and converting them into a dataframe 
that can be easily processed by any classifiers.

- importing the necessary libraries.

```python
import sys
import os
from nltk import PorterStemmer
import pandas as pd

```  
- Init method:

file_name defines the directory name such as `enron` and file_number defines the number of files there.
`SPAM_CODE` is the code assigned to the spam emails and `HAM_CODE` is the number assigned to the legit emails
```python
class EmailCollector:
    SPAM_CODE = 0
    HAM_CODE = 1
    START_POINT = 1
    NUMBER_OF_FOLDERS = 2

    def __init__(self, file_name, file_numbers):
        self.file_name = file_name
        self.training_set = []
        self.invalid_files_number = 0
        self.file_numbers = file_numbers
```

Example :
```python
emailCollector = EmailCollector("enron",[1,2,5])
# it will read the files inside the folders enron1, enron2, and enron5
```

- get_file_contents method:

This method is used to get the file contents of any file. It is a static method .
file name refers to the targeted file.
```python
class EmailCollector:
...
    @staticmethod
    def get_file_contents(file_name):
        with open(file_name, "r") as f:
            return f.read()
```
Example:-

```python
'''
file in directory
./main.py
./note.txt >> "Hello world"
'''
note = EmailCollector::get_file_contents("note.txt")
print(note)
# outputs note.txt content 'Hello world'
```

- get_files_in_path method:
Gets every file in a specific directory.

```python
class EmailCollector:
...
    @staticmethod
    def get_files_in_path(path):
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

```

Example :- 
```python
'''
file in directory
./main.py
./notes/.
./notes/note-1.txt
./notes/note-2.txt
'''
notes = EmailCollector::get_files_in_path("/notes")
print(notes)
# outputs ['note-1.txt', 'note-2.txt']
```

- get_email_data method:

Gets the data inside the files and converts it to a dataframe.

```python
emailCollector = EmailCollector("enron",[1,2,5])
emails = emailCollector.get_email_data("data")
print(emails)
# outputs emails with label
```

## TextFilter.py
A class that cleans the text of unnecessary data
- importing necessary libraries

```python
from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from EmailCollector import EmailCollector

```

- Init method

sets the text and the stemmer type for the email
```python
class TextCleaner:
    def __init__(self,text,stemmer):
        self.text = text
        self.stemmer = stemmer


```

- stem method

Stems the text.

```python
class TextFilter:
    ...
    def stem(self):
        text_array = [self.stemmer.stem(word) for word in self.tokenize()]
        return TextCleaner(" ".join(text_array), self.stemmer)

```
- remove_stop_words method:

removes stop words
```python
class TextCleaner:
    ...
    def remove_stop_words(self):
        filtered_words = [word for word in self.tokenize() if word not in stopwords.words('english')]
        return TextCleaner(" ".join(filtered_words),self.stemmer)

```

- remove_punctuation :

removes all punctuation
```python
class TextCleaner:
    ...
    def remove_punctuation(self):
        filtered_words = [word for word in self.tokenize() if word not in punctuation]
        return TextCleaner(" ".join(filtered_words),self.stemmer)

```

- \_\_str\_\_ :

returns string after being processed:

```python
class TextCleaner:
    ...
    def __str__(self):
        return self.text
```

## classfiers.py

- Importing necessary libraries:

```python
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

```

- feature_select

Selects a number of features
```python
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

``` 

- get_accuracy method

Takes confusion matrix as a parameter to calculate the accuracy. 
```python
def get_accuracy(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    result = (true_negative+true_positive)/(false_negative+false_positive+true_negative+true_positive)
    return result
```

- get_precision method

Takes confusion matrix as a parameter to calculate the precision.

```python

def get_precision(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    try:
        result = true_positive / (true_positive + false_positive)
    except:
        result = 0.404
    return result

```

- get_recall method

Takes confusion matrix as a parameter to calculate the recall.

```python

def get_recall(cm):
    true_positive, false_positive = cm[0]
    false_negative, true_negative = cm[1]
    try:
        result = true_positive / (true_positive + false_negative)
    except:
        result = 0.404
    return result
```


- Creating and training naive bayes model

```python

model = GaussianNB()
data = pd.read_csv("training_file.csv")
columns = data.columns.values

X=data[columns[0:len(columns)-1]]
y=data['email_label']
model.fit(X,y)
```

- Using the model to predict the test dataset

Also creating a confusion matrix from the test data set and the model predicted values
```python
y_pred=model.predict(test[columns[0:len(columns)-1]])
cm = confusion_matrix(test["email_label"], y_pred)
```
- printing out the accuracy ,recall and precision
```python
print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))

```
- Creating and training decision tree model: 
```python

model = DecisionTreeClassifier()
model.fit(X,y)
print("Decision tree :-")
y_pred=model.predict(test[new_features])
cm = confusion_matrix(data["email_label"], y_pred)
print(cm)

print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))

```

## NN.py

- Creating and training neural network model

```python
model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
model.fit(X_train,y_train)

```

- predicting the test dataset

```python
predictions = model.predict(X_test)

```

- showing confusion matrix

```python
print(confusion_matrix(y_test,predictions))

```

- showing recall, precision, and accuracy

```python

print("Precision = ",get_precision(cm))
print("recall = ",get_recall(cm))
print("Accuracy = ",get_accuracy(cm))

```