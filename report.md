comp723-text-mining
--------------------

Group member name | student id 
------------------|------------
Yousef Aldawoud | 18038023
Bilal Siddque | 17956171
 
# Abstract
This project is a small illustration of modern text mining and tools, 
and how it is use these days. The projects takes an email data set that has a large number of spam emails.
Our objective is to build a text mining (machine learning) model to classify spam emails from others.


# Introduction

As the lkdrjejtj increases for example dvgfgdfg.
Which generates all types of data. In this study we will be focusing on text mining. 
Text mining is defined as “the process of analyzing natural language text to glean information 
& patterns that are useful”. 
Emailing is a very common way of communication used by both organisations and ind
ery unstructured, which makes text mining challenging.

In this study we will be training different types of models using different libraries in python particularly Natural Language Tool-Kit (NLTK) and Scikit-Learn. 
These models are going to be trained on how to correctly identify legitimate emails from spam emails.
The models we used to train and test the data are: fdhgdkjfhg dkf iofhsdklh dsoifhjkld

## Data description

The data provided is a set of E-mails that were enlisted to 2 groups 
Spam emails (Emails that were useless to the people who received it) and 
Ham E-mails that mattered for the users.

The data contains `33,008` records in total. 
3 mails were defective (includes characters that can't be processed)
The emails were split into 5 files in no particular order.

#### Data statistics of total data set

Email type | Number of records | Percentage over-all
---------|---------------------|--------------------
Spam emails | 16464 | 49.879%
Ham (legitimate) emails | 16544 | 50.121% 


# Methods

## Data collection:

The data was provided to us by Auckland University of technology. 
The data is already separated into ham or spam folder respectfully.
 There are 5975 number of total email. 
 Number of ham emails are 3672 and the number of spam emails are 1500.
 the ratio of ham emails to spam emails is 1:3 (spam:ham). 
 The first ham email dates back to 10-12-1999, last ham email dates back to 11-01-2002. 
 The first spam email dates back to 18-12-2003, last spam email dates back to 06-09-2005.

## Data preparation: 

Below are the pre-processing step we took to get the data ready for our models to be trained on

1. Splitting data into training and testing
2. Adding labels to the emails respectfully (ham or spam)
3. Stemming the data
4. Removing stop words
5. Removing punctuations
6. Vectorization 

### Splitting data into training and testing

For spliting the data set we used 2 different type of data spliting methods. the 2 methods were:
* Using a 30:70 testing:training set whilst maintaining the ham:spam ratio
* Using enron1, enron3 & enron5 as training and enron2 & enron4 for testing


We did this to ensure we can minimise any loss in accuracy. which could be caused by spliting
the data sets.

####Method One 

When preforming the 30:70 used 30 percent of the email as testing about `9,903` emails and 70% 
of the emails about `23,108` for training the each of the classifiers. 
almost a 50:50 ratio of ham:spam was maintained between the test and training set.

##### Training set 

The training set was created by using 70% of the original data set. It had equal parts ham and
spam emails, containing `11,554` of both ham and spam emails. Containing `23,108` 
emails in total.

Email type | Number of records | Percentage over-all
---------|---------------------|--------------------
Spam emails | 11,554 | 50%|
Ham (legitimate) emails | 11,554 | 50%


##### Testing set 

The testing set was created by using 30% of the original data set. it had almost equal pasts
ham and spam, containing `4,910` spam emails & `4,900` ham emails with a total emails of 
`9,810`.

Email type | Number of records | Percentage over-all
---------|---------------------|--------------------
Spam emails | 4,910 | 50.01%
Ham (legitimate) emails | 4,900 |  49.99%

####Method Two 



##### Training set 

The training data set contain 3 folders out of 5. The total number of records of 
the training set is `15,865` the folders used in the training set are enron1, enron3 and enron5

Email type | Number of records | Percentage over-all
---------|---------------------|--------------------
Spam emails | 15,410 | 42.10%
Ham (legitimate) emails | 9,187 | 57.90% 


##### Testing set 

The testing data set contain 2 folders out of 5. The total number of records of 
the testing set is `17362` the folders used in the testing set are enron2 and, enron4

Email type | Number of records | Percentage over-all
---------|---------------------|--------------------
Spam emails | 10,499 | 60.47%
Ham (legitimate) emails | 7,362 | 39.13% 

### Adding labels
To ensure that the dataset doesn't take a large space in the storage we 
converted the labels to binary `(0 for spam, 1 for ham)`


### Stemming
    
Stemming is done in order to normalize textual data. 
It also helps in reducing the number of words in the corpus. 
Which in-turn helps machine learning algorithms to perform better.

### Removing stop words & punctuations

Stop words are words in sentences which do not add any additional meaning to the sentence. 
So, therefore they can be removed from the corpus in our case. Furthermore, 
removing punctuation helps in tokenization of the corpus, which we will get into next. 
Removal of stop words and punctuations both results in decreasing the size of the corpus which 
in-turn yields greater performance in machine learning algorithms.

### Vectorization
    
Since most machine learning models doesn't work with text directly we had 
to convert the data set to a vector to be able to process it.
We used ``TFIDF (Term Frequency Inverse Document Frequency)``

```math
TF = Number of time word appear in the document / Total number of word in document 
```

### Machine learning process

#### Algorithms

We used 3 different types of algorithms, to level out the playing field in order to 
create the model which best classifies the emails with the highest accuracy, precision & recall

The three algorithms we used:
* Naive Bayes
* Decision Tree
* Neural Network

All of the algorithms were used from `sklearn` library in order to create the models.

##### Naive Bayes

The naive Bayes algorithm is one of the most powerful and commonly used algorithm in machine
learning. this algorithms uses supervised leaning and classifies using the Bayes theorem reff.
It is particularly easy to build and, one of the advantages of using Naive Bayes algorithm
in our case, is that it works great of data sets which are large reffffff.

##### Decision Tree

The decision Tree is another very commonly used algorithm used in machine learning.
It uses a supervised approach which finds the best way to split the data set on different 
conditions. 

##### Neural Networks

The specific type of neural network well be using is from the `sklearn` library call 
`MLP` which is short for Multi-layer Perceptron. Has a minimum of three layers, which uses
a supervised approach to classify.

#### Feature selection

 Feature selection is the process of choosing a subset of features from the original data set.
 This effects the machine learning process in multiple ways, in consequence it also helps increase 
 the accuracy of the models. It helps the machine learning process by, reducing the overall corpus
 size, by decreasing the number of features the algorithm has to process. Making training and applying
 new algorithms easier. Another way feature selection positively affects the machine learning process
 is by getting rid of the feature which are noisy. Resulting in the algorithms preforming better.

# Results

Since we had split the data 2 different ways. We obtained 2 different set of results for the 
algorithms we ran.



##Splitting method One

As discussed earlier, method one of splitting. Splits the data set into 30% for testing and 
70% for training.

###Findings

 When running the algorithms, we found that the decision tree algorithm out preformed both
 Naive Bayes and the MLP classifier. With an accuracy of 0.94, recall of 0.83 & 
 precision of 0.94.

###Summary

| task one  | Naive Bayes  | Decision tree | NN  |
|-----------|--------------|---------------|-----|
| Accuracy  |     0.84375  |      0.9375   |  0.62444   | 
| recall    |     0.66666  |      0.83334  |  0.523174   |
| Precision |     0.89        |     0.94         |  0.611357   | 



##Splitting method Two

As discused earlier, method two of splitting. splits the enron folder for training and testing.
Enron1, enron3 & enron5 for training and enron2 & enron4 for testing.

###Findings

Similar finding were observed, when splitting the data set by method 2. The decision tree 
classifier had still the best performance. However a significant increase in the Naive Bayes classifier
was observed. On the other Hand, MLP classifier performed even worse than the performance in 
method 1.

###Summary

| task two  | Naive Bayes  | Decision tree | NN  
|-----------|--------------|---------------|-----|
| Accuracy  |    0.912489   |   0.92981      |  0.59484   |  
| recall    |    0.76248   |    0.85614    | 0.51156    |  
| Precision |     0.90632        |     0.93845         | 0.57591   |


# Discussion

## Explanation of Results
In this study we mainly focused on building the best classifier to correctly classify emails
as spam or ham (legitimate) emails. Looking at the problem, we wanted to create a classifier 
that will have a very less likelihood of misclassify ham (legitimate) emails as spam. In 
saying that, our main goal for finding the best classifier was having the best 
precision metric.

In our case precision of the classifier was more significant of a metric than accuracy and 
recall.

The best classifier we found was the decision tree classifier. Due to the fact, it scored the 
highest results in both different data splitting methods.

## Computational power || Problems we Faced
One of the main challenges we faced is not having 
enough processing power to run the algorithms for vectorization.
When we tried to run the vectorizing algorithms on our PCs we ended up getting 
`out of memory` error (An error that occurs when there's a lack of RAM memory for the running script)


###  Possible solutions 

#### Adding more RAM (Random access memory)

As discussed above the main problem we faced is not having enough RAM memory to carry out
the process of vectorizing the dataset. Adding more RAM was a possible solution however 
in our case this option wasn't feasible due to the uncertainty of the amount of the RAM we needed
to run the algorithms efficiently.


#### Reducing corpus size 

The reason that the algorithm raises a memory error is 
having a lot of documents to process. Hence reducing corpus size was a viable option to 
make the algorithms run our PCs. However, reducing the corpus size will have
negative effects on the machine learning model accuracy.

#### Process outsourcing

This approach seems to be the most viable option in our case. 
We used Azure cloud services, in order to run our algorithms.
Since we had experience in managing servers. We took this approach, as
our solutions, to the main problem we came across with.

# Conclusion

# Acknowledgements

# References

# Appendix