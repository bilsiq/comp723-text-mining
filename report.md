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

The data contains `33011` records in total. 
3 mails were defective (includes characters that can't be processed)
The emails were split into 5 files in no particular order.

#### Data statistics of total data set

Email type | number of records | percentage over all
---------|---------------------|--------------------
Spam emails | 16464 | 49.879%
Ham (legitimate) emails | 16544 | 50.121% 

### Training set 

The training data set contain 3 folders out of 5. The total number of records of 
the training set is `15865` the folders used in the training set are enron1, enron3 and enron5

Email type | number of records | percentage over all
---------|---------------------|--------------------
Spam emails | 15410 | 42.10%
Ham (legitimate) emails | 9187 | 57.90% 


### Testing set 

The testing data set contain 2 folders out of 5. The total number of records of 
the testing set is `17362` the folders used in the testing set are enron2 and, enron4

Email type | number of records | percentage over all
---------|---------------------|--------------------
Spam emails | 10499 | 60.471%
Ham (legitimate) emails | 7362 | 39.129% 

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

1. Adding labels to the emails respectfully (ham or spam)
2. Stemming the data
3. Removing stop words
4. Removing punctuations
5. vec of the emails  




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

### Machine learning process :-
 
#### Feature selection
 
### Executing the Algorithms
We had 2 sets of results which were generated using 2 different type of data slicing methods.
the first method was:
* Using a 30:70 testing:training set whilst maintaining the ham:spam ratio
* Using enron1, enron3 & enron5 as training and enron2 & enron4 for testing

By doing so we minimise any loss of accuracy which could occur when slicing the data set  

# Results

For the Results

| task one  | Naive Bayes  | decision tree | NN  |   |
|-----------|--------------|---------------|-----|---|
| Accuracy  |     0.84375  |      0.9375   |     |   |
| recall    |     0.66666  |      0.83334  |     |   |
| Pricision |     1        |     1         |     |   |

##30:70 split



###Naive Bayes

###Decision Tree

###Neural Network



# Discussion

## Computational power
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