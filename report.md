comp723-text-mining
--------------------

Group member name | student id 
------------------|------------
Yousef Aldawoud | 18038023
Bilal Siddque |
 
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
the training set is `21636` the folders used in the training set are enron1, enron3 and enron5

Email type | number of records | percentage over all
---------|---------------------|--------------------
Spam emails | 15410 | 40.402%
Ham (legitimate) emails | 9184 | 59.598% 


### Testing set 

The testing data set contain 2 folders out of 5. The total number of records of 
the testing set is `27836` the folders used in the testing set are enron2 and, enron4

Email type | number of records | percentage over all
---------|---------------------|--------------------
Spam emails | 17598 | 58.177%
Ham (legitimate) emails | 10238 | 41.833% 

# Methods

## Data collection:

The data was provided to us by Auckland University of technology. 
The data is already separated into ham or spam folder respectfully.
 There are 5975 number of total email. 
 Number of ham emails are 3672 and the number of spam emails are 1500.
 the ratio of ham emails to spam emails is 1:3 (spam:ham). 
 The first ham email dates back to 10-12-1999, last ham email dates back to 11-01-2002. 
 The first spam email dates back to 18-12-2003, last spam email dates back to 06-09-2005.

Data preparation: 

Below are the pre-processing step we took to get the data ready for our models to be trained on

1. Adding labels to the emails respectfully (ham or spam)
2. Stemming the data
3. Removing stop words
4. Removing punctuations
5. Tokenization of the emails  

1. Adding labels

    This is the first step in our data preparation and arguably the most important one as well. 
This step in vital because, in our approach we had done unsupervised learning. 
In unsupervised learning the presence on labels on each record is important. 
So, that the model can be correctly trained.

2. Stemming
    
    Stemming is done in order to normalize textual data. 
    It also helps in reducing the number of words in the corpus. 
    Which in-turn helps machine learning algorithms to perform better.

3. Removing stop words & punctuations

    Stop words are words in sentences which do not add any additional meaning to the sentence. 
    So, therefore they can be removed from the corpus in our case. Furthermore, 
    removing punctuation helps in tokenization of the corpus, which we will get into next. 
    Removal of stop words and punctuations both results in decreasing the size of the corpus which 
    in-turn yields greater performance in machine learning algorithms.

4. Tokenization
    
    


# Results

# Discussion

# Conclusion

# Acknowledgements

# References

# Appendix