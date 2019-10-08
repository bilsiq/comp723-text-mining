from EmailCollector import EmailCollector
import pandas as pd
from matplotlib import pyplot
from FeatureCreator import FeatureCreator
import numpy as np


def print_details(email_data: pd.DataFrame):
    print("Total number of E-mails :- " + str(len(email_data)))
    print("Number of spam E-mails :- " + str(len(email_data.loc[email_data['class'] == EmailCollector.SPAM_CODE])))
    print("Number of ham (not spam) E-mails :- " + str(len(email_data.loc[email_data['class'] == EmailCollector.HAM_CODE])))
    spam_emails_percentage = len(email_data.loc[email_data['class'] == EmailCollector.SPAM_CODE]) / len(email_data) * 100
    ham_emails_percentage = len(email_data.loc[email_data['class'] == EmailCollector.HAM_CODE]) / len(email_data) * 100
    print("Spam Emails percentage:- " + str(round(spam_emails_percentage, 3)) + " %")
    print("ham Emails percentage:- " + str(round(ham_emails_percentage, 3)) + " %")


def count_char(text, ignore_char=[]):
    char_length = len(text)
    for char in ignore_char:
        char_length -= text.count(char)
    return char_length

if __name__ == "__main__":

    print("Total Data set \n-----------------\n")
    data = EmailCollector("enron", range(1, 6)).get_email_data("./data")
    print_details(data)

    print("Training data set \n-----------------\n")
    training_data = EmailCollector("enron", [1, 3, 5]).get_email_data("./data")
    print_details(training_data)

    print("Testing data set \n-----------------\n")
    testing_data = EmailCollector("enron", [2, 4]).get_email_data("./data")
    print_details(testing_data)

    # testing features


    pins = np.linspace(0, 200, 40)
    t_spam = FeatureCreator(training_data.loc[training_data['class'] == EmailCollector.SPAM_CODE])
    t_ham = FeatureCreator(training_data.loc[training_data['class'] == EmailCollector.HAM_CODE])
    t_spam_data =t_spam.add_feature("cha_count", count_char, "emailContent").get_data()
    t_ham_data =t_ham.add_feature("cha_count", count_char, "emailContent").get_data()
    pyplot.hist(t_spam_data["cha_count"], pins, alpha=0.5,label="spam",normed=True)
    pyplot.hist(t_ham_data["cha_count"], pins, alpha=0.5,label="ham",normed=True)
    pyplot.legend(loc="upper left")
    pyplot.show()
