from EmailCollector import EmailCollector
import pandas as pd


def print_details(email_data: pd.DataFrame):
    print("Total number of E-mails :- " + str(len(email_data)))
    print("Number of spam E-mails :- " + str(len(email_data.loc[email_data['class'] == EmailCollector.SPAM_CODE])))
    print("Number of ham (not spam) E-mails :- " + str(len(email_data.loc[email_data['class'] == EmailCollector.HAM_CODE])))
    spam_emails_percentage = len(email_data.loc[email_data['class'] == EmailCollector.SPAM_CODE]) / len(email_data) * 100
    ham_emails_percentage = len(email_data.loc[email_data['class'] == EmailCollector.HAM_CODE]) / len(email_data) * 100
    print("Spam Emails percentage:- " + str(round(spam_emails_percentage, 3)) + " %")
    print("ham Emails percentage:- " + str(round(ham_emails_percentage, 3)) + " %")


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
