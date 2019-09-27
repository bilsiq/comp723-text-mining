import sys
import os


class EmailCollector:
    SPAM_CODE = 0
    HAM_CODE = 1
    START_POINT = 1
    NUMBER_OF_FOLDERS = 3

    def __init__(self,file_name):
        self.file_name = file_name
        self.training_set = []
        self.invalid_files_number = 0

    @staticmethod
    def get_file_contents(file_name):
        with open(file_name, "r") as f:
            return f.read()

    @staticmethod
    def get_files_in_path(path):
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def get_email_data(self, data_dir):

        for i in range(EmailCollector.START_POINT, EmailCollector.NUMBER_OF_FOLDERS + 1):
            current_dir = data_dir + "/" + self.file_name
            ham_file_list = EmailCollector.get_files_in_path(current_dir + str(i) + "/ham")
            spam_file_list = EmailCollector.get_files_in_path(current_dir + str(i) + "/spam")
            for spam_file in spam_file_list:
                try:
                    file_name = current_dir + str(i) + "/spam/" + spam_file
                    self.training_set.append({"emailContent": EmailCollector.get_file_contents(file_name), "class": EmailCollector.SPAM_CODE})
                except UnicodeDecodeError:
                    self.invalid_files_number += 1
            for ham_file in ham_file_list:
                try:
                    file_name = current_dir + str(i) + "/ham/" + ham_file
                    self.training_set.append({"emailContent": EmailCollector.get_file_contents(file_name), "class": EmailCollector.HAM_CODE})
                except UnicodeDecodeError:
                    self.invalid_files_number += 1
        return self.training_set


if __name__ == "__main__":
    e = EmailCollector("enron")
    t = e.get_email_data("./data")
    print(len(t))
