import string

import pandas as pd


class FeatureCreator:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def add_feature(self, feature_name: string, feature_method, used_feature):
        self.data[feature_name] = self.data[used_feature].apply(feature_method)
        return self

    def get_data(self):
        return self.data


def count_char(text, ignore_char=[]):
    char_length = len(text)
    for char in ignore_char:
        char_length -= text.count(char)
    return char_length


if __name__ == "__main__":
    print(count_char("Hello world", ["l", " "]))
