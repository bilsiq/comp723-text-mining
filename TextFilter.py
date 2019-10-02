from nltk import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from EmailCollector import EmailCollector


class TextCleaner:
    def __init__(self,text,stemmer):
        self.text = text
        self.stemmer = stemmer

    def stem(self):
        text_array = [self.stemmer.stem(word) for word in self.tokenize()]
        return TextCleaner(" ".join(text_array), self.stemmer)

    def remove_stop_words(self):
        filtered_words = [word for word in self.tokenize() if word not in stopwords.words('english')]
        return TextCleaner(" ".join(filtered_words),self.stemmer)

    def tokenize(self):
        return word_tokenize(self.text)

    def remove_punctuation(self):
        filtered_words = [word for word in self.tokenize() if word not in punctuation]
        return TextCleaner(" ".join(filtered_words),self.stemmer)

    def __str__(self):
        return self.text


def clean_text(text):
    tc = TextCleaner(text, PorterStemmer())
    return tc.remove_stop_words().remove_punctuation().stem().tokenize()


if __name__ == "__main__":
    s = "Hi my name is Zammel, and I am a runner I prefer running in wide areas."
    e = EmailCollector("enron", [1, 3, 5])
    data = e.get_email_data("./data")
    count_vect = CountVectorizer(analyzer=clean_text)
    X = count_vect.fit_transform(data["emailContent"])
    X_data_frame = pd.DataFrame(X.toarray())
    X_data_frame.columns = count_vect.get_feature_names()
    X_data_frame['email_label'] = data["class"]
    print(X_data_frame)
    X_data_frame.to_csv(r"./vectorized_data_set.csv")
