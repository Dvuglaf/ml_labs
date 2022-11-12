from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class TfidfTransformer:
    pass


def main():
    data = pd.read_csv("data/spam.csv", usecols=[0, 1], encoding='windows-1252')

    le = LabelEncoder()
    data['v1'] = le.fit_transform(data['v1'])

    x_train, x_test, y_train, y_test = train_test_split(data['v2'].values, data['v1'].values, test_size=0.33,
                                                        random_state=42)

    vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(2, 2))
    x_train = vectorizer.fit_transform(x_train).todense()
    x_test = vectorizer.transform(x_test).todense()


main()
