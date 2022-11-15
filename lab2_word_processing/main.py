from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


class CustomTfidfTransformer:
    def __init__(self):
        self._idf = None

    def fit(self, X: np.array) -> None:
        N = X.shape[0]
        df = np.count_nonzero(X, axis=0)
        self._idf = np.log((N + 1) / (df + 1)) + 1

    def transform(self, X: np.array) -> np.array:
        tf_idf_non_normalize = np.multiply(X, self._idf)  # tf - input X in terms of quantity not frequency
        return normalize(tf_idf_non_normalize, norm='l2')

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)


def precision_metric(conf_matrix: np.array) -> float:  # TP / (TP + FP)
    return conf_matrix[0, 0] / np.sum(conf_matrix, axis=1)[0]


def recall_metric(conf_matrix: np.array) -> float:  # TP / (TP + FN)
    return conf_matrix[0, 0] / np.sum(conf_matrix, axis=0)[0]


def compare_tfidf(X: np.array) -> None:
    custom_transformer = CustomTfidfTransformer()
    res_1 = custom_transformer.fit_transform(X)

    transformer = TfidfTransformer(norm='l2')
    res_2 = transformer.fit_transform(X).toarray()

    assert np.all(np.abs(res_1 - res_2) < 0.000001)


def plot_confusion_matrix(cf_matrix: np.array) -> None:
    class_names = ['spam', 'ham']
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in class_names],
                         columns=[i for i in class_names])
    print(f"Confusion matrix:\n{df_cm}")


def main():
    data = pd.read_csv("data/spam.csv", usecols=[0, 1], encoding='windows-1252')

    le = LabelEncoder()
    data['v1'] = le.fit_transform(data['v1'])  # transform labels to integers

    x_train, x_test, y_train, y_test = train_test_split(data['v2'].values, data['v1'].values, test_size=0.33,
                                                        random_state=42)

    # Transform text to feature matrix, also delete stop-words, tokenization is performed, normalization is performed
    vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(2, 2))
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    # Delete rows where no token occurs
    remove_zero_rows = np.where(np.sum(x_train, axis=1) == 0)
    x_train = np.delete(x_train, remove_zero_rows, axis=0)
    y_train = np.delete(y_train, remove_zero_rows, axis=0)

    # compare_tfidf(x_train)

    # Apply tfidf transformation
    transformer = CustomTfidfTransformer()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    # Fit model and prediction
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Getting metrics
    cf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])  # [ [TP, FP] , [TN, FN] ]
    plot_confusion_matrix(cf_matrix)

    print(f"\nPrecision: {precision_metric(cf_matrix)}\n"
          f"Recall: {recall_metric(cf_matrix)}")


main()
