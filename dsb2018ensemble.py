from sklearn.linear_model import LogisticRegression


class Ensembleer(object):
    """

    """

    def __init__(self, classifier):
        """

        :param classifier: a fit-transform object
        """
        self.classifier = classifier

    def fit(self, X, Y):
        """


        :param X: dict van id naar data
        :param Y: dict van id naar labeled mask
        :return:
        """
        X_descriptives = self.make_descriptives(X)
        Y_huub, Y_sander = self.get_predictions(X)

        #### WRITE CODE HERE (~ 10 lines)
        X_concat = None  # Concatenation of the descriptives and the predictions of two classifiers
        # TODO align X_concat and Y
        #### END CODE HERE
        self.classifier.fit(X_concat, Y)

    def transform(self, X):
        """

        :param X: dict van id naar data
        :return: Y weighted labels
        """
        #### WRITE CODE HERE (~20 lines)

        #### END CODE HERE
        return None

    def make_descriptives(self, X):
        """
        Makes features about the image, pixel intensity etc.

        :param X:
        :return:
        """
        #### WRITE CODE HERE (~10 lines)

        #### END CODE HERE

    def get_predictions(self, X):
        """
        Gets predictions

        :param X:
        :return:
        """
        ##### WRITE CODE HERE (~ 30 lines)
        Y_huub = None
        Y_sander = None
        ##### END CODE HERE
        return Y_huub, Y_sander


def uniform_data():
    """

    :return: X, Y aligned
    """
    return None, None


def write_predictions():
    """
    Writes a csv with the labels
    """


def main():
    ensemble = Ensembleer(LogisticRegression())
    X, Y = uniform_data()
    ensemble.fit(X, Y)
    ensemble.transform(X)


if __name__ == "__main__":
    main()
