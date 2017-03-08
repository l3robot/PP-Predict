import os
import sys

import numpy as np
from scipy.stats import f_oneway
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from others.macros import DATA
from others.data import load_data
from others.preprocessing import target2int

from xp.splitter import KFold_score

from bow.bagOfWords import BagOfWords
from bow.bowPreFilter import BowPreFilter
from bow.normBagOfWords import NormBagOfWords

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(' >> you need to give data')
        exit()
    else:
        file = sys.argv[1]

    """
    simple bow
    """

    X, y = load_data(os.path.join(DATA,file))
    y, reverse = target2int(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    low = 18
    high = 53

    X = X_train + X_test

    XX = BagOfWords().fit(X).transform(X, low, high)

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    learner = LinearSVC().fit(XX_train, y_train)

    print((learner.predict(XX_test) == y_test).mean())

    """
    bow ngram
    """

    X, y = load_data(os.path.join(DATA,file))
    y, reverse = target2int(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    low = 9
    high = 42

    X = X_train + X_test

    XX = BagOfWords(2).fit(X).transform(X, low, high)

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    learner = LinearSVC()

    learner = LinearSVC().fit(XX_train, y_train)

    print((learner.predict(XX_test) == y_test).mean())

    """
    bow norm
    """

    X, y = load_data(os.path.join(DATA,file))
    y, reverse = target2int(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    X = X_train + X_test

    XX = NormBagOfWords(1, 100.0).fit(X).transform(X)

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    learner = LinearSVC()

    learner = LinearSVC().fit(XX_train, y_train)

    print((learner.predict(XX_test) == y_test).mean())

    




