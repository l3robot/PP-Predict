import os
import sys

import numpy as np
from sklearn.svm import LinearSVC

from others.macros import DATA
from others.data import load_data
from others.preprocessing import target2int

from bow.bagOfWords import BagOfWords
from bow.bowAndLearner import BowAndLearner
from bow.normBagOfWords import NormBagOfWords

from xp.gridSearch import gridSearchNGramGamma

from visual.graphs import display2DGridSearch

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(' >> you need to give data')
        exit()
    else:
        file = sys.argv[1]

    X, y = load_data(os.path.join(DATA,file))
    y, reverse = target2int(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    learner = LinearSVC()

    ngrams = np.arange(1, 5)
    gammas = [0.01, 0.1, 1.0, 10.0, 100.0]

    ans = gridSearchNGramGamma(X, X_train, y_train, learner, ngrams, gammas, method='five2')
    display2DGridSearch(ans[1], ans[2], ans[3], 'ngram', 'gamma')

    best_ngrams, best_gamma = ans[0]

    print(ans[0])
    print(np.max(ans[1]))

    X = X_train + X_test

    XX = NormBagOfWords(best_ngrams, best_gamma).fit(X).transform(X)

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    learner = learner.fit(XX_train, y_train)

    print((learner.predict(XX_test) == y_test).mean())






