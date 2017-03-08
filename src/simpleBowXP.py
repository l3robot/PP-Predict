import os
import sys

import numpy as np
from sklearn.svm import LinearSVC

from others.macros import DATA
from others.data import load_data
from others.preprocessing import target2int

from bow.bagOfWords import BagOfWords
from bow.bowAndLearner import BowAndLearner

from xp.gridSearch import gridSearch2D

from visual.graphs import display2DGridSearch

from sklearn.model_selection import train_test_split


def condition(h1, h2):
    return h1 <= h2

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(' >> you need to give data')
        exit()
    else:
        file = sys.argv[1]

    X, y = load_data(os.path.join(DATA,file))
    y, reverse = target2int(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    bow = BagOfWords().fit(X)
    learner = LinearSVC()

    mixed_learner = BowAndLearner(bow, learner)

    lowBounds = np.arange(0, 30, 1)
    highBounds = np.arange(20, 70, 1)

    ans = gridSearch2D(X_train, y_train, mixed_learner, lowBounds, highBounds, condition)
    # display2DGridSearch(ans[1], ans[2], ans[3], 'borne basse', 'borne haute')

    best_low, best_high = ans[0]

    print(ans[0])
    print(np.max(ans[1]))

    X = X_train + X_test

    XX = mixed_learner.transform(X, best_low, best_high)

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    mixed_learner = mixed_learner.fit(XX_train, y_train)

    print((mixed_learner.predict(XX_test) == y_test).mean())






