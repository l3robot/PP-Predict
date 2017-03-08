import json

import numpy as np
from sklearn.svm import LinearSVC

from bow.bagOfWords import BagOfWords
from bow.bowAndLearner import BowAndLearner
from bow.bowPreFilter import BowPreFilter

from xp.gridSearch import gridSearch2D

from visual.graphs import save2DGridSearch


def condition(h1, h2):
    return h1 <= h2

def ngramXP(X, y, ngram):

    print(' +ngram : begin with ngram {}'.format(ngram))

    bow = BagOfWords(ngram).fit(X)
    learner = LinearSVC()

    mixed_learner = BowAndLearner(bow, learner)

    lowBounds = np.arange(0, 30, 1)
    highBounds = np.arange(20, 70, 1)

    ans = gridSearch2D(X, y, mixed_learner, lowBounds, highBounds, condition, method='five2')
    save2DGridSearch(ans[1], ans[2], ans[3], 'borne basse', 'borne haute', ngram)

    with open('results-{}.json'.format(ngram), 'wb') as f:
        json.dump('{}'.format(ans), f)

    return ans
