import os
import sys

import numpy as np
from sklearn.svm import LinearSVC

from others.macros import DATA
from others.data import load_data
from others.preprocessing import target2int

from bow.bagOfWords import BagOfWords

from xp.ngramXP import ngramXP

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

    ngs, ans, scores = [], [], []

    for ng in range(2,6):
        ret = ngramXP(X, y, ng)
        ngs.append(ng)
        ans.append(ret[0])
        scores.append(np.max(ret[1]))

    max_ = np.argmax(scores)

    best_ng = ngs[max_]
    best_low, best_high = ans[max_]

    print(scores[max_])
    print(best_ng)
    print(best_low)
    print(best_high)

    X = X_train + X_test

    bow = BagOfWords(best_ng).fit(X)
    XX = bow.transform(X, best_low, best_high)

    learner = LinearSVC()

    XX_train = XX[:len(X_train)]
    XX_test = XX[len(X_train):]

    learner = learner.fit(XX_train, y_train)

    print((learner.predict(XX_test) == y_test).mean())

    






