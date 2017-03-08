import os
import sys

import numpy as np
from sklearn.svm import LinearSVC

from others.macros import DATA
from others.data import load_data, load_data2
from others.preprocessing import target2int

from word2vec.word2vec import tosentences, learn, tovector

from bow.bagOfWords import BagOfWords

from xp.ngramXP import ngramXP

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(' >> you need to give data')
        exit()
    else:
        file1 = sys.argv[1]
        file2 = sys.argv[2]

    X, y = load_data2(os.path.join(DATA,file1))
    y, reverse = target2int(y)

    sentences = tosentences(X)
    model = learn(sentences)

    X, y = load_data(os.path.join(DATA,file2))
    y, reverse = target2int(y)

    print(tovector(X, model))

