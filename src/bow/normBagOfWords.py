import os
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class NormBagOfWords(BaseEstimator):

    def __init__(self, nGram=1, gamma=1):
        self.nGram_ = nGram
        self.gamma_ = gamma

    def hlim_w_(self, x):
        return range(len(x)-self.nGram_+1)

    def get_ngrams_(self, x, i):
        return ' '.join(x[i:i+self.nGram_])

    def toNGrams_(self, X):
        newX = []
        for x in X:
            newX.append(list(map(lambda i: self.get_ngrams_(x, i), self.hlim_w_(x))))
        return newX

    def check_bounds(self, x, low, high):
        return self.freqs_[x] >= low and self.freqs_[x] <= high

    def check_args(self, low, high):
        if high < low:
            print(' !BagOfWords : low bound must be lower than higher bound')
            raise ValueError

    def cutGrams_(self, X, low, high):
        newX = []
        for x in X:
            newX.append(list(filter(lambda gs: self.check_bounds(gs, low, high), x)))
        return newX

    def fit(self, X):
        self.freqs_ = Counter()
        for x in X:
            for i in self.hlim_w_(x):
                ngram = self.get_ngrams_(x, i)
                self.freqs_[ngram] += 1
        self.nbGrams_ = len(self.freqs_)
        return self

    def getFreq_(self, x):
        if x in self.freqs_:
            return self.freqs_[x]
        else:
            return 0

    def getFreqs(self, X):
        check_is_fitted(self, ['freqs_'])
        ret = []
        for x in X:
            ret.append(self.getFreq_(x))
        return ret

    def transform(self, X, low=0, high=-1, ret_dict=False):
        # regarder si le model a été ajusté
        check_is_fitted(self, ['freqs_'])

        if high > 0:
            self.check_args(low, high)
            newX = self.cutGrams_(self.toNGrams_(X), low, high)
        else:
            newX = self.toNGrams_(X)

        grams = set()

        for x in newX:
            grams |= set(x)

        grams2id = {g: i for i, g in enumerate(grams)}
        id2grams = {i: g for i, g in enumerate(grams)}

        vectorX = np.zeros((len(newX), len(grams)))

        for i, x in enumerate(newX):
            for g in x:
                vectorX[i,grams2id[g]] += 1 / (self.gamma_*self.freqs_[g])

        vectorX[vectorX != 0] = np.log(vectorX[vectorX != 0])
        vectorX = MinMaxScaler().fit_transform(vectorX)

        if ret_dict:
            return vectorX, id2grams
        else:
            return vectorX

    def transform_cut(self, X, low=0, high=-1):
        # regarder si le model a été ajusté
        check_is_fitted(self, ['freqs_'])

        if high > 0:
            self.check_args(low, high)
            newX = self.cutGrams_(self.toNGrams_(X), low, high)
            return newX
        else:
            newX = self.toNGrams_(X)
            return newX
