import numpy as np

from bow.bagOfWords import BagOfWords

class BowPreFilter(BagOfWords):

	def __init__(self, low, high, ngram=1):
		self.low_ = low
		self.high_ = high
		self.ngram_ = ngram
		super(BowPreFilter, self).__init__()

	def fit(self, X):
		self.X_ = X
		super(BowPreFilter, self).fit(X)
		return self

	def transform(self, X):
		newX = super(BowPreFilter, self).transform_cut(X, self.low_, self.high_)
		return BagOfWords(self.ngram_).fit(self.X_).transform(newX)

	def transform_cut(self, X):
		newX = super(BowPreFilter, self).transform_cut(X, self.low_, self.high_)
		bow = BagOfWords(self.ngram_).fit(self.X_)
		return bow.transform_cut(newX)

	def fit_transform(self, X):
		return self.fit(X).transform(X)
