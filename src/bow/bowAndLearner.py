import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BowAndLearner(BaseEstimator, ClassifierMixin):

	def __init__(self, bow, learner, low=0, high=-1):
		self.bow_ = bow
		self.learner_ = learner
	
	def fit(self, X, y):
		self.learner_.fit(X, y)
		self.fitted = True
		return self

	def predict(self, X):
		check_is_fitted(self, ['fitted'])
		return self.learner_.predict(X)

	def transform(self, X, low, high):
		return self.bow_.transform(X, low, high)