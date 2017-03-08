import numpy as np
from sklearn.model_selection import KFold, train_test_split

def KFold_score(X, y, learner, k=3, random_state=42, get_scores=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    scores = []
    
    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        l = learner.fit(X_train, y_train)
        scores.append((l.predict(X_valid) == y_valid).mean())

    if get_scores:
    	return scores
    else:
    	return np.mean(scores)

def five2_score(X, y, learner, random_state=42):

	scores = []

	for i in range(5):
		X_train, X_valid, y_train, y_valid = train_test_split(X, y,\
											 test_size=0.5, random_state=random_state+i)

		learner = learner.fit(X_train, y_train)
		score = (learner.predict(X_valid) == y_valid).mean()

		learner = learner.fit(X_valid, y_valid)
		score += (learner.predict(X_train) == y_train).mean()

	scores.append(score / 2.0)

	return np.mean(scores)
