import time

import numpy as np

from bow.bagOfWords import BagOfWords
from bow.normBagOfWords import NormBagOfWords

from xp.splitter import KFold_score, five2_score

def gridSearch2D\
(X, y, learner, hyp1_set, hyp2_set, condition, k=3, return_errors=False, random_state=42, method='kfold'):
	g_hyp1, g_hyp2, scores = [], [], []

	rs = random_state

	total = len(hyp1_set) * len(hyp2_set) 
	step = int(total / 10)

	start = time.time()

	for h1 in hyp1_set:
		for h2 in hyp2_set:
			rs += 1
			if condition(h1, h2):
				try:
					XX = learner.transform(X, h1, h2)
					if method == 'kfold' :
						scores.append(KFold_score(XX, y, learner, k=k, random_state=rs))
					elif method == 'five2' :
						scores.append(five2_score(XX, y, learner, random_state=rs))
					g_hyp1.append(h1)
					g_hyp2.append(h2)
				except (np.linalg.linalg.LinAlgError, ValueError):
					print(' !gridSearch2D : {}-{} rejected'.format(h1, h2))
			if (rs - random_state) % step == 0:
				now = time.time()
				print(' +gridSearch2D : {}% t:{:.2f}'.format(int((rs - random_state)/total*100), now - start))
				start = now

	best = np.argmax(scores)
	bests = (g_hyp1[best], g_hyp2[best])

	if return_errors:
		return bests, (1 - np.array(scores)), g_hyp1, g_hyp2
	else:
		return bests, np.array(scores), g_hyp1, g_hyp2

def gridSearchNGram\
(X, X_train, y_train, learner, ngrams, k=3, return_errors=False, random_state=42, method='kfold'):
	g_ngrams, scores = [], []

	rs = random_state

	total = len(ngrams)
	step = int(total / 10)

	start = time.time()

	for ng in ngrams:
		rs += 1
		bow = BagOfWords(ng).fit(X)
		XX = bow.transform(X_train)
		if method == 'kfold' :
			scores.append(KFold_score(XX, y_train, learner, k=k, random_state=rs))
		elif method == 'five2' :
			scores.append(five2_score(XX, y_train, learner, random_state=rs))
		g_ngrams.append(ng)
		if (rs - random_state) % step == 0:
			now = time.time()
			print(' +gridSearch2D : {}% t:{:.2f}'.format(int((rs - random_state)/total*100), now - start))
			start = now

	best = np.argmax(scores)
	bests = g_ngrams[best]

	if return_errors:
		return bests, (1 - np.array(scores)), g_ngrams
	else:
		return bests, np.array(scores), g_ngrams

def gridSearchNGramGamma\
(X, X_train, y_train, learner, ngrams, gamma, k=3, return_errors=False, random_state=42, method='kfold'):
	g_ngrams, g_gamma, scores = [], [], []

	rs = random_state

	total = len(ngrams) * len(gamma)
	step = int(total / 10)

	start = time.time()

	for ng in ngrams:
		for ga in gamma:
			rs += 1
			bow = NormBagOfWords(ng, ga).fit(X)
			XX = bow.transform(X_train)
			if method == 'kfold' :
				scores.append(KFold_score(XX, y_train, learner, k=k, random_state=rs))
			elif method == 'five2' :
				scores.append(five2_score(XX, y_train, learner, random_state=rs))
			g_ngrams.append(ng)
			g_gamma.append(ga)
			if (rs - random_state) % step == 0:
				now = time.time()
				print(' +gridSearch2D : {}% t:{:.2f}'.format(int((rs - random_state)/total*100), now - start))
				start = now

	best = np.argmax(scores)
	bests = (g_ngrams[best], g_gamma[best])

	if return_errors:
		return bests, (1 - np.array(scores)), g_ngrams, g_gamma
	else:
		return bests, np.array(scores), g_ngrams, g_gamma

