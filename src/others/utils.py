import numpy as np

from bow.bagOfWords import BagOfWords

def word_average(data, target, p):
    lenght = []

    for i, d in enumerate(data):
        if target[i] == p:
            lenght.append(len(d))

    return np.mean(lenght)

def get_top(data, target, p, n, ngram=1):
    p_data = []

    for i, d in enumerate(data):
        if target[i] == p:
            p_data.append(d)

    freqs = BagOfWords(ngram).fit(p_data).freqs_

    keys, values = list(zip(*(freqs.items())))

    top = np.argsort(values)[::-1][:n]
    return np.array(keys)[top], np.array(values)[top]