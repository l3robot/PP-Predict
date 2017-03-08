import numpy as np

def target2int(target):
    unique = np.unique(target)

    y = []

    y2t = {n: i for i, n in enumerate(unique)}
    t2y = {i: n for i, n in enumerate(unique)}

    for t in target:
        y.append(y2t[t])

    return np.array(y), t2y