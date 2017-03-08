import os

from others.macros import DATA
from others.data import formatData


if __name__ == '__main__':
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    formatData()