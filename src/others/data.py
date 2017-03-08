import os

import numpy as np

from others.macros import RAW, DATA


def filter_string(string):
    header = string[0:string.find('\n')]
    string = string[string.find('\n'):]
    translation_table = dict.fromkeys(map(ord, '—-:…•°«»<>[]\{\}()!@#$?.,\'\n'), ' ')
    string = string.translate(translation_table)
    string = string.lower()
    string = header + ' ' + string
    string = string.split(' ')

    content = list(filter(lambda x: len(x) > 0 and not x.isdigit(), string))

    return {'author': content[0],
            'subject': content[1],
            'party': content[2],
            'content': content[3:]}

def filter_string2(string):
    header = string[0:string.find('\n')]
    string = string[string.find('\n'):]
    translation_table = dict.fromkeys(map(ord, '•°«»<>[]\{\}@#$\n'), ' ')
    string = string.translate(translation_table)
    string = string.lower()
    string = header + ' ' + string

    author, subject, party = list(filter(lambda x: len(x) > 0, header.split(' ')))

    return {'author': author,
            'subject': subject,
            'party': party,
            'content': string}

def parse_data(raw_data):
    with open(os.path.join(RAW, raw_data), 'r') as f:
        raw_read = f.read()

    raw_split = raw_read.split('*****')

    raw_split = filter(lambda x: len(x) > 0, raw_split)

    return list(map(lambda x: filter_string(x), raw_split))

def parse_data2(raw_data):
    with open(os.path.join(RAW, raw_data), 'r') as f:
        raw_read = f.read()

    raw_split = raw_read.split('*****')

    raw_split = filter(lambda x: len(x) > 0, raw_split)

    return list(map(lambda x: filter_string2(x), raw_split))

def formatData():
    raw_data = os.listdir(RAW)

    data = []

    for rd in raw_data:
        data += parse_data(rd)

    with open(os.path.join(DATA, 'for_bow.csv'), 'w') as f:
        for d in data:
            f.write('{},{},{},{}\n'.format(d['author'],
                                         d['subject'],
                                         d['party'],
                                         ','.join(d['content'])))

def format4tocken():
    raw_data = os.listdir(RAW)

    data = []

    for rd in raw_data:
        data += parse_data2(rd)

    with open(os.path.join(DATA, 'for_w2v.csv'), 'w') as f:
        for d in data:
            f.write('{}@{}@{}@{}\n'.format(d['author'],
                                         d['subject'],
                                         d['party'],
                                         d['content']))

def load_data(file):
    with open(file, 'r') as f:
        raw = f.read()

    speeches = list(filter(lambda x: len(x) > 0, raw.split('\n')))

    data = list(map(lambda x: x.split(',')[3:], speeches))
    target = list(map(lambda x: x.split(',')[2], speeches))

    return data, target

def load_data2(file):
    with open(file, 'r') as f:
        raw = f.read()

    speeches = list(filter(lambda x: len(x) > 0, raw.split('\n')))

    data = list(map(lambda x: x.split('@')[3], speeches))
    target = list(map(lambda x: x.split('@')[2], speeches))

    return data, target
