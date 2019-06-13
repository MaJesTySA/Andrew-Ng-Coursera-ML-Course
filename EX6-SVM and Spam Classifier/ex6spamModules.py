import re
import numpy as np
def processVocab():
    p = re.compile(r'[0-9]')
    vocab = []
    with open('vocab.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(p, '', line)
            line = line.split('\t')
            line[1] = line[1].split('\n')
            vocab.append(line[1][0])
    return vocab

def processEmail(filename):
    words = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if (line != '\n'):
                line = line.split('\n')
                words.append(line[0])
    return words

def findWordIndex(words,vocabs):
    Index = []
    for word in words:
        if (word in vocabs):
            idx = np.where(vocabs == word)
            Index.append(idx[0][0])
    return Index

def emailFeatures(word_indices,vocabs):
    n=vocabs.shape[0]
    x=np.zeros(n)
    for word_index in word_indices:
        x[word_index]=1
    x=x.reshape((1,n))
    return x

def EmailToFeatures(filename):
    Email_words = np.array(processEmail(filename))
    vocabs = np.array(processVocab())
    word_indices = findWordIndex(Email_words, vocabs)
    x = emailFeatures(word_indices, vocabs)
    return x