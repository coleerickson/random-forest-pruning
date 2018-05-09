from collections import Counter
from math import log

def first(l):
    for x in l:
        return x

def mode(l):
    return Counter(l).most_common(1)[0][0]


def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)

def inner(x, y):
    ''' Returns the inner product (dot product) of vectors `x` and `y`, where `x` and `y` are represented as lists. '''
    result = 0
    for (xi, yi) in zip(x, y):
        result += xi * yi
    return result
