
import random


def change_base(b):
    x = random.choice("ACTG")
    while x == b:
        x = random.choice("ACTG")
    return x

def random_bases(n):
    return ''.join(random.choice('ACTG') for _ in range(n))



class DnaTransform:

    def _transform_one(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, s, **kwargs):
        if type(s) == str:
            return self._transform_one(s, **kwargs)
        elif type(s) == list or type(s) == tuple:
            return [self._transform_one(a, **kwargs) for a in s]


class RotateTransform(DnaTransform):
    """ Chop off a few bases at the beginning at add some random ones at the end, or vise-versa"""
    def __init__(self, min_len=5, max_len=10):
        self.min_len = min_len
        self.max_len = max_len

    def _transform_one(self, s, **kwargs):
        size = random.randint(self.min_len, self.max_len)
        if random.random() < 0.5:
            return s[size:] + random_bases(size)
        else:
            return random_bases(size) + s[0:-size]

class SnpTransform(DnaTransform):

    def _transform_one(self, s, **kwargs):
        snpcount = random.choice([1,1,1,1,2,2,3,4])
        for i in range(snpcount):
            j = random.randint(0, len(s)-1)
            s = s[0:j] + change_base(s[j]) + s[j+1:]
        return s


class InsTransform(DnaTransform):
    def __init__(self, min_len=1, max_len=4):
        self.min_len = min_len
        self.max_len = max_len

    def _transform_one(self, s, **kwargs):
        tcount = random.choice([1, 1, 1, 1, 2])
        for i in range(tcount):
            howmany = random.randint(self.min_len, self.max_len)
            j = random.randint(0, len(s) - 1)
            s2 = s[0:j] + random_bases(howmany) + s[j + 1:]
            s = s2[0:len(s)]
        return s

class DelTransform(DnaTransform):

    def _transform_one(self, s, **kwargs):
        tcount = random.choice([1, 1, 1, 1, 2])
        for i in range(tcount):
            minlen = kwargs.get('min_len', 1)
            maxlen = kwargs.get('max_len', 4)
            howmany = random.randint(minlen, maxlen)
            newbases = random_bases(howmany)
            j = random.randint(0, len(s) - 1)
            s2 = s[0:j] + s[j + howmany:] + newbases
            s = s2[0:len(s)]
        return s

class PickTransform(DnaTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def _transform_one(self, s, **kwargs):
        return random.choice(self.transforms)(s)


class SequentialTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, s, **kwargs):
        for t in self.transforms:
            s = t(s)
        return s
