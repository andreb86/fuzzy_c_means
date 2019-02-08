import numpy as np


class FCM(object):

    def __init__(self, dim: int, m: float, infile: str = './data/s1.txt'):
        raw = np.fromfile(infile, dtype=np.float64, count=-1, sep=" ")
        print(len(raw))
        self.n = int(len(raw) / dim)
        self.l = dim
        self.m = m
        self.data = np.reshape(raw, (self.n, self.l))
        self.dsum = np.zeros((self.n, 1))
        self.dist = None
        self.cent = None
        self.uold = None
        self.unew = None
        self.upow = None
        self.usum = None
        self.udif = None

    def centroids(self, y: np.ndarray):
        c = int(len(y) / self.l)
        self.cent = np.reshape(y, (c, self.l))
        self.dist = np.zeros((self.n, c))
        self.uold = np.zeros((self.n, c))
        self.unew = np.zeros((self.n, c))
        self.usum = np.zeros((c, 1))

    def distances(self):
        for i, x in enumerate(self.data):
            for j, y in enumerate(self.cent):
                self.dist[i, j] = np.power(
                    np.sum(np.power(x - y, 2)), 1 / (1 - self.m)
                )
        self.dsum = self.dist.sum(1)

    def weights(self):
        for i, d in enumerate(self.dist):
            self.unew[i] = d / self.dsum[i]
        self.upow = np.power(self.unew, self.m)
        self.usum = self.unew.sum(0)

    def check(self):
        err = np.abs(self.unew - self.uold)
        return np.max(err)

    def update(self):
        for j, y in enumerate(self.cent):
            num = 0
            for i, x in enumerate(self.data):
                num += self.upow[i, j] * x
            self.cent = num / self.usum[j]
