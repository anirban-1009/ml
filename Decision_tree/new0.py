from math import log

class Entropy:
    def __init__(self):
        self

    def ginni(self, p):
        g = 2*p*(1-p)
        return g

    def ent(self,p):
        e = -(p * log(p, 2) + (1-p) * log((1-p), 2))
        return e

    def p(self,n,t):
        p = n/(n+t)
        return p

m = Entropy()
print(m.p(45, 40))