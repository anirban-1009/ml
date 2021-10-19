from math import log

class purity:
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
        r = input('>')
        if r[0] == 'g':
            return self.ginni(p)

        elif r[0] == 'e':
            return self.ent(p)


m = purity()
purity = m.p(175, 217)
print(purity)