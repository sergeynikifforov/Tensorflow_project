from math import exp
from math import log
def eval(x,w11=1,w12=1,w13=1,w21=1,w22=1,w23=1):
    return w13*exp(w12*log(w11*x))+w23*exp(w22*log(w21*x))
x = 26.211588
w11 = 0.61921823
w12 = 0.44214845
w13 = -1.3587321
w21 = 0.61921823
w22 = -0.15614998
w23 = 1.0308944
print(x,eval(x,w11,w12,w13,w21,w22,w23))
