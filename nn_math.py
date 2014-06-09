from __future__ import division
import numpy as np

def clip_for_exp(x):
    return x.clip(-700.0,700.0) #Avoid overflows (Negitive bound not actually required)

from scipy.special import expit
def sigmoid(x):
    x = clip_for_exp(x)
    return expit(x)
def dSigmoid(y):return y*(1-y) #where y=sigmoid(x)

def softmax(x):
    x = clip_for_exp(x)
    numerators = np.exp(x)
    return numerators/np.sum(numerators)
def dSoftmax(y): return y*(1-y) #where y=softmax(x)

def rectifiedLinear(x):
    return np.maximum(0,x)
def dRectifiedLinear(y):
    return 0 if y<=0 else 1

def squaredMean(ts,ys): return 0.5*(np.sum(np.square(ts-ys))) #np.square is elementwise
def dSquaredMean(ts,ys): return ys-ts


def crossEntropy(ts,ys): return -np.sum((ts)*(np.log((ys))))
def dCrossEntropy(ts,ys): return -(ts)/(ys)


def winner_takes_all(prob_vect):
    ret = np.zeros_like(prob_vect, dtype=np.bool)
    ret[np.argmax(prob_vect)]=1
    return ret
    
