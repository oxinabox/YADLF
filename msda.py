import numpy as np
import numpyutil as nutil
import nn_math as nn_math
from  stacked_generative_model import StackedGenerativeModel


def append_bais_units(data):
    colshape=np.shape(data)[:-1] + (1,)
    return np.hstack((data, np.ones(colshape)))

class M_DA(object):
    def __init__(self, act_fun=lambda i:i, weight_bais=None):
        self.act_fun = act_fun
        self.weight_bais=weight_bais


    def train_unsupervised(self,data, noise_prob=0.2, reg=0.00001):
        '''Note: This is a full retrain. It destroys past learning'''
        #X in paper,  xx in article
        data = np.asarray(data)

        #X in paper, xxb in article
        data_with_bias = append_bais_units(data).T
        datum_len = data_with_bias.shape[0]

        #q in paper and aricle
        noise = np.ones((datum_len,1))*(1-noise_prob)
        noise[-1]=1

        #S in paper and article
        scater_data = np.dot(data_with_bias,data_with_bias.T)
        Q=scater_data*np.dot(noise,noise.T)
        Q.flat[::datum_len+1]=noise*np.diag(scater_data)

        P=scater_data*np.tile(noise,(1,datum_len)).T

        reg_mat = np.eye(datum_len)*reg
        numer = P[:-1,:]
        denom = Q+reg_mat

        self.weight_bais=np.dot(numer, np.linalg.pinv(denom)).T


    def get_output(self,x):
        xb=append_bais_units(x)
        y=np.dot(xb,self.weight)
        return self.act_fun(y)

    get_code=get_output

    @property
    def bias(self):
        return self.weight_bais[-1,:]

    @property
    def weight(self):
        return self.weight_bais[:-1,:]

    @property
    def knowledge(self):
        return self.weight_bais


class M_SDA(StackedGenerativeModel):
    def __init__(self, weights, act_fun):
            self.mdas = [M_DA(act_fun, w) for w in weights]


    @classmethod
    def random_init(cls, num_layers, act_fun=nn_math.sigmoid):
        ws = [None for _ in xrange(num_layers)]
        return cls(ws, act_fun)



    def train_unsupervised(self, data, noise_prob=0.2, reg=0.00001, silent=True):
        '''Note: This is a full retrain. It destroys past learning'''
        if not silent: print "started"
        output_below = data
        for layer_num,mda in enumerate(self.mdas[:-1], start=1):
            mda.train_unsupervised(output_below,noise_prob,reg)
            output_below = mda.get_output(output_below)
            if not silent: print "done layer " + str(layer_num)
        self.mdas[-1].train_unsupervised(output_below,noise_prob,reg)
        if not silent: print "done"

    @property
    def layer_models(self):
        self.mdas

    @property
    def weights(self):
        return map(lambda mda: mda.weight, self.mdas)

    @property
    def biases(self):
        return map(lambda mda: mda.bias, self.mdas)

    @property
    def knowledge(self):
        return (self.weights, self.biases)

