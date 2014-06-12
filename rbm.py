import numpyutil as nutil
import numpy as np
import nn_math as nn
import abc

def sample(probVector):
        return np.random.random_sample(probVector.shape)<probVector

#base on from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNPseudoCode (which is from bengio2009learning)
class RBM(object):
    def __init__(self, weight, hidden_bias, visible_bias, mean_field):
        self.visible_bias = visible_bias
        self.hidden_bias = hidden_bias
        self.weight = weight
        self.mean_field = mean_field


    @classmethod
    def random_init_params(cls,visible_layer_size, hidden_layer_size):
        weight= np.random.normal(0,0.01,
                                 (hidden_layer_size,visible_layer_size))
        hidden_bias = np.random.normal(0,0.01,(hidden_layer_size))
        visible_bias =np.random.normal(0,0.01,(visible_layer_size))
        return (weight, hidden_bias,visible_bias)

    @classmethod
    def random_init(cls,visible_layer_size, hidden_layer_size):
        params =  cls.random_init_params(visible_layer_size, 
                                         hidden_layer_size)
        return cls(*params)

    @abc.abstractmethod
    def sample_v_given_h(self,h):
        pass

    @abc.abstractmethod
    def mean_v_given_h(self,h):
        pass
    
    @abc.abstractmethod
    def prob_h_given_v(self,h):
        pass


    def do_CD1_once(self,v, mean_field=None):
        '''For optimistation reaasons this is seperate from CD-k
        Do ensure v is in appropriate domain (eg bernulli, gaussian).
        '''
        if mean_field==None:
            mean_field = self.mean_field

        assert type(v)!=np.void, "datum was np.void." + \
                "did you mistakenly use a labelled dataset?"

        h=nutil.sample(self.prob_h_given_v(v))
        posGrad=np.outer(h,v)

        vp=self.mean_v_given_h(h) if mean_field else self.sample_v_given_h(h)
        hp_prob=self.prob_h_given_v(vp)

        negGrad=np.outer(hp_prob,vp)

        delta_weight = (posGrad-negGrad)
        delta_hidden_bias = (h-hp_prob)
        delta_visible_bias = (v-vp)
        return delta_weight,delta_hidden_bias,delta_visible_bias

    def do_CDk_once(self,v, k):
        """ Does not currently support Mean Fiel"""
        h=nutil.sample(self.prob_h_given_v(v))
        posGrad=np.outer(h,v)

        vp_rec =  None
        hp_prob = None
        hp_rec = h
        for ii in range(0,k):
            vp_rec=self.sample_v_given_h(hp_rec)
            hp_prob=self.prob_h_given_v(vp_rec)
            hp_rec = nutil.sample(hp_prob)

        negGrad=np.outer(hp_prob,vp_rec)

        delta_weight = (posGrad-negGrad)
        delta_hidden_bias = (h-hp_prob)
        delta_visible_bias = (v-vp_rec)
        return delta_weight,delta_hidden_bias,delta_visible_bias

    get_training_updates = do_CD1_once

    @property
    def knowledge(self):
        return self.weight, self.hidden_bias, self.visible_bias

    @knowledge.setter
    def knowledge(self, value):
        (self.weight, self.hidden_bias, self.visible_bias) = value


    def generate_image_from_noise(self,pPixOn=0.2, mean_field=False):
        ''' If show_prob is false, will sample from the probability to get a concrete image, otherwise will display a probabilit "grey scale" '''
        v = nutil.sample(pPixOn*np.ones(len(self.visible_bias)))
        return self.generate_image(v, mean_field)

    def generate_image(self,v, mean_field=False):
        h=nutil.sample(self.prob_h_given_v(v))
        img = self.mean_v_given_h(h) if mean_field else self.sample_v_given_h(h)
        return (v,img)


class BbRBM(RBM):
    ''' Bernolli bernulii RBM
    Exepect inputs to be either 0 or 1
    '''

    def __init__(self, weight, hidden_bias, visible_bias):
        RBM.__init__(self, weight, hidden_bias, visible_bias, mean_field=False)
    
    #$p(h_j=1|\mathbf{v})=\sigma(b_j+\sum_{\forall i}{v_iw_{ij}})$
    def prob_h_given_v (self, v):
        return nn.sigmoid(self.hidden_bias + np.dot(self.weight,v)) #sum up rows (v is a column vector)

    def prob_v_given_h (self,h):
        return nn.sigmoid(self.visible_bias + np.dot(h,self.weight)) #sum up columns. (h is a row vector)

    mean_v_given_h = prob_v_given_h
    
    def sample_v_given_h(self,h):
        return nutil.sample(self.prob_v_given_h(h))



class GbRBM(RBM):
    ''' Gaussian bernulii RBM
    Exepect inputs zero mean, unit variancne normally distributed.
    '''

    def __init__(self, weight, hidden_bias, visible_bias):
        RBM.__init__(self, weight, hidden_bias, visible_bias, mean_field=True)
    
    #$p(h_j=1|\mathbf{v})=\sigma(b_j+\sum_{\forall i}{v_iw_{ij}})$
    def prob_h_given_v (self, v):
        return nn.sigmoid(self.hidden_bias + np.dot(self.weight,v)) 
            #sum up rows (v is a column vector)

    def mean_v_given_h (self,h):
        return self.visible_bias + np.dot(h,self.weight) #sum up columns. (h is a row vector)

    #$\mathcal{N}(\sigma(c_j+\sum_{\forall i}{v_iw^T_{ij}}),1.0)$
    def sample_v_given_h(self,h):
        mean = self.mean_v_given_h(h)
        return np.random.normal(mean, 1.00, mean.shape)

