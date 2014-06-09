import numpyutil as nutil
import numpy as np
import nn_math as nn

def random_weights_baises(layer_sizes):
    weights = [np.random.normal(0,0.01,(szNext,szThis)) for (szThis,szNext) in nutil.pairwise(layer_sizes)]
    biases = [np.random.normal(0,0.01,sz) for sz in layer_sizes[1:]]
    return weights, biases





class NeuralNet(object):
    def __init__(self,
                 weights,
                 biases,
                 post_process=lambda i: i, #Post Process is applied to final output only (Not used when training), but is used when checkign error rate
                 topErrorFunc=nn.squaredMean, d_topErrorFunc=nn.dSquaredMean,
                 actFunc=nn.sigmoid, d_actFunc=nn.dSigmoid,
                 topActFunc = nn.sigmoid, d_topActFunc = nn.dSigmoid):

        self.weights = weights
        self.biases = biases

        assert len(self.biases)==len(self.weights), "Must have as many bias vectors as weight matrixes"
        self.post_process = post_process
        self.topErrorFunc = topErrorFunc
        self.d_topErrorFunc = d_topErrorFunc
        self.actFunc = actFunc
        self.d_actFunc=d_actFunc
        self.topActFunc = topActFunc
        self.d_topActFunc = d_topActFunc


    @classmethod
    def random_init(cls,
                    layer_sizes,
                    post_process=lambda i: i, #Post Process is applied to final output only (Not used when training), but is used when checkign error rate
                    topErrorFunc=nn.squaredMean, d_topErrorFunc=nn.dSquaredMean,
                    actFunc=nn.sigmoid, d_actFunc=nn.dSigmoid,
                    topActFunc=nn.sigmoid, d_topActFunc=nn.dSigmoid):

        weights, biases = random_weights_baises(layer_sizes)
        return cls(weights=weights, biases=biases,
                   post_process=post_process,
                   topErrorFunc=topErrorFunc, d_topErrorFunc=d_topErrorFunc,
                   actFunc=actFunc, d_actFunc=d_actFunc,
                   topActFunc=topActFunc, d_topActFunc=d_topActFunc)

    @property
    def num_layers(self):
        return len(self.weights)+1

    @property
    def layer_sizes(self):
        yield np.shape(self.weights[0])[1] #input layer shape
        for w in self.weights:
            yield np.shape(w)[0]


    def _feedforward_all(self, v):
        """
        Returns: the output for each layer, starting with the input, and ending with the output (with the hidden layers inbetween)
        """
        def feedforward(v,w,b):
            return self.actFunc(b+np.dot(w,v))
        def feedforward_top(v,w,b):
            return self.topActFunc(b+np.dot(w,v))

        #assert len(self.weights) == self.num_layers-1

        ys = [v]
        for (w,b) in zip(self.weights,self.biases)[:-1]:
            y = feedforward(ys[-1],w,b)
            ys.append(y)
        y_top = feedforward_top(ys[-1],self.weights[-1], self.biases[-1])
        ys.append(y_top)
        return np.asarray(ys)

    def _get_layer_derivs(self,y_i, dEdx_j, w_ji):
            dEdb_j= dEdx_j #Bias Error Dervivitve = error signal

            dEdW_ji=np.outer(dEdx_j,y_i) #Weight Change
            #assert dEdW_ji.shape == w_ji.shape, "%s \n %s" % (nutil.arr_info(dEdW_ji,"dEdW_ji"), nutil.arr_info(w_ji,"w_ji"))

            dEdy_i= np.dot(dEdx_j,w_ji) #=Sum_j dEdx_j*w_ji
            dy_i_dx_i=self.d_actFunc(y_i)
            #assert not np.isnan(np.sum(dy_i_dx_i))
            dEdx_i=(dEdy_i*dy_i_dx_i)
            return (dEdW_ji, dEdb_j, dEdx_i)

    def _top_dEdx_j(self, target, output):
        y_j=output
        dEdy_j = self.d_topErrorFunc(target, output)
        dy_j_dx_j=self.d_topActFunc(y_j)
        dEdx_j=(dEdy_j*dy_j_dx_j)
        return dEdx_j

    def _get_dError_dWeighs(self, target, layerValues):
        """
        layerOutputs, starts with the input layer, continues through the hidden layers outputs and ends with the output layer
        Returns: a list of matrixes of the same size and shape as the weight matrixes
        """
        #assert len(layerValues)==self.num_layers, "%s == %i" % (nutil.arr_info(layerValues, "layerValues"), self.num_layers)

        output = layerValues[-1]

        dEdx_j = self._top_dEdx_j(target, output)
        dEdWs = []
        dEdBs = []
        y_is = layerValues[:-1]
        for (y_i, w_ji) in reversed(zip(y_is,self.weights)):
            (dEdW_ji, dEdb_j, dEdx_i) = self._get_layer_derivs(y_i, dEdx_j, w_ji)
            
            dEdWs.append(dEdW_ji)
            dEdBs.append(dEdb_j)
            dEdx_j = dEdx_i


        dEdWs.reverse() #put it the right way round
        dEdBs.reverse()
        return dEdWs, dEdBs


    def get_training_updates(self,vt): 
        v,t=vt

        layerOutputs = self._feedforward_all(v)
        error_derivs =  self._get_dError_dWeighs(t,layerOutputs)
        return nutil.uniop_nested(np.negative, error_derivs)




    @property
    def knowledge(self):
        return self.weights, self.biases

    @knowledge.setter
    def knowledge(self,value):
        self.weights, self.biases = value



    def get_raw_output(self, v):
        output = self._feedforward_all(v)[-1]
        return output

    def get_output(self, v):
        raw_output = self.get_raw_output(v)
        return self.post_process(raw_output)

    def error_rate(self, dataset):
        errors = 0.0
        #assert np.size(dataset[0][1])==np.shape(self.weights[-1])[0], "Label size doesn't match output layer size"
        for (v,lbl) in dataset:
            errors+= not all(self.get_output(v)==lbl)
        return errors/len(dataset)
