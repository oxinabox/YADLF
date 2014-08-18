
import abc

import numpy as np
import nn_math as nn


class StackedGenerativeModel(object):

    @abc.abstractproperty
    def biases(self):
        pass

    @abc.abstractproperty
    def weights(self):
        pass

    @abc.abstractproperty
    def knowledge(self):
        pass

    @abc.abstractproperty
    def layer_models(self):
        pass


    def get_code(self,x, depth=-1):
        code_depth = len(self.layer_models) if depth<0 else depth
        assert(code_depth<=len(self.layer_models))

        output_below = x
        for (cur_depth, model) in enumerate(self.layer_models, start=1):
            output_below = model.get_code(output_below)
            if cur_depth>=code_depth: break
        return output_below


    def as_neural_net(self, output_layer_size, constructor = None):
        '''
        Constructor: If given a function to construct a neural net,
        thak takes the weights and baises and returns a net
        '''
        from special_neuralnets import FixedTopErrorSignalNeuralNet

        weights = self.weights + [np.random.normal(0.0,0.01,
                                (output_layer_size, self.weights[-1].shape[0]))]
        biases = self.biases + [np.random.normal(0.0,0.01,
                                                        output_layer_size)]


        def default(ws,bs):
            return FixedTopErrorSignalNeuralNet(ws, bs,
                        post_process = nn.winner_takes_all,
                        topActFunc = nn.softmax, d_topActFunc=nn.dSoftmax)
        constructor = constructor or default
        return constructor(weights,biases)

