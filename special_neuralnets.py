import neuralnet
import nn_math as nn
import numpy as np

class FixedTopErrorSignalNeuralNet(neuralnet.NeuralNet):
    def __init__(self,
                 weights,
                 biases,
                 post_process=lambda i: i, #Post Process is applied to final output only (Not used when training), but is used when checkign error rate
                 actFunc=nn.sigmoid, d_actFunc=nn.dSigmoid,
                 topActFunc = nn.softmax, d_topActFunc = nn.dSoftmax):

        neuralnet.NeuralNet.__init__(self,
                           weights=weights,
                           biases=biases,
                           post_process=post_process,
                           topErrorFunc=None, d_topErrorFunc=None,
                           actFunc=actFunc, d_actFunc=d_actFunc,
                           topActFunc=topActFunc, d_topActFunc=d_topActFunc)

    @classmethod
    def random_init(cls,
                    layer_sizes,
                    post_process=lambda i: i, #Post Process is applied to final output only (Not used when training), but is used when checkign error rate
                    actFunc=nn.sigmoid, d_actFunc=nn.dSigmoid,
                    topActFunc=nn.softmax, d_topActFunc=nn.dSoftmax):

        weights, biases = neuralnet.random_weights_baises(layer_sizes)
        print cls
        return cls(weights, biases, post_process, actFunc, d_actFunc, topActFunc,d_topActFunc)


    def _top_dEdx_j(self, target, output):
        return output-target #CHEAT: This is always the best function. Skip the Calcus and just do it

def new_linear_classifier(input_size, output_size):
    return FixedTopErrorSignalNeuralNet.random_init([input_size, output_size],
                                                    post_process = nn.winner_takes_all)

