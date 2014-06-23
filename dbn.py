import numpyutil as nutil
import numpy as np
import nn_math as nn
import rbm
import trainer
import special_neuralnets


class DBN:
    def __init__(self, weights, hidden_biases, visible_biases, 
                 act_func=nn.sigmoid):
        self.rbms = []
        bottom_layer = True
        for (w,hb,vb) in zip(weights,hidden_biases,visible_biases):
            rbm_type = None
            if bottom_layer:
                rbm_type = rbm.GbRBM
                bottom_layer = False #Do this only once
            else:
                rbm_type = rbm.BbRBM
            new_rbm=rbm_type(w,hb,vb)
            self.rbms.append(new_rbm)


    @classmethod
    def random_init(cls,layer_sizes):

        params = [ rbm.RBM.random_init_params(sz_this,sz_next)
                    for (sz_this,sz_next) in nutil.pairwise(layer_sizes)]
                #Ask the RBM class what randomly intialised parameters looklike
        return cls(*zip(*params))


    def train_unsupervised(self, input_data, rep_freq=-1,
                               learning_rate = 0.001,
                               momentum= 0.9,
                               reg = 0.0002,
							   clip = None
                              ):
        trained_rbms = []
        for rbm_num, training_rbm in enumerate(self.rbms):
            this_rbm_trainer = trainer.Trainer(training_rbm, clip)
            for input_num, input_datum in enumerate(input_data):
                prev_layer_out = input_datum ##bottom RBM is a GbRBM so can take continuously valued inputs
                for trained_rbm in trained_rbms:
                    prob_prev = (trained_rbm.prob_h_given_v(prev_layer_out))
                    prev_layer_out = nutil.sample(prob_prev) #For BbRBM need 0 or 1
                #Train a batch of 1 
                this_rbm_trainer.train_online([prev_layer_out],
                                              learning_rate=learning_rate,
                                              momentum=momentum,
                                              reg = reg)
               
                if input_num%rep_freq==1:
                    print("one input %i/%i on RBM %i" % 
                        (input_num,len(input_data),rbm_num+1))

            #The one were were training is now trained, lock it it and use it
            trained_rbms.append(training_rbm) 
            if rep_freq>0:
                print "Done RBM: %i/%i" % (rbm_num+1, len(self.rbms))
    
    def as_neural_net(self, output_layer_size, constructor = None):
        '''
        Constructor: If given a function to construct a neural net, 
        thak takes the weights and baises and returns a net
        '''
        weights = self.weights + [np.random.normal(0.0,0.01,
                                (output_layer_size, self.weights[-1].shape[0]))]
        biases = self.upward_biases + [np.random.normal(0.0,0.01,
                                                        output_layer_size)]
        
        
        def default(ws,bs):
            return special_neuralnets.FixedTopErrorSignalNeuralNet(ws, bs,
                        post_process = nn.winner_takes_all,
                        topActFunc = nn.softmax, d_topActFunc=nn.dSoftmax)
        constructor = constructor or default
        return constructor(weights,biases)
        
        
            
    def generate_image_from_top(self, y, equib_dur=1):
        top_rbm = self.rbms[-1];
        output_above = y
        for ii in range(0,equib_dur): #HACK: insteaad of actually working out when we are in equiblium, just  do it a bunch of times
            vp = top_rbm.sample_v_given_h(output_above)
            output_above= nutil.sample(top_rbm.prob_h_given_v(vp))
        
        for rbm in self.rbms[::-1]: 
                #[1:]Don't do the bottom, so we can do that ot get prob
            output_above = rbm.sample_v_given_h(output_above)
            
        return output_above
    
    
    def get_code(self,input_vect, code_depth=None):
        code_depth = len(self.rbms) if code_depth==None else code_depth
        assert(code_depth<=len(self.rbms))
               
        output_below = input_vect
        for (cur_depth, rbm) in enumerate(self.rbms,start=1):
            output_below = nutil.sample(rbm.prob_h_given_v(output_below))
            if cur_depth>=code_depth: 
                break
        return output_below
    
    def generate_image_from_bottom(self,x, equib_dur):
        top_out = self.get_output(x)
        return x, self.generate_image_from_top(top_out, equib_dur=equib_dur)


    @property
    def weights(self):
        return list(map(lambda rbm: rbm.weight, self.rbms))
            
    @property
    def upward_biases(self):
        return list(map(lambda rbm: rbm.hidden_bias, self.rbms))
                
    @property
    def downward_biases(self):
        return list(map(lambda rbm: rbm.visible_bias, self.rbms))
        
    @property
    def knowledge(self):
        return [self.weights, self.upward_biases, self.downward_biases]
