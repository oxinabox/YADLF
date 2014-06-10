import itertools as it
import numpy as np
import numpyutil as nutil
import datasets.dataset_util as dutil
import copy 


class Trainer(object):
    def __init__(self, trainee):
        self.trainee = trainee
        self.prev_updates = nutil.uniop_nested(
            np.zeros_like, self.trainee.knowledge)

    def _get_batch_updates(self, batch):
        v0 = batch[0]
        update_totals = self.trainee.get_training_updates(v0)
        for v in batch[1:]:
            updates = self.trainee.get_training_updates(v)
            update_totals = nutil.add_nested(update_totals,updates)
          
        return update_totals

    def train_batch(self, batch, learning_rate, momentum, reg):
        batch_updates = self._get_batch_updates(batch)
        batch_size = len(batch)
        def calc_new_values(update,prev, cur_know):
            return (learning_rate/batch_size)*(update -reg*cur_know) + momentum*prev

        updates = nutil.triop_nested(calc_new_values, 
                                     batch_updates, 
                                     self.prev_updates,
                                     self.trainee.knowledge)
        self.trainee.knowledge = nutil.add_nested(self.trainee.knowledge,updates)
        self.prev_updates = updates 

    def train_minibatch(self,
                        dataset,
                        batch_size,
                        learning_rate=0.01, 
                        momentum=0.9,
                        reg = 0.0002,
                        report_freq=None):
        silent = not report_freq>0 #If we don;'t have a positive reporting frequency, then be siolent
        num_batches = dutil.num_batches(dataset, batch_size)

        for batch_num, batch in enumerate(dutil.batch_split(dataset, batch_size)):
            self.train_batch(batch, learning_rate, momentum, reg)
            if not silent and (batch_num%report_freq == 0):
                print "Done batch %i/%i" % (batch_num, num_batches)
        if not silent: print "Done All"

    def early_stopping(self,
                       train_set, 
                       valid_set,
                       batch_size,
                       validation_period=1,
                       init_patience = np.inf,
                       learning_rate=0.01,
                       momentum=0.9, 
                       reg = 0.0002,
                       silent=False):
        num_batches = dutil.num_batches(train_set, batch_size)

        best_at_batch_num =  None
        best_error_rate = np.inf
        best_knowledge = copy.deepcopy(self.trainee.knowledge)
        try:
            patience_rem = init_patience
            for batch_num, batch in enumerate(dutil.batch_split(train_set, batch_size)):
                self.train_batch(batch, learning_rate, momentum, reg)
                if batch_num%validation_period == 0:
                    error_rate = self.trainee.error_rate(valid_set)
                    if error_rate<=best_error_rate:
                        best_error_rate = error_rate
                        patience_rem = init_patience
                        best_knowledge = copy.deepcopy(self.trainee.knowledge)
                        best_at_batch_num = batch_num
                        if not silent: print("!"),
                    else:
                        patience_rem-=1
                        if patience_rem==0:
                            print "Out of Patience"
                            break
                    if error_rate==0.0:
                        #CONSIDER: This is not ideal as valid_set<test_set
                        print "Perfection Obtained in batch: " + str(batch_num)
                        return
                    
                    if not silent:
                         print "Error Rate: %f \t\t\t(%i/%i)" % (
                            error_rate,
                            batch_num, 
                            num_batches)
            print "Done All"
        finally:
            print "Validation Error:\t%f\n From Batch: \t %i" %(best_error_rate, best_at_batch_num)
            self.trainee.knowledge = best_knowledge
            return best_at_batch_num, best_error_rate


    def early_stopping_repreating(self,
                       train_set, 
                       valid_set,
                       batch_size,
                       validation_period=1,
                       init_patience = np.inf,
                       learning_rates=[0.1,0.01, 0.001],
                       momentum=0.9, 
                       reg = 0.0002,
                       silent=False):
        """Use the reduces learning rate to home in on the solution by repreated early stopping."""
        last_ret = None
        for learning_rate in learning_rates:
            last_ret = self.early_stopping(train_set, 
                               valid_set,
                               batch_size,
                               validation_period,
                               init_patience,
                               learning_rate,
                               momentum,
                               reg,
                               silent=False)
        return last_ret



    def train_online(self, 
                     training_data, 
                     learning_rate=0.1, 
                     momentum=0.9, 
                     reg = 0.0002,
                     report_freq=None):
        self.train_minibatch(training_data,
                             batch_size=1,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             reg=reg,
                             report_freq=report_freq)

