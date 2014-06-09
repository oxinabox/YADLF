from __future__ import division
import numpy as np
import itertools as it
import math
import collections

def _create_sym_datum(top_maker,non_sym_bottom_maker):
    top = top_maker()
    isSym = np.random.random()>0.5
    if isSym:
        bottom = top[::-1]
    else:
        bottom=non_sym_bottom_maker(top)
    datum = np.concatenate((top,bottom))
    return (datum, isSym)


#Lets make a syetric dataset based of MNIST
def make_sym(base_data):
    num_elements = len(base_data)
    basedata_iter = it.cycle(base_data)
    def non_sym_bottom_maker(top):
        return basedata_iter.next()
    
    datalist = [_create_sym_datum(basedata_iter.next,non_sym_bottom_maker ) for ii in xrange(num_elements)]
    return datalist

def normalize(vectors):
    old_settings = np.seterr(invalid='ignore')
        #We can divide 0/0, sell comment below
    try:
        mat = np.asarray(vectors, dtype=float)
        normed = (mat - mat.mean(axis=0)) /mat.std(axis=0)
        #If there are any NaNs in the result, then a zero/zero division was done
        #If the std dev is zero then the value must be equal to its mean
        #Which means it should normalise to 0
        normed[np.isnan(normed)]=0
    finally:
        np.seterr(**old_settings)
    return normed

def balance_ordering(labelled_data):
    from collections import defaultdict
    groups = defaultdict(list)
    for (d,l) in labelled_data:
        groups[l].append((d,l))
    group_iters = map(iter, groups.values())

    for data_iter in it.cycle(group_iters): #Infinite Loop
        yield data_iter.next() 
        # data_iter.next() will throw a StopIteration which is good 
        # that will stop the generator

def num_batches(data, batch_size):
    return math.ceil(len(data)/batch_size)

def batch_split(data, batch_size):
    for start in xrange(0,len(data),batch_size):
         yield data[start:start+batch_size]

DatasetTrio = collections.namedtuple('DatasetTrio',['train','valid','test'])

def divvy_dataset(full, valid_size,test_size):
    '''Splits up the dataset into a training, a validation as a test set.
    If test data is to be seperated off, in same order, place it at the end.
    '''
    assert(valid_size>0)
    assert(test_size>0)
    assert(valid_size+test_size<len(full))
    
    valid = full[:valid_size]
    train = full[valid_size:-test_size]
    test  = full[-test_size:]
    return DatasetTrio(train=train,valid=valid,test=test)



def normalise_dataset_trio(trio):
    '''Normalises the dataset trio, but makes sure it does not cheat.
    ie Only makes normalising characteristic off the training data.
    To improve validation, scales validation same as test
    '''
    import sklearn.preprocessing as preproc
    def data_of(dataset):
        return  np.array(list(dataset.data[:,0]))

    scaler = preproc.StandardScaler().fit(data_of(trio.train))

    def norm_dataset(dataset):
        data_raw = data_of(dataset)
        lbls_raw = dataset.data[:,1]
        
        data_raw_norm = scaler.transform(data_raw)
        return dataset.make_from(zip(data_raw_norm, lbls_raw))

    return DatasetTrio(*map(norm_dataset, trio))

def balance_all(trio):
    for ds in trio:
        ds.balance()
