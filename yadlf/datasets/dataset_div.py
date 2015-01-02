import numpy as np
from .. import numpyutil as nutil
import dataset_util as dutil
import mnist_loaded as data_module #TODO Make this more generic


def split(data, source_lbls, target_lbls):
    source = []
    target = []
    for (v,t) in data:
        if in_group(source_lbls,t):
            source.append((v,t))
        elif in_group(target_lbls,t):
            target.append((v,t))
        
    source_set = data.make_from(source)
    target_set = data.make_from(target)
    return source_set, target_set

def in_group(group,lbl):
    lbl_value=nutil.unsparse(lbl)
    return lbl_value in group

    
def rand_group():
    import random
    candidates = range(0,10)
    random.shuffle(candidates)
    return frozenset(candidates[:5]), frozenset(candidates[5:])

def rand_groups(num):
    groups = set()
    while(len(groups)<num):
        groups.add(rand_group())
    return groups

def split_data_cases(source_lbls, target_lbls):
    base_datas = data_module.base
    source_datas, target_datas = zip(*map(lambda ds: 
                                    split(ds, source_lbls, target_lbls),
                                    base_datas))
    source_trio=dutil.DatasetTrio(*source_datas)
    dutil.balance_all(source_trio)
    
    target_trio=dutil.DatasetTrio(*target_datas)
    dutil.balance_all(target_trio)
    return source_trio, target_trio

def subdivisions(dataset,subdivision_sizes):
    start = 0
   
    for end in subdivision_sizes:
       if end>=len(dataset):
            yield len(dataset), dataset[start:-1] # Yield the remainder
            return
       else:
            yield end, dataset[start:end]
            start = end

 

def subdivide_trio(base, subdivision_sizes):
    '''
    Makes a bunch of dataset_trios for each dataset.
    Each dataset trio assumes the learning algorithm
    has already been trained on the earlier ones.
    If not, the the second thing returned is
    The total training data, normalised.
    '''
    import sklearn.preprocessing as preproc
    total_training_used = np.empty_like(base.train.data['data'])
    for sz, train_subdivision in subdivisions(base.train, subdivision_sizes):
        #Normalise Training Data
        train_normed_data = preproc.StandardScaler()\
                            .fit_transform(train_subdivision.data['data'])

        train_normed = zip(train_normed_data, train_subdivision.data['lbls'])
        total_training_used = np.append(total_training_used, 
                                        train_subdivision.data['data'])
        #Renormalise Test, Validataion Data
        overall_scaler = preproc.StandardScaler().fit(total_training_used)
        
        def overall_norm(dataset):
            return zip(overall_scaler.transform(dataset.data['data']),
                           dataset.data['lbls'])
        
        valid_normed = overall_norm(base.valid)
        test_normed = overall_norm(base.test)

        yield sz, dutil.DatasetTrio(*map(base.train.make_from,
                                     [train_normed,
                                      valid_normed,
                                      test_normed]))

