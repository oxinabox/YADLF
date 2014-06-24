import numpy as np
import functools
from  datasets.generic_dataset import LabelledDataset
import datasets.dataset_util as dutil

def code_dataset(dbn, depth, dataset):
    return LabelledDataset([ (dbn.get_code(data,depth), lbl) 
                                    for (data, lbl) in dataset])

def input_and_code_dataset(dbn, depth,dataset):
    def encode(datum):
        code = dbn.get_code(datum,depth)
        return np.concatenate((code,datum))

    return LabelledDataset([(encode(data),lbl)
                                    for (data, lbl) in dataset])

def rerep_trio(rerep_dataset_func, dbn, depth, datatrio):
    encode_func = functools.partial(rerep_dataset_func,dbn,depth)
    return dutil.DatasetTrio(*map(encode_func, datatrio))



def code_datatrio(dbn, depth, datatrio):
    return rerep_trio(code_dataset, dbn, depth, datatrio)

def input_and_code_datatrio(dbn, depth,datatrio):
    return rerep_trio(input_and_code_dataset, dbn, depth, datatrio)



