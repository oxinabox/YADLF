import numpy as np
import functools
from  datasets.generic_dataset import LabelledDataset
import datasets.dataset_util as dutil

def code_dataset(model, depth, dataset):
    data = dataset.data['data']
    lbls = dataset.data['lbl']
    coded_datalist = zip(model.get_code(data,depth), lbls)
    #OPTIMIMISE: this could be better, makign a list then it is turned ino an array in thethe LavelledDataset        

    return LabelledDataset(coded_datalist)

def input_and_code_dataset(model, depth,dataset):
    data = dataset.data['data']
    lbls = dataset.data['lbl']
    codes = model.get_code(data,depth)
    enhanced_data = np.concatenate((data,codes), axis=1)
    data_list = zip(enhanced_data, lbls) 

    #OPTIMIMISE: this could be better, makign a list then it is turned ino an array in thethe LavelledDataset        

    return LabelledDataset(data_list)
       

def rerep_trio(rerep_dataset_func, dbn, depth, datatrio):
    encode_func = functools.partial(rerep_dataset_func,dbn,depth)
    return dutil.DatasetTrio(*map(encode_func, datatrio))



def code_datatrio(dbn, depth, datatrio):
    return rerep_trio(code_dataset, dbn, depth, datatrio)

def input_and_code_datatrio(dbn, depth,datatrio):
    return rerep_trio(input_and_code_dataset, dbn, depth, datatrio)



