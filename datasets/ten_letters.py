import numpyutil as nutil
# data sourced from https://archive.ics.uci.edu/ml/datasets/Artificial+Characters
import numpy as np
import image_dataset as idc
import dataset_util as dutil
import os

_raw_filename = os.path.join(os.path.dirname(__file__),'ten_letters.raw')
def _raw_data():
    raw_file = open(_raw_filename)
    for line in raw_file:
        data_str, label_str = line.split("    ")
        data = np.fromstring(data_str, sep=" ")
        label = tuple(np.fromstring(label_str.strip(), sep=" "))
        yield (data, label)

datashape = (12,8)
full = idc.LabelledImageDataset(list(_raw_data()), datashape)
_test_size = 250
_valid_size = 250
base = dutil.divvy_dataset(full, _valid_size, _test_size)

letters = list("acdefghlpr")

def unsparce_letter(vect):
    index = nutil.unsparse(vect)
    print vect
    print index
    return letters[index]

def sparse_letters(letter):
    index = letters.index(letter)
    return nutil.sparse(10,index)


#symetric varients
_sym_datashape = (12*2,8)
def  _make_sym_letters(basedata):
    datalist = dutil.make_sym(basedata)
    return idc.LabelledImageDataset(datalist,_sym_datashape, balance=True)

sym_full = _make_sym_letters(full.as_unlabelled())
sym = dutil.divvy_dataset(sym_full, _valid_size, _test_size)
#Normalised varients
_full_patterns, _full_labels = zip(*list(full))
_norm_full_patterns = dutil.normalize(_full_patterns)
norm_full = idc.LabelledImageDataset(zip(_norm_full_patterns, _full_labels), datashape)

norm = dutil.divvy_dataset(norm_full, _valid_size, _test_size)


#normalised systic varients
sym_norm_full = _make_sym_letters(norm_full.as_unlabelled())
sym_norm = dutil.divvy_dataset(sym_norm_full, _valid_size, _test_size)
