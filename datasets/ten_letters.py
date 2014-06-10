from .. import numpyutil as nutil
# data sourced from https://archive.ics.uci.edu/ml/datasets/Artificial+Characters
import numpy as np
import image_dataset as idc
import dataset_util as dutil
import os
import copy


_raw_filename = os.path.join(os.path.dirname(__file__),'ten_letters.raw')
def _raw_data():
    raw_file = open(_raw_filename)
    for line in raw_file:
        data_str, label_str = line.split("    ")
        data = np.fromstring(data_str, sep=" ")
        label = tuple(np.fromstring(label_str.strip(), sep=" "))
        #Note: The data in RAW form is already Sparse repressented
        yield (data, label)

datashape = (12,8)
full = idc.LabelledImageDataset(list(_raw_data()), datashape)
_test_size = 250
_valid_size = 250
base = dutil.divvy_dataset(full, _valid_size, _test_size)

norm = copy.deepcopy(base)
dutil.normalise_dataset_trio(norm)

letters = list("acdefghlpr")

def unsparce_letter(vect):
    index = nutil.unsparse(vect)
    return letters[index]

def sparse_letters(letter):
    index = letters.index(letter)
    return nutil.sparse(len(letters), index)
