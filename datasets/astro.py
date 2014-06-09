from __future__ import division
from .. import numpyutil as nutil
import numpy as np
import dataset_util as dutil
import generic_dataset as dc
import itertools as it

path = '/home/wheel/oxinabox/Uni_Notes/honours/prototypes/datasets/uwa_astro/rand_phase/'
suffix = '_snr5_size1000'
_data_raw=lbls_raw = np.loadtxt(path+'data'+suffix).T
#_lbls_phase_raw = np.loadtxt(path+'labels_phase'+suffix)
_lbls_freq_raw = np.loadtxt(path+'labels'+suffix)
n_freqs = 64


def sparse_phase_shifts(raw_shifts):
    int_shifts = raw_shifts*(16/(2*np.pi))
    return map(lambda p: tuple(nutil.sparse(16,p)), int_shifts)

def unsparse_phase_shifts(sparse_shifts):
    int_shifts = map(nutil.unsparse, sparse_shifts)
    return (2*np.pi/4)*int_shifts


_fstart = _lbls_freq_raw[0]
_f_diff = _lbls_freq_raw[1]-_fstart


def sparse_freq(raw_rep):
    int_rep = (raw_rep-_fstart)/_f_diff
    return map(lambda i: tuple(nutil.sparse(n_freqs,i)), int_rep)

def unsparse_freq(sparse_rep):
    int_rep = map(nutil.unsparse, sparse_rep)
    return int_rep*_f_diff+_fstart

#_lbls_phase = sparse_phase_shifts(_lbls_phase_raw)
_lbls_freq = sparse_freq(_lbls_freq_raw)

_norm_data_raw = dutil.normalize(_data_raw)

#Frequency
test_size = n_freqs*int(0.001*len(_norm_data_raw))
valid_size = test_size

#freq_full = idc.LabelledDataset(it.izip(_data_raw,_lbls_freq),True)


norm_freq_full = dc.LabelledDataset(list(it.izip(_norm_data_raw,_lbls_freq)),True)

norm_freq=dutil.divvy_dataset(norm_freq_full, valid_size,test_size)

#Phase
#phase_full = idc.LabelledDataset(it.izip(_data_raw,_lbls_phase),datashape,True)

#norm_phase_full = dc.LabelledDataset(list(it.izip(_norm_data_raw,_lbls_phase)),True)

#norm_phase=dutil.divvy_dataset(norm_phase_full, valid_size,test_size)
