import image_dataset as idc
from .. import numpyutil as nutil
import numpy as np
import cPickle, gzip
import dataset_util as dutil
import os.path


def bind_labels(data,lbls_raw):
    def label (lbl_raw):
        return tuple(nutil.sparse(10,lbl_raw))
    data_list = [(np.asarray(dv),label(lbl)) 
                 for (dv, lbl) in zip(data, lbls_raw)]
    return idc.LabelledImageDataset(data_list, balance=False)

def _balance_all(datasets):
    for dd in datasets:
        dd.balance()


_raw_filename = os.path.join(os.path.dirname(__file__),'mnist_dln.pkl.gz')
with  gzip.open(_raw_filename, 'rb') as pickled:
    (_train_raw, _train_labels_raw), \
    (_valid_raw, _valid_labels_raw),\
    (_test_raw, _test_labels_raw) = cPickle.load(pickled)


base = dutil.DatasetTrio(
            bind_labels(_train_raw, _train_labels_raw),
            bind_labels(_valid_raw, _valid_labels_raw),
            bind_labels(_test_raw, _test_labels_raw))

_balance_all(base)

norm = dutil.normalise_dataset_trio(base)


