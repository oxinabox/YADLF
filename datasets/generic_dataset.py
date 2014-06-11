import numpy as np
from .. import numpyutil as nutil
import dataset_util
import abc



class Dataset(object):
    def __init__(self, data, dtype):
        self.dtype=dtype
        self.data = data

    def make_from(self, data):
        '''
        returns a new one of these, with same parameters but different data
        class should override this so that slices are the right type

        '''
        return self.__class__(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,value):
        self._data=np.asarray(value,self.dtype)


    @abc.abstractmethod
    def __getitem__(self,sliceIndex):
        data = self.data[sliceIndex]
        if np.size(data) and not isinstance(sliceIndex, (int,long)):
            return self.make_from(data)
        else:
            return data
    
    

    @abc.abstractproperty
    def datum_size(self):
        pass

    def __len__(self):
        return len(self.data)


class UnlabelledDataset(Dataset):
    def __init__(self, data):
        if not(len(data[0].shape)==1): #if it isn't flat, flatten it
            data = map(nutil.flatten, data)
        dtype = np.float32 
        Dataset.__init__(self,data, dtype)


    @property
    def datum_size(self):
        return np.size(self.data[0][0])



class LabelledDataset(Dataset):
    def __init__(self, data):
        dtype = np.dtype([('data', np.float32, len(data[0][0])),
                          ('lbls', np.float32, len(data[0][1]))
                         ])
        Dataset.__init__(self,data,dtype)
    
    def balance(self):
        '''Balances the dataset. Warning: This will delete elements if there is not a equal numbers of each class. '''
        orig_data = self.data
        balanced_gen = dataset_util.balance_ordering(orig_data)
        self.data = list(balanced_gen)


    def make_from(self, data):
        return LabelledDataset(data)

    def as_unlabelled(self):
        unlabelled_data = self.data['data']
        return UnlabelledDataset(unlabelled_data)

    @property
    def label_size(self):
        return np.size(self.data[0][1])
    @property
    def datum_size(self):
        return np.size(self.data[0][0])
