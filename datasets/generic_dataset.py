import numpy as np
import numpyutil as nutil
import dataset_util
import abc



class Dataset(object):
    def __init__(self, data):
        self.data = np.asarray(data)

    def make_from(self, data):
        '''
        returns a new one of these, with same parameters but different data
        class should override this so that slices are the right type

        '''
        return self.__class__(data)


    @abc.abstractmethod
    def __getitem__(self,sliceIndex):
        data = self.data[sliceIndex]
        if np.size(data) and not isinstance(sliceIndex, (int,long)):
            return self.make_from(data)
        else:
            return data


    @abc.abstractproperty
    def datasize(self):
        pass

    def __len__(self):
        return len(self.data)


class UnlabelledDataset(Dataset):
    def __init__(self,data):
        if not(len(data[0].shape)==1): #if it isn't flat, flatten it
            data = map(nutil.flatten, data)
        Dataset.__init__(self,data)


    @property
    def datasize(self):
        return np.size(self.data[0][0])



class LabelledDataset(Dataset):
    def __init__(self,data, balance=False):
        Dataset.__init__(self,data)
        if balance:
            self.balance()

    
    def balance(self):
        '''Balances the dataset. Warning: This will delete elements if there is not a equal numbers of each class. '''
        orig_data = self.data
        balanced_gen = dataset_util.balance_ordering(orig_data)
        self.data = np.asarray(list(balanced_gen))


    def make_from(self, data):
        return LabelledDataset(data, False)

    def as_unlabelled(self):
        unlabelled_data = zip(*self.data)[0]
        return UnlabelledDataset(unlabelled_data)

    @property
    def label_size(self):
        return np.size(self.data[0][1])
    @property
    def datasize(self):
        return np.size(self.data[0][0])
