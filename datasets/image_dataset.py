import numpy as np

import itertools as it
from .. import numpyutil as nutil


import generic_dataset


def show_titled_data(titled_data_vectors, datashape):
    import pylab as pl
    print datashape
    pl.subplots_adjust(wspace=2)
    for ii, dvt in enumerate(titled_data_vectors):
        (dv,titletext) = dvt
        pl.subplot(1,len(titled_data_vectors), ii+1)
        pl.imshow(1-np.reshape(dv, datashape)) #,  cmap = pl.cm.Greys_r)
        pl.title(str(titletext))
        pl.show()


def show_data(data_vectors, datashape):
    dvt=it.izip_longest(data_vectors,[],fillvalue="")
    dvt = list(dvt)
    show_titled_data(dvt, datashape)


def _default_datashape(vector, datashape=None):
        if not datashape:
            sqlen= round(np.size(vector)**0.5) #Assume Square
            assert sqlen**2==np.size(vector), \
                    "if datashape not given, the vector must be square"
            datashape=(sqlen, sqlen)
        else:
           assert datashape[0]*datashape[1]==np.size(vector), \
                  """dataShape is not a possible shape of the data. Shape of data %i""" % np.size(vector)  
        return datashape




class UnlabelledImageDataset(generic_dataset.UnlabelledDataset):
    def __init__(self,data,datashape=None):
        self.datashape = _default_datashape(data[0], datashape)
        generic_dataset.UnlabelledDataset.__init__(self,data)

    def make_from(self, data):
        return UnlabelledImageDataset(data, self.datashape)


    def show_data(self, sliceIndex):
        ''' good to use numpy.s_[slicenotation] to declare the sliceIndex'''
        dataVectors = self.data[sliceIndex]
        show_data(dataVectors, self.datashape)





class LabelledImageDataset(generic_dataset.LabelledDataset):
    def __init__(self,data,datashape=None):
        self.datashape = _default_datashape(data[0][0], datashape)
        generic_dataset.LabelledDataset.__init__(self,data)

    def make_from(self, data):
        return LabelledImageDataset(data, self.datashape)

    def show_data(self, sliceIndex):
        ''' good to use numpy.s_[slicenotation] to declare the sliceIndex'''
        data_vectors_and_titles = self.data[sliceIndex]        
        show_titled_data(data_vectors_and_titles, self.datashape)

    def as_unlabelled(self):
        unlabelled_data = self.data['data']
        return UnlabelledImageDataset(unlabelled_data, self.datashape)



