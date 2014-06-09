#!python
#cython: embedsignature=True


cimport cython
import numpy as np
cimport numpy as np
import itertools as it



def sparse(base,value):
    assert value>=0
    assert value<base
    ret = np.zeros(base)
    ret[value]=1
    return ret

def unsparse(sparse_value):
    return np.nonzero(sparse_value)[0][0] #Nonzero returns a tupple of array indexs which are nonzero

def flatten(datum): 
        return np.reshape(datum,-1)

def arr_info(var, name=None):
    if type(name) is str:
        name = name + " is: "
    else:
        name=""
     
    return (name + str(type(var)) + " " + str(np.shape(var)) + " len="+ str(len(var)))

def print_nest_info(ele, name=None, max_child=3 ,depth=0):
    if type(name) is str:
        name = name + " is: "
    else:
        name=""
    print('\t'*depth + str(type(ele)))
    try:
        print('\t'*depth + "len: "+ str(len(ele)))
        for subele in ele[:max_child]:
            print_nest_info(subele, None, max_child, depth+1)
    except (IndexError, TypeError):
        pass #We have hit the bottom





def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return it.izip(a, b)

def sample(probVector):
    return np.random.random_sample(probVector.shape)<probVector

#Nested functions to operate on lists of ndarrays

@cython.boundscheck(False)
@cython.wraparound(False)
def uniop_nested(func,o_list):
    def inner(i_list):
        if isinstance(i_list[0],np.ndarray):
             return map(func, i_list)
        else:
             return map(inner, i_list)
    return inner(o_list)
   
  
@cython.boundscheck(False)
@cython.wraparound(False)
def binop_nested(func, o1, o2):
    if not isinstance(o1,np.ndarray):
        return [binop_nested(func, i1, i2)  for (i1,i2) in zip(o1,o2)]
    else:
        return func(o1,o2)

  
@cython.boundscheck(False)
@cython.wraparound(False)
def triop_nested(func, o1, o2, o3):
    if not isinstance(o1,np.ndarray):
        return [triop_nested(func, i1, i2, i3)  for (i1,i2, i3) in zip(o1,o2, o3)]
    else:
        return func(o1,o2, o3)




@cython.boundscheck(False)
@cython.wraparound(False)
def add_nested(s1,s2):
    return binop_nested(np.add,s1,s2)
