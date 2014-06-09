import sys
import os
import importlib

sys.path.append('/datasets')
sys.path.append('/..')
import datasets.generic_dataset as generic_dataset
from dataset_util import DatasetTrio
import cPickle as pickle

def convert(module_filename):
#    module_filename= os.path.join(os.path.dirname(__file__), module_filename)
    module_name = ".".join(os.path.basename(module_filename).split(".")[:-1])
    module_dir = os.path.dirname(module_filename)

    pickle_filename = os.path.join(module_dir,module_name + ".pkl")
    output_filename = os.path.join(module_dir,module_name +"_loaded" + ".py")
    

    
    #module_obj = imp.load_source(module_name,module_filename)
    module_obj = importlib.import_module(module_name,'datasets')



    datas = dict((name, val) for (name,val) in module_obj.__dict__.iteritems() 
                if isinstance(val,(generic_dataset.Dataset, DatasetTrio)))

    def pickle_data():
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(datas.values(),pickle_file, 1)


    def make_module():
        template = """
import os
import sys
_module_dir = os.path.dirname(__file__)
_filename = os.path.join(_module_dir,'%(pickle_filename)s')
if not _module_dir in sys.path:
    sys.path.append(_module_dir)

import cPickle as pickle
with open(_filename,'rb') as _pickle_file:
    [%(varnames)s]=pickle.load(_pickle_file)
"""         
        with open(output_filename,'w') as module_file:
            content = template % dict(
                                module_dir= module_dir,
                                varnames=", ".join(datas.keys()),
                                pickle_filename = pickle_filename
                                     )
            module_file.write(content)

    pickle_data()
    make_module()

print "Pickling ", sys.argv[1],
convert(sys.argv[1])
print "Done"



