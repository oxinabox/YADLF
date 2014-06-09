

#atasets = datasets/ten_letters.py datasets/mnist.py


numpyutil.so: numpyutil.pyx
	python setup.py build_ext --inplace
	gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o $@ numpyutil.c

#(datasets) 

