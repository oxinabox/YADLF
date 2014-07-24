

#atasets = datasets/ten_letters.py datasets/mnist.py


numpyutil.so: numpyutil.pyx
	python setup.py build_ext --inplace
	gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I/home/stud/20361362/local/x86_64/python/venv/include/python2.6 -o $@ numpyutil.c

#(datasets) 

.PHONY : clean
clean:
	rm numpyutil.c
	rm numpyutil.so


