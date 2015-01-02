DATASETS = ten_letters_loaded.py mnist_loaded.py 
#astro_loaded.py

all: $(DATASETS)

$(DATASETS): %_loaded.py: %.py
	python dataset_module_pickler.py $<

	

