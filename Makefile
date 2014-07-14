

all:
	python setup.py build_ext --inplace
	cython -a src/*.pyx
	mv *.so speedup

html:
	cython -a src/*.pyx

run:
	python setup.py build_ext --inplace
	./differential.py



