

all:
	python setup.py build_ext --inplace
	mv *.so speedup

html:
	cython -a src/*.pyx

