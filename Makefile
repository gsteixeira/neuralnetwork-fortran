SHELL := /bin/bash

default: neural.f90
	gfortran neural.f90 -o neuralnet

run:
	gfortran neural.f90 -o neuralnet
	./neuralnet

clean:
	rm -f neuralnet
	rm -f *.mod
