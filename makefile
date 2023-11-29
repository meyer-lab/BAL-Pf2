SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard pf2/figures/figure*.py)
allOutput = $(patsubst pf2/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: pf2/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*
