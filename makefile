SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard pf2/figures/figure*.py)
allOutput = $(patsubst pf2/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: pf2/figures/figure%.py factor_cache/factors.h5ad
	@ mkdir -p ./output
	poetry run fbuild $*

output/figure2.svg:
    : by default, don't build $@

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports pf2

factor_cache/factors.h5ad:
	@ mkdir -p ./factor_cache
	poetry run factor
