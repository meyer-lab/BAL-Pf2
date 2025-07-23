.PHONY: clean test pyright

flist = $(wildcard pf2/figures/figure*.py)
allOutput = $(patsubst pf2/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: pf2/figures/figure%.py
	@ mkdir -p ./output
	rye run bal_fbuild $*

test: .venv
	rye run pytest -s -v -x

.venv:
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=pf2 --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright pf2

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache