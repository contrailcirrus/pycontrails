# pycontrails automated tasks

SHELL := /bin/bash  # override default /bin/sh
TAG ?= $(shell git describe --tags)

# Put first so that "make" without argument is like "make help".
help:
	echo "See Makefile for recipe list"

.PHONY: help

# -----------
# Pip / Setup
# -----------

# generic pip install all dependencies
# the latest open3d and accf packages often don't support the latest
# versions of python
pip-install:
	pip install -U pip wheel
	pip install -e ".[ecmwf,gcp,gfs,jupyter,vis,zarr]"

	# these still must be installed manually for Python < 3.10
	# -pip install -e ".[open3d]"

# development installation
dev-install: pip-install

	pip install -e ".[dev,docs]"

	# install pre-commit
	pre-commit install

clean: docs-clean
	rm -rf .mypy_cache \
		   .pytest_cache \
		   .ruff_cache \
		   build \
		   dist \
		   pycontrails.egg-info \
		   pycontrails/__pycache__ \
		   pycontrails/data/__pycache__ \
		   pycontrails/datalib/__pycache__ \
		   pycontrails/models/__pycache__ \
		   pycontrails/models/cocip/__pycache__

remove: clean
	pip uninstall pycontrails

licenses:
	deplic .

check-licenses:
	deplic -c setup.cfg .

# -----------
# Extensions
# -----------

dev-pycontrails-bada:
	git clone git@github.com:contrailcirrus/pycontrails-bada.git ../pycontrails-bada
	cd ../pycontrails-bada && make dev-install

# -----------
# QC, Test
# -----------

ruff: black-check
	ruff pycontrails tests

black:
	black pycontrails

black-check:
	black pycontrails --check
	
mypy:
	mypy pycontrails

pytest:
	pytest tests/unit

pytest-cov:
	pytest \
		-v \
		--cov=pycontrails \
		--cov-report=html:coverage \
		--cov-report=term-missing \
		--durations=10 \
		--ignore=tests/unit/test_zarr.py \
		tests/unit

test: ruff mypy black-check nb-black-check pytest doctest nb-test

profile:
	python -m cProfile -o $(script).prof $(script)

# -----------
# Release
# -----------

changelog:
	git log $(shell git describe --tags --abbrev=0)..HEAD --pretty=format:'- (%h) %s' 

main-test-status:
	curl -s https://api.github.com/repos/contrailcirrus/pycontrails/actions/workflows/test.yaml/runs?branch=main \
		| jq -e -r '.workflow_runs[0].status == "completed" and .workflow_runs[0].conclusion == "success"'

	curl -s https://api.github.com/repos/contrailcirrus/pycontrails/actions/workflows/doctest.yaml/runs?branch=main \
		| jq -e -r '.workflow_runs[0].status == "completed" and .workflow_runs[0].conclusion == "success"'

# ----
# Docs
# ----

DOCS_DIR = docs
DOCS_BUILD_DIR = docs/_build

# Common ERA5 data for nb-tests and doctests
ensure-era5-cached:
	python -c 'from pycontrails.datalib.ecmwf import ERA5; \
		time = "2022-03-01", "2022-03-01T23"; \
		lev = [300, 250, 200]; \
		met_vars = ["t", "q", "u", "v", "w", "ciwc", "z", "cc"]; \
		rad_vars = ["tsr", "ttr"]; \
		ERA5(time=time, variables=met_vars, pressure_levels=lev).download(); \
		ERA5(time=time, variables=rad_vars).download()'

cache-era5-gcp:
	python -c 'from pycontrails.datalib.ecmwf import ERA5; \
		from pycontrails import DiskCacheStore; \
		cache = DiskCacheStore(cache_dir=".doc-test-cache"); \
		time = "2022-03-01", "2022-03-01T23"; \
		lev = [300, 250, 200]; \
		met_vars = ["t", "q", "u", "v", "w", "ciwc", "z", "cc"]; \
		rad_vars = ["tsr", "ttr"]; \
		ERA5(time=time, variables=met_vars, pressure_levels=lev, cachestore=cache).download(); \
		ERA5(time=time, variables=rad_vars, cachestore=cache).download()'

	gcloud storage cp -r -n .doc-test-cache/* gs://contrails-301217-unit-test/doc-test-cache/

doctest: ensure-era5-cached
	pytest --doctest-modules \
		--ignore-glob=pycontrails/ext/* \
		pycontrails -vv

doc8:
	doc8 docs

nb-black:
	black docs/examples/*.ipynb

nb-black-check:
	black docs/examples/*.ipynb --check

nb-test: ensure-era5-cached nb-black-check nb-check-links
	pytest --nbval-lax --ignore-glob=*/ACCF.ipynb docs/examples

# execute notebooks for docs output
# Note that nb-test will fail after running this locally
# TODO: this currently breaks the xarray notebook output styling somehow
nb-execute: nb-black-check
	jupyter nbconvert --inplace \
		--ClearMetadataPreprocessor.enabled=True \
		--to notebook --execute docs/examples/[!ACCF]*.ipynb docs/tutorials/*.ipynb

# Check for broken links in notebooks
# https://github.com/jupyterlab/pytest-check-links
nb-check-links:
	python -m pytest --check-links \
		--check-links-ignore "https://doi.org/10.1021/acs.est.9b05608" \
		--check-links-ignore "https://doi.org/10.1021/acs.est.2c05781" \
		--check-links-ignore "https://github.com/contrailcirrus/pycontrails-bada" \
		docs/examples docs/tutorials

docs-build: doc8
	sphinx-build -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf $(DOCS_DIR)/api/*

docs-serve: doc8
	sphinx-autobuild \
		--re-ignore .*api\/.* \
		--re-ignore CHANGELOG.md \
		--re-ignore _build\/.* \
		-b html \
		$(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-pdf: doc8
	sphinx-build -b latex $(DOCS_DIR) $(DOCS_BUILD_DIR)/latex
	cd $(DOCS_BUILD_DIR)/latex && make
