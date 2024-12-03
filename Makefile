# pycontrails automated tasks

SHELL := /bin/bash  # override default /bin/sh
TAG ?= $(shell git describe --tags)

# Put first so that "make" without argument is like "make help".
help:
	echo "See Makefile for recipe list"

.PHONY: help

# Colors
COLOR_YELLOW = \033[0;33m
END_COLOR = \033[0m

# -----------
# Pip / Setup
# -----------

# generic pip install all dependencies
# the latest open3d and accf packages often don't support the latest
# versions of python
pip-install:
	python -m pip install -U pip wheel
	python -m pip install -e ".[complete]"

	# open3d wheels not available for latest python versions; install manually
	# pip install -e ".[open3d]"

# development installation
dev-install: pip-install

	python -m pip install -e ".[dev,docs]"

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

PYCONTRAILS_BADA_DIR = ../pycontrails-bada

dev-pycontrails-bada:
	git -C $(PYCONTRAILS_BADA_DIR) pull || git clone git@github.com:contrailcirrus/pycontrails-bada.git $(PYCONTRAILS_BADA_DIR)
	cd $(PYCONTRAILS_BADA_DIR) && make dev-install

# -----------
# QC, Test
# -----------

lint:
	ruff check pycontrails tests docs
	ruff format --check pycontrails tests docs

format:
	ruff format pycontrails tests docs

# https://taplo.tamasfe.dev/
taplo:
	taplo format pyproject.toml --option indent_string='    '

# https://yamllint.readthedocs.io/en/stable/configuration.html
yamllint:
	yamllint -d "{extends: default, rules: {line-length: {max: 100}}}" .

mypy:
	mypy pycontrails

pytest:
	pytest tests/unit

pytest-regenerate-results:
	pytest tests/unit --regenerate-results

pytest-cov:
	pytest \
		-v \
		--cov=pycontrails \
		--cov-report=html:coverage \
		--cov-report=term-missing \
		--durations=10 \
		--ignore=tests/unit/test_zarr.py \
		tests/unit

test: lint mypy pytest doctest nb-test

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

# Check that numpy 2 is being used
ensure-numpy-2:
	[[ $$(python -c 'import numpy as np; print(np.__version__)') == 2.* ]] \
	|| ( \
		echo -e "$(COLOR_YELLOW)NumPy 2 required for doctests$(END_COLOR)"; \
		exit 1)

# Check for GCP credentials
ensure-gcp-credentials:
	python -c 'from google.cloud import storage; storage.Client()' \
		|| ( \
		echo -e \
			"$(COLOR_YELLOW)Google application credentials required to run doc tests\n" \
			"See https://cloud.google.com/docs/authentication/application-default-credentials$(END_COLOR)"; \
		exit 1)

# Cache common ERA5 data for nb-tests and doctests in DiskCacheStore
ensure-era5-cached:
	python -c 'from pycontrails.datalib.ecmwf import ERA5; \
		from pycontrails import DiskCacheStore; \
		cache = DiskCacheStore(); \
		time = "2022-03-01", "2022-03-01T23"; \
		lev = [300, 250, 200]; \
		met_vars = ["t", "q", "u", "v", "w", "ciwc", "z", "cc"]; \
		rad_vars = ["tsr", "ttr"]; \
		ERA5(time=time, variables=met_vars, pressure_levels=lev, cachestore=cache).download(); \
		ERA5(time=time, variables=rad_vars, cachestore=cache).download()' \
		|| ( \
		echo -e \
			"$(COLOR_YELLOW)CDS Credentials required to run notebooks and doc tests\n" \
			"See https://py.contrails.org/api/pycontrails.datalib.ecmwf.ERA5.html$(END_COLOR)"; \
		exit 1)

# Cache ERA5 files to GCP for use in Actions
cache-era5-gcp: ensure-era5-cached
	gcloud storage cp -r -n .doc-test-cache/* gs://contrails-301217-unit-test/doc-test-cache/

doctest: ensure-numpy-2 ensure-era5-cached ensure-gcp-credentials
	pytest --doctest-modules \
		--ignore-glob=pycontrails/ext/* \
		pycontrails -vv

doc8:
	doc8 docs

nb-format-check:
	ruff format --check docs/**/*.ipynb

# Note must be kept in sync with 
# `.pre-commit-config.yaml` and `make nb-clean-check`
nb-clean:
	nb-clean clean docs/**/*.ipynb \
        --remove-empty-cells \
		--preserve-cell-metadata tags \
		--preserve-cell-outputs \
		--preserve-execution-counts

nb-clean-check:
	nb-clean check docs/**/*.ipynb \
        --remove-empty-cells \
		--preserve-cell-metadata tags \
		--preserve-cell-outputs \
		--preserve-execution-counts

# Makes sure that all cells can execute without errors
# Add `nbval-skip` cell tag if you want to skip a cell
# Add `nbval-check-output` cell tag if you want to specifically compare cell output
# TODO: 
#   - Install eccodes on Github Action so GFS notebook can run
#   - Provide meteorology data for `run-cocip-on-flight` tutorial
nb-test: ensure-era5-cached nb-clean-check nb-format-check nb-check-links
	python -m pytest --nbval-lax \
		--ignore=docs/integrations/ACCF.ipynb \
		--ignore=docs/integrations/APCEMM.ipynb \
		--ignore=docs/notebooks/specific-humidity-interpolation.ipynb \
		--ignore=docs/notebooks/GFS.ipynb \
		--ignore=docs/notebooks/run-cocip-on-flight.ipynb \
		--ignore=docs/notebooks/model-levels.ipynb \
		--ignore=docs/notebooks/ARCO-ERA5.ipynb \
		docs/notebooks docs/integrations

# Check for broken links in notebooks
# https://github.com/jupyterlab/pytest-check-links
nb-check-links:
	python -m pytest --check-links \
		--check-links-ignore "https://doi.org/10.1021/acs.est.9b05608" \
		--check-links-ignore "https://doi.org/10.1021/acs.est.2c05781" \
		--check-links-ignore "https://doi.org/10.1175/JAMC-D-11-0242.1" \
		--check-links-ignore "https://github.com/contrailcirrus/pycontrails-bada" \
		--check-links-ignore "https://ourairports.com" \
		--check-links-ignore "https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast" \
		docs/notebooks/*.ipynb

# Execute all notebooks in docs
# NOTE notebooks from docs/integrations/ manually
# Add `skip-execution` cell tag if you want to skip a cell
# Add `raises-exception` cell tag if you know the cell raises exception
nb-execute: ensure-era5-cached nb-format-check nb-check-links
	jupyter nbconvert --inplace \
		--to notebook \
		--execute \
		docs/notebooks/*.ipynb

	# clean notebooks after execution
	make nb-clean

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

	# GIF in notebooks/advection.ipynb kills the pdflatex process
	# If advection notebook changes, this will have to get updated
	rm docs/_build/latex/notebooks_advection_20_0.png

	# this needs to get run twice
	cd $(DOCS_BUILD_DIR)/latex && make
	cd $(DOCS_BUILD_DIR)/latex && make

	echo "PDF exported to $(DOCS_BUILD_DIR)/latex/pycontrails.pdf"
