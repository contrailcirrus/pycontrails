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
	pip install -e .[ecmwf,gcp,gfs,jupyter,vis,zarr]

	# these still must be installed manually for Python < 3.10
	# -pip install -e .[open3d]
	# -pip install -e .[accf]

# development installation
dev-install: pip-install

	pip install -e .[dev,docs]

	# install pre-commit
	pre-commit install

clean: docs-clean
	rm -rf .mypy_cache \
		   .pytest_cache \
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

flake8: black-check
	flake8 pycontrails tests

black:
	black pycontrails

black-check:
	black pycontrails --check

nbblack:
	black docs/examples/*.ipynb

nbblack-check:
	black docs/examples/*.ipynb --check
	
mypy:
	mypy pycontrails

pytest:
	pytest tests/unit

pydocstyle:
	pydocstyle pycontrails --match='[^_deprecated].*.py' --verbose

pytest-cov:
	pytest \
		-v \
		--cov=pycontrails \
		--cov-report=html:coverage \
		--cov-report=term-missing \
		--durations=10 \
		--ignore=tests/unit/test_zarr.py \
		tests/unit

pytest-integration:
	pytest tests/integration

# Common ERA5 data for nbtests and doctests
ensure-era5-cached:
	python -c 'from pycontrails.datalib.ecmwf import ERA5; \
		time = "2022-03-01", "2022-03-01T23"; \
		lev = [300, 250, 200]; \
		met_vars = ["t", "q", "u", "v", "w", "ciwc", "z", "cc"]; \
		rad_vars = ["tsr", "ttr"]; \
		ERA5(time=time, variables=met_vars, pressure_levels=lev).download(); \
		ERA5(time=time, variables=rad_vars).download()'

doctest: ensure-era5-cached
	pytest --doctest-modules pycontrails -vv

nbtest: ensure-era5-cached nbblack-check
	pytest -W ignore --nbval-lax -p no:python docs/examples

test: flake8 mypy pytest doctest black-check nbblack-check pydocstyle

profile:
	python -m cProfile -o $(script).prof $(script)



# -----------
# Release
# -----------

changelog:
	git log $(shell git describe --tags --abbrev=0)..HEAD --pretty=format:'- (%h) %s' 


preversion:
	# make sure we're on the main branch
	[ `git rev-parse --abbrev-ref HEAD` = main ] || (printf "\n--> must be on the main branch\n\n" && exit 1)

	# ensure nothing outstanding
	git diff --exit-code || (printf "\n--> commit or stash outstanding changes\n" && exit 1)
	git diff --cached --exit-code || (printf "\n--> commit or stash outstanding changes\n" && exit 1)

bump: preversion
	# usage: 
	# 
	# $ make bump version=0.12.5

	# bump version
	python -c 'import re, sys; \
		f = open("pyproject.toml", "r"); \
		p = r"version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\""; \
		txt = f.read(); \
		m = re.search(p, txt); \
		major, minor, patch = [int(v) for v in m.groups()]; \
		target = "$(version)".split("."); \
		tmajor, tminor, tpatch = [int(v) for v in target]; \
		case1 = tmajor == major and tminor == minor and tpatch == patch + 1; \
		case2 = tmajor == major and tminor == minor + 1 and tpatch == 0; \
		case3 = tmajor == major + 1 and tminor == 0 and tpatch == 0; \
		(case1 or case2 or case3) or sys.exit("Versions must be consecutive"); \
        updated = txt.replace(m.group(), f"version = \"{tmajor}.{tminor}.{tpatch}\""); \
		open("pyproject.toml", "w").write(updated)'

	# re-install package to update package version
	pip install -e .

	# commit version
	git add pyproject.toml
	git commit -m "release($(version)): bump version"

release: preversion

	# fetch all tags
	git fetch --tags
	
	# get version
	version=`python -c 'import pycontrails; print(pycontrails.__version__);'`; \
		git tag -a v$$version -m ""; \
		git push origin; \
		git push origin v$$version;

# ----
# Docs
# ----

DOCS_DIR = docs
DOCS_BUILD_DIR = docs/_build

doc8:
	doc8 docs

docs-build: doc8
	sphinx-build -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)
	rm -rf $(DOCS_DIR)/api/*

docs-serve: doc8
	sphinx-autobuild \
		--re-ignore .*api\/.* \
		--re-ignore *CHANGELOG.md \
		-b html \
		$(DOCS_DIR) $(DOCS_BUILD_DIR)/html

docs-pdf: doc8
	sphinx-build -b latex $(DOCS_DIR) $(DOCS_BUILD_DIR)/latex
	cd $(DOCS_BUILD_DIR)/latex && make
