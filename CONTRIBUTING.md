# Contributing

Contributions (bug reports, fixes, documentation,
enhancements, ideas, ...) are welcome and appreciated.

To get started, find the best path for your contribution:

- Ask questions, discuss models, and present ideas in [Discussions](https://github.com/contrailcirrus/pycontrails/discussions).
- Report bugs or suggest changes as [Issues](https://github.com/contrailcirrus/pycontrails/issues).
- Contribute fixes or improvements as [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls).

Model updates should first be reviewed in [Discussions](https://github.com/contrailcirrus/pycontrails/discussions) before creating issues or submitting PRs.

Please follow the [Github Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines) when participating in any of these forums.

## Contributing to documentation

Documentation is written in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) and built into website using [Sphinx](https://www.sphinx-doc.org/en/master/).

For small changes, you can edit files within the `docs/` directly in Github and submit
as a [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls) within the Github interface.

For larger changes:

- Setup a [local development environment](https://py.contrails.org/install#develop).
- Edit documents and notebooks following the conventions in the existing documentation.
- Build and review the documentation locally:

```bash
# docs build to directory docs/_build/html
$ make docs-build

# automatically build docs on changes
# docs will be served at `http://127.0.0.1:8000`
$ make docs-serve
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).

## Contributing to the code base

We (mostly) adhere to the Contributing guidelines published by [xarray](https://docs.xarray.dev/en/stable/contributing.html).

If you are new to development, see xarray's [Working with the code](https://docs.xarray.dev/en/stable/contributing.html#working-with-the-code) to get started with
version control, [git](http://git-scm.com/), [Github](https://github.com/contrailcirrus/pycontrails), [Forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo), and [creating branches](https://docs.xarray.dev/en/stable/contributing.html#creating-a-branch).

For more involved updates, make sure a [Github Issue](https://github.com/contrailcirrus/pycontrails/issues) describing the intended changes.

Once ready to develop:

- Setup a [local development environment](https://py.contrails.org/install#develop).
- Make sure [pre-commit hooks](https://pre-commit.com/) are installed:

```bash
$ make dev-install

# or if manually installed
$ pre-commit install 
```

- Implement updates.
   Make sure all code is well documented using the [numpy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
- Ensure tests pass locally:

```bash
$ make test
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).
