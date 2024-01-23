
# Contributing

Contributions (bug reports, fixes, documentation,
enhancements, ideas, ...) are welcome and appreciated.

To get started, find the best path for your contribution:

- Ask questions, discuss models, and present ideas in [Discussions](https://github.com/contrailcirrus/pycontrails/discussions).
- Report bugs or suggest changes as [Issues](https://github.com/contrailcirrus/pycontrails/issues).
- Contribute fixes or improvements as [Pull Requests](https://github.com/contrailcirrus/pycontrails/pulls).

Please follow the [Github Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines) when participating in any of these forums.

The following emulates the [xarray contributing guidelines](https://docs.xarray.dev/en/stable/contributing.html).

## Contributing to documentation

Documentation is written in [reStructuredText](http://docutils.sourceforge.net/rst.html) and synthesized with [Sphinx](https://www.sphinx-doc.org/en/master/).

For small changes, [fork and edit](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) files directly in the Github interface.

For larger changes:

- Set up a [local development environment](https://py.contrails.org/develop.html).
- Edit documents and notebooks following [existing conventions](https://py.contrails.org/develop.html#conventions).
- Build and review the documentation locally:

```bash
# docs build to directory docs/_build/html
$ make docs-build
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).

## Contributing to the code base

If you are new to development, see xarray's [Working with the code](https://docs.xarray.dev/en/stable/contributing.html#working-with-the-code).
This reference provides an introduction to version control, [git](http://git-scm.com/), [Github](https://github.com/contrailcirrus/pycontrails),
[Forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo), and [creating branches](https://docs.xarray.dev/en/stable/contributing.html#creating-a-branch).

For more involved changes, create a [Github Issue](https://github.com/contrailcirrus/pycontrails/issues) describing the intended changes first.

Once you're ready to develop:

- Set up a [local development environment](https://py.contrails.org/develop.html).
- Implement updates.
  Make sure code is documented using [existing conventions](https://py.contrails.org/develop.html#conventions).
- Ensure tests pass locally:

```bash
$ make test
```

- Submit changes as a [Pull Request](https://github.com/contrailcirrus/pycontrails/pulls).
