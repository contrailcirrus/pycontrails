# Release

> Current release process is manual

1. Open a merge request on [Github](https://github.com/contrailcirrus/pycontrails/). The overview summary should be informed by the `git` commit messages. See the `make changelog` helper.

```bash
make changelog
```

1. Update the changelog (`CHANGELOG.md`). This should be similar to the merge request summary in the previous step.

1. Ensure all tests are passing *locally*.

```bash
make test
```

1. Ensure all [tests are passing](https://github.com/contrailcirrus/pycontrails/actions).

1. Test the docstring and examples documentation.

```bash
make doctest
make nbtest
```

1. Merge into `main`

1. Bump version, commit, and tag with semantic release number. Push tag to remote:

```bash
make bump version=X.Y.Z  # update pycontrails.__version__ and commit locally
make release      # create tag and push tag
```
