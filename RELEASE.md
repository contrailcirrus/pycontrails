# Release

> ``pycontrails`` is tagged and pushed to [pypi](https://pypi.org/project/pycontrails/) through GitHub Actions.

1. Ensure all changes to be released are committed and pushed to `main`.
1. Ensure [Unit tests](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml) and [Doc / Notebook tests](https://github.com/contrailcirrus/pycontrails/actions/workflows/doctest.yaml) have passed on `main`.
   The [Release Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml) will fail if these actions have not completed successfully.
1. Review and update the [CHANGELOG](CHANGELOG.md).
   Ensure the most recent entry is for the tag about to be released.
1. Create a [new release on Github](https://github.com/contrailcirrus/pycontrails/releases).
   Double check that the release tag is consistent and consecutive with
   the previous release tag following [PEP 440](https://peps.python.org/pep-0440/#version-scheme).
   This often but not always agrees with [Semantic Versioning](https://semver.org/).
1. Copy the CHANGELOG content for the release into the *Description*.
1. Publish the new release.
   This will create a tag on `main` and trigger the [Release Action](https://github.com/contrailcirrus/pycontrails/actions/workflows/release.yaml).
1. Confirm that the release is successfully deployed to [PyPI](https://pypi.org/project/pycontrails/).
1. Confirm that the documentation is successfully deployed to [py.contrails.org](https://py.contrails.org).
