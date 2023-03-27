# Release

> ``pycontrails`` is released through GitHub Actions

Releases are triggered by cutting a [new release](https://github.com/contrailcirrus/pycontrails/releases) through the GitHub UI. The releaser is responsible for providing a release tag, title, and description.

- The release tag should be a semantic version number prefixed with a `v` (e.g. `v0.1.0`).
- The release title should be the same as the release tag.
- The release description can be a summary of the changes since the last release (for example, key features from the [changelog](CHANGELOG.md)). It's recommended to use the **Generate release notes** feature of the GitHub UI to generate a list of changes since the last release.

## Release checklist

1. Ensure all changes to be released are committed and pushed to `main`.
1. Ensure all tests have passed on [Github Actions](https://github.com/contrailcirrus/pycontrails/actions/workflows/test.yaml).
1. Check and update the [changelog](CHANGELOG.md). In particular, ensure that the most recent entry is for the tag about to be released.
1. Create the [new release on Github](https://github.com/contrailcirrus/pycontrails/releases). Double check that the release tag is consistent and consecutive with the previous release tag following [PEP 440](https://peps.python.org/pep-0440/#version-scheme). This often but not always agrees with [Semantic Versioning](https://semver.org/).
1. Confirm that the release is successfully deployed to [PyPI](https://pypi.org/project/pycontrails/).
1. Confirm that the documentation is successfully deployed to [py.contrails.org](https://py.contrails.org).
