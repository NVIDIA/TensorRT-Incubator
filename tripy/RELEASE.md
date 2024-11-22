# Releasing Tripy

This document explains how to release a new version of Tripy.

1. Update version numbers in [`pyproject.toml`](./pyproject.toml) and
    [`__init__.py`](./tripy/__init__.py) (make sure they match!).

    Often, updates to Tripy will also require updates to dependencies,
    like MLIR-TRT, so make sure to update those version numbers as well.

2. Add a new entry to [`packages.html`](./docs/packages.html).
    This ensures that we will be able to `pip install` Tripy.

3. If there were any other functional changes since the most recent
    L1, make sure to run L1 testing locally.

4. Create a PR with the above changes.

5. Once the PR created in (4) is merged, **WAIT FOR THE POST-MERGE PIPELINES TO COMPLETE**.
    This is a very important step as otherwise the release pipeline could fail.

    Once the post-merge pipelines have succeeded, create a new tag with:
    ```bash
    git tag tripy-vX.Y.Z
    ```
    replacing `X.Y.Z` with the version number and push it to the repository.

    This should trigger our release pipeline, which will build and deploy
    the documentation and create a GitHub release with the wheel.


## Additional Notes

The public build instructions use the usual isolated build flow to create the
wheels. However, in our release pipelines, we would also like to include stub
files in the generated wheels. `stubgen`, which we use in [setup.py](./setup.py)
requires all dependencies of of the package to be installed for the `--inspect-mode`
option to work. Obviously, this would be a bit circular for external users, but it
is easy for us to do in the development container which already includes the dependencies.
