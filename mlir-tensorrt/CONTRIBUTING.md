# Contributing to MLIR-TensorRT

Contributions are welcome!

## Coding Standard

We will strive to follow the same coding standard as upstream MLIR, which is
described here: https://llvm.org/docs/CodingStandards.html

Python files are formatted using the [`black` formatter](https://black.readthedocs.io/en/stable/).

## Development Environment

This project provides a pre-configured CUDA 12.5 development environment using [Dev Containers](https://containers.dev/). We offer configurations for both `ubuntu` and `rockylinux8`, located in the `.devcontainer` directory.

### VS Code (Recommended)
1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open the project in VS Code.
3. When prompted, click "Reopen in Container" and select your preferred OS configuration.

VS Code will automatically build the container and connect to the development environment.

### Dev Containers CLI
If you are not using VS Code, you can manage the environment with the [Dev Containers CLI](https://github.com/devcontainers/cli).

1.  Install the CLI.
2.  Choose one of the available configurations from the `.devcontainer` directory (e.g., `cuda12.5-ubuntu-llvm17`).
3.  From the project root, build and start the container by running the `up` command. Replace `<config-name>` with your chosen configuration.
    ```bash
    devcontainer up --workspace-folder . --config .devcontainer/<config-name>/devcontainer.json
    ```
    For example:
    ```bash
    devcontainer up --workspace-folder . --config .devcontainer/cuda12.5-ubuntu-llvm17/devcontainer.json
    ```
4.  To open a shell inside the running container, use the `exec` command:
    ```bash
    devcontainer exec --workspace-folder . --config .devcontainer/<config-name>/devcontainer.json /bin/bash
    ```

## How to Submit a PR

- Fork the repo on GitHub
- Use GitHub CLI to submit PRs.

Make sure you push your branches to your fork instead of the main repo.

```bash
git remote add fork [url to your fork]
```

1. Checkout a branch ```git switch -c my-feature```
2. Do work
3. `git clang-format HEAD~1`
4. `git stage` your files and `git commit -s` with a message following guidelines
   below. Commits must contain a "sign off" message to indicate that the code you are
   contributing is your own work and contributed under our existing license.
5. `git push fork`
6. `gh pr create`


### Testing

Until we have some sort of automated externally-facing continuous testing setup, we will test PRs
using our internal infrastructure. In the meantime, developers should post their own test
results to the PR as a comment for the most basic sanity check. Just post the output of

```
ninja -C build check-mlir-tensorrt check-mlir-executor check-mlir-tensorrt-dialect
```

### Merging PRs

After review, when it is time to merge the change, simply do:

1. `gh pr merge --squash --delete-branch`

## Guidelines for PRs

Each PR is one commit. Multiple commits will be automatically squashed. We are
operating in git fast-forward-only mode, so you will need to rebase periodically
if you are on a long-lived branch.

Each commit should generally  follow a multi-line style as follows:

```
[SomeTag] A short headline

Explanation of what the commit entails. It can be as long as you like, the more
detail the better. No functional change should be made without an explanation.
```

The exception to this policy is for "Non-Functional-Changes" (NFC)
commits which are things like correcting typos and updating documentation
in that case it's fine to have something like:

```NFC: fix typo in documentation```

Functional changes and bug fixes should have an accompanying test.
Usually this will be in the form of a LIT test.

Each commit should be focused and atomic. Break your work up into small commits
as much as possible.

#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies
  that the contribution is your original work, or you have rights to submit it
  under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be
    accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
  ```