# How to contribute

Thanks for taking the time to contribute!

From opening a bug report to creating a pull request: every contribution is
appreciated and welcome. If you're planning to implement a new feature or change
the API please create an [issue first](https://github.com/FOR-sight-ai/interpreto/issues/new). This way we can ensure that your precious
work is not in vain.

## Setup and dependency installation

- Clone the repo `git clone https://github.com/FOR-sight-ai/interpreto.git`.
- Go to your freshly downloaded repo `cd interpreto`
- Create a virtual environment and install the necessary dependencies for development.

We use [`uv`](https://github.com/astral-sh/uv) to manage Interpreto dependencies.
If you dont have `uv`, you should install with `make uv-download`.

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run:

```bash
make install # Regular dependencies for normal usage

or

make install-dev # Dev dependencies including docs and linting
```

To activate your `.venv` run `source .venv/bin/activate`.

Welcome to the team!

## Codestyle

After installation you may execute code formatting.

```bash
make lint
```

Many checks are configured for this project. Command `make lint` will check style with `ruff`.
We use Google style for docstrings.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Run `make fix-style` to format your changes.
5. Run `make lint` to ensure that formats are okay.
6. Write a [good commit message](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

## Other help

You can contribute by spreading a word about this library. It would also be a huge contribution to write a
short article on how you are using this project. You can also share your best practices with us.
