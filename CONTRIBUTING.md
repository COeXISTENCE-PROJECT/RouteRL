# Contributing to RouteRL

Thank you for contributing. Please keep changes focused and explain their purpose clearly.

## Branch workflow

- Create your branch from the latest `dev` branch.
- Open pull requests against `dev`, not `main`.
- Maintainers periodically merge `dev` into `main` for releases.
- Do not open a pull request against `main` unless a maintainer asks you to.

```bash
git switch dev
git pull
git switch -c short-description
```

## Local setup

Install [SUMO](https://sumo.dlr.de/docs/Installing/index.html), then create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make sure `SUMO_HOME` is set for your installation.

## Making changes

- Follow the style of the surrounding code.
- Keep unrelated changes out of the same pull request.
- Add or update tests when behavior changes.
- Update documentation or tutorials when public behavior changes.
- Do not commit generated simulation records, plots, caches, or environment files unless they are required reference files.

## Pull requests

In the pull request, describe what changed, why it was needed, and how it was tested. If you did not run a relevant test, state that clearly and explain why.

Please follow the repository's [Code of Conduct](CODE_OF_CONDUCT.md).
