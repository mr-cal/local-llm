# Agents

## Overview

`mr-cal/local-llm` is a Python CLA that manages a local llama.cpp server and clients
on the same local network for connecting to the server.

This tool sets up clients directly on a host or within a LXD VM. It also has
benchmarking and model management for the server.

## Development

local-llm uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Running tests

```bash
make test           # Full test suite
uv run pytest tests/unit/path/to/test_file.py::test_name  # run a specific test
```

### Formatting and linting

```bash
make format
make lint
```

## Practices

- Make the smallest safe change necessary to resolve the issue. Avoid unrelated bug
  fixes, opportunistic cleanup, and refactoring unless required. The right amount of
  complexity is the minimum needed for the current task.
- Never speculate about code you haven't inspected.
- Follow the project's existing conventions regarding style, docstrings, logging,
  comments, and testing.
- Comments should explain complex business logic, non-obvious algorithms, regex, and
  other "gotchas". Comments should brief, explain "why" not "how", and be helpful for
  future maintainers.
- Update relevant documentation and release notes to reflect code changes.

## Processes

- If you're contributing to a specific release, target the upstream
  `hotfix/<major.minor>` branch, if it exists. Otherwise, target the `main` branch.
- Commit headers are no more than 80 characters, follow [Conventional
  Commits](https://www.conventionalcommits.org/en/v1.0.0/), and use the following types:
    - ci, build, feat, fix, perf, refactor, style, test, docs, chore
- Always run `make format`, `make lint`, and `make test` before completing your
  work.
