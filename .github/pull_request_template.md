Before submitting, please check the following:

- Make sure you have tests for the new code and that test passes (run `uv run pytest`)
- If applicable, add a line to the [unreleased] part of CHANGELOG.md, following [keep-a-changelog](https://keepachangelog.com/en/1.0.0/).
- Format added code by `uv run ruff`
- Type checking by `uv run mypy` and `uv run pyright`
- Make sure the checks (github actions) pass.
- Check that the docs compile without errors (run `uv run sphinx-build -W docs/source docs/build` after `uv sync --extra doc`.)

Then, please fill in below:

**Context (if applicable):**

**Description of the change:**

**Related issue:**
