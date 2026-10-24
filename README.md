# GLM-5.3-Flash drafter

Build and evaluate a DSpark-style speculative drafter for
`zai-org/GLM-5.3-Flash`. The first milestone is a correct target-feature and
rejection-recovery integration.

## Start here

- [Target contract](docs/target-contract.md): verified architecture facts and
  integration boundaries.
- [Source manifest](UPSTREAMS.toml): exact upstream revisions and reading paths.
- [Setup research](docs/research/repository-setup.md): rationale and source links.

`upstream/deepspec` is the complete pinned algorithm checkout.
`reference/glm-5.3-flash` holds model metadata and tokenizer assets.
`reference/transformers` holds selected target/framework source files.
Default searches exclude these upstream trees to keep agent context focused;
inspect them with explicit paths and `rg --no-ignore`.

## Local development

Use Python 3.12+ from a Git worktree:

```sh
git submodule update --init upstream/deepspec
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m glm_flash_drafter
pytest
./run-gates.sh
```

The checksum command verifies every frozen upstream reference. To reconstruct
the snapshots, fetch each allowlisted file at its manifest revision, then check
it against `reference/SHA256SUMS`; Hugging Face downloads must use resolved URLs
so tokenizer assets are not replaced by Git LFS pointers.

The lockfile installs bootstrap tools, not PyTorch, Transformers, or a training
environment. DeepSpec's dependency pins and the Flash implementation still need
runtime reconciliation. Keep model weights, feature caches, data, and checkpoints
on the compute host or in an external cache.

## Checks

The setup comes from the canonical
[Python strict baseline](https://github.com/PhillipChaffee/67-sus-95-clean/tree/main/python),
pinned in `UPSTREAMS.toml`. The PR checks enforce lint/docstrings, formatting,
strict types, branch-inclusive coverage of at least 95%, cognitive complexity,
file length, and the baseline hygiene/dependency gates. Install the non-Python
tool versions listed in `.github/workflows/ci.yml` before using the full runner.

Third-party snapshots retain upstream formatting and are excluded from
first-party code/prose checks; their contents are checksum-verified. Secret
scanning still covers the working tree. Mutation testing runs nightly with an
85% score floor, as in the canonical baseline.

Dependency gates scan project-owned manifests/locks. DeepSpec's uninstalled
reference requirements have known advisory/license failures; audit them with
`osv-scanner scan -r upstream/deepspec`. A project-owned, audited GPU lockfile is
required before training. See [bootstrap verification](docs/bootstrap-verification.md)
for results and baseline adaptations.

## Current milestone

Choose the serving backend and compute environment, then prove feature/logit
alignment and correct continuation after speculative rejection. A tiny cache
and one training step come before large-scale data generation. See the target
contract for the acceptance criteria.
