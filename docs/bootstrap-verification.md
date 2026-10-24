# Bootstrap verification

## Scope

The bootstrap uses Python 3.12.13 and the hash-pinned development tools in
`requirements-lock.txt`. It includes 18 frozen reference files and the DeepSpec
submodule at the revisions recorded in `../UPSTREAMS.toml`.

The tests verify reference integrity, missing/modified/malformed inputs, CLI
exit behavior, and the snapshot's target identity/dimensions. They do not load
model tensors or claim runtime compatibility.

All 21 local PR gates passed after the project/reference dependency boundary
was made explicit. This includes the advisory/license gates for the bootstrap
environment, working-tree secret scanning, link checks, and workflow checks.

## Coverage-gate proof

On 2026-10-23:

- Full suite: 9 tests passed; branch-inclusive coverage 100%.
- `pytest -k 'not test_reports_modified_reference'`: all 8 selected tests passed,
  but coverage fell to 94.29% and pytest exited 1 because the 95% gate failed.
- The test was deselected for the proof, not deleted or disabled in configuration.

## Canonical baseline adaptations

The baseline revision and template location are recorded in `../UPSTREAMS.toml`.
Project-specific changes rename package tokens, place workflows under
`.github/workflows/`, and scope first-party checks to owned code. Snapshots and
the submodule retain upstream formatting. The import-layer contract covers the
CLI above the reference-verification module.

The CI jobs use distinct concurrency groups so the test and hygiene jobs cannot
cancel each other. The local workflow checks use this repository's workflow
directory. All coverage, complexity, duplication, and mutation thresholds retain
their canonical values.

The reference repository's own `scripts/verify-sync.sh` fails at its inspected
revision: `.markdownlint-cli2.jsonc` differs between the root and each language's
template copy. This is pre-existing upstream drift. This repository starts from
the pinned Python templates and records its local adaptations explicitly.

## Upstream dependency audit

An initial recursive OSV 2.5.1 scan also included the uninstalled DeepSpec
reference requirements. On 2026-10-23 it reported 105 known vulnerabilities
across 8 package versions, plus license-classification/allowlist failures.
Examples include torch 2.9.1 and pillow 9.5.0. These are not dependencies of
the bootstrap environment.

The project dependency gates exclude the `upstream` reference tree and continue
to scan project-owned dependency manifests and lockfiles recursively. There are
no vulnerability suppressions. Reproduce the separate reference audit with:

```sh
osv-scanner scan -r upstream/deepspec
```

Before running training, create a compatible project-owned GPU environment,
resolve its advisory/license issues, and commit its audited lockfile. The
reference requirements are provenance, not an approved installation recipe.

## Execution boundary

GPU loading, feature/logit parity, recurrent-state recovery, training, latency,
and the scheduled mutation workflow have not been executed in this bootstrap.
