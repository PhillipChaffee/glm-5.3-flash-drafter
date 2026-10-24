# Project tracking and Liquid AI relevance

Reviewed 2026-10-24. Tracking guidance was checked against the installed
Ask Matt, Wayfinder, and setup skills. Two `openai/gpt-6.1-sol` researchers
inspected the Liquid source groups. This document recommends a workflow and
reference material; it does not configure a tracker or adopt dependencies.

## Recommendation

Use GitHub Issues for work and decisions, versioned repository documents for
evidence and technical contracts, and a small Wayfinder map for the uncertain
route to training and studying our own drafter. The repository already lives on
GitHub. No current requirement justifies adding GitLab as a separate tracker.

The user clarified the goal on 2026-10-24: this is a fun research project.
Custom training is the destination. The map resolves how to conduct the work,
not whether performance gains justify doing it.

Liquid4All is Liquid AI's GitHub organization. Its most useful contributions
here are measurement and run-record patterns, cache-state correctness examples,
and an optional structured-output evaluator. None of the inspected components
removes the need to qualify the exact Flash checkpoint and serving runtime.

## 1. One home for each kind of information

| Information | Canonical home | Rule |
| --- | --- | --- |
| Current question, owner, blockers, next action | GitHub issue | Close against that issue's stated completion criterion. |
| Route through unresolved decisions | One Wayfinder map issue | Keep an index and short decision pointers; query open children. |
| Sources, investigations, competing findings | `docs/research/` | Date findings, pin sources, and mark superseded conclusions. |
| Adopted target/data/feature semantics | `docs/target-contract.md` | Distinguish adopted contracts from candidates and historical behavior. |
| Durable architectural choices | Short ADRs when warranted | Link the resolving issue; avoid duplicating every research note. |
| Experiment identity and result | One record under `docs/experiments/<run-id>/` | Preserve resolved configuration, raw-result locations, checksums, and result limits. |
| Large model/data/cache artifacts | Modal or Hugging Face | Reference immutable revisions or artifact digests from the run record. |

The issue resolution is the decision record for the question it answers. An ADR
captures an enduring adopted design when needed, rather than copying the whole
discussion. A map indexes those records instead of becoming another report.

At inspection, the new GitHub repository had Issues enabled and no issues. The
recent salvage, inventory, and feasibility reports were local and uncommitted.
Persist the notes in Git before relying on them across sessions.

## 2. Use Wayfinder for decisions, then ordinary implementation tickets

The installed Wayfinder skill is explicitly planning-first. Its map is a
`wayfinder:map` parent issue; child questions use `research`, `prototype`,
`grilling`, or decision-unblocking `task` types. Once the route is clear, the
installed flow hands off to a specification and implementation tickets.

Recommended map destination:

> Define an executable plan for building, training, and studying our own
> DSpark-style GLM-5.3-Flash drafter through reproducible experiments.

Start with a small decision sequence:

1. Choose the first drafter recipe and define its feature and output contracts.
2. Choose the initial data, reasoning/history policy, and compute setup.
3. Select reusable code and artifacts, with the repairs each one needs.
4. Define the checks and small training run that establish an interpretable starting point.
5. Choose the first comparisons that can explain the drafter's behavior.

Use public checkpoints as optional comparison points. Keep correctness checks
so that results answer the experiment's question. Treat performance gains and
negative results as findings, without an adoption gate for the project.

Existing research answers parts of these questions. Link it rather than
automatically commissioning another literature review. Use native GitHub
[sub-issues][sub-issues] and [blocking relationships][dependencies] to express
the remaining dependencies. Reference old issues selectively; do not import
all 48 into the active plan.

Run `/setup-matt-pocock-skills` before formally invoking this flow. Its installed
setup records the tracker and domain-document conventions in `docs/agents/`.
When decisions become concrete buildable work, use `/to-spec`, `/to-tickets`,
and `/implement` rather than extending the map indefinitely.

### Make evidence strength visible

Every resolution/result should distinguish:

- Inspected source or published claims.
- Executed local CPU checks.
- Reproduced target GPU/runtime results.
- Historical measurements with recoverable artifacts.
- Unverified claims or missing artifacts.

An issue asking whether a public checkpoint exists can close on metadata and
source inspection. That closure does not satisfy a separate issue requiring
correct state recovery or measured speedup.

For an experiment, record the question, fixed conditions, comparator, planned
pass/fail interpretation, actual execution, artifact links, failures, result,
and remaining uncertainty. Record failed runs as evidence instead of silently
dropping them from summaries.

## 3. Liquid AI: what is useful

### Pipette: measurement and experiment-record patterns

Best immediate use: borrow the design, without adopting its service stack.

The inspected client measures repeated runs, preserves samples and dispersion,
keeps preparation outside timing, validates output token counts and finite
measurements, and stops a cell when a repetition is invalid. Its management
layer separately records model/runtime identity, generation configuration,
resolved harness settings, harness version, and individual completions/failures.
It normalizes and hashes descriptors/configuration.
[Measurement code][pipette-measure] · [Record contracts][pipette-storage] ·
[Canonicalization][pipette-json]

These patterns directly address our earlier difficulties reconstructing run
settings and interpreting aggregate numbers. For this project, a small
versioned manifest and result files are sufficient initially.

Pipette has Linux vLLM/SGLang launchers, but its inspected server latency path
owns server lifecycle, uses nonstreaming `/v1/completions`, and requires
`/tokenize` plus `ignore_eos`. It is not an arbitrary-endpoint streaming
first-token/inter-token or load-testing harness. Server dispatch rejects some
prefill/decode/VL-throughput benchmark types available elsewhere in its catalog.
[Dispatch][pipette-dispatch] · [Latency implementation][pipette-latency]

Its scoring service is a useful interface reference: prompts are retrieved by
ID, completions are submitted for scoring, and duplicate/unknown IDs are
rejected. Its inspected evaluations are IFBench, IFStruct, GPQA-Diamond, and
MATH-500, not coding execution. Five repetitions are a smoke protocol, not a
tail-latency study. [Scoring API][pipette-scores]

### Hybrid cache handling: useful invariants, not a GLM rollback library

Liquid's SGLang diffusion extension snapshots and restores pre-block
short-convolution state, gates KV writes during tentative work, and reconstructs
state at a partial commit. Its correction forward restores the previous state
before encoding corrected tokens. These are relevant examples of preventing
rejected suffixes from contaminating subsequent execution.
[LFM cache implementation][lfm-cache]

The implementation is specific to LFM projections and its width-three
convolution. Flash also has KDA recurrent matrices and sparse-attention indexer
state; this is not a reusable GLM rollback implementation.

The repository also contains genuine DSpark verifier/parity-test machinery,
separate from its native diffusion algorithm. The inspected DSpark lineage
exists in upstream SGLang, so prefer the canonical upstream subsystem over
adopting this entire fork. The inspected DSpark integration test targets Qwen3,
not proof of this exact Flash integration.
[DSpark verifier][liquid-dspark] · [Parity tests][liquid-parity] ·
[Upstream lineage][sg-lineage]

### IFStruct: optional output-format smoke tests

IFStruct provides a frozen 2,000-prompt JSON/YAML dataset and
`validate_response(...) -> ValidationResult`. Its validator can score saved
target-only and speculative outputs under one controlled generation harness.
[Dataset][ifstruct-data] · [Validator][ifstruct-validator]

It measures unconstrained structural compliance, not correctness of code,
reasoning, or executed tool calls. The validator strips thinking tags and
implements a schema subset; frontier models may approach its ceiling.

Use it as a narrow later regression suite, not the main success benchmark. Its
CLI does not forward reasoning-effort controls or tools; it is nonstreaming and
its latency accounting excludes retries and assigns zero to exhausted failures.
Its average latency is unsuitable for our speedup decision.
[Client][ifstruct-client] · [CLI/artifacts][ifstruct-eval]

### Broader coding evaluation: canonical OpenBench

Liquid's inspected OpenBench fork has no fork-only commits; canonical
`groq/openbench` is ten commits ahead at the compared revisions. Its upstream
vLLM provider accepts an existing endpoint and tool parameters; HumanEval adds
executed coding checks. There is no demonstrated reason to choose Liquid's fork
for this project. [Exact comparison][openbench-compare] ·
[Provider][openbench-provider] · [HumanEval][openbench-humaneval]

HumanEval is a small-function benchmark, not a complete proxy for multi-turn
max-effort repository coding. Tool-use tasks can require simulator/grader
models, so adopt only the evaluations needed for the initial comparison.

## 4. What does not replace our training/serving plan

`lfm-diffusion` implements native DUO block diffusion for adapted Liquid models.
Its sampler requires model-specific noise levels, block positions, and
self-conditioning. The inspected reference path recomputes committed context
with caching disabled. Its README describes training/fine-tuning integration as
future work, not a released GLM drafter-training recipe. [Sampler][lfm-sampler]

The related DuoBlock verifier checks proposals against the same adapted LFM's
causal head. Its positive-threshold mode explicitly does not preserve the
autoregressive distribution; its documentation says diffusion readout logits
are not the proposal density required for speculative rejection sampling.
It cannot replace the frozen-GLM DSpark verifier unchanged.
[DuoBlock algorithm][duoblock]

The serving documentation has useful benchmark controls: tuned baselines,
back-to-back runs, fixed workload shape, CPU-allocation accounting, and
multiple machines. Its reported speedups concern a 350M LFM setup, not Flash.
Some denoising settings are slower than autoregressive decoding; overhead must
be measured rather than inferred from parallel token generation.
[Serving protocol][lfm-serving]

`liquid-finetune` provides generic causal-LM/SFT loading and training. An
arbitrary-HF-ID loader is not demonstrated GLM5Next support; the inspected
paths do not provide DSpark target-feature distillation or block-training
objectives. Adopting its orchestration would add integration work before
supplying the missing contract. `lfm-inference` is deployment-example material.
[Fine-tuning loader][finetune-loader] · [Inference repository][lfm-inference]

## 5. Source pins and licenses

These are inspected revisions, not compatibility-tested dependency pins.
Code licensing is separate from model weights, benchmark datasets, and
third-party code bundled in a project.

| Repository | Revision | Code license |
| --- | --- | --- |
| `Liquid4All/lfm-diffusion` | `3fa72a115f56f527915929dc8310442ee0ac7b73` | Apache-2.0 |
| `Liquid4All/sglang-diffusion-lfm` | `5a4acc883a72552ac9661cbc385005de9d64385d` | Apache-2.0 |
| `Liquid4All/liquid-finetune` | `d9cc23dae5a16498ab7dc11958b80ca803e0440a` | MIT |
| `Liquid4All/lfm-inference` | `93f6ebbab8e432a9aa74e4ff12dcc7a2c4645576` | MIT |
| `Liquid4All/pipette-clients` | `d93700a707815f3d73c57a0c1dc1a25aac5eb04f` | Apache-2.0 |
| `Liquid4All/pipette-mgmt` | `798be574c59e15a94fb8ab48053c480eee55d635` | Apache-2.0 |
| `Liquid4All/pipette-scores` | `d93e2e1007ebe73ddbdde5a19816c9acbaf9938e` | Apache-2.0 original work; third-party notices |
| `Liquid4All/ifstruct` | `73e03a6d3a8508357ba430675a106d4887abedf3` | Apache-2.0 |
| `Liquid4All/openbench` | `6a080a2f59a02500a63a6f24919b6b03cdb22081` | MIT |
| `groq/openbench` | `d03922ccbbd9004e802c556ef9c2a40ed870659c` | MIT |

The diffusion model's linked weight/config/license endpoints returned HTTP 401
to the researcher; its exact artifact terms were not established. LFM model
licenses must be checked separately rather than inferred from Apache/MIT code.
IFStruct's inspected dataset is Apache-2.0; other evaluation datasets carry
their own terms. Pipette-scores pins an older IFStruct validator revision than
the standalone repository inspected here.

## 6. Minimal next step

Persist the existing notes, configure GitHub as the skill tracker, and chart
the small decision map. For each upcoming pilot run, create a resolved manifest
covering target/drafter/runtime/tokenizer revisions, data digest, rendering and
reasoning/history policy, sampling, hardware, precision, parallelism, context
lengths, speculation settings, and harness version.

Record raw outputs, failures, acceptance/verification counters, and timing
samples. Measure realistic coding separately from fixed-token microbenchmarks.
Borrow Pipette's discipline now; defer new evaluation dependencies until a
specific validation question requires them.

## Sources and verification boundary

Tracker behavior is documented by GitHub and the installed skills. Liquid
conclusions come from read-only source inspection at the revisions above;
negative support claims are scoped to those inspected components. No Liquid
models, GPU kernels, live endpoints, or hardware benchmarks were executed.

[sub-issues]: https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues
[dependencies]: https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies
[pipette-measure]: https://github.com/Liquid4All/pipette-clients/blob/d93700a707815f3d73c57a0c1dc1a25aac5eb04f/crates/pipette-ops/src/measurement.rs
[pipette-storage]: https://github.com/Liquid4All/pipette-mgmt/blob/798be574c59e15a94fb8ab48053c480eee55d635/docs/storage.md
[pipette-json]: https://github.com/Liquid4All/pipette-mgmt/blob/798be574c59e15a94fb8ab48053c480eee55d635/src/canonical_json.rs
[pipette-dispatch]: https://github.com/Liquid4All/pipette-clients/blob/d93700a707815f3d73c57a0c1dc1a25aac5eb04f/crates/pipette-torch-oai/src/execute/mod.rs#L289-L316
[pipette-latency]: https://github.com/Liquid4All/pipette-clients/blob/d93700a707815f3d73c57a0c1dc1a25aac5eb04f/crates/pipette-torch-oai/src/execute/end_to_end_latency.rs
[pipette-scores]: https://github.com/Liquid4All/pipette-scores/blob/d93e2e1007ebe73ddbdde5a19816c9acbaf9938e/packages/pipette-scores/pipette_scores/api/app.py
[lfm-cache]: https://github.com/Liquid4All/sglang-diffusion-lfm/blob/5a4acc883a72552ac9661cbc385005de9d64385d/python/sglang/srt/models/lfm2_diffusion.py#L75-L228
[liquid-dspark]: https://github.com/Liquid4All/sglang-diffusion-lfm/blob/5a4acc883a72552ac9661cbc385005de9d64385d/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L697-L757
[liquid-parity]: https://github.com/Liquid4All/sglang-diffusion-lfm/blob/5a4acc883a72552ac9661cbc385005de9d64385d/test/registered/spec/dspark/test_dspark_kernel_parity.py
[sg-lineage]: https://github.com/sgl-project/sglang/commit/733c05c887da9aad4b2e6c46292588ba5110a8e0
[ifstruct-data]: https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0/blob/2e342ca5f2673fb1287411fdbadcdfde82b35682/README.md
[ifstruct-validator]: https://github.com/Liquid4All/ifstruct/blob/73e03a6d3a8508357ba430675a106d4887abedf3/ifstruct/validator.py
[ifstruct-client]: https://github.com/Liquid4All/ifstruct/blob/73e03a6d3a8508357ba430675a106d4887abedf3/ifstruct/client.py
[ifstruct-eval]: https://github.com/Liquid4All/ifstruct/blob/73e03a6d3a8508357ba430675a106d4887abedf3/ifstruct/eval.py
[openbench-compare]: https://github.com/groq/openbench/compare/6a080a2f59a02500a63a6f24919b6b03cdb22081...d03922ccbbd9004e802c556ef9c2a40ed870659c
[openbench-provider]: https://github.com/groq/openbench/blob/d03922ccbbd9004e802c556ef9c2a40ed870659c/src/openbench/model/_providers/vllm.py
[openbench-humaneval]: https://github.com/groq/openbench/blob/d03922ccbbd9004e802c556ef9c2a40ed870659c/src/openbench/evals/humaneval.py
[lfm-sampler]: https://github.com/Liquid4All/lfm-diffusion/blob/3fa72a115f56f527915929dc8310442ee0ac7b73/lfm_diffusion/sampler.py#L154-L243
[duoblock]: https://github.com/Liquid4All/sglang-diffusion-lfm/blob/5a4acc883a72552ac9661cbc385005de9d64385d/python/sglang/srt/dllm/algorithm/duo_block.py
[lfm-serving]: https://github.com/Liquid4All/lfm-diffusion/blob/3fa72a115f56f527915929dc8310442ee0ac7b73/docs/serving.md
[finetune-loader]: https://github.com/Liquid4All/liquid-finetune/blob/d9cc23dae5a16498ab7dc11958b80ca803e0440a/src/liquid_finetune/checkpointing/model_loading.py#L228-L300
[lfm-inference]: https://github.com/Liquid4All/lfm-inference/tree/93f6ebbab8e432a9aa74e4ff12dcc7a2c4645576
