# Issue tracker: GitHub

Issues and specifications live in `PhillipChaffee/glm-5.3-flash-drafter`.
Use `gh` with `--repo PhillipChaffee/glm-5.3-flash-drafter` for tracker operations.
Read the project goal in `README.md` before choosing work.

## Issue operations

- Search existing open and closed issues before creating another issue.
- Create issues with `gh issue create --title "..." --body-file <file>`.
- Read the body, labels, state, and all comments with `gh issue view`.
- Post decisions with `gh issue comment --body-file <file>`.
- Change labels with `gh issue edit --add-label` or `--remove-label`.
- Close resolved work with `gh issue close --reason completed`.
- Close dropped work with `gh issue close --reason "not planned"`.
- When a skill says to publish a specification, create a GitHub issue.

## Pull requests as a triage surface

PRs as a request surface: no.

## Wayfinding operations

Active map: [Plan our first custom GLM-5.3-Flash drafter experiment](https://github.com/PhillipChaffee/glm-5.3-flash-drafter/issues/2).

- Map: one issue labelled `wayfinder:map`. Keep its body as a decision index.
- Children: create with `gh issue create --parent <map-number>` and one
  `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or
  `wayfinder:task` label.
- Dependencies: after creation, use
  `gh issue edit <child> --add-blocked-by <blocker-numbers>`.
- Frontier: query the map's native sub-issues with
  `gh api --paginate repos/PhillipChaffee/glm-5.3-flash-drafter/issues/<map-number>/sub_issues`.
  Keep open, unassigned children whose blockers are all closed.
- Claim: assign a ticket to the driving developer when its work starts.
- Resolve: post a resolution comment, close with an explicit reason, and add
  a short named link to the map's Decisions so far section.

Read the issue titles back before wiring relationships after parallel creation.
Use native relationships as the source of truth for open work and blocking.
Refer to tickets by linked titles in human-facing explanations.

## Evidence and completion

Link research files, commits, and experiment records from issues.
Distinguish inspected source, published claims, executed CPU checks, executed
GPU checks, and historical results. Close against the question's own evidence
requirement. Performance results guide research rather than justify whether
custom training proceeds.
