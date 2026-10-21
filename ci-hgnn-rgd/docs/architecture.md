# Stage 01 architecture, assumptions and handoff

This document records the original Stage 01 design. Stages 02, 03, and 04 are now implemented;
see [the Stage 02 implementation and executed findings](stage02.md) for current
acquisition, staging, persistent inspection state and readiness. The checklist below
is retained as the original handoff, rather than a claim that every source is ready.
See [Stage 03 integration and temporal controls](stage03.md) for the current data layer.
See [Stage 04 citation-intent classification](stage04.md) for the added modelling
boundary, controls, artifacts, and actual baseline-only execution.

## Repository integration

Initial inspection found only `README.md` with the repository title, with no
services, dependencies, tests, schemas or datasets. The first uncommitted scaffold
was under `components/ci_hgnn_rgd`. The expanded Stage 01 specification explicitly
requested `ci-hgnn-rgd/`; that same component was moved and revised, leaving one
package rather than two competing implementations. Only the root README link
changes outside this component. No teammate code or existing shared interface was
present to replace. The existing authorized branch is `feature/-IT23239470-Gap`.

The package exposes typed Python contracts and a thin CLI. Team services
can call `load_config`, `builtin_registry`, `validate_corpus`, and `prepare_run`.
Stage 04 extends the `nlp/` boundary without changing the frozen Stage 01 graph
contracts: `CitationIntentEvidence` preserves context provenance and nullable
endpoints, while a later graph stage must decide whether a verified citation pair
can consume it. Interfaces may still need scientific review and explicit
schema-version changes before publication or external integration.

## Contracts and evidence

Pydantic models are frozen, reject unexpected fields, and use immutable tuples for
collections. Adapters must construct/revalidate models through normal constructors
or `model_validate`; `model_construct` and `model_copy(update=...)` bypass validation
and must not be used for untrusted input. JSON schemas are available through
`model_json_schema()`. No database persistence layer is implied.

`EvidenceReference` links a specific source record/version, locator, URL, source
payload checksum, retrieval time and known/unknown availability time. Optional
source text can retain the permitted supporting excerpt. Synthetic evidence must
be explicitly tagged. `validate_corpus` rejects synthetic evidence by default,
duplicate entity IDs, dangling references, inconsistent context-edge links,
conflicting reference IDs, future source observations and future model training.
Unknown publication dates and source availability fail temporal eligibility.

A well-formed URL or checksum is not verification of authenticity. Adapters must
retain permitted raw bytes and independently verify availability, versions, hashes,
record locators and source rights. `verify_checksum` compares a caller-selected
local file to an independently supplied SHA-256; it does not authenticate the origin
of that expected hash. Record-payload hashes and archive hashes can differ and must
not be silently substituted for one another.

Paper and author external identifiers retain namespaces; internal IDs are supplied
by explicit resolution decisions. Source-specific IDs and versions are not joined
by string equality. Ambiguous mappings must remain unresolved until supported by
evidence, method version and confidence. The resolver is a stub.

## Scientific separation

The SciCite taxonomy contains exactly Background, Method and Result. Raw lowercase
source labels have an explicit mapping constant, not a trained classifier. Unknown
intent uses a null label/confidence or an absent prediction. Extended taxonomies
have a separate namespace and explicit classes; they are not SciCite extensions.
Classifier predictions, provided source annotations and human ground truth must
remain distinguishable in later adapters and evaluation stores.

`PaperTopicEvidence` is independent of citation-intent predictions. It distinguishes
mentions, direct investigation and unknown, retaining source references, basis,
method/version and confidence. Models require a training cutoff. Frequency-only
investigation claims fail validation. This check does not validate the scientific
adequacy of a human or model's textual evidence; a later protocol must do that.

The graph schema requires paper/author/topic types, allows optional institution
and research-area types, and constrains directed relation endpoints. Citation
intent stays an optional, provenance-bearing edge attribute. Unknown investigation
must not create a positive `investigates_topic` edge. Zero counts preserve required
node-type declarations in empty graph manifests; an actual graph is not built.

`ResearchGapCandidate` requires attention and investigation signals, a finite score,
supporting evidence, model/scoring versions, explicit coverage counts, cutoff and
uncertainty. Score normalization and a scoring formula are deliberately undefined.
Unquantified uncertainty has no fabricated interval. Estimated intervals require
a method and ordered bounds containing the score. Candidates always remain
unvalidated hypotheses. The schema is not a source of research-gap ground truth.

## Time and leakage

All event timestamps require timezones; cutoff/source times normalize to UTC.
Publication dates are compared against the UTC cutoff date, not the host timezone.
Year-only dates must not be converted into invented exact dates. Adapters must
preserve missing dates until a defensible precision-aware policy is established.

The current observation policy is conservative: original evidence must have both
known availability and retrieval at or before the cutoff. This rejects later
retrievals even for older papers. No immutable historical snapshot is assumed.
The foundation checks the adapter's declarations; it cannot independently prove
their truth. Output creation/inference may occur after a historical observation
cutoff; future model training data are still forbidden. A training cutoff alone
does not prove that a pretrained model or feature vocabulary is uncontaminated.

Temporal windows use `[start, end)` and must be chronological and disjoint. Configured
split boundaries must be complete, ordered and no later than the run cutoff. These
helpers do not allocate papers or verify train/test entity overlap. Later stages
must freeze raw snapshots, resolve preprint/published duplicates, allocate splits,
then fit all learned preprocessing on training data only. Graph topology and
attributes must respect split-specific observation boundaries. Do not treat a
benchmark's full transductive graph as automatically safe for prospective discovery.
Gold labels, gap ground truth and expert assessments must remain separate from inputs.

## Runtime and audit boundaries

Only Pydantic and PyYAML are runtime requirements. The verified development pins
include pytest, Ruff and build tools; no ML backend is required or imported at
package import time. PyTorch is imported only by an explicitly requested optional
seed/device inspection when installed. CPU fallback is valid without a torch
installation; explicitly requesting an unavailable accelerator fails.

`set_random_seeds` can seed Python, optional legacy NumPy RNGs and optional torch
RNGs. It does not seed future independent Generator instances, retroactively set
Python hash randomization, or guarantee GPU determinism. Library/device versions,
training-data fingerprints and determinism settings must accompany future runs.

`prepare_run` validates inputs and returns an in-memory Stage 01 manifest. It
does not write artifacts, initialize global seeds or train. Record collections are
sorted by validated IDs before hashing; nested order remains significant (e.g.
author order and context order). Config hashes include resolved project paths,
so relocation changes the configuration fingerprint. Run UUID/time vary by design.
The host must retain source snapshots, source-code revision/dirty-tree snapshot,
environment pins and actual data bytes; fingerprints alone are insufficient for
reproduction. Generated artifacts from earlier local checks remain ignored and
are not scientific evidence or distribution contents.

The CLI config validator validates declared configuration and catalogue references,
not real datasets or empirical readiness. Planned datasets have null versions and
checksums. `DatasetMetadata.require_ready` blocks use without pinned version,
checksum, provenance and declared schema verification. Those declarations still
require source audits. The built-in catalogue has no ready datasets. Acquired
versions require explicit host registration; no implicit latest release is selected.

## Stage 02 readiness checklist

The following is a handoff checklist, not authorization to execute Stage 02:

- [ ] Agree on source access, storage limits and dataset-specific licence/attribution notes.
- [ ] Select explicit source releases/snapshots and retain genuine acquisition provenance.
- [ ] Independently verify archive/record checksums and availability/retrieval timestamps.
- [ ] Audit actual field names, identifier namespaces, record versions and missingness.
- [ ] Verify SciCite raw text keys/three-label mapping and retain original labels.
- [ ] Verify the exact MAG-Year/H2GB variant, node/edge schema, dates and release terms.
- [ ] Establish evidence-backed cross-source identifier/version mapping; reject ambiguous joins.
- [ ] Define temporal precision, historical-snapshot policy and train/validation/test allocation.
- [ ] Implement source-specific loaders against the existing `DatasetLoader`/`Corpus` boundary.
- [ ] Keep synthetic fixtures separate; add bounded tests against permitted real source samples.
- [ ] Add verified dependency pins only when a source adapter actually needs them.
- [ ] Design independent investigation annotations, gap ground truth and expert assessment
      protocols before any scientific scoring or performance claim.

No item above is marked complete merely because a typed interface exists. Stage 01
software verification is separate from data acquisition and scientific validation.
