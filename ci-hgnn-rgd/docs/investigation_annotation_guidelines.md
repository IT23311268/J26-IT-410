# Independent investigation annotation — stage05-v1

This protocol labels a **paper–topic pair**, not a citation context. Topic labels
are provisional c-TF-IDF/component descriptions, so inspect the topic keywords
and representative papers before accepting the definition. Request clarification
if the topic definition is ambiguous. A cluster assignment, title match, cosine
score, rule cue, or SciCite Result prediction cannot supply the answer.

## Operational labels

**INVESTIGATES_TOPIC:** the available paper text affirmatively establishes that
the topic is a central research target: an objective, question, design, experiment,
analysis, or substantive result directly addressing it. Identify the topic-specific
research action and quote its exact supporting span.

**MENTIONS_TOPIC:** the available text discusses the topic as background, related
work, comparison, motivation, limitation, or future possibility without affirmative
evidence that the paper investigates it. Quote the contextual mention. This label
describes the available text; it is not proof that the full paper contains no
investigation.

**UNKNOWN:** available text cannot reliably distinguish the alternatives. Use it
for insufficient/title-only text, uncertain topic alignment, ambiguous aims,
conflicting evidence, or an unclear topic definition. Lack of evidence in an
abstract is not proof of absence. A supporting span is optional for UNKNOWN.

## Worked examples — synthetic illustrations only

These examples are written for the guideline, not real supervision or research
measurements. Do not import them into an empirical annotation set.

| Topic | Available text | Illustrative decision and reason |
| --- | --- | --- |
| Graph sparsification | “We evaluate graph sparsification algorithms on controlled connectivity experiments.” | INVESTIGATES_TOPIC: affirmative topic-specific evaluation and design. |
| Graph sparsification | “Prior work on graph sparsification motivates our unrelated scheduling study.” | MENTIONS_TOPIC: explicit background role. |
| Graph sparsification | “Future work could investigate graph sparsification.” | MENTIONS_TOPIC: future proposal, not this study's investigation. |
| Graph sparsification | Title: “Graphs for efficient systems.” No abstract. | UNKNOWN: insufficient topic-specific research evidence. |
| Graph sparsification | “We investigate improved runtime.” | UNKNOWN: generic investigation cue without topic alignment. |
| Any topic | A citing context has SciCite label Result. | UNKNOWN unless independent paper text establishes the relationship. |

Negation matters: “we do not investigate X” cannot establish investigation of X.
Distinguish the current authors' work from quoted or attributed prior work.
Rules offer candidates and may miss attribution/negation nuances; they are not
substitutes for this protocol. Broad lexical topics may need human refinement.

## Independent annotation and source spans

1. Freeze the topic model version, corpus, paper ID/key, original title/abstract,
   topic definition, source provenance, and temporal snapshot before annotation.
2. Generate a stratified task sample using topic, publication year (including
   unknown), category, source, completeness, and assignment-score level. The
   score bins are sampling strata, not calibrated confidence bins.
3. Assign each pair to at least two different annotators. They work independently,
   without seeing each other's labels, model guesses, or rule candidate labels.
   Operational independence must be enforced by the research process; distinct
   IDs alone cannot prove it.
4. Fill `paper_key`, nullable `canonical_paper_id`, `topic_id`, frozen
   `topic_definition`, exact `title`, nullable `abstract`, `label`, `annotator_id`,
   timezone-aware `annotation_timestamp`, `guideline_version=stage05-v1`,
   confidence in `[0,1]`, notes, and the preassigned train/dev/test split.
5. For known labels, select an exact source span. `text_field` is `title` or
   `abstract`. Offsets are zero-based Python Unicode **character** indices,
   with inclusive start and exclusive end. They index original text, not normalized
   text, tokens, UTF-8 bytes, or a title–abstract concatenation. The source slice
   must exactly equal `relevant_text_span`.
6. Human confidence is self-reported confidence, not a calibrated probability.
   Do not let a high cosine score or softmax value dictate it.

The task export is a blank template, not a completed annotation. The import file
contains one flat JSON object per completed annotator response, not the task's
empty `annotations` array. Source text/definition, IDs, guideline version, spans,
and duplicate responses are validated at import.

## Agreement, disagreement, and adjudication

Compute Cohen's kappa separately for each annotator pair on their independently
shared paper–topic pairs. Report sample size and raw agreement alongside kappa.
Constant identical labels yield undefined kappa (`null`), not a fabricated perfect
score. No overlapping independent responses means `NOT_EVALUABLE`. Do not claim
agreement from blank tasks or model labels. Synthetic test agreement is explicitly
marked and is not empirical agreement.

Unanimous labels from two or more distinct annotators can enter the final export.
Disagreements require a different adjudicator, notes, and a label supported by an
existing grounded response. If a new interpretation is required, collect the
additional grounded response and preserve the original disagreements. An
adjudication JSONL record has `paper_key`, `topic_id`, `final_label`,
`adjudicator_id`, and `notes`. Unresolved pairs remain excluded from supervised
training; their original responses remain in the annotation table.

## Supervised evaluation

Assign splits by canonical paper identity (or the preserved source-local paper
key) before model fitting. Every topic pair for a paper belongs to the same split.
The workflow rejects paper overlap, missing split assignments, inconsistent source
text, and missing independent validation. By default at least 30 validated train
pairs, all three train labels, and nonempty dev/test groups are required. This is
a software minimum, not evidence of adequate statistical power or representativity.
Maintain separate source/author/version groups where independently justified.

Train on train only, select checkpoints using dev macro F1, and run test inference
only through the explicit investigation evaluation command. Report per-class
support/precision/recall/F1 and the complete three-class confusion matrix. No
pseudo-label, rule cue, cluster membership, or citation intent can replace human
validation. With no validated annotation data, the classifier is `NOT_TRAINED`.
