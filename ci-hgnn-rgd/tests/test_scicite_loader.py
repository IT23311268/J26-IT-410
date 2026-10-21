import json

from stage02_helpers import scicite_record

from ci_hgnn_rgd.data.inspection import inspect_text
from ci_hgnn_rgd.data.sources.scicite import adapt
from ci_hgnn_rgd.data.streaming import events


def test_valid_scicite_record_preserves_text_label_and_provenance(tmp_path):
    path = tmp_path / "train.jsonl"
    raw = scicite_record(string="  SYNTHETIC text with original whitespace.  ")
    path.write_text(json.dumps(raw) + "\n")
    canonical, issues = adapt(next(events(path, "scicite")), synthetic=True)
    assert not issues
    assert canonical.citation_text == raw["string"]
    assert canonical.original_intent_label == "background"
    assert canonical.normalized_intent_label == "Background"
    assert canonical.source_split == "train"
    assert canonical.identifier_provenance.synthetic
    assert canonical.identifier_provenance.dataset_version is None


def test_unsupported_label_is_retained_without_expanding_scicite(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(scicite_record(label="limitation")))
    canonical, issues = adapt(next(events(path, "scicite")), synthetic=True)
    assert "unsupported_scicite_label" in issues
    assert canonical.original_intent_label == "limitation"
    assert canonical.normalized_intent_label is None


def test_missing_citation_ids_are_not_invented(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(scicite_record(citingPaperId=None, citedPaperId=None)))
    canonical, issues = adapt(next(events(path, "scicite")), synthetic=True)
    assert not issues
    assert canonical.citing_paper_identifier is None and canonical.cited_paper_identifier is None
    assert canonical.source_split == "dev"


def test_split_preservation_duplicate_contexts_and_overlap(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "dev", "test"):
        (source / f"{split}.jsonl").write_text(json.dumps(scicite_record(id=split)))
    result = inspect_text("scicite", source, tmp_path / "out", synthetic=True)
    assert set(result["quality"]["label_distribution_by_split"]) == {"train", "dev", "test"}
    assert result["quality"]["duplicate_contexts"] == 2
    assert result["quality"]["cross_split_exact_text_overlap_groups"] == 1
    assert result["status"] == "PARTIALLY_VALIDATED"
    assert len(list(source.iterdir())) == 3


def test_nonstandard_numeric_constant_is_flagged_but_text_is_retained(tmp_path):
    path = tmp_path / "train.jsonl"
    raw = scicite_record(citeStart=float("nan"))
    path.write_text(json.dumps(raw))
    result = inspect_text("scicite", path, tmp_path / "out", synthetic=True)
    assert result["quality"]["issues"]["nonstandard_json_constant"] == 1
    staged = json.loads(open(result["staging_path"], encoding="utf-8").readline())
    assert staged["record"]["citation_text"] == raw["string"]
    assert staged["validation_status"] == "invalid"
