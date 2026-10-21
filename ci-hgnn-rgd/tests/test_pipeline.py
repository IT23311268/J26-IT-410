from pathlib import Path

import pytest
from conftest import revised

from ci_hgnn_rgd.data.loaders import DatasetLoader
from ci_hgnn_rgd.pipeline import STAGES, canonical_json, prepare_run, require_stage


def test_future_stage_guards_and_loader_are_explicit():
    assert require_stage(1).implemented
    assert require_stage(2).implemented
    assert require_stage(3).implemented
    assert require_stage(4).implemented
    for number in range(5, 13):
        with pytest.raises(NotImplementedError):
            require_stage(number)
    for number in (0, 13, True):
        with pytest.raises(ValueError):
            require_stage(number)
    assert len(STAGES) == 12
    with pytest.raises(NotImplementedError):
        DatasetLoader().load(None, Path("not-an-existing-dataset"))


def test_empty_manifest_never_claims_scientific_results(config):
    manifest = prepare_run(config)
    assert all(count == 0 for count in manifest["counts"].values())
    assert manifest["scientific_stages_executed"] == []
    assert manifest["stage"] == 1
    assert "accuracy" not in canonical_json(manifest)


def test_fingerprints_stable_for_collection_order_and_sensitive_to_input(config, synthetic_corpus):
    first = prepare_run(config, synthetic_corpus, allow_synthetic=True)
    reverse = revised(synthetic_corpus, papers=tuple(reversed(synthetic_corpus.papers)))
    second = prepare_run(config, reverse, allow_synthetic=True)
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["config_sha256"] == second["config_sha256"]
    assert first["run_id"] != second["run_id"]
    paper = revised(synthetic_corpus.papers[0], title="Changed synthetic fixture title")
    changed = revised(synthetic_corpus, papers=(paper, synthetic_corpus.papers[1]))
    third = prepare_run(config, changed, allow_synthetic=True)
    assert first["corpus_sha256"] != third["corpus_sha256"]


def test_all_future_interfaces_fail_instead_of_returning_fake_results():
    from ci_hgnn_rgd.data.entity_resolution import EntityResolver
    from ci_hgnn_rgd.evaluation.metrics import evaluate_intent
    from ci_hgnn_rgd.explainability.evidence import EvidenceExplainer
    from ci_hgnn_rgd.graph.builder import GraphBuilder
    from ci_hgnn_rgd.models.baselines import BaselineRunner
    from ci_hgnn_rgd.models.hetero_gnn import HeteroGNN
    from ci_hgnn_rgd.nlp.investigation_detection import InvestigationDetector
    from ci_hgnn_rgd.nlp.topic_discovery import TopicDiscoverer
    from ci_hgnn_rgd.reporting.generator import ReportGenerator
    from ci_hgnn_rgd.scoring.citation_asymmetry import CitationAsymmetryScorer

    operations = (
        lambda: EntityResolver().resolve(()),
        lambda: TopicDiscoverer().fit((), None),
        lambda: InvestigationDetector().detect((), ()),
        lambda: GraphBuilder().build(None, None, None),
        lambda: HeteroGNN().fit(None, None),
        lambda: BaselineRunner().evaluate(None),
        lambda: CitationAsymmetryScorer().score(None, ()),
        lambda: evaluate_intent((), {}),
        lambda: EvidenceExplainer().explain(None),
        lambda: ReportGenerator().generate((), Path("not-created.md")),
    )
    for operation in operations:
        with pytest.raises(NotImplementedError):
            operation()
