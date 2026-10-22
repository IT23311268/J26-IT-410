"""Offline input, temporal and scientific encoder tests; no remote weights requested."""

from unittest.mock import patch

import numpy as np
import pytest
from stage05_helpers import POLICY, configuration, papers, source_row, stage03_fixture

from ci_hgnn_rgd.nlp.topics.config import EmbeddingSettings
from ci_hgnn_rgd.nlp.topics.data import prepare_corpus, prepare_paper
from ci_hgnn_rgd.nlp.topics.embeddings import (
    EmbeddingCache,
    ScientificEncoder,
    _quantize_cpu_encoder,
    cache_key,
    embed_corpus,
    encoder_identity,
    validate_embeddings,
)
from ci_hgnn_rgd.nlp.topics.workflow import _history_guard


def test_stage03_compatibility_and_duplicate_canonical_exclusion(tmp_path):
    config, _ = stage03_fixture(tmp_path, [source_row(0), source_row(0), source_row(1)])
    view = prepare_corpus(config, allow_synthetic_fixtures=True)
    assert len(view.papers) == 2
    assert view.profile["counts"]["duplicate_paper_identity"] == 1
    assert view.papers[0].canonical_paper_id == "synthetic:p0"


def test_real_workflow_rejects_synthetic_corpus(tmp_path):
    config, _ = stage03_fixture(tmp_path)
    assert prepare_corpus(config).papers == ()


@pytest.mark.parametrize(
    "abstract,title,mode",
    [
        (None, "fixture", "title_only"),
        ("  ", "fixture", "title_only"),
        ("abstract", "fixture", "title_abstract"),
    ],
)
def test_missing_abstract_is_never_invented(abstract, title, mode):
    row = prepare_paper(
        source_row(abstract=abstract, title=title),
        configuration_value(),
        policy=POLICY,
        cutoff=None,
    )
    assert row.original_abstract == abstract
    assert row.text_mode == mode


def configuration_value():
    from ci_hgnn_rgd.nlp.topics.config import Stage05Config

    return Stage05Config()


def test_title_only_can_be_disabled_and_empty_text_rejected(tmp_path):
    config = configuration(tmp_path, dataset={"allow_title_only": False})
    with pytest.raises(ValueError, match="title_only_disabled"):
        prepare_paper(source_row(abstract=None), config, policy=POLICY, cutoff=None)
    with pytest.raises(ValueError, match="empty_scientific"):
        prepare_paper(source_row(title="", abstract="  "), config, policy=POLICY, cutoff=None)


def test_symbols_negation_unicode_and_original_content_are_preserved():
    raw = "  A\u030angstro\u0308m \u03b1 + \u03b2 = \u03b3; H\u2082O is not NaCl.\n"
    row = prepare_paper(source_row(title=raw), configuration_value(), policy=POLICY, cutoff=None)
    assert row.original_title == raw
    assert row.title == "\u00c5ngstr\u00f6m \u03b1 + \u03b2 = \u03b3; H\u2082O is not NaCl."


def test_source_local_ids_and_missing_identifier_handling():
    row = prepare_paper(
        source_row(canonical_paper_id=None), configuration_value(), policy=POLICY, cutoff=None
    )
    assert row.canonical_paper_id is None and row.paper_key.startswith("source-local:")
    with pytest.raises(ValueError, match="missing_source_record"):
        prepare_paper(
            source_row(canonical_paper_id=None, source_record_id=None, record_key=None),
            configuration_value(),
            policy=POLICY,
            cutoff=None,
        )


def test_hash_sampling_and_order_are_independent_of_source_order(tmp_path):
    config, run = stage03_fixture(tmp_path)
    values = config.model_dump(mode="python")
    values["dataset"]["max_records"] = 3
    config = type(config).model_validate(values)
    first = prepare_corpus(config, allow_synthetic_fixtures=True)
    path = run / "papers.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    import json

    from ci_hgnn_rgd.data.provenance import file_sha256

    manifest_path = run / "integration_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_checksums"][path.name] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    second = prepare_corpus(config, allow_synthetic_fixtures=True)
    assert [p.paper_key for p in first.papers] == [p.paper_key for p in second.papers]
    assert first.manifest["corpus_sha256"] == second.manifest["corpus_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"available_at": None},
        {"publication_date": "2030-01-01"},
        {"enrichment_date": "2030-01-01"},
        {"version_date": "2030-01-01"},
    ],
)
def test_historical_cutoff_rejects_unknown_and_future_features(change):
    with pytest.raises(ValueError, match="temporally_ineligible"):
        prepare_paper(
            source_row(**change),
            configuration_value(),
            policy="STRICT_POINT_IN_TIME",
            cutoff="2021-01-01",
        )


def test_publication_proxy_does_not_claim_verified_availability():
    paper = prepare_paper(
        source_row(available_at=None),
        configuration_value(),
        policy="PUBLICATION_DATE_PROXY",
        cutoff="2021-01-01",
    )
    assert not paper.temporal_assessment["strict_historical_claim"]


@pytest.mark.parametrize(
    "changes",
    [
        {"publication_date": "invalid"},
        {"publication_year": True},
        {"title": "invalid\ufffdencoding"},
    ],
)
def test_invalid_metadata_and_unrecoverable_encoding_are_reported(changes):
    with pytest.raises(ValueError):
        prepare_paper(source_row(**changes), configuration_value(), policy=POLICY, cutoff=None)


def test_encoder_temporal_availability_is_checked_independently():
    with pytest.raises(ValueError, match="verified_model_available_at"):
        _history_guard(
            configuration_value(), {"policy": "STRICT_POINT_IN_TIME", "cutoff": "2021-01-01"}
        )


@pytest.mark.parametrize("kind", ["dimension", "count", "nan", "zero", "normalization", "order"])
def test_embedding_validation_rejects_invalid_vectors(kind):
    matrix = np.eye(3, dtype="float32")
    keys = ["a", "b", "c"]
    dimension, expected = 3, keys.copy()
    if kind == "dimension":
        dimension = 4
    elif kind == "count":
        keys = keys[:2]
    elif kind == "nan":
        matrix[0, 0] = np.nan
    elif kind == "zero":
        matrix[0] = 0
    elif kind == "normalization":
        matrix *= 2
    else:
        expected.reverse()
    with pytest.raises(ValueError):
        validate_embeddings(
            matrix, keys, dimension=dimension, expected_keys=expected, normalized=True
        )


def test_duplicate_vectors_are_reported_without_fake_quality_claim():
    report = validate_embeddings(np.ones((2, 3)), ["a", "b"], dimension=3)
    assert report["duplicate_vectors"] == 1


def test_cache_identity_changes_with_text_revision_adapter_and_normalization():
    paper = papers()[0]
    original = EmbeddingSettings()
    base = cache_key(paper, original)
    for field, value in (
        ("specter2_revision", "a" * 40),
        ("adapter_revision", "b" * 40),
        ("normalize", False),
        ("max_length", 64),
        ("quantization", "int8_dynamic"),
    ):
        settings = original.model_dump()
        settings[field] = value
        assert cache_key(paper, EmbeddingSettings.model_validate(settings)) != base
    changed = paper.model_dump()
    changed["text_sha256"] = "changed-source-text"
    assert cache_key(type(paper).model_validate(changed), original) != base


def test_incremental_cache_and_batched_mmap_alignment(tmp_path):
    settings = EmbeddingSettings.model_validate({"dimension": 3, "batch_size": 2})
    corpus = papers(count=5)
    calls = []

    class Encoder:
        def encode(self, rows):
            calls.append(len(rows))
            return np.tile([1, 0, 0], (len(rows), 1)).astype("float32"), [False] * len(rows)

    first = embed_corpus(
        corpus, tmp_path / "first.npy", settings, tmp_path / "cache", encoder=Encoder()
    )
    assert calls == [2, 2, 1]
    second = embed_corpus(
        corpus, tmp_path / "second.npy", settings, tmp_path / "cache", encoder=Encoder()
    )
    assert second["quality"]["cache_hits"] == 5 and calls == [2, 2, 1]
    assert [r["paper_key"] for r in first["index"]] == [p.paper_key for p in corpus]
    np.testing.assert_equal(np.load(tmp_path / "first.npy"), np.load(tmp_path / "second.npy"))


def test_cache_checksum_tampering_is_rejected(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("fixture-key", np.array([1, 0, 0]), False)
    (tmp_path / "fixture-key.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        cache.get("fixture-key")


@pytest.mark.parametrize(
    "backend,pooling", [("specter2", "cls"), ("scibert", "masked_mean_including_special_tokens")]
)
@pytest.mark.parametrize("quantization", ["none", "int8_dynamic"])
def test_encoder_interface_and_pooling_with_tiny_local_model(
    tmp_path, backend, pooling, quantization
):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    torch.set_num_threads(1)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nphysics\nbiology\nfixture\npaper\n", encoding="utf-8"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    model = BertModel(
        BertConfig(
            vocab_size=9,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    settings = EmbeddingSettings.model_validate(
        {
            "backend": backend,
            "dimension": 8,
            "device": "cpu",
            "max_length": 16,
            "quantization": quantization,
        }
    )
    encoder = ScientificEncoder(settings, model=model, tokenizer=tokenizer)
    vectors, truncated = encoder.encode(papers(count=2))
    assert vectors.shape == (2, 8) and vectors.dtype == np.float32
    assert len(truncated) == 2 and all(truncated)
    assert encoder_identity(settings)["pooling"] == pooling
    validate_embeddings(vectors, ["a", "b"], dimension=8, normalized=True)


def test_quantization_preserves_bottleneck_adapter_and_reduces_parameter_storage():
    import torch

    pytest.importorskip("adapters")
    from adapters import AutoAdapterModel
    from transformers import BertConfig

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = AutoAdapterModel.from_config(
        BertConfig(
            vocab_size=64,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
        )
    )
    model.add_adapter("synthetic_proximity")
    model.set_active_adapters("synthetic_proximity")
    model.eval()
    inputs = {"input_ids": torch.tensor([[2, 6, 7, 3]]), "attention_mask": torch.ones(1, 4)}
    with torch.inference_mode():
        original = model(**inputs).last_hidden_state
        quantized, report = _quantize_cpu_encoder(model)
        output = quantized(**inputs).last_hidden_state
    assert report["quantized_linear_layers"] >= 16
    assert report["estimated_parameter_bytes"] < 0.7 * report["original_parameter_bytes"]
    assert model.active_adapters == quantized.active_adapters
    assert torch.nn.functional.cosine_similarity(original, output, dim=-1).min() > 0.98
    assert torch.isfinite(output).all()


def test_quantization_does_not_discard_active_lora_weights():
    pytest.importorskip("adapters")
    from adapters import AutoAdapterModel
    from transformers import BertConfig

    model = AutoAdapterModel.from_config(
        BertConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    model.add_adapter("synthetic_lora", config="lora")
    with pytest.raises(ValueError, match="active LoRA"):
        _quantize_cpu_encoder(model)


@pytest.mark.parametrize("values", [{"device": "cuda"}, {"precision": "bf16"}])
def test_quantized_encoder_rejects_incompatible_device_or_precision(values):
    with pytest.raises(ValueError, match="dynamic int8"):
        EmbeddingSettings(quantization="int8_dynamic", **values)


def test_offline_specter_loader_uses_pinned_base_and_proximity_adapter(tmp_path):
    import sys
    from types import SimpleNamespace

    calls = []

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return cls()

        def load_adapter(self, path, **kwargs):
            calls.append((path, kwargs))

        def to(self, device):
            return self

        def eval(self):
            return self

    settings = EmbeddingSettings(device="cpu", min_available_ram_gib=0)
    with (
        patch.dict(sys.modules, {"adapters": SimpleNamespace(AutoAdapterModel=Model)}),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download,
    ):
        ScientificEncoder(settings, offline=True).load()
    assert calls[0][0] == "allenai/specter2_base"
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["revision"] == settings.specter2_revision
    assert calls[0][1]["weights_only"] is True
    assert calls[0][1]["use_safetensors"] is False
    assert download.call_args.kwargs["local_files_only"] is True
    assert download.call_args.kwargs["revision"] == settings.adapter_revision
    assert calls[1][1]["set_active"] is True
