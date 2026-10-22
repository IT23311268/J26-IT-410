"""Pinned scientific encoders, content-addressed safe caches and aligned mmap storage."""

import gc
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile

from ...data.manifests import atomic_json
from ...data.provenance import file_sha256
from ...pipeline import fingerprint
from ...runtime import detect_device
from ..intent_transformer import _available_system_memory_gib
from .config import EmbeddingSettings

LOGGER = logging.getLogger(__name__)


def encoder_identity(settings: EmbeddingSettings) -> dict:
    specter = settings.backend == "specter2"
    return {
        "backend": settings.backend,
        "model_id": settings.specter2_model if specter else settings.scibert_model,
        "model_revision": settings.specter2_revision if specter else settings.scibert_revision,
        "adapter_id": settings.adapter if specter else None,
        "adapter_revision": settings.adapter_revision if specter else None,
        "pooling": "cls" if specter else "masked_mean_including_special_tokens",
        "tokenizer_revision": settings.specter2_revision if specter else settings.scibert_revision,
        "max_length": settings.max_length,
        "normalization": settings.normalize,
        "dimension": settings.dimension,
        "precision": settings.precision,
        "quantization": settings.quantization,
        "preprocessing": "NFC_whitespace_title_sep_abstract_v1",
        "output_dtype": "float32",
    }


def cache_key(paper, settings: EmbeddingSettings) -> str:
    return fingerprint(
        [paper.canonical_paper_id, paper.paper_key, paper.text_sha256, encoder_identity(settings)]
    )


def validate_embeddings(
    values, keys, *, dimension: int, expected_keys=None, normalized=False
) -> dict:
    import numpy as np

    if values.ndim != 2 or values.shape != (len(keys), dimension):
        raise ValueError("embedding dimensions or row count do not match paper IDs")
    if len(keys) != len(set(keys)) or (
        expected_keys is not None and list(keys) != list(expected_keys)
    ):
        raise ValueError("embedding ID alignment or uniqueness failed")
    norms, hashes = [], set()
    duplicates = 0
    for start in range(0, len(keys), 512):
        block = np.asarray(values[start : start + 512])
        if not np.isfinite(block).all():
            raise ValueError("non-finite embedding values")
        block_norms = np.linalg.norm(block, axis=1)
        if (block_norms <= 1e-12).any():
            raise ValueError("zero embedding vector")
        if normalized and not np.allclose(block_norms, 1, atol=1e-4):
            raise ValueError("embedding normalization check failed")
        norms.extend(float(v) for v in block_norms)
        for row in block:
            import hashlib

            digest = hashlib.sha256(row.tobytes()).hexdigest()
            duplicates += digest in hashes
            hashes.add(digest)
    return {
        "rows": len(keys),
        "dimension": dimension,
        "dtype": str(values.dtype),
        "duplicate_vectors": duplicates,
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
        "norm_mean": sum(norms) / len(norms) if norms else None,
        "interpretation": (
            "Intrinsic checks only; duplicates or category similarity "
            "are not definitive quality judgments."
        ),
    }


def _quantize_cpu_encoder(model):
    """Quantize in place, avoiding a second full float32 copy of the encoder."""
    import torch

    # Adapters wraps BERT projections in LoRA-capable Linear subclasses even
    # for a bottleneck adapter. Empty LoRA containers are plain linear maps;
    # sharing their existing Parameters avoids copying the base weights.
    def unwrap_empty_lora(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear) and type(child) is not torch.nn.Linear:
                from adapters.methods.lora import LoRALinearTorch

                if not isinstance(child, LoRALinearTorch) or len(child.loras):
                    raise ValueError("dynamic int8 encoding cannot flatten active LoRA projections")
                replacement = torch.nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device="meta",
                )
                replacement.weight = child.weight
                replacement.bias = child.bias
                setattr(parent, name, replacement)
            else:
                unwrap_empty_lora(child)

    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    unwrap_empty_lora(model)
    model = torch.ao.quantization.quantize_dynamic(
        model.eval(), {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    layers = [m for m in model.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
    if not layers:
        raise ValueError("dynamic int8 encoding found no quantizable linear layers")
    estimated_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        m.in_features * m.out_features + 4 * m.out_features for m in layers
    )
    return model, {
        "method": "torch_dynamic_qint8_inplace",
        "quantized_linear_layers": len(layers),
        "original_parameter_bytes": original_bytes,
        "estimated_parameter_bytes": estimated_bytes,
        "interpretation": "Parameter storage estimate; excludes runtime and activation memory.",
    }


class ScientificEncoder:
    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        offline=True,
        model=None,
        tokenizer=None,
        cache_dir=None,
    ):
        self.settings, self.offline = settings, offline
        self.model, self.tokenizer = model, tokenizer
        self.device = None
        self.quantization_report = None
        self.cache_dir = str(cache_dir) if cache_dir is not None else None

    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        selected = detect_device(self.settings.device).device
        if self.settings.quantization != "none" and selected != "cpu":
            raise ValueError("dynamic int8 encoding requires CPU")
        if selected == "cpu" and self.settings.cpu_threads is not None:
            torch.set_num_threads(self.settings.cpu_threads)
        if self.settings.precision == "fp16" and selected != "cuda":
            raise ValueError("fp16 embedding computation requires CUDA")
        if self.model is None:
            ram = _available_system_memory_gib()
            if ram is not None and ram < self.settings.min_available_ram_gib:
                raise RuntimeError(
                    f"embedding load blocked: {ram:.2f} GiB RAM available; "
                    f"need {self.settings.min_available_ram_gib}"
                )
            identity = encoder_identity(self.settings)
            common = {
                "revision": identity["model_revision"],
                "local_files_only": self.offline,
                "trust_remote_code": False,
                "cache_dir": self.cache_dir,
            }
            self.tokenizer = AutoTokenizer.from_pretrained(identity["model_id"], **common)
            if self.settings.backend == "specter2":
                try:
                    from adapters import AutoAdapterModel
                    from huggingface_hub import snapshot_download
                except ImportError as error:
                    raise RuntimeError(
                        "SPECTER2 requires adapters==1.3.0 and the pinned base/proximity "
                        "adapter; no fallback occurred"
                    ) from error
                # Legacy official SPECTER2 weights are tensor-only loaded by
                # torch>=2.6 / adapters>=1.3. No arbitrary pickle object loading.
                if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
                    raise RuntimeError("SPECTER2 legacy checkpoint loading requires torch>=2.6")
                self.model = AutoAdapterModel.from_pretrained(
                    # The pinned release has tensor-only .bin weights. Explicitly
                    # disable Transformers' background safetensors conversion,
                    # which can otherwise fetch an unrelated conversion revision.
                    identity["model_id"],
                    weights_only=True,
                    use_safetensors=False,
                    **common,
                )
                path = snapshot_download(
                    identity["adapter_id"],
                    revision=identity["adapter_revision"],
                    local_files_only=self.offline,
                    cache_dir=self.cache_dir,
                    allow_patterns=[
                        "adapter_config.json",
                        "pytorch_adapter.bin",
                        "adapter.safetensors",
                    ],
                )
                self.model.load_adapter(
                    path, source="hf", load_as="stage05_proximity", set_active=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    identity["model_id"], use_safetensors=True, **common
                )
        self.device = torch.device(selected)
        self.model.to(self.device).eval()
        if self.settings.quantization == "int8_dynamic":
            self.model, self.quantization_report = _quantize_cpu_encoder(self.model)
            LOGGER.info("Stage 05 encoder quantization: %s", self.quantization_report)

    def encode(self, papers):
        import numpy as np
        import torch

        if self.device is None:
            self.load()
        texts = [p.title + self.tokenizer.sep_token + (p.abstract or "") for p in papers]
        lengths = [len(self.tokenizer(text, truncation=False)["input_ids"]) for text in texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        precision = self.settings.precision
        cast = (
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
            )
            if precision != "fp32"
            else nullcontext()
        )
        with torch.inference_mode(), cast:
            hidden = self.model(**inputs).last_hidden_state
            if self.settings.backend == "specter2":
                vectors = hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors = vectors.float()
            if self.settings.normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32), [
            n > self.settings.max_length for n in lengths
        ]


class EmbeddingCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def get(self, key: str):
        import numpy as np

        meta = self.root / (key + ".json")
        path = self.root / (key + ".npy")
        if not meta.exists():
            return None
        document = json.loads(meta.read_text(encoding="utf-8"))
        if (
            document.get("key") != key
            or not path.exists()
            or file_sha256(path) != document["sha256"]
        ):
            raise ValueError("embedding cache checksum mismatch")
        return np.load(path, allow_pickle=False), bool(document["truncated"])

    def put(self, key: str, vector, truncated: bool):
        import numpy as np

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (key + ".npy")
        temporary = None
        try:
            with NamedTemporaryFile(dir=self.root, suffix=".npy", delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, vector.astype("float32"), allow_pickle=False)
            os.replace(temporary, path)
            atomic_json(
                self.root / (key + ".json"),
                {"key": key, "sha256": file_sha256(path), "truncated": truncated},
            )
        finally:
            if temporary and temporary.exists():
                temporary.unlink()


def embed_corpus(
    papers,
    output: Path,
    settings: EmbeddingSettings,
    cache_root: Path,
    *,
    offline=True,
    encoder=None,
) -> dict:
    import numpy as np

    if not papers:
        raise ValueError("no eligible scholarly paper text to embed")
    if Path(output).exists():
        raise ValueError("embedding output exists; choose a new versioned run")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    owns_encoder = encoder is None
    encoder = encoder or ScientificEncoder(
        settings, offline=offline, cache_dir=cache_root / "models"
    )
    cache = EmbeddingCache(cache_root)
    matrix = np.lib.format.open_memmap(
        output, mode="w+", dtype="float32", shape=(len(papers), settings.dimension)
    )
    index, hits = [], 0
    for start in range(0, len(papers), settings.batch_size):
        batch = papers[start : start + settings.batch_size]
        cached = [
            cache.get(cache_key(p, settings)) if settings.cache_enabled else None for p in batch
        ]
        missing = [p for p, value in zip(batch, cached, strict=True) if value is None]
        encoded, truncated = encoder.encode(missing) if missing else ([], [])
        if missing and (
            np.asarray(encoded).shape != (len(missing), settings.dimension)
            or len(truncated) != len(missing)
        ):
            raise ValueError("encoder returned misaligned vectors or truncation flags")
        cursor = 0
        for offset, (paper, value) in enumerate(zip(batch, cached, strict=True)):
            key = cache_key(paper, settings)
            if value is None:
                vector, flag = encoded[cursor], truncated[cursor]
                validate_embeddings(
                    np.asarray([vector]),
                    [paper.paper_key],
                    dimension=settings.dimension,
                    normalized=settings.normalize,
                )
                if settings.cache_enabled:
                    cache.put(key, vector, flag)
                cursor += 1
            else:
                vector, flag = value
                hits += 1
                validate_embeddings(
                    np.asarray([vector]),
                    [paper.paper_key],
                    dimension=settings.dimension,
                    normalized=settings.normalize,
                )
            matrix[start + offset] = vector
            index.append(
                {
                    "row": start + offset,
                    "paper_key": paper.paper_key,
                    "canonical_paper_id": paper.canonical_paper_id,
                    "cache_key": key,
                    "text_sha256": paper.text_sha256,
                    "truncated": flag,
                }
            )
        matrix.flush()
        if start == 0 or len(index) == len(papers) or start // 100 != len(index) // 100:
            LOGGER.info(
                "Stage 05 embedded %s/%s papers (%s cache hits)", len(index), len(papers), hits
            )
    quality = validate_embeddings(
        matrix,
        [p.paper_key for p in papers],
        dimension=settings.dimension,
        expected_keys=[row["paper_key"] for row in index],
        normalized=settings.normalize,
    )
    quality.update(
        cache_hits=hits,
        truncated_records=sum(row["truncated"] for row in index),
        truncation_rate=sum(row["truncated"] for row in index) / len(index),
    )
    del matrix
    result = {
        "encoder": encoder_identity(settings),
        "index": index,
        "quality": quality,
        "quantization_report": getattr(encoder, "quantization_report", None),
    }
    if owns_encoder:
        # Release cyclic adapter/module references before UMAP/HDBSCAN are
        # imported when embedding and topic fitting share a process.
        del encoder
        gc.collect()
    return result
