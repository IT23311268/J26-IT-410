"""Official SciCite principal splits; preserve source labels, text, and missing IDs."""

from ...schemas import SCICITE_RAW_LABELS, ExternalIdentifier
from ..provenance import reference
from ..staging import StagedCitationContext

SOURCE_URL = "https://github.com/allenai/scicite"
ARCHIVE_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"


def adapt(event, *, version=None, synthetic=False, acquired_at=None):
    raw = event.record
    issues = []
    text = raw.get("string", raw.get("context"))
    if not isinstance(text, str):
        text = None
    label = raw.get("label")
    if label is not None and not isinstance(label, str):
        issues.append("invalid_label_type")
        label = None
    if not text or not text.strip():
        issues.append("missing_citation_text")
    if label is None:
        issues.append("missing_label")
    elif label not in SCICITE_RAW_LABELS:
        issues.append("unsupported_scicite_label")
    ids = []
    for field in ("citingPaperId", "citedPaperId"):
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(f"invalid_{field}_type")
            value = None
        ids.append(
            ExternalIdentifier(namespace="scicite-paper-id", value=value)
            if isinstance(value, str) and value.strip()
            else None
        )
    provenance = reference(
        "scicite", SOURCE_URL, version, event, synthetic=synthetic, acquired_at=acquired_at
    )
    record_id = raw.get("id", raw.get("_id"))
    record_id = record_id if isinstance(record_id, str) and record_id.strip() else None
    canonical = StagedCitationContext(
        source_dataset="scicite",
        source_record_id=record_id,
        citation_text=text,
        original_intent_label=label,
        normalized_intent_label=SCICITE_RAW_LABELS[label].value
        if label in SCICITE_RAW_LABELS
        else None,
        citing_paper_identifier=ids[0],
        cited_paper_identifier=ids[1],
        identifier_provenance=provenance,
        source_split=event.split,
        text_availability="present" if text and text.strip() else "missing",
        validation_status="invalid" if issues else "valid",
    )
    return canonical, issues
