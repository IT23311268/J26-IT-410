"""Cornell metadata JSONL staging. Exact raw IDs and date precision are retained."""

from datetime import date, datetime
from email.utils import parsedate_to_datetime

from ...schemas import ExternalIdentifier
from ..identifier_audit import normalize_arxiv, normalize_doi
from ..provenance import reference
from ..staging import StagedPaper

SOURCE_URL = "https://www.kaggle.com/datasets/Cornell-University/arxiv"


def adapt(event, *, version=None, synthetic=False, acquired_at=None):
    raw = event.record
    issues = []

    def text(field):
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            issues.append(f"invalid_{field}_type")
            return None
        return value

    raw_id, title, abstract = text("id"), text("title"), text("abstract")
    normalized = normalize_arxiv(raw_id)
    if raw_id is None or normalized is None:
        issues.append("missing_or_invalid_arxiv_id")
    for field, value in (("title", title), ("abstract", abstract)):
        if not value or not value.strip():
            issues.append(f"missing_{field}")
    author_text = text("authors")
    parsed_authors = raw.get("authors_parsed")
    authors = ()
    if isinstance(parsed_authors, list) and all(
        isinstance(a, list) and all(isinstance(s, str) for s in a) for a in parsed_authors
    ):
        authors = tuple(" ".join(a) for a in parsed_authors if any(a))
    elif parsed_authors is not None:
        issues.append("invalid_authors_parsed_type")
    if not authors and author_text and author_text.strip():
        authors = (author_text,)  # Do not invent author segmentation from a free-text string.
    if not authors:
        issues.append("missing_authors")
    categories_raw = text("categories")
    categories = tuple(categories_raw.split()) if categories_raw else ()
    if not categories:
        issues.append("missing_categories")
    versions_raw = raw.get("versions", [])
    versions = (
        tuple(versions_raw)
        if isinstance(versions_raw, list) and all(isinstance(v, dict) for v in versions_raw)
        else ()
    )
    if versions_raw and not versions:
        issues.append("invalid_versions_type")
    published = None
    v1 = [v.get("created") for v in versions if v.get("version") == "v1"]
    if v1:
        try:
            published = parsedate_to_datetime(v1[0]).date()
        except (ValueError, TypeError, OverflowError):
            issues.append("invalid_publication_date")
    if published is None:
        issues.append("missing_publication_date")
    update = None
    update_raw = text("update_date")
    if update_raw:
        try:
            update = date.fromisoformat(update_raw)
        except ValueError:
            issues.append("invalid_update_date")
    if published and published.year > datetime.now().year + 1:
        issues.append("future_publication_year")
    doi_raw = text("doi")
    doi = normalize_doi(doi_raw)
    if doi_raw and doi is None:
        issues.append("invalid_doi")
    identifiers = []
    if normalized:
        identifiers.append(ExternalIdentifier(namespace="arxiv", value=normalized.base))
    if doi:
        identifiers.append(ExternalIdentifier(namespace="doi", value=doi))
    result = StagedPaper(
        source_dataset="arxiv-metadata",
        source_record_id=raw_id,
        title=title,
        abstract=abstract,
        authors=authors,
        publication_year=published.year if published else None,
        publication_date=published,
        update_date=update,
        doi_raw=doi_raw,
        doi_normalized=doi,
        arxiv_id_raw=raw_id,
        arxiv_id_normalized=normalized.base if normalized else None,
        arxiv_version=normalized.version if normalized else None,
        versions=versions,
        external_ids=tuple(identifiers),
        subject_categories=categories,
        source_provenance=reference(
            "arxiv-metadata",
            SOURCE_URL,
            version,
            event,
            synthetic=synthetic,
            acquired_at=acquired_at,
        ),
        metadata_completeness={
            "title": bool(title and title.strip()),
            "abstract": bool(abstract and abstract.strip()),
            "authors": bool(authors),
            "categories": bool(categories),
            "date": published is not None,
            "doi": doi is not None,
            "arxiv_id": normalized is not None,
        },
    )
    return result, issues
