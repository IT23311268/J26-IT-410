# Claim-Level Hallucination Detection & Citation Verification

An academic research prototype for analyzing literature reviews at the claim level, verifying citation links, checking claim support against reference papers, and auditing bibliographies for potentially fabricated references.

> This project is developed as part of an academic research project and is intended for research and educational use.

## Overview

The system helps researchers inspect citations and factual claims contained in academic literature reviews.

It performs two main tasks:

1. **Claim-Level Hallucination Detection & Citation Verification**
2. **Bibliography Audit — Real vs Fake Reference Check**

The application uses a FastAPI backend and a React frontend.

## Main Features

### 1. Literature Review Analysis

The system can:

- Detect cited claims from literature review text
- Extract numeric citations such as `[5]`, `[5, 7]`, and `[5-7]`
- Support author-year citations
- Resolve citations against bibliography entries
- Match references with uploaded research papers
- Retrieve relevant evidence from reference papers
- Classify claim support using Natural Language Inference

Possible content-verification results include:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT_EVIDENCE`
- `NOT_AVAILABLE`

## Two-Layer Verification

The system separates **citation linkage** from **claim support**.

### Layer 1 — Citation Link Verification

Checks whether a citation used by a literature review claim can be resolved to an entry in the paper's reference list.

Example:

```text
Claim → [5]
[5] → Reference entry
Citation Link → VALID
```

### Layer 2 — Content Support Verification

If the corresponding reference paper is available, the system retrieves evidence from it and checks whether that evidence supports or contradicts the claim.

```text
Reference Found ≠ Claim Supported
```

A citation may be correctly linked to a real reference while there is not enough evidence available to verify the content of the claim.

## Bibliography Audit

The Bibliography Audit analyzes references contained in an uploaded research paper.

It:

- Detects the References/Bibliography section
- Extracts numbered bibliography entries
- Preserves multi-line references
- Checks scholarly metadata using multiple sources
- Attempts DOI and arXiv matching where available
- Uses fallback scholarly metadata searches when needed

References are classified as:

- `Real`
- `Fake`
- `Uncertain`

The `Uncertain` category is used when the available metadata is not strong enough to confidently classify a reference.

## Technology Stack

### Backend

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- Sentence-Transformers
- FAISS
- PyMuPDF
- Pydantic
- Uvicorn

### Frontend

- React
- Axios
- JavaScript
- CSS

### Models

Evidence embeddings:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Natural Language Inference:

```text
cross-encoder/nli-deberta-v3-small
```

## System Flow

```text
Literature Review / Research Paper
            │
            ▼
   Claim & Citation Extraction
            │
            ▼
    Citation Link Resolution
            │
            ▼
   Reference Paper Matching
            │
            ▼
       PDF Processing
            │
            ▼
   Sentence-Level Chunking
            │
            ▼
     MiniLM Embeddings
            │
            ▼
       FAISS Retrieval
            │
            ▼
      DeBERTa-v3 NLI
            │
            ▼
 Claim Verification Result
```

The Bibliography Audit follows a separate pipeline:

```text
Research Paper PDF
        │
        ▼
References Section Detection
        │
        ▼
Reference Extraction
        │
        ▼
Scholarly Metadata Verification
        │
        ▼
Real / Fake / Uncertain
```

## Project Structure

```text
Hallucination Detection & Citation Verification/
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   │   ├── analysis_pipeline.py
│   │   ├── claim_extractor.py
│   │   ├── embedding_service.py
│   │   ├── literature_service.py
│   │   ├── paper_service.py
│   │   ├── pdf_processor.py
│   │   ├── reference_verifier.py
│   │   └── verifier.py
│   ├── uploads/
│   └── requirements.txt
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── components/
    │   ├── services/
    │   ├── App.js
    │   └── App.css
    └── package.json
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "Hallucination Detection & Citation Verification"
```

## Backend Setup

Move to the backend directory:

```bash
cd backend
```

Create a virtual environment if one does not already exist:

```bash
python -m venv venv
```

### Windows PowerShell

Activate it using:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python main.py
```

Alternatively:

```bash
uvicorn main:app --reload
```

If the default port is unavailable:

```bash
uvicorn main:app --reload --port 8001
```

## Frontend Setup

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm start
```

The frontend normally runs on:

```text
http://localhost:3000
```

Make sure the frontend API configuration points to the port being used by the FastAPI backend.

## Basic Usage

### Literature Review Analysis

1. Start the backend.
2. Start the frontend.
3. Provide the literature review/reference content required by the interface.
4. Upload the relevant research/reference PDF files.
5. Select **Analyze Literature Review**.
6. Review the extracted claims, citations, citation-link status, content-verification status, confidence, evidence source, and source page where available.

### Bibliography Audit

1. Open the **Bibliography Audit** section.
2. Upload a research paper PDF.
3. Select **Verify References**.
4. Review each extracted reference and its verification result.

## Important Limitations

This system is a research prototype.

Results should not be interpreted as definitive proof that a claim or citation is correct or incorrect.

Possible limitations include:

- incomplete or missing scholarly metadata
- inaccessible reference papers
- PDF extraction errors
- ambiguous citations
- differences between preprints and published versions
- imperfect semantic retrieval
- Natural Language Inference model errors
- external scholarly service availability

`Uncertain` and `Insufficient Evidence` results should be manually reviewed.

The final academic judgment should remain with the researcher.

## Privacy & Security

Do not commit:

- API keys
- passwords
- access tokens
- private research documents
- personally identifiable information
- local environment files containing secrets

Recommended entries for `.gitignore` include:

```gitignore
venv/
__pycache__/
*.pyc
.env
uploads/*
!uploads/.gitkeep
node_modules/
build/
*.log
```

## Research Purpose

The project explores how claim-level verification, semantic retrieval, Natural Language Inference, citation resolution, and scholarly metadata verification can be combined into an explainable academic verification workflow.

The goal is to assist researchers by providing evidence and verification signals rather than replacing manual scholarly review.

## Disclaimer

This software is provided for academic and research purposes.

The classifications produced by the system are automated predictions based on available documents, machine-learning models, and scholarly metadata services. They may contain errors and should not be treated as authoritative academic, legal, or publication decisions.

Always verify important findings against the original scholarly sources.

## License

Add the license selected for your repository here, or keep the repository private if it is not yet intended for public distribution.
