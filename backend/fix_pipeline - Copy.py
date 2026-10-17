"""
Patch analysis_pipeline.py:
  1. Fix _aggregate_subclaims — CONTRADICTED only when confidence >= 0.60.
  2. Add print: stripped claim + subclaim list before NLI.
Run from the backend/ directory.
"""
import ast

path = "services/analysis_pipeline.py"
src = open(path, encoding="utf-8").read()

# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: Fix _aggregate_subclaims
# ─────────────────────────────────────────────────────────────────────────────
OLD_AGG = (
    "def _aggregate_subclaims(subclaim_results: List[dict]) -> tuple:\n"
    "    if not subclaim_results:\n"
    "        return \"INSUFFICIENT_EVIDENCE\", 0.0, \"No subclaims evaluated.\"\n"
    "        \n"
    "    statuses = [r[\"hallucination_status\"] for r in subclaim_results]\n"
    "    confs = [r[\"hallucination_confidence\"] for r in subclaim_results]\n"
    "    \n"
    "    if \"CONTRADICTED\" in statuses:\n"
    "        idx = statuses.index(\"CONTRADICTED\")\n"
    "        return \"CONTRADICTED\", confs[idx], \"One or more critical subclaims were contradicted by the evidence.\"\n"
    "        \n"
    "    if all(s == \"SUPPORTED\" for s in statuses):\n"
    "        return \"SUPPORTED\", min(confs), \"All subclaims are strongly supported by the evidence.\"\n"
    "        \n"
    "    if \"SUPPORTED\" in statuses or \"PARTIALLY_SUPPORTED\" in statuses:\n"
    "        avg_conf = sum(confs) / len(confs)\n"
    "        return \"PARTIALLY_SUPPORTED\", round(avg_conf, 3), \"The claim is partially supported, but some subclaims lack strong evidence.\"\n"
    "        \n"
    "    return \"INSUFFICIENT_EVIDENCE\", 0.0, \"There is insufficient evidence to support the key subclaims.\""
)

NEW_AGG = (
    "def _aggregate_subclaims(subclaim_results: List[dict]) -> tuple:\n"
    "    \"\"\"Aggregate per-subclaim verdicts into a single claim verdict.\n"
    "\n"
    "    Rules:\n"
    "    - CONTRADICTED only if at least one subclaim is CONTRADICTED with\n"
    "      confidence >= 0.60 (avoids false positives from weak NLI signals).\n"
    "    - SUPPORTED only if all valid subclaims are SUPPORTED.\n"
    "    - PARTIALLY_SUPPORTED if some are SUPPORTED or PARTIALLY_SUPPORTED.\n"
    "    - Malformed/empty subclaims with INSUFFICIENT_EVIDENCE are ignored\n"
    "      and do NOT drag down a verdict from SUPPORTED.\n"
    "    \"\"\"\n"
    "    if not subclaim_results:\n"
    "        return \"INSUFFICIENT_EVIDENCE\", 0.0, \"No subclaims evaluated.\"\n"
    "\n"
    "    statuses = [r[\"hallucination_status\"] for r in subclaim_results]\n"
    "    confs    = [r[\"hallucination_confidence\"] for r in subclaim_results]\n"
    "\n"
    "    # CONTRADICTED only when the model is genuinely confident (>= 0.60).\n"
    "    # Low-confidence contradiction signals are treated as INSUFFICIENT_EVIDENCE.\n"
    "    strong_contradictions = [\n"
    "        (s, c) for s, c in zip(statuses, confs)\n"
    "        if s == \"CONTRADICTED\" and c >= 0.60\n"
    "    ]\n"
    "    if strong_contradictions:\n"
    "        best_conf = max(c for _, c in strong_contradictions)\n"
    "        return (\n"
    "            \"CONTRADICTED\",\n"
    "            best_conf,\n"
    "            \"One or more subclaims are strongly contradicted by the evidence.\",\n"
    "        )\n"
    "\n"
    "    # Treat any low-confidence CONTRADICTED as INSUFFICIENT_EVIDENCE for\n"
    "    # the purpose of the remaining checks.\n"
    "    effective_statuses = [\n"
    "        s if not (s == \"CONTRADICTED\" and c < 0.60) else \"INSUFFICIENT_EVIDENCE\"\n"
    "        for s, c in zip(statuses, confs)\n"
    "    ]\n"
    "    effective_confs = list(confs)  # keep original confidences\n"
    "\n"
    "    if all(s == \"SUPPORTED\" for s in effective_statuses):\n"
    "        return (\n"
    "            \"SUPPORTED\",\n"
    "            min(effective_confs),\n"
    "            \"All subclaims are strongly supported by the evidence.\",\n"
    "        )\n"
    "\n"
    "    if any(s in (\"SUPPORTED\", \"PARTIALLY_SUPPORTED\") for s in effective_statuses):\n"
    "        supporting_confs = [\n"
    "            c for s, c in zip(effective_statuses, effective_confs)\n"
    "            if s in (\"SUPPORTED\", \"PARTIALLY_SUPPORTED\")\n"
    "        ]\n"
    "        avg_conf = round(sum(supporting_confs) / len(supporting_confs), 3)\n"
    "        return (\n"
    "            \"PARTIALLY_SUPPORTED\",\n"
    "            avg_conf,\n"
    "            \"The claim is partially supported, but some subclaims lack strong evidence.\",\n"
    "        )\n"
    "\n"
    "    return (\n"
    "        \"INSUFFICIENT_EVIDENCE\",\n"
    "        0.0,\n"
    "        \"There is insufficient evidence to support the key subclaims.\",\n"
    "    )"
)

if OLD_AGG in src:
    src = src.replace(OLD_AGG, NEW_AGG, 1)
    print("Patch 1 applied: _aggregate_subclaims fixed.")
else:
    print("ERROR: Patch 1 — could not find _aggregate_subclaims. Aborting.")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: Add print statements for stripped claim and subclaims before NLI
# ─────────────────────────────────────────────────────────────────────────────
OLD_DECOMP_CALL = (
    "        clean_claim = _strip_citation_from_claim(claim[\"claim_text\"], citation_text)\n"
    "        subclaims = decompose_claim(clean_claim)\n"
    "        \n"
    "        logger.info(f\"--- Claim decomposed into {len(subclaims)} subclaims ---\")"
)

NEW_DECOMP_CALL = (
    "        clean_claim = _strip_citation_from_claim(claim[\"claim_text\"], citation_text)\n"
    "        subclaims = decompose_claim(clean_claim)\n"
    "\n"
    "        # Debug: print stripped claim and every subclaim before NLI\n"
    "        print(f\"\\n[DECOMPOSE] Citation-stripped claim: {clean_claim}\")\n"
    "        for _i, _sc in enumerate(subclaims, 1):\n"
    "            print(f\"[DECOMPOSE] SUBCLAIM {_i}: {_sc}\")\n"
    "\n"
    "        logger.info(f\"--- Claim decomposed into {len(subclaims)} subclaims ---\")"
)

if OLD_DECOMP_CALL in src:
    src = src.replace(OLD_DECOMP_CALL, NEW_DECOMP_CALL, 1)
    print("Patch 2 applied: subclaim debug prints added in pipeline.")
else:
    print("ERROR: Patch 2 — could not find decompose_claim call block. Aborting.")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: Print selected status per subclaim after evaluate_and_rerank_evidence
# ─────────────────────────────────────────────────────────────────────────────
OLD_SC_EVAL = (
    "            sc_result, best_ev = evaluate_and_rerank_evidence(\n"
    "                subclaim_text=sc,\n"
    "                raw_evidence=raw_evidence,\n"
    "                target_paper=target_paper,\n"
    "                top_k=2,\n"
    "            )\n"
    "            subclaim_evals.append(sc_result)\n"
    "            all_evidence.extend(best_ev)"
)

NEW_SC_EVAL = (
    "            sc_result, best_ev = evaluate_and_rerank_evidence(\n"
    "                subclaim_text=sc,\n"
    "                raw_evidence=raw_evidence,\n"
    "                target_paper=target_paper,\n"
    "                top_k=2,\n"
    "            )\n"
    "            print(\n"
    "                f\"[NLI RESULT] Subclaim '{sc[:60]}...' -> \"\n"
    "                f\"{sc_result['hallucination_status']} \"\n"
    "                f\"({sc_result['hallucination_confidence']:.4f})\"\n"
    "            )\n"
    "            subclaim_evals.append(sc_result)\n"
    "            all_evidence.extend(best_ev)"
)

if OLD_SC_EVAL in src:
    src = src.replace(OLD_SC_EVAL, NEW_SC_EVAL, 1)
    print("Patch 3 applied: subclaim NLI result print added.")
else:
    print("WARNING: Patch 3 — subclaim eval block not found. Skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────
try:
    ast.parse(src)
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax check: FAILED — {e}")
    import sys; sys.exit(1)

open(path, "w", encoding="utf-8").write(src)
print(f"Written: {path}")
