"""
Patch verifier.py with corrected decompose_claim and _decide_status fallback.
Run from the backend/ directory.
"""
import re

path = "services/verifier.py"
src = open(path, encoding="utf-8").read()

# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: Replace decompose_claim
# ─────────────────────────────────────────────────────────────────────────────

OLD_DECOMPOSE_START = "def decompose_claim(\n    claim_text: str,\n) -> List[str]:"
OLD_DECOMPOSE_END   = "\n    return [claim_text]\n"

start_idx = src.find(OLD_DECOMPOSE_START)
assert start_idx != -1, "Could not find decompose_claim start"

# Find the end of the function (the last `return [claim_text]` followed by blank lines)
end_idx = src.find(OLD_DECOMPOSE_END, start_idx)
assert end_idx != -1, "Could not find decompose_claim end sentinel"
end_idx += len(OLD_DECOMPOSE_END)   # include the sentinel itself

NEW_DECOMPOSE = '''def decompose_claim(
    claim_text: str,
) -> List[str]:
    """
    Split a complex claim into simpler, self-contained atomic claims.

    Pattern handled:
        "X [verb] Y that is Z"
        -> Subclaim 1: "X [verb] Y."
        -> Subclaim 2: "Y is Z."

    The noun phrase for Subclaim 2 is taken as the words immediately before
    "that/which" in the sentence.  Walking backwards from the end of the main
    clause, we stop at recognisable verb-phrase boundary tokens so that we
    capture the actual referent noun phrase (e.g. "hallucinated content")
    rather than generic last-two-words.

    If a reliable decomposition cannot be produced the original full claim is
    returned as a single subclaim so that NLI still runs against something
    meaningful.
    """
    if not claim_text:
        return []

    claim_text = claim_text.strip()

    if len(claim_text) < 10:
        return [claim_text]

    # ----------------------------------------------------------------
    # Pattern: "... [noun phrase] that/which [copula/aux] ..."
    # e.g. "Deep learning systems can produce hallucinated content
    #        that is not supported by the source information."
    # ----------------------------------------------------------------
    rel_match = re.search(
        r"^(.*?)\\s+(that|which)\\s+"
        r"(is|are|can be|could be|has|have|was|were|will be)"
        r"\\s+(.*)$",
        claim_text,
        flags=re.IGNORECASE,
    )

    if rel_match:
        main_clause = rel_match.group(1).strip()   # "X can produce Y"
        verb        = rel_match.group(3).strip()   # "is"
        rest        = rel_match.group(4).strip()   # "not supported by Z."

        # Walk backwards through main_clause words and stop at verb-like tokens
        # to isolate the noun phrase that "that" refers to.
        _VERB_STOPS = {
            "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "do", "does", "did",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had",
            "produce", "produces", "produced",
            "generate", "generates", "generated",
            "contain", "contains", "contained",
            "include", "includes", "included",
            "create", "creates", "created",
            "make", "makes", "made",
        }
        main_words = main_clause.split()
        noun_words = []
        for w in reversed(main_words):
            if w.lower().rstrip(".,;") in _VERB_STOPS:
                break
            noun_words.insert(0, w)

        # Fallback: last 2 words if extraction fails
        if not noun_words:
            noun_words = main_words[-2:] if len(main_words) >= 2 else main_words

        noun_phrase = " ".join(noun_words)

        # Build subclaims
        sc1 = main_clause.rstrip(".").strip() + "."
        rest_clean = rest.rstrip(".").strip()
        sc2 = f"{noun_phrase} {verb} {rest_clean}."

        subclaims = []
        if noun_phrase and len(sc1.split()) > 3:
            subclaims.append(sc1)
        if noun_phrase and len(sc2.split()) > 3:
            subclaims.append(sc2)

        if len(subclaims) == 2:
            return subclaims
        # If only 1 or 0 valid subclaims, fall through to conjunction check or full claim.

    # ----------------------------------------------------------------
    # Generic conjunction splitting.
    # ----------------------------------------------------------------
    parts = re.split(
        r",\\s+and\\s+|\\s+and\\s+",
        claim_text,
        flags=re.IGNORECASE,
    )

    valid_parts = [
        part.strip()
        for part in parts
        if len(part.strip().split()) > 3
    ]

    if len(valid_parts) > 1:
        return valid_parts

    # Conservative fallback.
    return [claim_text]
'''

src = src[:start_idx] + NEW_DECOMPOSE + src[end_idx:]
print("Patch 1 applied: decompose_claim replaced.")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: Fix _decide_status fallback — remove spurious CONTRADICTED branch
# ─────────────────────────────────────────────────────────────────────────────
# The old fallback picked "CONTRADICTED" if contradiction was the max even at
# ~0.33.  Replace the entire fallback block with INSUFFICIENT_EVIDENCE always.

OLD_FALLBACK = (
    "    # Fallback:\n"
    "    # select the strongest probability instead of returning 0%.\n"
    "    strongest = max(\n"
    "        entailment,\n"
    "        contradiction,\n"
    "        neutral,\n"
    "    )\n"
    "\n"
    "    if strongest == contradiction:\n"
    "        return (\n"
    "            \"CONTRADICTED\",\n"
    "            round(\n"
    "                contradiction,\n"
    "                4,\n"
    "            ),\n"
    "            (\n"
    "                \"The strongest NLI signal \"\n"
    "                \"indicates contradiction.\"\n"
    "            ),\n"
    "        )\n"
    "\n"
    "    if strongest == entailment:\n"
    "        return (\n"
    "            \"PARTIALLY_SUPPORTED\",\n"
    "            round(\n"
    "                entailment,\n"
    "                4,\n"
    "            ),\n"
    "            (\n"
    "                \"The strongest available signal \"\n"
    "                \"indicates partial support.\"\n"
    "            ),\n"
    "        )\n"
    "\n"
    "    return (\n"
    "        \"INSUFFICIENT_EVIDENCE\",\n"
    "        round(\n"
    "            neutral,\n"
    "            4,\n"
    "        ),\n"
    "        (\n"
    "            \"The strongest available signal \"\n"
    "            \"is neutral.\"\n"
    "        ),\n"
    "    )"
)

NEW_FALLBACK = (
    "    # Fallback: none of the thresholds were reached.\n"
    "    # Use neutral probability as confidence — never fabricate CONTRADICTED\n"
    "    # at low confidence values.\n"
    "    return (\n"
    "        \"INSUFFICIENT_EVIDENCE\",\n"
    "        round(neutral, 4),\n"
    "        (\n"
    "            \"The retrieved evidence does not provide \"\n"
    "            \"enough information to verify the claim.\"\n"
    "        ),\n"
    "    )"
)

if OLD_FALLBACK in src:
    src = src.replace(OLD_FALLBACK, NEW_FALLBACK, 1)
    print("Patch 2 applied: _decide_status fallback fixed.")
else:
    print("ERROR: Patch 2 — could not find old fallback text. Aborting.")
    import sys; sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: Add debug prints inside _run_nli (evidence page + premise + hyp + logits)
# The NLI debug prints already exist from the previous session's verifier.py.
# This version of the file (1039 lines) has logger.info only, so we add prints.
# ─────────────────────────────────────────────────────────────────────────────
# Find the logger.info block after the probability extraction and add prints.
OLD_NLI_LOG = (
    "        logger.info(\n"
    "            \"NLI | %s | \"\n"
    "            \"contradiction=%.4f \"\n"
    "            \"entailment=%.4f \"\n"
    "            \"neutral=%.4f\",\n"
    "            evidence_page or \"unknown\",\n"
    "            contradiction,\n"
    "            entailment,\n"
    "            neutral,\n"
    "        )\n"
    "\n"
    "        return {\n"
    "            \"entailment\": entailment,\n"
    "            \"contradiction\": contradiction,\n"
    "            \"neutral\": neutral,\n"
    "        }"
)

NEW_NLI_LOG = (
    "        logger.info(\n"
    "            \"NLI | %s | \"\n"
    "            \"contradiction=%.4f \"\n"
    "            \"entailment=%.4f \"\n"
    "            \"neutral=%.4f\",\n"
    "            evidence_page or \"unknown\",\n"
    "            contradiction,\n"
    "            entailment,\n"
    "            neutral,\n"
    "        )\n"
    "\n"
    "        # Debug print (requirement 9)\n"
    "        print(\"\\n[NLI DEBUG]\")\n"
    "        print(f\"  Evidence page : {evidence_page or '(unknown)'}\")\n"
    "        print(f\"  Premise       : {premise[:120]}{'...' if len(premise) > 120 else ''}\")\n"
    "        print(f\"  Hypothesis    : {hypothesis}\")\n"
    "        print(f\"  Raw logits    : [{logits[0].item():.4f}, {logits[1].item():.4f}, {logits[2].item():.4f}]\")\n"
    "        print(f\"  Contradiction : {contradiction:.4f}\")\n"
    "        print(f\"  Entailment    : {entailment:.4f}\")\n"
    "        print(f\"  Neutral       : {neutral:.4f}\")\n"
    "        # Selected status is printed by the caller\n"
    "\n"
    "        return {\n"
    "            \"entailment\": entailment,\n"
    "            \"contradiction\": contradiction,\n"
    "            \"neutral\": neutral,\n"
    "        }"
)

if OLD_NLI_LOG in src:
    src = src.replace(OLD_NLI_LOG, NEW_NLI_LOG, 1)
    print("Patch 3 applied: NLI debug prints added.")
else:
    print("WARNING: Patch 3 — NLI debug log block not found. Skipping (may already be present).")

# ─────────────────────────────────────────────────────────────────────────────
# Write patched file
# ─────────────────────────────────────────────────────────────────────────────
import ast
try:
    ast.parse(src)
    print("Syntax check: PASSED")
except SyntaxError as e:
    print(f"Syntax check: FAILED — {e}")
    import sys; sys.exit(1)

open(path, "w", encoding="utf-8").write(src)
print(f"Written: {path}")
