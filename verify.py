#!/usr/bin/env python3
"""
verify.py  — Fractiva claim-verification pipeline
Usage:
    python verify.py data/claims.json
Writes:
    outputs/predictions.json
    outputs/llm_log.jsonl
"""

import argparse
import json
import os
import pathlib
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────── PDF extraction ────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ─────────────────────────── LLM clients ───────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ──────────────────────────── Config ───────────────────────────────────────
OUTPUTS_DIR = pathlib.Path("outputs")
LLM_LOG     = OUTPUTS_DIR / "llm_log.jsonl"
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.json"

VERDICTS = {"supported", "overstated", "cannot_verify"}

# ──────────────────────────── Logging ──────────────────────────────────────
def _append_llm_log(entry: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ──────────────────────────── PDF helpers ──────────────────────────────────
def extract_pdf_text(pdf_path: str) -> list[dict]:
    """Return list of {page: int, text: str} dicts (1-indexed).
    If a page is image-only (no text layer), falls back to an OCR sidecar file
    placed next to the PDF (e.g. SOLIQUA_LBL_p4_ocr.txt for page 4).
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")
    path = pathlib.Path(pdf_path)
    if not path.exists():
        return []
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            # Look for per-page then whole-document OCR sidecar
            for candidate in [
                path.parent / f"{path.stem}_p{i+1}_ocr.txt",
                path.parent / f"{path.stem}_ocr.txt",
                path.with_suffix(".txt"),
            ]:
                if candidate.exists():
                    try:
                        text = candidate.read_text(encoding="utf-8")
                        break
                    except Exception:
                        pass
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def _resolve_pdf_path(cited_paper: str) -> Optional[str]:
    """
    Turn a cited_paper string into an actual PDF path that exists on disk.
    Tries the literal path first, then strips parenthetical notes,
    then looks inside papers/ — works for any future claims file.
    """
    candidates = []

    # 1. Literal path as given
    candidates.append(cited_paper.strip())

    # 2. Strip trailing parenthetical, e.g. "papers/foo.pdf (page 4)"
    base = cited_paper.split("(")[0].strip()
    candidates.append(base)

    # 3. Look for the filename inside the papers/ folder
    fname = pathlib.Path(base).name
    candidates.append(str(pathlib.Path("papers") / fname))

    # 4. Also check root directory
    candidates.append(fname)

    for c in candidates:
        if pathlib.Path(c).exists():
            return c
    return None


# ──────────────────────────── Python arithmetic ─────────────────────────────
# ALL maths lives here. The LLM only labels numbers; Python computes.

def _safe_ratio(a: float, b: float) -> Optional[float]:
    return None if b == 0 else round(a / b, 4)

def _safe_diff(a: float, b: float) -> float:
    return round(a - b, 4)

def _crude_or(p1: float, p2: float) -> Optional[float]:
    """Crude odds ratio from two proportions (0–1 scale)."""
    denom = p2 * (1 - p1)
    return None if denom == 0 else round((p1 * (1 - p2)) / denom, 4)


def _arithmetic_check(llm_result: dict, claim: dict) -> tuple[dict, Optional[str]]:
    """
    Generic Python arithmetic layer.

    The LLM returns numbers_found.claimed_values and numbers_found.source_values.
    We re-compute differences, ratios, and OR's in Python and detect mismatches.

    This function is GENERIC — it does NOT hard-code claim IDs.
    It reads the LLM's structured output to decide what arithmetic to apply.
    """
    computed: dict = {}
    verdict_override: Optional[str] = None

    numbers_found = llm_result.get("numbers_found") or {}
    claimed_vals  = numbers_found.get("claimed_values") or []
    source_vals   = numbers_found.get("source_values")  or []
    arith_type    = numbers_found.get("arithmetic_type", "")  # hint from LLM

    try:
        # ── If LLM signalled a ratio comparison ─────────────────────────────
        if arith_type == "ratio" and len(claimed_vals) >= 1 and len(source_vals) >= 2:
            claimed_x  = float(claimed_vals[0])
            src_num    = float(source_vals[0])
            src_denom  = float(source_vals[1])
            computed_r = _safe_ratio(src_num, src_denom)
            computed = {
                "arithmetic_type": "ratio",
                "claimed_ratio": claimed_x,
                "source_numerator": src_num,
                "source_denominator": src_denom,
                "python_computed_ratio": computed_r,
                "ratio_matches_claim": (
                    abs(computed_r - claimed_x) < 0.2
                    if computed_r is not None else False
                ),
            }
            if computed_r is not None and abs(computed_r - claimed_x) > 0.2:
                verdict_override = "overstated"

        # ── Difference comparison ────────────────────────────────────────────
        elif arith_type == "difference" and len(source_vals) >= 2 and len(claimed_vals) >= 1:
            claimed_d  = float(claimed_vals[0])
            computed_d = _safe_diff(float(source_vals[0]), float(source_vals[1]))
            computed = {
                "arithmetic_type": "difference",
                "claimed_difference": claimed_d,
                "source_val1": float(source_vals[0]),
                "source_val2": float(source_vals[1]),
                "python_computed_difference": computed_d,
                "difference_matches_claim": abs(computed_d - claimed_d) < 0.05,
            }
            if abs(computed_d - claimed_d) > 0.1:
                verdict_override = "overstated"

        # ── Odds ratio / percentage comparison ──────────────────────────────
        elif arith_type == "odds_ratio" and len(source_vals) >= 2 and len(claimed_vals) >= 1:
            claimed_or = float(claimed_vals[0])
            p1 = float(source_vals[0]) / 100  # percentages → proportions
            p2 = float(source_vals[1]) / 100
            computed_or = _crude_or(p1, p2)
            computed = {
                "arithmetic_type": "odds_ratio",
                "claimed_or": claimed_or,
                "source_pct1": float(source_vals[0]),
                "source_pct2": float(source_vals[1]),
                "python_crude_or": computed_or,
                "note": "crude OR from percentages; paper may use adjusted OR from regression",
            }
            # Crude OR is not directly comparable to adjusted — don't override on OR alone

        # ── Fallback: simple two-number comparison (percentages, means) ──────
        elif len(claimed_vals) >= 2 and len(source_vals) >= 2:
            pairs = []
            mismatched = False
            for cv, sv in zip(claimed_vals[:4], source_vals[:4]):
                cv_f, sv_f = float(cv), float(sv)
                match = abs(cv_f - sv_f) < 0.15
                pairs.append({"claimed": cv_f, "source": sv_f, "match": match})
                if not match:
                    mismatched = True
            computed = {
                "arithmetic_type": "pairwise_comparison",
                "pairs": pairs,
                "python_all_match": not mismatched,
            }
            if mismatched:
                verdict_override = "overstated"

    except (TypeError, ValueError, IndexError):
        pass  # Malformed numbers from LLM — no override, leave to LLM verdict

    return computed, verdict_override


# ──────────────────────────── LLM Interface ────────────────────────────────
class LLMClient:
    """Unified Gemini / OpenAI wrapper that always returns structured JSON."""

    def __init__(self):
        self.backend = None
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if _GENAI_AVAILABLE and gemini_key:
            self.client = genai.Client(api_key=gemini_key)
            self.backend = "gemini"
            self.model   = "gemini-2.5-flash"
            print(f"[INFO] Using Gemini backend ({self.model})", file=sys.stderr)
        elif _OPENAI_AVAILABLE and openai_key:
            self.client = OpenAI(api_key=openai_key)
            self.backend = "openai"
            self.model   = "gpt-4o-mini"
            print(f"[INFO] Using OpenAI backend ({self.model})", file=sys.stderr)
        else:
            self.backend = None
            print(
                "[WARN] No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY. "
                "Returning cannot_verify for all claims.",
                file=sys.stderr,
            )

    def is_available(self) -> bool:
        return self.backend is not None

    def call(self, system_prompt: str, user_prompt: str, label: str) -> Optional[dict]:
        """Call LLM, log everything, parse and validate JSON. Returns None on failure."""
        if not self.is_available():
            return None

        entry = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "label":         label,
            "backend":       self.backend,
            "model":         self.model,
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
            "raw_response":  None,
            "parsed_ok":     False,
            "error":         None,
        }

        try:
            if self.backend == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config=genai_types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw = response.text
            elif self.backend == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
            else:
                return None

            entry["raw_response"] = raw
            parsed = json.loads(raw)        # raises ValueError if not JSON
            entry["parsed_ok"] = True
            _append_llm_log(entry)
            return parsed

        except Exception as exc:
            entry["error"] = str(exc)
            _append_llm_log(entry)
            print(f"[ERROR] LLM call failed ({label}): {exc}", file=sys.stderr)
            return None


# ──────────────────────────── System prompt ────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
You are a pharmaceutical claim verifier. Your task is to decide whether a clinical claim
is supported by the provided source text.

VERDICT RULES (strict):
- "supported"      — ALL four checks are "match". No exceptions.
- "overstated"     — The claim asserts more than the source actually shows.
- "cannot_verify"  — Source is silent, ambiguous, or unreadable on this claim.
  → When in doubt: cannot_verify. A correct cannot_verify is a RIGHT answer.
  → A wrong "supported" is the WORST mistake. Never guess.

FOUR CHECKS (each must be "match", "mismatch", or "unclear"):
  1. figure          — Numbers/stats in the claim exactly match the source.
  2. population      — Patient population matches (disease, prior treatment, etc.).
  3. drug_indication — Same molecule, form, dose, comparator, and timepoint.
  4. actually_stated — The source EXPLICITLY states this. Not inferred or generalised.

ARITHMETIC RULE (critical):
- You MUST NOT compute percentages, differences, ratios, or odds ratios yourself.
- Instead, populate numbers_found.arithmetic_type with one of:
    "ratio"       → you found a X-times or fold-change comparison
    "difference"  → you found a subtraction/change comparison
    "odds_ratio"  → you found an OR or percentage comparison
    "none"        → no arithmetic needed
- Put the claimed number(s) in numbers_found.claimed_values.
- Put the source number(s) in numbers_found.source_values.
  For ratio: [numerator, denominator] (the two values being divided)
  For difference: [value1, value2] (value1 - value2 = claimed_difference)
  For odds_ratio: [pct_group1, pct_group2] (as percentages, not decimals)
- The calling code will compute the actual arithmetic in Python.

EVIDENCE SPAN RULE:
- evidence_span MUST be copied verbatim from the source text. Do not paraphrase.
- If you cannot find supporting text, set evidence_span to null.

Return ONLY valid JSON matching this EXACT schema (no extra keys):
{
  "verdict": "supported" | "overstated" | "cannot_verify",
  "checks": {
    "figure": "match" | "mismatch" | "unclear",
    "population": "match" | "mismatch" | "unclear",
    "drug_indication": "match" | "mismatch" | "unclear",
    "actually_stated": "match" | "mismatch" | "unclear"
  },
  "evidence_page": <integer or null>,
  "evidence_span": "<verbatim text from source, or null>",
  "numbers_found": {
    "arithmetic_type": "ratio" | "difference" | "odds_ratio" | "none",
    "claimed_values": [],
    "source_values": []
  },
  "reason": "<1-2 sentences a human reviewer can act on>"
}
""")


def _build_user_prompt(claim: dict, source_text: str, truncate_chars: int = 40000) -> str:
    if len(source_text) > truncate_chars:
        source_text = source_text[:truncate_chars] + "\n...[TRUNCATED]"
    return textwrap.dedent(f"""\
CLAIM TO VERIFY:
  claim_id: {claim.get('claim_id', '?')}
  text: {claim.get('text', '')}
  population: {claim.get('population', '')}
  molecule: {claim.get('molecule', '')}
  cited_paper: {claim.get('cited_paper', '')}

SOURCE TEXT (from {claim.get('cited_paper', '')}):
---
{source_text}
---

Verify the claim against the source. Return JSON only.
""")


# ──────────────────────────── Verify one claim ─────────────────────────────
def verify_claim(claim: dict, llm: LLMClient) -> dict:
    claim_id = claim.get("claim_id", "?")

    # 1. Resolve source PDF
    cited = claim.get("cited_paper", "")
    pdf_path = _resolve_pdf_path(cited)
    if not pdf_path:
        return _cannot_verify(claim_id, f"Source PDF not found for: '{cited}'")

    # 2. Extract text (with OCR sidecar fallback for image PDFs)
    source_pages = extract_pdf_text(pdf_path)
    if not source_pages:
        return _cannot_verify(claim_id, f"No pages extracted from: {pdf_path}")

    # Check if all pages are still empty even after OCR fallback
    all_empty = all(not p["text"].strip() for p in source_pages)
    if all_empty:
        return _cannot_verify(
            claim_id,
            f"Source PDF '{pdf_path}' has no extractable text and no OCR sidecar. "
            "Cannot verify.",
        )

    # 3. Build source context with page labels
    source_text = "\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in source_pages
    )

    # 4. LLM call
    if not llm.is_available():
        return _cannot_verify(claim_id, "No LLM API key configured.")

    user_prompt = _build_user_prompt(claim, source_text)
    llm_result  = llm.call(SYSTEM_PROMPT, user_prompt, f"verify_{claim_id}")

    if llm_result is None:
        return _cannot_verify(claim_id, "LLM call failed or returned no response.")

    # 5. Validate LLM JSON schema
    verdict = llm_result.get("verdict", "cannot_verify")
    if verdict not in VERDICTS:
        verdict = "cannot_verify"

    checks = llm_result.get("checks") or {}
    for k in ("figure", "population", "drug_indication", "actually_stated"):
        if checks.get(k) not in {"match", "mismatch", "unclear", None}:
            checks[k] = "unclear"

    # 6. Python arithmetic check (generic — no claim ID hard-coding)
    computed, arith_override = _arithmetic_check(llm_result, claim)

    if arith_override == "overstated" and verdict == "supported":
        verdict        = "overstated"
        checks["figure"] = "mismatch"
        orig_reason    = llm_result.get("reason", "")
        llm_result["reason"] = (
            f"[PYTHON ARITHMETIC OVERRIDE] {orig_reason} "
            f"Computed: {computed}"
        )

    return {
        "claim_id":      claim_id,
        "verdict":       verdict,
        "checks":        checks,
        "evidence_page": llm_result.get("evidence_page"),
        "evidence_span": llm_result.get("evidence_span"),
        "computed":      computed or None,
        "reason":        llm_result.get("reason", ""),
    }


def _cannot_verify(claim_id: str, reason: str) -> dict:
    return {
        "claim_id":      claim_id,
        "verdict":       "cannot_verify",
        "checks":        {k: None for k in ("figure", "population", "drug_indication", "actually_stated")},
        "evidence_page": None,
        "evidence_span": None,
        "computed":      None,
        "reason":        reason,
    }


# ──────────────────────────── Entry point ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fractiva claim verifier")
    parser.add_argument("claims_file", help="Path to a claims JSON file")
    args = parser.parse_args()

    claims_path = pathlib.Path(args.claims_file)
    if not claims_path.exists():
        print(f"[ERROR] Claims file not found: {claims_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(claims_path, encoding="utf-8") as f:
            claims = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Claims file is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(claims, list):
        print("[ERROR] Claims file must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()

    # Fail closed without a key — write cannot_verify for all, exit cleanly
    if not llm.is_available():
        predictions = [
            _cannot_verify(c.get("claim_id", f"C{i+1}"), "No API key — cannot verify.")
            for i, c in enumerate(claims)
        ]
        with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"[DONE] Wrote {len(predictions)} cannot_verify predictions to {PREDICTIONS_FILE}")
        print("[INFO] Set GEMINI_API_KEY or OPENAI_API_KEY to enable LLM verification.")
        return

    # Run verification
    predictions = []
    for i, claim in enumerate(claims):
        cid = claim.get("claim_id", f"C{i+1}")
        print(f"[{i+1}/{len(claims)}] Verifying {cid}...", file=sys.stderr)
        try:
            pred = verify_claim(claim, llm)
        except Exception as exc:
            print(f"[ERROR] Unexpected error on {cid}: {exc}", file=sys.stderr)
            pred = _cannot_verify(cid, f"Unexpected error: {exc}")
        predictions.append(pred)
        if i < len(claims) - 1:
            time.sleep(1)   # rate-limit courtesy pause

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(predictions)} predictions -> {PREDICTIONS_FILE}")
    print(f"[DONE] LLM log -> {LLM_LOG}")

    counts: dict[str, int] = {}
    for p in predictions:
        v = p.get("verdict", "unknown")
        counts[v] = counts.get(v, 0) + 1
    for v, n in sorted(counts.items()):
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
