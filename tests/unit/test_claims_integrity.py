"""Step 9 tests: claims / metrics integrity of ACTIVE public-facing files.

Scans the active source-of-truth documents and frontend copy for
unsupported or stale claims. Historical records are deliberately OUT of
scope (they legitimately document what was once claimed/corrected):

  * CHANGELOG.md  — historical change record
  * ROADMAP.md    — contains deliberate strikethrough corrections of the
                    old unverified figures
  * models/       — static training artifacts (JSON metric values)
  * src/ (evaluation/training code) — computes metrics, makes no claims

In-scope scan set (must be clean of prohibited claims):

  README.md, SPEC.md, ARCHITECTURE.md, FRONTEND_DESIGN_SPEC.md,
  PRIVACY_DATA_LIFECYCLE.md, frontend/index.html, frontend/metrics.html

Also asserts that verified metrics appear only with synthetic-labelled
context in the places they are displayed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCAN_FILES = [
    "README.md",
    "SPEC.md",
    "ARCHITECTURE.md",
    "FRONTEND_DESIGN_SPEC.md",
    "PRIVACY_DATA_LIFECYCLE.md",
    "frontend/index.html",
    "frontend/metrics.html",
]

# Prohibited stale / unsupported claims (regex, case-insensitive).
PROHIBITED = [
    (r"(?<![\d.])99\.96", "stale 99.96% classifier accuracy"),
    (r"(?<![\d.])92 ?%", "stale 92% accuracy claim"),
    (r"(?<![\d.])80\.9(?![\d])", "stale 80.9% budget-acceptance claim"),
    (r"bank[ -]grade", "unsupported bank-grade security claim"),
    (r"military[ -]grade", "unsupported military-grade claim"),
    (r"SOC ?2", "unsupported SOC 2 claim"),
    (r"ISO ?27001", "unsupported ISO 27001 claim"),
    (r"\bPCI\b", "unsupported PCI compliance claim"),
    (r"industry[ -]leading", "marketing puffery"),
    (r"best[ -]in[ -]class", "marketing puffery"),
    (r"zero competitors", "marketing puffery"),
    (r"trusted by", "fake social-proof claim"),
    (r"AI Accuracy", "stale marketing badge phrasing"),
    (
        r"every\s+(decision|recommendation)[^.]{0,80}\bSHAP\b",
        "SHAP-on-every-recommendation overclaim",
    ),
    (
        r"personaliz(ed|ed)?\s+forecast\w*\s+(is\s+)?(live|complete|production)",
        "personalized forecasting presented as complete",
    ),
]


def _texts() -> dict[str, str]:
    return {name: (REPO / name).read_text(encoding="utf-8") for name in SCAN_FILES}


def test_scan_files_exist():
    for name in SCAN_FILES:
        assert (REPO / name).is_file(), f"missing scan target: {name}"


@pytest.mark.parametrize("name", SCAN_FILES)
def test_no_prohibited_claims(name):
    text = _texts()[name]
    for pattern, label in PROHIBITED:
        hits = re.findall(pattern, text, flags=re.IGNORECASE)
        assert not hits, f"{name}: prohibited claim ({label}): {pattern!r} -> {hits}"


def test_index_metrics_are_verified_and_synthetic_labelled():
    text = _texts()["frontend/index.html"]
    # Verified synthetic-labelled values present…
    assert "89.10%" in text
    assert "6.10%" in text
    assert "78.86%" in text
    # …and explicitly labelled as synthetic evaluation.
    assert "synthetic-labelled evaluation data" in text
    assert "not\n        production or real-customer performance" in text


def test_metrics_page_carries_synthetic_banner():
    text = _texts()["frontend/metrics.html"]
    assert "SYNTHETIC-DATASET EVALUATION" in text
    assert "not real-world production performance" in text


def test_spec_classifier_table_is_synthetic_labelled():
    spec = _texts()["SPEC.md"]
    for metric in ("89.10%", "85.56%", "89.11%", "98.07%", "7.81%", "4.56%"):
        assert metric in spec
    assert "Synthetic evaluation" in spec
    assert "NOT production customer metrics" in spec


def test_readme_forecast_claims_are_honest():
    """STEP 13 replaced the global artifact with user-specific forecasting; the
    docs must describe what actually exists and never pass a synthetic
    benchmark off as production accuracy."""
    readme = _texts()["README.md"]
    # The personalized path is documented…
    assert "own persisted transaction history" in readme
    assert "limited_history" in readme and "insufficient_history" in readme
    # …the legacy artifact is still named as global/static…
    assert "global/static" in readme
    assert "NOT genuinely personalized" in readme
    assert "not_personalized" in readme
    # …and synthetic benchmark numbers carry their caveat.
    assert "synthetic" in readme.lower()
    assert "not customer or production accuracy" in readme
    assert "not comparable" in readme


def test_no_fake_customers_or_production_status():
    for name, text in _texts().items():
        low = text.lower()
        for phrase in ("paying users", "our customers", "revenue:", "bank integration is live"):
            assert phrase not in low, f"{name}: unsupported production claim {phrase!r}"
