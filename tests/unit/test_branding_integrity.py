"""Step 11 tests: active product branding is consistently "Planwisely".

FinWise / FinSight were the product's earlier names. Any *active* surviving
reference is a branding bug, so this suite scans the active, product-facing
source-of-truth files and fails on stale branding.

Deliberate exclusions (documented so the exclusion is auditable rather than
accidental):

* ``CHANGELOG.md`` — historical record. History is not rewritten for cosmetic
  consistency; see :func:`test_changelog_is_present_but_excluded`.
* ``tests/**`` — negative fixtures legitimately assert legacy *absence*.
* ``huggingface_demo/**`` — separate delivery artifact (untouched by Step 11).
* ``fin-advisor`` occurrences (Dockerfile, render.yaml, MLflow experiment and
  registered-model names, ``frontend/config.js`` API_BASE, repo URL) are
  repository/infrastructure identifiers, NOT FinWise/FinSight product
  branding. Renaming them would break the deployed service, the model
  registry and the published repo link, so they are intentionally retained.
  The stale-brand patterns below are therefore scoped to the brand names
  themselves and never match ``fin-advisor``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "frontend"

# Active product-facing files that must never carry stale branding.
ACTIVE_FILES = (
    "frontend/index.html",
    "frontend/auth.js",
    "frontend/config.js",
    "frontend/metrics.html",
    "README.md",
    "SPEC.md",
    "ARCHITECTURE.md",
    "FRONTEND_DESIGN_SPEC.md",
    "PRIVACY_DATA_LIFECYCLE.md",
)

# Files that must positively identify the product as "Planwisely".
CORE_BRANDING_FILES = (
    "frontend/index.html",
    "frontend/auth.js",
    "frontend/config.js",
    "frontend/metrics.html",
    "README.md",
    "SPEC.md",
    "ARCHITECTURE.md",
    "FRONTEND_DESIGN_SPEC.md",
)

# Case-insensitive, tolerant of "Fin Wise" / "FinSight" spacing variants.
STALE_BRAND_PATTERNS = (
    re.compile(r"fin\s*wise", re.IGNORECASE),
    re.compile(r"fin\s*sight", re.IGNORECASE),
)


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _index() -> str:
    return _read("frontend/index.html")


# ── 1. scan surface ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("rel", ACTIVE_FILES)
def test_scan_files_exist(rel: str) -> None:
    assert (REPO / rel).is_file(), f"branding scan target missing: {rel}"


def test_changelog_is_present_but_excluded() -> None:
    """The historical file exists and is intentionally NOT scanned above."""
    assert (REPO / "CHANGELOG.md").is_file()
    assert "CHANGELOG.md" not in ACTIVE_FILES


# ── 2-3. no stale active branding ────────────────────────────────────────────


@pytest.mark.parametrize("rel", ACTIVE_FILES)
def test_no_stale_active_branding(rel: str) -> None:
    text = _read(rel)
    for pattern in STALE_BRAND_PATTERNS:
        found = pattern.search(text)
        assert found is None, f"stale branding {found.group(0)!r} found in {rel}"


def test_no_stale_branding_in_any_frontend_asset() -> None:
    """Every shipped frontend asset (JS/CSS included), not just the scan set."""
    for path in sorted(FRONTEND.glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in STALE_BRAND_PATTERNS:
            found = pattern.search(text)
            assert found is None, f"stale branding {found.group(0)!r} in {path.name}"


def test_no_stale_storage_key_prefix() -> None:
    """No browser storage key may still use the old brand prefix."""
    assert "_finwise" not in _read("frontend/auth.js").lower()


# ── 4-7. positive Planwisely branding ────────────────────────────────────────


@pytest.mark.parametrize("rel", CORE_BRANDING_FILES)
def test_planwisely_named_as_product(rel: str) -> None:
    assert "Planwisely" in _read(rel), f"'Planwisely' missing from {rel}"


def test_index_title_is_planwisely() -> None:
    assert re.search(r"<title>\s*Planwisely\s*</title>", _index())


def test_login_brand_is_planwisely() -> None:
    assert re.search(
        r'class="login-brand">\s*💰\s*Planwisely\s*</div>', _index()
    ), "login brand must read Planwisely"


def test_app_shell_brand_is_planwisely() -> None:
    assert re.search(
        r'class="sidebar-brand">\s*💰\s*Planwisely\s*</div>', _index()
    ), "app shell brand must read Planwisely"


def test_about_page_product_name_is_planwisely() -> None:
    index = _index()
    assert re.search(r'<h1 class="page-title">About Planwisely</h1>', index)
    assert "Planwisely uses a LightGBM classifier" in index


def test_frontend_config_object_is_planwisely_namespaced() -> None:
    assert "window.PLANWISELY_CONFIG" in _read("frontend/config.js")
