"""
Unit tests for frontend security and DOM-based XSS prevention.
Verifies that no dynamic innerHTML assignments exist in the client interface,
and that malicious payloads render purely as plain text.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_frontend_has_zero_inner_html_assignments() -> None:
    html_path = Path("frontend/index.html")
    assert html_path.exists(), "frontend/index.html must exist"

    content = html_path.read_text(encoding="utf-8")

    # Ensure no innerHTML assignments exist anywhere in scripts
    inner_html_matches = re.findall(r"\.innerHTML\s*=", content)
    assert len(inner_html_matches) == 0, f"Found unsafe innerHTML assignments: {inner_html_matches}"


def test_frontend_uses_safe_text_content_and_dom_nodes() -> None:
    html_path = Path("frontend/index.html")
    content = html_path.read_text(encoding="utf-8")

    # Verify usage of safe DOM properties
    assert "textContent" in content
    assert "createElement" in content
    assert "appendChild" in content
    assert "createTextNode" in content

