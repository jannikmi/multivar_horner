"""Shared helpers for reading and writing the release metadata of this package.

The release version is defined in ``pyproject.toml`` and mirrored in ``CITATION.cff``
(``version`` and ``date-released``). The release date is taken from ``CHANGELOG.rst``.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.rst"

# top level keys only (no indentation), so nested keys of "preferred-citation" are not matched
CFF_KEY_PATTERN = r'^{key}:[ \t]*"?([^"\n]*?)"?[ \t]*$'
PYPROJECT_VERSION_PATTERN = re.compile(r'^(version\s*=\s*)"[^"]*"', re.MULTILINE)
# e.g. "3.2.1 (2026-09-17)" followed by an underline
CHANGELOG_ENTRY_PATTERN = re.compile(
    r"^(\d+\.\d+\.\d+\S*)\s+\((\d{4}-\d{2}-\d{2})\)\s*\n[_=\-~^]{3,}", re.MULTILINE
)
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9.+-]*)$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def pyproject_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["tool"]["poetry"]["version"]


def _cff_pattern(key: str) -> re.Pattern:
    return re.compile(CFF_KEY_PATTERN.format(key=re.escape(key)), re.MULTILINE)


def citation_value(key: str) -> str | None:
    match = _cff_pattern(key).search(CITATION.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def changelog_entries() -> list[tuple[str, str]]:
    """:return: (version, date) of all changelog entries, newest first"""
    return CHANGELOG_ENTRY_PATTERN.findall(CHANGELOG.read_text(encoding="utf-8"))


def set_pyproject_version(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new_text, count = PYPROJECT_VERSION_PATTERN.subn(
        rf'\g<1>"{version}"', text, count=1
    )
    if count != 1:
        raise ValueError(f"no version entry found in {PYPROJECT}")
    PYPROJECT.write_text(new_text, encoding="utf-8")


def set_citation_value(key: str, value: str) -> None:
    text = CITATION.read_text(encoding="utf-8")
    new_text, count = _cff_pattern(key).subn(f'{key}: "{value}"', text, count=1)
    if count != 1:
        raise ValueError(f"no top level '{key}' entry found in {CITATION}")
    CITATION.write_text(new_text, encoding="utf-8")
