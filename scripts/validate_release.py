"""Check that the release metadata is consistent across all files.

Usage::

    python scripts/validate_release.py

Checks that
* the ``version`` in ``CITATION.cff`` equals the version in ``pyproject.toml``,
* the newest entry in ``CHANGELOG.rst`` is the entry of this version and
* the ``date-released`` in ``CITATION.cff`` equals the date of this changelog entry.

Exits with a non-zero status if any check fails.
"""

import sys

from release_metadata import changelog_entries, citation_value, pyproject_version


def validate() -> list[str]:
    errors = []
    version = pyproject_version()

    cff_version = citation_value("version")
    if cff_version != version:
        errors.append(
            f"CITATION.cff version {cff_version!r} does not match the pyproject.toml version {version!r}"
        )

    entries = changelog_entries()
    if not entries:
        errors.append(
            "CHANGELOG.rst contains no entries of the form '<version> (<YYYY-MM-DD>)'"
        )
        return errors

    latest_version, latest_date = entries[0]
    if latest_version != version:
        errors.append(
            f"the newest CHANGELOG.rst entry is {latest_version!r}, expected an entry for version {version!r}"
        )
        return errors

    cff_date = citation_value("date-released")
    if cff_date != latest_date:
        errors.append(
            f"CITATION.cff date-released {cff_date!r} does not match the CHANGELOG.rst date {latest_date!r} of {version}"
        )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("release metadata is inconsistent:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"release metadata is consistent (version {pyproject_version()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
