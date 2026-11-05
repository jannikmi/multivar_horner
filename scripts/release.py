"""Set the version of a new release in all files which contain it.

Usage::

    python scripts/release.py 3.2.2 [--date 2026-11-05]

Updates the version in ``pyproject.toml`` and the ``version`` and ``date-released``
entries in ``CITATION.cff``. The release date defaults to today.
Add the corresponding entry ("<version> (<date>)") to ``CHANGELOG.rst`` manually
and check the result with ``python scripts/validate_release.py``.
"""

import argparse
import datetime
import sys

from release_metadata import (
    DATE_PATTERN,
    VERSION_PATTERN,
    set_citation_value,
    set_pyproject_version,
)
from validate_release import validate


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="the version of the new release, e.g. 3.2.2")
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="the release date in ISO format (YYYY-MM-DD), defaults to today",
    )
    args = parser.parse_args()

    if not VERSION_PATTERN.match(args.version):
        parser.error(f"invalid version: {args.version!r}")
    if not DATE_PATTERN.match(args.date):
        parser.error(f"invalid date (expected YYYY-MM-DD): {args.date!r}")
    datetime.date.fromisoformat(args.date)

    set_pyproject_version(args.version)
    set_citation_value("version", args.version)
    set_citation_value("date-released", args.date)
    print(
        f"set version {args.version} (released {args.date}) in pyproject.toml and CITATION.cff"
    )

    errors = validate()
    if errors:
        print("\nremaining release inconsistencies:")
        for error in errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
