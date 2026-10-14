"""`python -m hidrometros.check`: one PASS/FAIL line per acceptance criterion, non-zero on any FAIL."""

from __future__ import annotations

import sys

from hidrometros.criteria import CRITERIA, c12_git_clean


def audit() -> int:
    """Pre-push audit (spec 9.1): nothing forbidden is tracked, figures are plots, notebooks hold no images."""
    r = c12_git_clean()
    print(f"{'PASS' if r.ok else 'FAIL'}  pre-push audit  [{r.detail}]")
    return 0 if r.ok else 1


def main() -> int:
    if sys.argv[1:] == ["audit"]:
        return audit()
    failed = 0
    for n, (title, fn) in CRITERIA.items():
        r = fn()
        failed += not r.ok
        print(f"{'PASS' if r.ok else 'FAIL'}  {n:>2}. {title}  [{r.detail}]")
    print(f"\n{len(CRITERIA) - failed}/{len(CRITERIA)} criteria pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
