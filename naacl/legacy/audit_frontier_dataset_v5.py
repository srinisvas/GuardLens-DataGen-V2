#!/usr/bin/env python3
"""Run the canonical frontier dataset audit with the v5 protocol-chain validator."""
from __future__ import annotations

import audit_frontier_dataset as afd
from prepare_frontier_dataset_v5 import assert_expected_provenance_v5


def main() -> None:
    original = afd.assert_expected_provenance
    try:
        afd.assert_expected_provenance = assert_expected_provenance_v5
        afd.main()
    finally:
        afd.assert_expected_provenance = original


if __name__ == "__main__":
    main()
