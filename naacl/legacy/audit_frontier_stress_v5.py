#!/usr/bin/env python3
"""Run the hard-benign stress audit with the v5 protocol-chain validator."""
from __future__ import annotations

import audit_frontier_stress as afs
from prepare_frontier_dataset_v5 import assert_expected_provenance_v5


def main() -> None:
    original = afs.assert_expected_provenance
    try:
        afs.assert_expected_provenance = assert_expected_provenance_v5
        afs.main()
    finally:
        afs.assert_expected_provenance = original


if __name__ == "__main__":
    main()
