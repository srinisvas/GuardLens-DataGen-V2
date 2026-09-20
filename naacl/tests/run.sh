#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m compileall -q naacl
python -m unittest discover -s naacl/tests -p 'test_*.py' -v
for script in \
  naacl/launch_*.slurm \
  naacl/launch_stage.sh \
  naacl/rebuild_semantic_turn_repair_candidate.sh
do
  bash -n "$script"
done
