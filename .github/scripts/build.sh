#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p .output/payload
python3 ../automation/.github/scripts/dependencies.py
uv venv .output/test --python 3.13
uv pip install --python .output/test/bin/python .output/deps/*/*.whl
make check test PYTHON="$PWD/.output/test/bin/python"
uv build
mkdir -p .output/payload/abilities
for project in abilities/*; do
 [[ -f "$project/ability.manifest.yaml" ]] || continue
 .output/test/bin/ability-scaffold pack "$project" -o ".output/payload/abilities/$(basename "$project").zip"
done
test "$(find .output/payload/abilities -name '*.zip' | wc -l)" -eq 7
