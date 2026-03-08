#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <primitive_name>"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMITIVE_NAME="$1"
TARGET_DIR="${ROOT_DIR}/primitives/${PRIMITIVE_NAME}"

mkdir -p "${TARGET_DIR}"

cat > "${TARGET_DIR}/README.md" <<EOF
# ${PRIMITIVE_NAME}

## Goal

Describe what this primitive computes and why it matters.

## Model

- FLOPs:
- Bytes:
- AI:

## Measurement

- Command:
- Dataset:
- Threads:

## Results

- AI:
- Achieved GFLOPS:
- Roofline bound:
- Efficiency:
EOF

cat > "${TARGET_DIR}/meta.json" <<EOF
{
  "primitive": "${PRIMITIVE_NAME}",
  "category": "todo",
  "status": "draft",
  "notes": ""
}
EOF

echo "created: ${TARGET_DIR}"

