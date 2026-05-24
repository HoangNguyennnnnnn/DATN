#!/usr/bin/env bash
echo "DEPRECATED: Phase B removed. Use: bash scripts/train_imf_v8.sh"
exec "$(dirname "$0")/train_imf_v8.sh" "$@"
