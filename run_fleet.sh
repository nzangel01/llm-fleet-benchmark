#!/usr/bin/env bash
set -euo pipefail

# Ollama Fleet Benchmark Wrapper
# Usage: ./run_fleet.sh [model] [concurrency_levels]
# Default: qwen3:8b  "1 4 8 15 30"

MODEL="${1:-qwen3:8b}"
LEVELS="${2:-1 4 8 15 30}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  uv venv "$VENV"
  source "$VENV/bin/activate"
  uv pip install -r "$SCRIPT_DIR/requirements.txt"
else
  source "$VENV/bin/activate"
fi

echo "=========================================="
echo "  Ollama Fleet Benchmark"
echo "  Model: $MODEL"
echo "  Concurrency levels: $LEVELS"
echo "=========================================="

# Run benchmark across fleet
python3 "$SCRIPT_DIR/fleet_bench.py" \
  --models "$MODEL" \
  --endpoints \
    "http://192.168.1.5:11434" \
    "http://localhost:11434" \
  --endpoint-labels \
    kokkoro \
    silvia \
  --concurrency-levels $LEVELS \
  --requests 10 \
  --prompt "Explain quantum entanglement in one sentence." \
  --warmup 2 \
  --out-json "/tmp/fleet_bench_${MODEL//[:\/]/_}.json" \
  $([ -t 1 ] && echo "--tui")
