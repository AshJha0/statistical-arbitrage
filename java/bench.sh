#!/usr/bin/env bash
# Latency/throughput harness (not JMH). Run `bash build.sh` first. Optional arg: data dir.
set -euo pipefail
cd "$(dirname "$0")"
java -cp out/main com.quant.statarb.Bench "$@"
