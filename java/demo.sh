#!/usr/bin/env bash
# Run the end-to-end demo (run `bash build.sh` first); any cwd.
set -euo pipefail
cd "$(dirname "$0")"

java -Dstdout.encoding=UTF-8 -cp out/main com.quant.statarb.Demo "$@"
