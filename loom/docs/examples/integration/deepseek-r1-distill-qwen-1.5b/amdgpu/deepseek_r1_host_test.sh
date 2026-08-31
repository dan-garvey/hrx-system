#!/usr/bin/env bash
set -euo pipefail

binary="$1"
output="$("$binary" --host-self-test)"
grep -Fq "queue_reservation=host_mocked" <<<"$output"
grep -Fq "host self-test: ok" <<<"$output"
