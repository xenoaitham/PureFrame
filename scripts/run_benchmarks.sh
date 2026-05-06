#!/usr/bin/env bash
set -e
INPUT=/tmp/pureframe_bench/bench_30s_1080p.mp4
OUT=/tmp/pureframe_bench/results.txt
> "$OUT"

for PROFILE in HIGH MEDIUM LOW CPU; do
    echo "=== Profile: $PROFILE ===" | tee -a "$OUT"
    rm -rf ~/.cache/pureframe 2>/dev/null || true
    START=$(date +%s.%N)
    pureframe process "$INPUT" \
        --profile "$PROFILE" \
        --output /tmp/pureframe_bench/out_$PROFILE.mp4 2>&1 | tail -10 | tee -a "$OUT"
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "Profile $PROFILE: ${ELAPSED}s on 30s clip" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done

cat "$OUT"
