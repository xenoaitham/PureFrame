#!/usr/bin/env bash
# Generates a 30-second test clip for end-to-end verification.
set -e
mkdir -p /tmp/pureframe_smoke
cd /tmp/pureframe_smoke

ffmpeg -y -f lavfi -i color=c=blue:size=1280x720:duration=10:rate=24 \
       -f lavfi -i sine=frequency=440:duration=10 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac scene1.mp4 2>/dev/null

ffmpeg -y -f lavfi -i color=c=red:size=1280x720:duration=10:rate=24 \
       -f lavfi -i sine=frequency=880:duration=10 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac scene2.mp4 2>/dev/null

ffmpeg -y -f lavfi -i color=c=green:size=1280x720:duration=10:rate=24 \
       -f lavfi -i sine=frequency=1320:duration=10 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac scene3.mp4 2>/dev/null

printf "file 'scene1.mp4'\nfile 'scene2.mp4'\nfile 'scene3.mp4'\n" > list.txt
ffmpeg -y -f concat -safe 0 -i list.txt -c copy smoke_test.mp4 2>/dev/null
rm scene1.mp4 scene2.mp4 scene3.mp4 list.txt

echo "Created /tmp/pureframe_smoke/smoke_test.mp4"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 smoke_test.mp4
