set -e
mkdir -p dc_blocked
for i in *.mp3; do sox "$i" "dc_blocked/dcb_${i}"; done
# for i in *.wav; do sox "$i" "dc_blocked/dcb_${i}"; done
