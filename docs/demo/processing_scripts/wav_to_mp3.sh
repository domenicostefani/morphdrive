mkdir -p mp3
for i in *.wav; do lame -V0 "$i" "mp3/${i%.wav}.mp3"; done
