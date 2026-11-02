#!/usr/bin/env bash
# Optional RAM guard: run one command and stop it (SIGTERM) if the machine's available RAM (MemAvailable) drops under
# RAM_GUARD_MIN_MB (default 3000), so other work on a shared machine keeps its memory. Polls every 2 s, at most
# RAM_GUARD_MAX_S seconds (default 21600 = 6 h; after that it stops watching and waits for the command). Linux only.
#   bash tools/ram_guard.sh <command ...>
ulimit -c 0
[ $# -ge 1 ] || { echo "usage: ram_guard.sh <command ...>"; exit 2; }
MIN=${RAM_GUARD_MIN_MB:-3000}
"$@" & CHILD=$!
for i in $(seq 1 $(( ${RAM_GUARD_MAX_S:-21600} / 2 ))); do
  kill -0 $CHILD 2>/dev/null || break
  a=$(awk '/^MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$a" -lt "$MIN" ] && { echo "$(date +%T) RAM GUARD: available $a MB < $MIN MB, stopping pid $CHILD"; kill $CHILD; }
  sleep 2
done
wait $CHILD
