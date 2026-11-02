#!/usr/bin/env bash
# Optional GPU guard: run one command on one GPU under an exclusive lock, so jobs that share a GPU take turns.
#   bash tools/gpu_guard.sh GPU_INDEX "<who and what>" <command ...>
# - GPU_INDEX is nvidia-smi's index. The command sees only that GPU, selected by UUID with CUDA_DEVICE_ORDER=PCI_BUS_ID
#   (without PCI order, CUDA numbers GPUs fastest-first, so a plain index can pick another card).
# - Lock: flock on $GPU_GUARD_LOCK_DIR/gpu<N>.lock (default /tmp); it is released when the command ends, also when it
#   is killed. Waits at most GPU_GUARD_WAIT_S seconds (default 3600) for the lock. The holder is in gpu<N>.lock.owner.
# - GPU_GUARD_EXPECT_POWER_W (optional): stop unless every GPU's power limit is this value (read only; limits and
#   clocks are never changed).
# Needs flock (util-linux) and nvidia-smi.
set -uo pipefail
ulimit -c 0
[ $# -ge 3 ] || { echo "usage: gpu_guard.sh GPU_INDEX \"<owner text>\" <command ...>"; exit 2; }
GPU=$1 OWNER=$2; shift 2
[[ $GPU =~ ^[0-9]+$ ]] || { echo "STOP: GPU_INDEX must be a number, not $GPU"; exit 2; }
DIR=${GPU_GUARD_LOCK_DIR:-/tmp}
LOCK=$DIR/gpu$GPU.lock
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i { print $2 }')
[ -n "$UUID" ] || { echo "STOP: no GPU with nvidia-smi index $GPU"; exit 2; }
exec 9>>"$LOCK"
echo "gpu$GPU: waiting for $LOCK (holder: $(cat "$LOCK.owner" 2>/dev/null))"
flock -w "${GPU_GUARD_WAIT_S:-3600}" 9 || { echo "STOP: $LOCK not free after ${GPU_GUARD_WAIT_S:-3600} s"; exit 3; }
echo "$OWNER $(date +%FT%T) pid $$" > "$LOCK.owner"
nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader
if [ -n "${GPU_GUARD_EXPECT_POWER_W:-}" ]; then
  nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits | awk -F, -v w="$GPU_GUARD_EXPECT_POWER_W" \
    '{ if ($2+0 < w-0.5 || $2+0 > w+0.5) bad=1 } END { exit bad }' \
    || { echo "STOP: a GPU power limit is not $GPU_GUARD_EXPECT_POWER_W W"; exit 1; }
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$UUID
"$@"
