#!/usr/bin/env bash
# Records every outbound network connection of a CLEAN container build of this repository (docs/BUILD_NETWORK.md).
# Run as root on the Linux machine that runs Docker (with Colima or another VM: inside the VM). Needs docker buildx,
# tcpdump and python3.
#   sudo bash tools/build_netscan.sh OUT_DIR [docker buildx build options, e.g. --build-arg TORCH_CUDA_ARCH_LIST=8.6]
# Clean build: a new buildx builder (empty BuildKit cache, empty apt and uv cache mounts), --no-cache --pull. The
# builder and its cache are removed at the end. The image is not kept unless you add: -t NAME --load
# OUT_DIR gets build.pcap (connection starts, DNS, TLS ClientHello and HTTP request lines only), build.log, report.md.
# Not seen: traffic on port 22 (left out, so that the VM's own SSH is not recorded), and the builder's own image
# (moby/buildkit), which is pulled before the capture starts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: build_netscan.sh OUT_DIR [docker buildx build options]}; shift
BUILDER=conjure3d-netscan
[ "$(id -u)" = 0 ] || { echo "STOP: run as root (tcpdump needs it)"; exit 2; }
command -v tcpdump >/dev/null || { echo "STOP: install tcpdump"; exit 2; }
[ ! -e "$OUT/build.pcap" ] || { echo "STOP: $OUT/build.pcap exists; use a new OUT_DIR"; exit 2; }
if docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "STOP: builder $BUILDER exists (an earlier run did not finish); remove it: docker buildx rm $BUILDER"; exit 2
fi
mkdir -p "$OUT"
docker buildx create --name "$BUILDER" --driver docker-container >/dev/null
DUMP=
cleanup() {
  [ -z "$DUMP" ] || { kill -INT "$DUMP" 2>/dev/null || true; wait "$DUMP" 2>/dev/null || true; DUMP=; }
  docker buildx rm "$BUILDER" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker buildx inspect --bootstrap "$BUILDER" >/dev/null
FILTER='not port 22 and (port 53
  or (tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0)
  or (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16 and tcp[((tcp[12:1] & 0xf0) >> 2) + 5:1] = 0x01)
  or (tcp dst port 80 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420))'
tcpdump -i any -nn -s 2000 -U -w "$OUT/build.pcap" "$FILTER" 2>"$OUT/tcpdump.log" &
DUMP=$!
sleep 2
echo "CLEAN BUILD (new builder $BUILDER, --no-cache --pull, empty cache mounts) started $(date -u +%FT%TZ)"
status=0
docker buildx build --builder "$BUILDER" --no-cache --pull --progress=plain "$@" "$ROOT" >"$OUT/build.log" 2>&1 || status=$?
echo "build exit $status, ended $(date -u +%FT%TZ)"
cleanup
trap - EXIT
python3 -I "$ROOT/tools/netscan_report.py" "$OUT/build.pcap" "$OUT/build.log" >"$OUT/report.md"
echo "report: $OUT/report.md"
exit "$status"
