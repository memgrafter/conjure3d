#!/usr/bin/env bash
# Records every outbound network connection of a CLEAN container build of this repository (docs/BUILD_NETWORK.md).
# Run on the Linux machine that runs Docker (with Colima or another VM: inside the VM), as a user who can use Docker.
# Needs docker buildx and python3; root is not needed.
#   bash tools/build_netscan.sh OUT_DIR [docker buildx build options, e.g. --build-arg TORCH_CUDA_ARCH_LIST=8.6]
# Clean build: a new buildx builder (empty BuildKit cache, empty apt and uv cache mounts), --no-cache --pull. The
# builder and its cache are removed at the end. The image is not kept unless you add: -t NAME --load
# tcpdump runs in a small container (Debian, pinned by digest) in the builder's network namespace, so it records only
# the build's traffic, and nothing of the host. BUILDER_OPTS gives options to `docker buildx create`, for example
# BUILDER_OPTS="--driver-opt cpuset-cpus=2-5 --driver-opt cpu-shares=2" to keep cores free for other work.
# OUT_DIR gets build.pcap (connection starts, DNS, TLS ClientHello and HTTP request lines only), build.log, report.md.
# Not seen: the builder's own image (moby/buildkit) and the tcpdump image, which are pulled before the capture starts.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: build_netscan.sh OUT_DIR [docker buildx build options]}; shift
BUILDER=conjure3d-netscan
DUMP_IMAGE=conjure3d-netscan-tcpdump
DUMP_BASE=debian:trixie-slim@sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4
[ ! -e "$OUT/build.pcap" ] || { echo "STOP: $OUT/build.pcap exists; use a new OUT_DIR"; exit 2; }
if docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  echo "STOP: builder $BUILDER exists (an earlier run did not finish); remove it: docker buildx rm conjure3d-netscan"; exit 2
fi
mkdir -p "$OUT"
printf 'FROM %s\nRUN apt-get update && apt-get install -y --no-install-recommends tcpdump\n' \
  "$DUMP_BASE" | docker build -q -t "$DUMP_IMAGE" - >/dev/null
# shellcheck disable=SC2086
docker buildx create --name "$BUILDER" --driver docker-container ${BUILDER_OPTS:-} >/dev/null
DUMP=
cleanup() {
  if [ -n "$DUMP" ]; then
    docker kill -s INT "$BUILDER-tcpdump" >/dev/null 2>&1 || true; wait "$DUMP" 2>/dev/null || true; DUMP=
  fi
  docker buildx rm conjure3d-netscan >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker buildx inspect --bootstrap "$BUILDER" >/dev/null
FILTER='port 53
  or (tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0)
  or (tcp[((tcp[12:1] & 0xf0) >> 2):1] = 0x16 and tcp[((tcp[12:1] & 0xf0) >> 2) + 5:1] = 0x01)
  or (tcp dst port 80 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420)'
# the pcap goes to stdout, so the file belongs to the user who runs this script
docker run --rm --name "$BUILDER-tcpdump" --network "container:buildx_buildkit_${BUILDER}0" \
  --cap-add NET_RAW --cap-add NET_ADMIN "$DUMP_IMAGE" \
  tcpdump -i any -nn -s 2000 -U -Z root -w - "$FILTER" >"$OUT/build.pcap" 2>"$OUT/tcpdump.log" &
DUMP=$!
sleep 3
kill -0 "$DUMP" 2>/dev/null || { echo "STOP: tcpdump did not start: $(cat "$OUT/tcpdump.log")"; exit 2; }
echo "CLEAN BUILD (new builder $BUILDER, --no-cache --pull, empty cache mounts) started $(date -u +%FT%TZ)"
status=0
docker buildx build --builder "$BUILDER" --no-cache --pull --progress=plain "$@" "$ROOT" >"$OUT/build.log" 2>&1 || status=$?
echo "build exit $status, ended $(date -u +%FT%TZ)"
cleanup
trap - EXIT
python3 -I "$ROOT/tools/netscan_report.py" "$OUT/build.pcap" "$OUT/build.log" >"$OUT/report.md"
echo "report: $OUT/report.md"
exit "$status"
