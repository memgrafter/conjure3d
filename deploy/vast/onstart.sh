#!/usr/bin/env bash
# vast.ai on-start script for the conjure3d image (SSH or Jupyter launch mode). Steps: deploy/vast/README.md.
# Template "On-start Script" field:  bash /opt/conjure3d/repo/deploy/vast/onstart.sh
# - Makes the image's settings visible in SSH sessions (vast.ai starts them without the image's ENV).
# - With HF_TOKEN set in your vast.ai account's environment variables, and the weights not in /data yet, it starts
#   setup.sh weights in the background. Log: /data/setup-weights.log. Without HF_TOKEN, log in yourself:
#   hf auth login && setsid -w bash trellis2/setup.sh weights
set -uo pipefail
cd /opt/conjure3d/repo
# The volume holds only the pipeline's files, so setup's free-space floors (20 GB each by default) are lowered
export SETUP_MIN_FREE_GB=${SETUP_MIN_FREE_GB:-1} T2_WEIGHTS_MIN_FREE_GB=${T2_WEIGHTS_MIN_FREE_GB:-2}
# The image's settings, never HF_TOKEN or other secrets
for v in SETUP_MIN_FREE_GB T2_WEIGHTS_MIN_FREE_GB PATH T2_VENV T2_SRC T2_WEIGHTS HF_HOME CUDA_HOME TORCH_CUDA_ARCH_LIST CUDA_DEVICE_ORDER HF_HUB_DISABLE_TELEMETRY \
         PYTHONDONTWRITEBYTECODE UV_PYTHON_INSTALL_DIR LD_LIBRARY_PATH NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
  grep -q "^$v=" /etc/environment 2>/dev/null || echo "$v=${!v-}" >> /etc/environment
done
grep -q "^cd /opt/conjure3d/repo" /root/.bashrc 2>/dev/null || echo "cd /opt/conjure3d/repo" >> /root/.bashrc
mkdir -p /data
if [ -e /data/.conjure3d-weights-ok ]; then
  echo "conjure3d: weights in /data/weights"
elif [ -n "${HF_TOKEN:-}" ]; then
  ( setsid -w bash trellis2/setup.sh weights > /data/setup-weights.log 2>&1 \
      && touch /data/.conjure3d-weights-ok ) &
  echo "conjure3d: setup.sh weights started, log /data/setup-weights.log"
else
  echo "conjure3d: no weights yet; run: hf auth login && setsid -w bash trellis2/setup.sh weights"
fi
