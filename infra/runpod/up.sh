#!/usr/bin/env bash
# up.sh — provision 1x A100 80GB on-demand, attach the Network Volume.
# Writes the pod id to .pod_id so serve.sh / down.sh can find it.
#
# Adjust GPU_TYPE / IMAGE / VOLUME_ID to your RunPod account. This is a template;
# the exact runpodctl flags depend on your CLI version (`runpodctl create pod -h`).
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY in .env}"
: "${VOLUME_ID:?set VOLUME_ID (the persistent Network Volume) in .env}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
POD_NAME="${POD_NAME:-clinical-graphrag}"

echo "[up] creating pod ($GPU_TYPE)…"
POD_ID="$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" \
  --imageName "$IMAGE" \
  --volumeId "$VOLUME_ID" \
  --volumePath /vol \
  --ports '8000/http,7474/http,7687/tcp' \
  --containerDiskSize 40 \
  --cost on-demand \
  | grep -oE 'pod "[^"]+"' | sed -E 's/pod "([^"]+)"/\1/')"

echo "$POD_ID" > .pod_id
echo "[up] pod_id=$POD_ID (saved to .pod_id)"
echo "[up] waiting for SSH/HTTP…"; sleep 20
