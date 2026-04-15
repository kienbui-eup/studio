#!/usr/bin/env bash
# =============================================================================
# eStudio - One-liner bootstrap for a fresh vast.ai instance
#
# Usage (inside the instance shell):
#   curl -sSL https://raw.githubusercontent.com/kienbui-eup/studio/master/bootstrap.sh | bash
#
# This does two things:
#   1. Clone the studio repo to /workspace/studio (packaging: workflow JSONs,
#      generator, custom_nodes/ComfyUI-Lovis-Node, input assets, scripts).
#   2. Invoke setup-vastai.sh from that clone.
# =============================================================================
set -euo pipefail

STUDIO_REPO="${STUDIO_REPO:-https://github.com/kienbui-eup/studio.git}"
STUDIO_BRANCH="${STUDIO_BRANCH:-master}"
STUDIO_DIR="${STUDIO_DIR:-/workspace/studio}"

log() { echo -e "\033[1;36m[bootstrap]\033[0m $*"; }

log "Cloning ${STUDIO_REPO} (${STUDIO_BRANCH}) → ${STUDIO_DIR}"
if [ -d "${STUDIO_DIR}/.git" ]; then
    log "  [skip] already cloned, pulling latest"
    git -C "${STUDIO_DIR}" fetch --depth 1 origin "${STUDIO_BRANCH}"
    git -C "${STUDIO_DIR}" reset --hard "origin/${STUDIO_BRANCH}"
else
    mkdir -p "$(dirname "${STUDIO_DIR}")"
    git clone --depth 1 --branch "${STUDIO_BRANCH}" "${STUDIO_REPO}" "${STUDIO_DIR}"
fi

log "Running setup-vastai.sh $*"
cd "${STUDIO_DIR}"
chmod +x setup-vastai.sh
exec bash setup-vastai.sh "$@"
