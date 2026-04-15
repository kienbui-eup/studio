#!/usr/bin/env bash
# =============================================================================
# eStudio - Vast.ai GPU Instance Bootstrap Script (v2 - optimized)
# Sets up ComfyUI with the full video production workflow (WanVideo I2V pipeline)
#
# Target image: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
# Expected runtime: ~12 minutes (vs ~30 minutes in v1)
#
# Usage:
#   bash setup-vastai.sh                  # Full setup
#   bash setup-vastai.sh --skip-models    # Skip large model downloads
#   bash setup-vastai.sh --skip-nodes     # Skip custom node cloning
#   bash setup-vastai.sh --only-models    # Only download models
#   bash setup-vastai.sh --only-nginx     # Only configure nginx
#   bash setup-vastai.sh --no-start       # Don't auto-start ComfyUI
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_DIR="/workspace/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
CUSTOM_NODES_DIR="${COMFYUI_DIR}/custom_nodes"
INPUT_DIR="${COMFYUI_DIR}/input"
WORKFLOWS_DIR="${COMFYUI_DIR}/user/default/workflows"

NGINX_PORT=8288
COMFYUI_PORT=8188
NGINX_USER="admin"
NGINX_PASS="EupStudio2026@"

# Parallelism controls
CLONE_JOBS=8          # Concurrent git clones
DL_CONCURRENT=4       # Concurrent file downloads (aria2 -j)
DL_CONNECTIONS=16     # Connections per file (aria2 -x, -s)

SKIP_MODELS=false
SKIP_NODES=false
ONLY_MODELS=false
ONLY_NGINX=false
NO_START=false

# ---------------------------------------------------------------------------
# Custom nodes registry — format: "repo_url|target_dirname"
# Keep kienbui-eup repos first (forks we control), then upstream.
# ---------------------------------------------------------------------------
CUSTOM_NODES=(
    # --- kienbui-eup controlled forks ---
    "https://github.com/kienbui-eup/ComfyUI-Lovis-Node.git|ComfyUI-Lovis-Node"
    "https://github.com/kienbui-eup/ComfyUI_ChatterBox_Voice.git|ComfyUI_ChatterBox_Voice"
    "https://github.com/kienbui-eup/DirectorsConsole.git|ComfyUI-DirectorsConsole"

    # --- WanVideo I2V pipeline ---
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git|ComfyUI-WanVideoWrapper"
    "https://github.com/kijai/ComfyUI-KJNodes.git|ComfyUI-KJNodes"

    # --- Core utility nodes ---
    "https://github.com/yolain/ComfyUI-Easy-Use.git|ComfyUI-Easy-Use"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git|ComfyUI-Custom-Scripts"
    "https://github.com/theUpsider/ComfyUI-Logic.git|ComfyUI-Logic"
    "https://github.com/WASasquatch/was-node-suite-comfyui.git|was-node-suite-comfyui"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git|ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/jamesWalker55/comfyui-various.git|comfyui-various"

    # --- Video / frames ---
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git|ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git|ComfyUI-Frame-Interpolation"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git|ComfyUI-AnimateDiff-Evolved"

    # --- Images / IP-Adapter ---
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git|ComfyUI_IPAdapter_plus"
    "https://github.com/cubiq/ComfyUI_essentials_mb.git|ComfyUI_essentials_mb"
    "https://github.com/chflame163/ComfyUI_LayerStyle.git|ComfyUI_LayerStyle"
    "https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git|ComfyUI_LayerStyle_Advance"
    "https://github.com/banodoco/ComfyUI-RMBG.git|ComfyUI-RMBG"
    "https://github.com/djbielejeski/a-person-mask-generator.git|a-person-mask-generator"

    # --- Audio / subtitles ---
    "https://github.com/yuvraj108c/ComfyUI-Whisper.git|ComfyUI-Whisper"
    "https://github.com/a1lazydog/ComfyUI-AudioBatch.git|ComfyUI-AudioBatch"

    # --- Misc ---
    "https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside.git|ComfyUI_RyanOnTheInside"
    "https://github.com/Extraltodeus/ComfyUI-Workflow-Encrypt.git|ComfyUI-Workflow-Encrypt"
    "https://github.com/jaimemartinez-99/ComfyUI-JM-MiniMax-API.git|ComfyUI-JM-MiniMax-API"
)

# ---------------------------------------------------------------------------
# Models registry — format: "url|subdir|filename"
# Remove or comment lines you don't need for faster setup.
# ---------------------------------------------------------------------------
MODELS=(
    # SDXL base
    "https://civitai.com/api/download/models/344487|checkpoints|realvisxlV50_v50LightningBakedvae.safetensors"

    # IP-Adapter Plus SDXL
    "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors|ipadapter|ip-adapter-plus_sdxl_vit-h.safetensors"
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors|clip_vision|clip_vision_h.safetensors"

    # WanVideo I2V 14B FP8 (main, ~14GB)
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors|diffusion_models|wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors|vae|wan_2.1_vae.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors|text_encoders|umt5_xxl_fp16.safetensors"
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-models)  SKIP_MODELS=true ;;
        --skip-nodes)   SKIP_NODES=true ;;
        --only-models)  ONLY_MODELS=true ;;
        --only-nginx)   ONLY_NGINX=true ;;
        --no-start)     NO_START=true ;;
        --help|-h)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg (use --help)"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo -e "\033[1;32m[eStudio]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Wait for up to N background jobs to stay under CLONE_JOBS concurrency
throttle() {
    local max="$1"
    while [ "$(jobs -rp | wc -l)" -ge "$max" ]; do
        sleep 0.1
    done
}

# =========================================================================
# STEP 1: System Dependencies
# =========================================================================
install_system_deps() {
    log "STEP 1: System dependencies..."
    if command -v nginx &>/dev/null \
        && command -v htpasswd &>/dev/null \
        && command -v aria2c &>/dev/null \
        && command -v ffmpeg &>/dev/null; then
        log "  [skip] all tools already installed"
        return
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq nginx apache2-utils aria2 ffmpeg git-lfs > /dev/null
    log "  [done] nginx, apache2-utils, aria2, ffmpeg, git-lfs installed"
}

# =========================================================================
# STEP 2: Clone ComfyUI (pinned to v0.18.1 for stability)
# =========================================================================
clone_comfyui() {
    log "STEP 2: ComfyUI core..."
    if [ -d "${COMFYUI_DIR}/.git" ]; then
        log "  [skip] ${COMFYUI_DIR} already a git repo"
        return
    fi
    mkdir -p "$(dirname "${COMFYUI_DIR}")"
    git clone --depth 1 --branch v0.18.1 \
        https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}" \
        2>/dev/null || git clone --depth 1 \
        https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_DIR}"
    log "  [done] ComfyUI cloned"
}

# =========================================================================
# STEP 3: Install ComfyUI core requirements
# =========================================================================
install_comfyui_requirements() {
    log "STEP 3: ComfyUI core requirements..."
    if [ -f "${COMFYUI_DIR}/requirements.txt" ]; then
        pip install -r "${COMFYUI_DIR}/requirements.txt" --quiet 2>/dev/null \
            || warn "some ComfyUI requirements failed"
        log "  [done]"
    else
        warn "requirements.txt not found"
    fi
}

# =========================================================================
# STEP 4: Clone ALL custom nodes in parallel
# =========================================================================
clone_custom_nodes() {
    log "STEP 4: Parallel clone of ${#CUSTOM_NODES[@]} custom nodes (${CLONE_JOBS} jobs)..."
    mkdir -p "${CUSTOM_NODES_DIR}"

    local tmp_log
    tmp_log=$(mktemp)
    trap "rm -f $tmp_log" RETURN

    for entry in "${CUSTOM_NODES[@]}"; do
        local url="${entry%%|*}"
        local name="${entry##*|}"
        local target="${CUSTOM_NODES_DIR}/${name}"

        if [ -d "${target}/.git" ] || [ -d "${target}" ]; then
            echo "  [skip] ${name}" >> "$tmp_log"
            continue
        fi

        throttle "$CLONE_JOBS"
        (
            if git clone --depth 1 --quiet "$url" "$target" 2>/dev/null; then
                echo "  [ok]   ${name}" >> "$tmp_log"
            else
                echo "  [fail] ${name}" >> "$tmp_log"
            fi
        ) &
    done
    wait
    sort "$tmp_log"
    log "  [done] parallel clone complete"
}

# =========================================================================
# STEP 5: Install node requirements (batched + deduplicated)
# =========================================================================
install_node_requirements() {
    log "STEP 5: Installing node requirements (batched)..."

    local combined="/tmp/all_node_requirements.txt"
    : > "$combined"

    for req in "${CUSTOM_NODES_DIR}"/*/requirements.txt; do
        [ -f "$req" ] || continue
        # Skip comments and empty lines
        grep -v '^\s*#' "$req" | grep -v '^\s*$' >> "$combined" || true
    done

    # Add extra deps not in any requirements.txt
    cat >> "$combined" <<'EOF'
conformer
EOF

    # Sort + unique to reduce duplicate resolution work
    sort -u "$combined" -o "${combined}.uniq"
    local count
    count=$(wc -l < "${combined}.uniq")
    log "  [pip] Installing ${count} unique packages in one pass..."
    pip install -r "${combined}.uniq" --quiet 2>/dev/null \
        || warn "some packages failed (likely version conflicts, non-fatal)"

    # Run node-specific post-install scripts that need special handling
    if [ -f "${CUSTOM_NODES_DIR}/ComfyUI-Frame-Interpolation/install.py" ]; then
        log "  [post] RIFE install.py"
        ( cd "${CUSTOM_NODES_DIR}/ComfyUI-Frame-Interpolation" \
            && python install.py >/dev/null 2>&1 ) \
            || warn "RIFE install.py failed"
    fi

    log "  [done] node requirements installed"
}

# =========================================================================
# STEP 6: Download models in parallel via aria2c -i
# =========================================================================
download_models() {
    log "STEP 6: Parallel model download (${DL_CONCURRENT} files × ${DL_CONNECTIONS} connections)..."

    # Pre-create target dirs
    local dirs
    dirs=$(printf '%s\n' "${MODELS[@]}" | awk -F'|' '{print $2}' | sort -u)
    while IFS= read -r d; do mkdir -p "${MODELS_DIR}/${d}"; done <<< "$dirs"

    local input_file="/tmp/aria2_models.txt"
    : > "$input_file"

    local total=0 skipped=0
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r url subdir filename <<< "$entry"
        local dest="${MODELS_DIR}/${subdir}/${filename}"
        if [ -f "$dest" ] && [ -s "$dest" ]; then
            skipped=$((skipped+1))
            continue
        fi
        # aria2 input file format: URL\n  out=name\n  dir=path
        printf '%s\n  out=%s\n  dir=%s\n' \
            "$url" "$filename" "${MODELS_DIR}/${subdir}" >> "$input_file"
        total=$((total+1))
    done

    log "  [info] ${total} to download, ${skipped} already present"
    if [ "$total" -eq 0 ]; then
        log "  [done] all models present"
        return
    fi

    aria2c -i "$input_file" \
        -j "$DL_CONCURRENT" \
        -x "$DL_CONNECTIONS" \
        -s "$DL_CONNECTIONS" \
        -k 1M \
        --continue=true \
        --auto-file-renaming=false \
        --summary-interval=30 \
        --console-log-level=warn \
        || warn "some downloads failed (check files manually)"

    log "  [done] model download complete"
}

# =========================================================================
# STEP 7: Copy input assets from studio repo
# =========================================================================
upload_input_assets() {
    log "STEP 7: Copy input assets..."
    mkdir -p "${INPUT_DIR}"

    local src="${SCRIPT_DIR}/input"
    if [ ! -d "$src" ]; then
        warn "no input/ directory in ${SCRIPT_DIR} — skipping"
        return
    fi

    # Sync all subdirectories (heyj_linh_vat, 3d, etc.)
    cp -rn "$src"/* "${INPUT_DIR}/" 2>/dev/null || true
    local count
    count=$(find "${INPUT_DIR}" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) 2>/dev/null | wc -l)
    log "  [done] ${count} image assets in place"
}

# =========================================================================
# STEP 8: Copy workflow JSONs + generator
# =========================================================================
copy_workflow() {
    log "STEP 8: Workflow files..."
    mkdir -p "${WORKFLOWS_DIR}"

    local copied=0
    for wf in "${SCRIPT_DIR}"/workflow-video-production*.json; do
        [ -f "$wf" ] || continue
        cp "$wf" "${WORKFLOWS_DIR}/$(basename "$wf")"
        copied=$((copied+1))
    done
    log "  [done] ${copied} workflow JSONs copied"

    # Generator scripts
    for gen in "${SCRIPT_DIR}"/generate_video_workflow*.py; do
        [ -f "$gen" ] || continue
        cp "$gen" "${COMFYUI_DIR}/$(basename "$gen")"
    done
}

# =========================================================================
# STEP 9: Configure Nginx reverse proxy with basic auth
# =========================================================================
configure_nginx() {
    log "STEP 9: Configure nginx..."

    mkdir -p /etc/nginx
    htpasswd -bc /etc/nginx/.htpasswd "${NGINX_USER}" "${NGINX_PASS}" 2>/dev/null

    cat > /etc/nginx/sites-available/comfyui <<NGINX_CONF
server {
    listen ${NGINX_PORT};
    server_name _;
    client_max_body_size 0;

    location /ws {
        proxy_pass http://127.0.0.1:${COMFYUI_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
        auth_basic "eStudio - ComfyUI";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }

    location / {
        proxy_pass http://127.0.0.1:${COMFYUI_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        auth_basic "eStudio - ComfyUI";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
NGINX_CONF

    mkdir -p /etc/nginx/sites-enabled
    ln -sf /etc/nginx/sites-available/comfyui /etc/nginx/sites-enabled/comfyui
    rm -f /etc/nginx/sites-enabled/default

    if ! grep -q "sites-enabled" /etc/nginx/nginx.conf 2>/dev/null; then
        sed -i '/http {/a \    include /etc/nginx/sites-enabled/*;' /etc/nginx/nginx.conf 2>/dev/null \
            || warn "could not auto-edit nginx.conf"
    fi

    nginx -t 2>/dev/null && {
        nginx -s stop 2>/dev/null || true
        sleep 0.5
        nginx
        log "  [done] nginx on :${NGINX_PORT} -> :${COMFYUI_PORT} (${NGINX_USER}/${NGINX_PASS})"
    } || err "nginx config invalid"
}

# =========================================================================
# STEP 10: Start ComfyUI
# =========================================================================
start_comfyui() {
    log "STEP 10: Start ComfyUI..."

    if pgrep -f "main.py.*--listen" > /dev/null 2>&1; then
        warn "ComfyUI already running — skip"
        return
    fi

    cd "${COMFYUI_DIR}"
    nohup python main.py \
        --listen 127.0.0.1 \
        --port "${COMFYUI_PORT}" \
        --enable-cors-header \
        > /workspace/comfyui.log 2>&1 &

    local pid=$!
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        log "  [done] ComfyUI running (PID ${pid})"
        log "  [url]  http://<INSTANCE_IP>:${NGINX_PORT}  (${NGINX_USER}/${NGINX_PASS})"
        log "  [log]  tail -f /workspace/comfyui.log"
    else
        err "ComfyUI exited — check /workspace/comfyui.log"
    fi
}

# =========================================================================
# Main
# =========================================================================
main() {
    local t_start=$SECONDS

    log "=========================================="
    log "  eStudio Vast.ai Setup v2 (optimized)"
    log "=========================================="
    log "Target:   ${COMFYUI_DIR}"
    log "Options:  skip_models=${SKIP_MODELS} skip_nodes=${SKIP_NODES} no_start=${NO_START}"
    log ""

    if [ "$ONLY_MODELS" = true ]; then download_models; exit 0; fi
    if [ "$ONLY_NGINX"  = true ]; then configure_nginx; exit 0; fi

    install_system_deps
    clone_comfyui
    install_comfyui_requirements

    if [ "$SKIP_NODES" = false ]; then
        clone_custom_nodes
        install_node_requirements
    else
        log "STEP 4-5: [SKIPPED] custom nodes"
    fi

    if [ "$SKIP_MODELS" = false ]; then
        download_models
    else
        log "STEP 6: [SKIPPED] model download"
    fi

    upload_input_assets
    copy_workflow
    configure_nginx

    if [ "$NO_START" = false ]; then
        start_comfyui
    fi

    local elapsed=$((SECONDS - t_start))
    log ""
    log "=========================================="
    log "  Setup complete in $((elapsed/60))m $((elapsed%60))s"
    log "=========================================="
}

main "$@"
