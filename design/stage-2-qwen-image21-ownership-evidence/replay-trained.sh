#!/usr/bin/env bash
set -euo pipefail
export IMAGINE_PARITY_ROOT=/tmp/imagine-rust-comfy-parity-20261017
export IMAGINE_REFERENCE_SITE=/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages
cd /home/thhel/ChatGPT/imagine
prefix=${1:?Supply a fresh isolated prefix}
bash inference/tools/check_qwen_portrait_editing.sh /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models "$prefix-int8" inference/profiles/qwen21-lora-portrait-1600-int8.json
bash inference/tools/check_qwen_lora.sh /home/thhel/git/ComfyUI /mnt/media/gemensam/AI/ComfyUI/models "$prefix-small" bf16 turbo
