#!/usr/bin/env zsh
# Runs st-stack.zsh's proxy port release for one port: exit 0 when the port is
# free afterwards, 1 when startup refused it.
port=$1
export ST_STACK_FUNCTIONS_ONLY=1 ST_PROXY_CHAT_PORT=$port ST_PROXY_IMAGE_PORT=$port ST_STACK_STOP_TIMEOUT=5
source "${0:A:h}/../st-stack.zsh"
ensure_proxy_ports_available
