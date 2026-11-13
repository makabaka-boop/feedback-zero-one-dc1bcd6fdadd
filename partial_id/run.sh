#!/usr/bin/env bash
# 部分识别精确求解器入口。
# 用法: ./run.sh <输入.json> <输出.json>
# 使用 -I 隔离模式，避免从当前目录/脚本目录加载任何第三方模块。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -I "$SCRIPT_DIR/pid.py" "$@"
