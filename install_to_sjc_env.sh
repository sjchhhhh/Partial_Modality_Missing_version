#!/usr/bin/env bash
# =============================================================================
# 将 new 项目的 environment.yml（除 PyTorch）安装到 sjc 环境
# 用法：
#   1) 如果 sjc 环境已存在：bash install_to_sjc_env.sh
#   2) 如果 sjc 环境不存在：脚本会自动创建
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="sjc"
YML_FILE="environment.yml"

echo "检查 conda 环境 ${ENV_NAME} 是否存在..."
if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "环境 ${ENV_NAME} 已存在，将更新现有环境..."
    conda env update -n ${ENV_NAME} -f ${YML_FILE} --prune
else
    echo "环境 ${ENV_NAME} 不存在，将创建新环境..."
    conda env create -n ${ENV_NAME} -f ${YML_FILE}
fi

echo ""
echo "========== 安装完成 =========="
echo "环境名: ${ENV_NAME}"
echo "请激活环境：conda activate ${ENV_NAME}"
echo ""
echo "注意：environment.yml 中不包含 PyTorch，请根据本机 CUDA 自行安装，例如："
echo "  pip install torch torchvision torchaudio"
echo "或访问 https://pytorch.org 选择对应 CUDA 版本安装。"
echo "=============================="
