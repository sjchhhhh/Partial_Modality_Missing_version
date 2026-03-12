#!/usr/bin/env bash
# =============================================================================
# 安装 environment.yml 中除 PyTorch 以外的依赖
# 用法：
#   1) 先根据 environment.yml 创建并激活环境，例如：
#        conda env create -f environment.yml
#        conda activate sjc1
#   2) 再运行本脚本（在当前已激活的 conda 环境下）：
#        bash install_deps_no_pytorch.sh
#
# 本脚本会安装 environment.yml 里的 pip 依赖（不包含 torch/torchvision/torchaudio）。
# conda 依赖请通过 environment.yml 创建环境时已安装；若你未用 yml 创建环境，
# 请先执行：conda env create -f environment.yml  （环境名为 sjc）
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "当前 conda 环境: ${CONDA_DEFAULT_ENV:-未激活 conda 环境}"
echo ""

# 仅安装 pip 依赖（environment.yml 中 conda 部分已通过 env create 安装，这里只补 pip）
if [ ! -f "requirements_no_torch.txt" ]; then
  echo "错误: 未找到 requirements_no_torch.txt，请确保与 install_deps_no_pytorch.sh 同目录。"
  exit 1
fi

echo "正在安装 requirements_no_torch.txt 中的包（不含 PyTorch）..."
pip install -r requirements_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "========== 安装完成 =========="
echo "未安装 PyTorch。请根据本机 CUDA 自行安装，例如："
echo "  pip install torch torchvision torchaudio"
echo "或访问 https://pytorch.org 选择对应 CUDA 版本安装。"
echo "=============================="
