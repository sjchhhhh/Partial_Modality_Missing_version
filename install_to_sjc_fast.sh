#!/usr/bin/env bash
# =============================================================================
# 快速安装：使用国内镜像源（清华/阿里云）加速安装到 sjc 环境
# 用法：conda activate sjc 后运行 bash install_to_sjc_fast.sh
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "当前 conda 环境: ${CONDA_DEFAULT_ENV:-未激活 conda 环境}"
echo ""

# 配置 pip 使用清华镜像（如果还没配置）
if [ ! -f ~/.pip/pip.conf ]; then
    echo "配置 pip 使用清华镜像源..."
    mkdir -p ~/.pip
    cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
fi

# 只安装核心 pip 包（跳过一些可选的大包如 mmcv、opencv 等，可后续按需安装）
echo "正在安装核心 pip 依赖（使用清华镜像）..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    addict==2.4.0 \
    click==8.1.3 \
    einops==0.6.0 \
    h5py==3.8.0 \
    huggingface-hub==0.14.1 \
    pandas==1.5.1 \
    pyyaml==6.0 \
    regex==2023.5.5 \
    requests==2.28.1 \
    tqdm==4.64.1 \
    transformers==4.29.0 \
    tokenizers==0.13.3 \
    scikit-learn==1.1.3 \
    numpy==1.23.4

echo ""
echo "========== 核心包安装完成 =========="
echo "如需安装其他包（如 mmcv、opencv 等），可手动运行："
echo "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_no_torch.txt"
echo ""
echo "请根据本机 CUDA 自行安装 PyTorch："
echo "  pip install torch torchvision torchaudio"
echo "=============================="
