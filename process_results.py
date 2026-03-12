import matplotlib.pyplot as plt
import numpy as np
import os
import re

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据添加区域 ====================
# 实验/图名：画图时显示在标题和文件名中，方便以后记住这次做了什么（可随意修改）
EXPERIMENT_NAME = 'new_SimSiam'

# 在这里添加你的多组数据
# 格式说明：
#   - 每组数据包含：数据名称（用于图例）和数据列表
#   - 每行数据格式: [缺失率, Has0_acc_2, Has0_F1_score, Non0_acc_2, Non0_F1_score, Mult_acc_5, Mult_acc_7, MAE, Corr, Loss, Loss(pred_m)]
#   - 指标值格式: "(平均值, 标准差)"

datasets = {
    # 第一组数据（原始数据）
    'backbone': [
        [0.0,"(82.75, 0.38)","(82.68, 0.37)","(84.45, 0.54)","(84.44, 0.51)","(51.65, 0.66)","(45.82, 0.56)","(71.18, 0.63)","(79.6, 0.26)","(-226.29, 1.24)","(71.48, 0.43)"],
        [0.1,"(80.27, 0.25)","(80.19, 0.27)","(82.17, 0.22)","(82.15, 0.2)","(50.78, 0.56)","(45.58, 0.66)","(76.81, 0.57)","(75.97, 0.02)","(-188.46, 1.3)","(76.86, 0.83)"],
        [0.2,"(77.94, 0.27)","(77.97, 0.28)","(79.17, 0.36)","(79.26, 0.37)","(45.04, 0.31)","(40.48, 0.6)","(86.15, 0.29)","(69.93, 0.28)","(-167.68, 1.25)","(86.24, 0.43)"],
        [0.3,"(76.14, 0.38)","(76.15, 0.4)","(77.39, 0.26)","(77.47, 0.27)","(42.47, 1.2)","(38.58, 0.97)","(92.02, 0.94)","(64.89, 0.75)","(-138.21, 8.31)","(92.14, 1.22)"],
        [0.4,"(72.21, 0.89)","(72.23, 0.86)","(73.17, 0.82)","(73.28, 0.77)","(38.39, 0.73)","(34.21, 0.76)","(103.48, 1.36)","(56.63, 0.63)","(-102.27, 6.21)","(102.73, 1.57)"],
        [0.5,"(70.41, 0.72)","(70.47, 0.71)","(70.78, 0.73)","(70.94, 0.7)","(35.03, 1.44)","(31.1, 1.44)","(112.29, 3.14)","(49.29, 1.6)","(-78.88, 5.71)","(111.59, 3.33)"],
        [0.6,"(66.81, 0.59)","(66.89, 0.6)","(67.33, 0.26)","(67.51, 0.27)","(34.01, 0.88)","(31.1, 0.88)","(120.33, 1.91)","(40.13, 1.09)","(-60.79, 4.09)","(119.75, 2.46)"],
        [0.7,"(63.07, 0.92)","(63.09, 0.97)","(63.57, 0.86)","(63.71, 0.92)","(30.47, 0.12)","(27.55, 0.12)","(129.17, 1.55)","(35.52, 1.27)","(-30.78, 8.21)","(128.16, 2.09)"],
        [0.8,"(62.29, 0.97)","(61.94, 1.14)","(61.74, 1.26)","(61.56, 1.44)","(24.64, 0.63)","(22.89, 0.31)","(134.01, 1.16)","(32.77, 0.48)","(-4.0, 8.75)","(133.69, 1.1)"],
        [0.9,"(56.22, 0.84)","(55.91, 1.13)","(56.05, 1.12)","(55.89, 1.41)","(20.85, 0.31)","(20.12, 0.31)","(143.64, 2.27)","(18.95, 0.88)","(29.04, 8.83)","(144.34, 2.04)"],
        [1.0,"(53.25, 1.91)","(51.83, 3.3)","(52.39, 2.49)","(51.15, 3.86)","(16.62, 0.48)","(16.62, 0.48)","(144.87, 1.52)","(6.97, 1.65)","(8.41, 34.96)","(144.92, 1.72)"],
    ],
    
    # 第二组数据（在这里添加你的新数据）
    'ours': [
    [0.0,"(83.14, 0.77)","(83.05, 0.81)","(85.06, 0.45)","(85.04, 0.48)","(52.28, 0.84)","(46.6, 0.45)","(70.92, 0.2)","(79.59, 0.22)","(-231.43, 2.7)","(71.09, 0.7)"],
[0.1,"(81.34, 0.86)","(81.26, 0.83)","(82.98, 0.83)","(82.95, 0.8)","(50.49, 0.5)","(45.09, 0.68)","(76.16, 1.39)","(76.21, 0.72)","(-184.63, 11.63)","(76.72, 1.02)"],
[0.2,"(79.2, 0.66)","(79.22, 0.63)","(80.44, 0.76)","(80.52, 0.73)","(45.09, 1.34)","(40.33, 1.36)","(84.69, 1.09)","(71.02, 0.18)","(-172.0, 2.3)","(84.83, 1.58)"],
[0.3,"(76.04, 0.38)","(76.02, 0.37)","(77.39, 0.4)","(77.44, 0.4)","(42.86, 0.52)","(38.58, 0.28)","(91.6, 1.91)","(65.02, 1.09)","(-147.24, 3.33)","(91.2, 1.51)"],
[0.4,"(71.86, 0.83)","(71.83, 0.84)","(72.86, 0.99)","(72.92, 0.99)","(38.63, 0.52)","(34.79, 0.42)","(102.33, 0.6)","(56.74, 0.18)","(-121.57, 8.51)","(102.19, 0.72)"],
[0.5,"(71.57, 1.34)","(71.61, 1.36)","(72.2, 1.46)","(72.32, 1.47)","(35.61, 1.31)","(31.63, 1.44)","(110.18, 0.63)","(51.21, 1.36)","(-97.85, 3.43)","(110.55, 0.34)"],
[0.6,"(66.04, 0.86)","(66.08, 0.86)","(66.46, 0.9)","(66.61, 0.9)","(33.33, 0.89)","(30.57, 0.84)","(122.77, 0.76)","(38.96, 0.57)","(-73.55, 7.14)","(123.15, 1.12)"],
[0.7,"(63.75, 1.5)","(63.41, 2.03)","(64.03, 1.76)","(63.79, 2.29)","(28.86, 0.78)","(26.14, 0.34)","(132.44, 4.55)","(34.8, 0.43)","(-31.84, 12.0)","(131.94, 4.73)"],
[0.8,"(62.88, 0.5)","(62.71, 0.36)","(62.7, 0.51)","(62.68, 0.39)","(24.49, 0.36)","(22.55, 0.38)","(135.36, 1.55)","(31.85, 0.65)","(-12.21, 2.36)","(134.78, 1.62)"],
[0.9,"(57.34, 0.61)","(57.43, 0.59)","(57.67, 0.64)","(57.91, 0.61)","(22.16, 0.41)","(21.43, 0.43)","(139.14, 3.47)","(18.48, 2.17)","(-6.94, 5.07)","(139.12, 3.43)"],
[1.0,"(54.18, 1.22)","(53.65, 1.92)","(53.56, 1.67)","(53.21, 2.34)","(17.06, 1.04)","(17.06, 1.04)","(143.86, 1.84)","(7.26, 0.27)","(-77.09, 30.18)","(144.29, 1.77)"]
    ],
}
# ==================== 数据添加区域结束 ====================

# 指标名称
metric_names = ['Has0_acc_2', 'Has0_F1_score', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7', 'MAE', 'Corr', 'Loss', 'Loss(pred_m)']

# 解析所有数据集
all_datasets_parsed = {}

for dataset_name, raw_data in datasets.items():
    missing_rates = []
    metric_data = {name: [] for name in metric_names}
    
    for row in raw_data:
        missing_rate = row[0]
        missing_rates.append(missing_rate)
        
        # 解析每个指标的值（格式为 "(值, 标准差)"）
        for idx, metric_name in enumerate(metric_names):
            value_str = row[idx + 1].strip()
            # 提取括号中的第一个数字（平均值）
            match = re.search(r'\(([\d.]+)', value_str)
            if match:
                value = float(match.group(1))
                metric_data[metric_name].append(value)
            else:
                metric_data[metric_name].append(None)
    
    all_datasets_parsed[dataset_name] = {
        'missing_rates': missing_rates,
        'metric_data': metric_data
    }

# 创建图表目录（如果权限不足，请修改此路径）
plot_dir = '/mnt/HDD2/teng/Master-23/sjc/work2/EMT-DLFR-master-new/results/mosi/emt-dlfr/run_once/results/results_plots'
os.makedirs(plot_dir, exist_ok=True)

# 为每个指标创建趋势图（多组数据对比）
# 定义不同数据组的颜色和标记样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'P']

# 创建单独的图表（每个指标一张图）
for metric_name in metric_names:
    plt.figure(figsize=(10, 6))
    
    # 为每组数据绘制一条线
    for idx, (dataset_name, dataset_info) in enumerate(all_datasets_parsed.items()):
        missing_rates = dataset_info['missing_rates']
        values = dataset_info['metric_data'][metric_name]
        
        # 确保values和missing_rates长度一致
        min_len = min(len(values), len(missing_rates))
        values = values[:min_len]
        rates = missing_rates[:min_len]
        
        # 过滤掉None值
        valid_data = [(r, v) for r, v in zip(rates, values) if v is not None]
        
        if len(valid_data) == 0:
            continue
        
        valid_missing_rates, valid_values = zip(*valid_data)
        
        # 绘制线条，使用不同的颜色和标记
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(valid_missing_rates, valid_values, marker=marker, linewidth=2, 
                markersize=8, label=dataset_name, color=color)
    
    plt.xlabel('Missing Rate', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} vs Missing Rate — {EXPERIMENT_NAME}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc='best', fontsize=10)
    
    # 设置x轴刻度
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # 保存图片（文件名带实验名便于区分）
    safe_name = metric_name.replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
    safe_exp = re.sub(r'[^\w\-]', '_', EXPERIMENT_NAME)[:40]
    plot_path = os.path.join(plot_dir, f'{safe_name}_{safe_exp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已创建图表: {plot_path}")

# 创建包含所有指标的合并图表
num_metrics = len(metric_names)
# 计算子图布局：尽量接近正方形
cols = int(np.ceil(np.sqrt(num_metrics)))
rows = int(np.ceil(num_metrics / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
fig.suptitle(f'All Metrics vs Missing Rate — {EXPERIMENT_NAME}', fontsize=16, fontweight='bold', y=0.995)

# 如果只有一行或一列，确保axes是二维数组
if num_metrics == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = axes.reshape(1, -1)
elif cols == 1:
    axes = axes.reshape(-1, 1)

for idx, metric_name in enumerate(metric_names):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 and cols > 1 else (axes[row] if rows > 1 else axes[col])
    
    # 为每组数据绘制一条线
    for dataset_idx, (dataset_name, dataset_info) in enumerate(all_datasets_parsed.items()):
        missing_rates = dataset_info['missing_rates']
        values = dataset_info['metric_data'][metric_name]
        
        # 确保values和missing_rates长度一致
        min_len = min(len(values), len(missing_rates))
        values = values[:min_len]
        rates = missing_rates[:min_len]
        
        # 过滤掉None值
        valid_data = [(r, v) for r, v in zip(rates, values) if v is not None]
        
        if len(valid_data) == 0:
            continue
        
        valid_missing_rates, valid_values = zip(*valid_data)
        
        # 绘制线条，使用不同的颜色和标记
        color = colors[dataset_idx % len(colors)]
        marker = markers[dataset_idx % len(markers)]
        ax.plot(valid_missing_rates, valid_values, marker=marker, linewidth=1.5, 
                markersize=5, label=dataset_name, color=color)
    
    ax.set_xlabel('Missing Rate', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.set_title(metric_name, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8)
    
    # 只在第一个子图显示图例
    if idx == 0:
        ax.legend(loc='best', fontsize=8)

# 隐藏多余的子图
for idx in range(num_metrics, rows * cols):
    row = idx // cols
    col = idx % cols
    ax = axes[row, col] if rows > 1 and cols > 1 else (axes[row] if rows > 1 else axes[col])
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.99])
safe_exp = re.sub(r'[^\w\-]', '_', EXPERIMENT_NAME)[:40]
combined_plot_path = os.path.join(plot_dir, f'all_metrics_combined_{safe_exp}.png')
plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"已创建合并图表: {combined_plot_path}")
print(f"\n所有图表已保存到: {plot_dir}")
