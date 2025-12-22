import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径（根据你的路径修改）
file_path = r'D:\Bin\谷歌浏览器下载\extracted_features_and_pretrained_models\extracted_features_and_pretrained_models\data\features\Ball_rgb_VST.npy'

# 加载 .npy 文件并允许 pickle（处理包含 object 类型的数据）
data = np.load(file_path, allow_pickle=True)

# 打印数据的基本信息
print(f"数据类型: {type(data)}")
print(f"数组形状: {data.shape}")
print(f"数组数据类型: {data.dtype}")
print(f"数组维度: {data.ndim}")

# 如果数据是 object 类型，进一步处理它
if data.dtype == 'object':
    print("\n数据类型为 'object'，正在尝试转换数据...")

    # 打印 data 的内容，以查看具体是什么数据
    print("\ndata 内容的前几个元素（如果存在的话）：")

    # 打印字典中的每个条目
    for key, value in data.item().items():  # item() 获取字典对象
        print(f"键: {key}")
        print(f"值（即 numpy 数组的形状和部分数据）：")
        print(f"数组形状: {value.shape}")
        print(f"数组内容的前几项：\n{value[:5]}")  # 打印数组前五项，避免输出过多
        print("\n" + "-" * 50)  # 分隔不同的条目

    # 获取 'Ball_001' 特征数据
    ball_001_features = data.item().get('Ball_001')

    # 如果 'Ball_001' 的特征矩阵存在，绘制热图
    if ball_001_features is not None:
        # 绘制 Ball_001 的特征热图
        plt.figure(figsize=(10, 6))
        sns.heatmap(ball_001_features, cmap='viridis', xticklabels=50, yticklabels=10)
        plt.title('Ball_001 特征热图')
        plt.show()
    else:
        print("未找到 'Ball_001' 的特征数据。")
else:
    print("数据不是 'object' 类型，无法进行处理。")
