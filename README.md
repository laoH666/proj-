# REFUGE 眼底图像分割项目

本项目用于完成 REFUGE 眼底图像中的视盘/视杯分割实验。核心代码基于 PyTorch 实现了一个简化版 U-Net，并提供训练、损失曲线绘制和预测结果可视化功能。

## 项目结构

```text
project/
├── README.md
├── .gitignore
├── data/                         # 本地数据集目录，不上传到 GitHub
└── src/
    ├── task2_segmentation.py      # 主训练与可视化脚本
    └── results/                   # 已生成的实验结果图
```

## 数据集说明

数据集文件不放在 GitHub 仓库里，请队友在本地自行准备。默认代码期望数据目录类似下面这样：

```text
project/data/refuge/REFUGE/train/
├── Images/    # 训练图片
└── gts/       # 对应 mask 标注
```

如果你的数据集路径不同，请打开 `src/task2_segmentation.py`，修改文件底部的这两个变量：

```python
IMG_DIR = r"你的图片目录"
MASK_DIR = r"你的标注目录"
```

注意：`.gitignore` 已经排除了 `data/` 和常见数据文件格式，避免误把数据集上传到 GitHub。

## 环境依赖

建议使用 Python 3.9+。需要安装以下依赖：

```powershell
pip install torch torchvision pillow matplotlib numpy tqdm
```

如果使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch 版本，参考 PyTorch 官网安装命令。

## 运行方法

在项目根目录执行：

```powershell
cd src
python task2_segmentation.py
```

脚本会自动：

1. 读取 REFUGE 训练图片和 mask。
2. 使用 U-Net 进行二分类分割训练。
3. 计算训练 loss 和 IoU。
4. 在 `src/results/` 中保存 loss 曲线和预测对比图。

## 主要代码内容

`src/task2_segmentation.py` 包含以下部分：

- `REFUGEDataset`：读取图片和 mask。
- `UNet`：分割模型结构。
- `DiceLoss` 和 `CombinedLoss`：训练损失函数。
- `calculate_metrics`：计算 IoU 和 Dice。
- `train_model`：模型训练主流程。
- `show_results`：保存 loss 曲线和预测可视化图片。

## 常见问题

### 1. 找不到图片或 mask

检查 `IMG_DIR` 和 `MASK_DIR` 是否指向正确目录。图片默认按 `.jpg` 读取，mask 默认由同名 `.bmp` 匹配，如果数据集命名规则不同，需要修改 `REFUGEDataset.__getitem__` 中的 mask 路径生成逻辑。

### 2. 训练很慢

先确认是否安装了 GPU 版本的 PyTorch。脚本启动时会打印当前使用的设备：`cuda` 表示使用 GPU，`cpu` 表示使用 CPU。

### 3. 不要上传数据集

数据集应放在 `data/` 下，Git 会自动忽略。提交代码前可以运行：

```powershell
git status --ignored
```

如果看到 `!! data/`，说明数据集已经被正确忽略。
