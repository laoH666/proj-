"""
Task 1.3: AI辅助编写ResNet-18训练代码
本脚本包含：数据加载、模型构建、训练、评估
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import medmnist
from medmnist import INFO
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ==================== 设置随机种子（保证结果可复现）====================
def set_seed(seed=42):
    """固定随机种子，让每次运行结果一致"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 1. 配置参数 ====================
class Config:
    """所有可调参数集中管理，方便后续做对比实验"""
    # 数据相关
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    image_size = 64          # ResNet通常用224，但OCTMNIST原始只有28，64是合理的
    batch_size = 32          # 每批训练多少张图
    num_workers = 0          # 数据加载线程数（Windows上设0最稳定）
    
    # 模型相关
    num_classes = 4          # CNV, DME, DRUSEN, NORMAL 四分类
    pretrained = False       # 不用预训练权重（OCTMNIST是灰度图，ImageNet是RGB）
    
    # 训练相关
    learning_rate = 0.001    # 学习率
    num_epochs = 30          # 训练轮数
    optimizer_name = 'Adam'  # 优化器：'Adam' 或 'SGD'
    loss_name = 'CrossEntropy'  # 损失函数
    
    # ===== 数据缩减（用于快速调参）=====
    data_fraction = 0.2   # 1.0 表示用全部数据；0.1 表示只用10%

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

print("=" * 60)
print("Task 1.3: ResNet-18 训练脚本")
print("=" * 60)
print(f"使用设备: {config.device}")
print(f"训练轮数: {config.num_epochs}")
print(f"批次大小: {config.batch_size}")
print(f"学习率:   {config.learning_rate}")
print(f"优化器:   {config.optimizer_name}")

# ==================== 2. 数据加载 ====================
print("\n" + "=" * 60)
print("2. 加载数据")
print("=" * 60)

# 训练集的数据增强
train_transform = transforms.Compose([
    transforms.RandomRotation(10),        # 随机旋转±10度（数据增强）
    transforms.RandomHorizontalFlip(),    # 随机水平翻转（数据增强）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])

# 验证集和测试集：不需要数据增强，只做归一化
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载三个数据集
train_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='train', 
    transform=train_transform, download=True, size=config.image_size
)

# ==================== 数据集缩减（分层采样版本）====================
from sklearn.model_selection import train_test_split

if config.data_fraction < 1.0:
    num_train = len(train_dataset)
    indices = np.arange(num_train)

    # 🔥 关键：提取所有标签
    labels = np.array([train_dataset[i][1] for i in indices]).squeeze()

    # 🔥 分层采样（保证类别比例不变）
    selected_indices, _ = train_test_split(
        indices,
        train_size=config.data_fraction,
        stratify=labels,
        random_state=42  # 保证可复现
    )

    train_dataset = Subset(train_dataset, selected_indices)

    # 打印信息
    print("\n⚠️ 调参模式开启！（分层采样）")
    print(f"训练集从 {num_train} → {len(selected_indices)}")

    # 🔍 可选：打印采样后类别分布（强烈建议保留）
    sampled_labels = labels[selected_indices]
    print("采样后类别分布:", Counter(sampled_labels))


val_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='val', 
    transform=eval_transform, download=True, size=config.image_size
)
test_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='test', 
    transform=eval_transform, download=True, size=config.image_size
)

# 用DataLoader打包，让数据可以分批取出
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                        shuffle=False, num_workers=config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                         shuffle=False, num_workers=config.num_workers)

print(f"训练集: {len(train_dataset)} 张, {len(train_loader)} 批")
print(f"验证集: {len(val_dataset)} 张, {len(val_loader)} 批")
print(f"测试集: {len(test_dataset)} 张, {len(test_loader)} 批")


# ==================== 计算类别权重 ====================
print("\n" + "=" * 60)
print("计算类别权重（用于解决DRUSEN问题）")
print("=" * 60)

train_labels = []

# ⚠️ 注意：这里用的是“已经分层采样后的 train_dataset”
for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    train_labels.append(int(label[0]))

train_labels = np.array(train_labels)

# 统计每类数量
class_counts = np.bincount(train_labels, minlength=config.num_classes)
print("各类别样本数:", class_counts)

# ================= 核心：手动增强 DRUSEN =================
# 类别顺序：CNV=0, DME=1, DRUSEN=2, NORMAL=3

class_weights = np.ones(config.num_classes)

# 🔥 关键：强制提高 DRUSEN 权重
class_weights[2] = 3.0   # 可以尝试 3~8

print("初始类别权重:", class_weights)

# 转为 tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)



# ==================== 3. 构建ResNet-18模型 ====================
print("\n" + "=" * 60)
print("3. 构建ResNet-18模型")
print("=" * 60)

def create_resnet18(num_classes=4, pretrained=False):
    """
    创建ResNet-18模型
    参数:
        num_classes: 分类数
        pretrained: 是否使用预训练权重
    """
    # 从torchvision加载ResNet-18结构
    model = models.resnet18(pretrained=pretrained)
    
    # 关键：OCTMNIST是灰度图(1通道)，但ResNet默认接受RGB(3通道)
    # 修改第一层卷积，从3通道改为1通道
    model.conv1 = nn.Conv2d(
        in_channels=1,          # 输入1个通道（灰度图）
        out_channels=64,        # 输出64个特征图
        kernel_size=7,          # 卷积核大小
        stride=2,
        padding=3,
        bias=False
    )
    
    # 修改最后的全连接层，输出4个类别（原来是1000类）
    in_features = model.fc.in_features  # 获取全连接层输入维度（512）
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# 创建模型并移到设备（CPU或GPU）
model = create_resnet18(num_classes=config.num_classes, pretrained=config.pretrained)
model = model.to(config.device)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"模型结构:\n{model}")

# ==================== 4. 定义损失函数和优化器 ====================
print("\n" + "=" * 60)
print("4. 损失函数与优化器")
print("=" * 60)

# 损失函数：交叉熵损失（分类任务标配）
# 它内部已经包含了Softmax，所以模型输出Logits即可
criterion = nn.CrossEntropyLoss(weight=class_weights)
print(f"损失函数: CrossEntropyLoss（内部包含Softmax）")

# 优化器
if config.optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
elif config.optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
print(f"优化器: {config.optimizer_name}")

# 学习率调度器：训练过程中逐渐降低学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print(f"学习率调度: 每10轮降为原来的0.1倍")

# ==================== 5. 训练函数 ====================
from tqdm import tqdm  # 文件开头加上这个导入

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用tqdm包装loader，显示进度条
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.squeeze().to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条显示
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{correct/total:.3f}'
        })
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """在验证集/测试集上评估"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.squeeze().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets

# ==================== 6. 训练循环 ====================
print("\n" + "=" * 60)
print("5. 开始训练")
print("=" * 60)

# 记录每个epoch的指标（用于后续画图）
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

best_val_acc = 0.0  # 记录最佳验证准确率
best_val_preds = None  # 保存最佳验证集的预测结果
best_val_targets = None  # 保存最佳验证集的真实标签


for epoch in range(config.num_epochs):
    # 训练一个epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, config.device
    )
    # 在验证集上评估
    val_loss, val_acc, val_preds, val_targets = evaluate(
        model, val_loader, criterion, config.device
    )

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # 保存历史记录
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_preds = val_preds  # 保存最佳时的预测结果
        best_val_targets = val_targets  # 保存最佳时的真实标签
        torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
    
    # 打印进度
    print(f"Epoch [{epoch+1:2d}/{config.num_epochs}] "
          f"LR: {current_lr:.5f} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
          f"{'★' if val_acc == best_val_acc else ''}")
    
    # ===== 新增：每10轮epoch打印一次当前最佳validation的混淆矩阵 =====
    if (epoch + 1) % 10 == 0 and best_val_preds is not None:
        print("\n" + "=" * 60)
        print(f"第 {epoch+1} 轮 - 当前最佳Validation混淆矩阵 (Val Acc: {best_val_acc:.4f})")
        print("=" * 60)
        
        # 计算并打印混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(best_val_targets, best_val_preds)
        label_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        print("\n混淆矩阵:")
        # 打印表头
        print(f"{'':>10}", end="")
        for name in label_names:
            print(f"{name:>10}", end="")
        print(f"{'总计':>10}")
        
        # 打印每一行
        for i, name in enumerate(label_names):
            print(f"{name:>10}", end="")
            for j in range(len(cm[i])):
                print(f"{cm[i][j]:>10}", end="")
            print(f"{np.sum(cm[i]):>10}")
        
        # 打印总计行
        print(f"{'总计':>10}", end="")
        for j in range(len(cm)):
            print(f"{np.sum(cm[:, j]):>10}", end="")
        print(f"{np.sum(cm):>10}")
        
        # 计算并打印类别准确率
        print("\n各类别准确率:")
        for i, name in enumerate(label_names):
            class_acc = cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0
            print(f"  {name:>8}: {class_acc:.4f} ({cm[i][i]}/{np.sum(cm[i])})")
        
        print("\n简要分类报告:")
        print(classification_report(best_val_targets, best_val_preds, 
                                  target_names=label_names, digits=4))
        print("=" * 60 + "\n")


print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")

# ==================== 7. 测试集评估 ====================
print("\n" + "=" * 60)
print("6. 测试集评估")
print("=" * 60)

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth')))
test_loss, test_acc, test_preds, test_targets = evaluate(
    model, test_loader, criterion, config.device
)
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.4f}")

# ==================== 8. 绘制训练曲线 ====================
print("\n" + "=" * 60)
print("7. 绘制训练曲线")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss曲线
axes[0].plot(range(1, config.num_epochs+1), history['train_loss'], 
             'b-', label='Train Loss', linewidth=2)
axes[0].plot(range(1, config.num_epochs+1), history['val_loss'], 
             'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy曲线
axes[1].plot(range(1, config.num_epochs+1), history['train_acc'], 
             'b-', label='Train Acc', linewidth=2)
axes[1].plot(range(1, config.num_epochs+1), history['val_acc'], 
             'r-', label='Val Acc', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.save_dir, 'training_curves.png'), dpi=150)
plt.show()
print("✓ 训练曲线已保存")

# ==================== 9. 分类报告 ====================
print("\n" + "=" * 60)
print("8. 分类评估报告")
print("=" * 60)

from sklearn.metrics import classification_report, confusion_matrix

label_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
print("\n分类报告:")
print(classification_report(test_targets, test_preds, target_names=label_names, digits=4))

# 混淆矩阵
cm = confusion_matrix(test_targets, test_preds)
print("\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
fig2, ax2 = plt.subplots(figsize=(8, 7))
im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.figure.colorbar(im, ax=ax2)
ax2.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel='预测类别',
        ylabel='真实类别',
        title='混淆矩阵')
# 在每个格子里写数字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(config.save_dir, 'confusion_matrix.png'), dpi=150)
plt.show()
print("✓ 混淆矩阵已保存")

print("\n" + "=" * 60)
print("Task 1.3 基础训练完成！")
print(f"最佳模型保存在: {os.path.join(config.save_dir, 'best_model.pth')}")
print("=" * 60)