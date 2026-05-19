"""
Task 1.3: AI辅助编写ResNet-18训练代码
本脚本包含：数据加载、模型构建、训练、评估、实验对比分析
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
from torchvision.models.resnet import BasicBlock, ResNet
import medmnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 设置随机种子 ====================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 配置参数 ====================
class Config:
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    image_size = 64
    batch_size = 32
    num_workers = 0
    num_classes = 4
    pretrained = False
    learning_rate = 0.001
    num_epochs = 10
    data_fraction = 0.1
    
    # 实验对比配置
    enable_comparison = True
    
    # 定义对比实验配置
    experiments = [
        {
            'name': 'Baseline (ReLU + CrossEntropy + Adam)',
            'activation': 'relu',
            'loss': 'CrossEntropy',
            'optimizer': 'Adam',
            'lr': 0.001
        },
        {
            'name': 'LeakyReLU + CrossEntropy + Adam',
            'activation': 'leaky_relu',
            'loss': 'CrossEntropy',
            'optimizer': 'Adam',
            'lr': 0.001
        },
        {
            'name': 'ELU + CrossEntropy + Adam',
            'activation': 'elu',
            'loss': 'CrossEntropy',
            'optimizer': 'Adam',
            'lr': 0.001
        },
        {
            'name': 'ReLU + FocalLoss + Adam',
            'activation': 'relu',
            'loss': 'FocalLoss',
            'optimizer': 'Adam',
            'lr': 0.001
        },
        {
            'name': 'ReLU + CrossEntropy + SGD',
            'activation': 'relu',
            'loss': 'CrossEntropy',
            'optimizer': 'SGD',
            'lr': 0.01
        },
        {
            'name': 'ReLU + CrossEntropy + AdamW',
            'activation': 'relu',
            'loss': 'CrossEntropy',
            'optimizer': 'AdamW',
            'lr': 0.001
        }
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    comparison_dir = os.path.join(save_dir, 'comparison_results')

config = Config()
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.comparison_dir, exist_ok=True)

print("=" * 60)
print("Task 1.3: ResNet-18 训练脚本（含实验对比）")
print("=" * 60)
print(f"使用设备: {config.device}")
print(f"训练轮数: {config.num_epochs}")
print(f"将运行 {len(config.experiments)} 个对比实验")

# ==================== 数据加载 ====================
print("\n" + "=" * 60)
print("加载数据")
print("=" * 60)

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='train', 
    transform=train_transform, download=True, size=config.image_size
)

if config.data_fraction < 1.0:
    num_train = len(train_dataset)
    indices = np.arange(num_train)
    labels = np.array([train_dataset[i][1] for i in indices]).squeeze()
    selected_indices, _ = train_test_split(
        indices, train_size=config.data_fraction, stratify=labels, random_state=42
    )
    train_dataset = Subset(train_dataset, selected_indices)
    print(f"\n训练集从 {num_train} → {len(selected_indices)}")

val_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='val', 
    transform=eval_transform, download=True, size=config.image_size
)
test_dataset = medmnist.OCTMNIST(
    root=config.data_root, split='test', 
    transform=eval_transform, download=True, size=config.image_size
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                          shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                        shuffle=False, num_workers=config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                         shuffle=False, num_workers=config.num_workers)

print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

# 计算类别权重
train_labels = []
for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    train_labels.append(int(label[0]))
train_labels = np.array(train_labels)
class_counts = np.bincount(train_labels, minlength=config.num_classes)
print(f"\n各类别样本数: {class_counts}")

class_weights = torch.tensor([1.0, 1.0, 3.0, 1.0], dtype=torch.float32).to(config.device)

# ==================== 修改ResNet-18以支持不同激活函数 ====================
class FlexibleResNet(ResNet):
    """继承官方ResNet，只修改激活函数"""
    def __init__(self, block, layers, num_classes=1000, activation='relu', zero_init_residual=False):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)
        
        # 保存激活函数类型
        self.activation_name = activation
        
        # 修改第一层卷积为1通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替换所有的激活函数
        self._replace_activations(self, activation)
        
        # 修改最后的全连接层
        self.fc = nn.Linear(512, num_classes)
        
    def _replace_activations(self, module, activation):
        """递归替换模块中的所有ReLU为指定激活函数"""
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                if activation == 'relu':
                    new_act = nn.ReLU(inplace=True)
                elif activation == 'leaky_relu':
                    new_act = nn.LeakyReLU(0.1, inplace=True)
                elif activation == 'elu':
                    new_act = nn.ELU(alpha=1.0, inplace=True)
                elif activation == 'prelu':
                    new_act = nn.PReLU()
                else:
                    new_act = nn.ReLU(inplace=True)
                setattr(module, name, new_act)
            else:
                self._replace_activations(child, activation)

def create_resnet18(num_classes=4, pretrained=False, activation='relu'):
    """
    创建ResNet-18模型，支持不同激活函数
    使用torchvision官方实现，保证可复现
    """
    # 使用官方的ResNet-18结构
    model = FlexibleResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, activation=activation)
    return model

# ==================== 损失函数 ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def get_loss_function(loss_name, class_weights=None, device='cpu'):
    if loss_name == 'CrossEntropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'FocalLoss':
        return FocalLoss(alpha=1, gamma=2, weight=class_weights)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)

def get_optimizer(optimizer_name, model, lr):
    """获取优化器"""
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        return optim.Adam(model.parameters(), lr=lr)

# ==================== 训练函数 ====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.squeeze().to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    
    pbar = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.squeeze().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
    
    return total_loss / total, correct / total, all_preds, all_targets

# ==================== 单个实验执行函数 ====================
def run_single_experiment(exp_config, exp_name, train_loader, val_loader, test_loader, 
                         class_weights_tensor, base_config):
    """
    执行单个实验
    返回: history, best_val_acc, test_acc, test_preds, test_targets
    """
    print("\n" + "=" * 70)
    print(f"实验: {exp_name}")
    print("=" * 70)
    print(f"激活函数: {exp_config['activation']}")
    print(f"损失函数: {exp_config['loss']}")
    print(f"优化器: {exp_config['optimizer']} (LR={exp_config['lr']})")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建模型 - 注意函数名是 create_resnet18
    model = create_resnet18(
        num_classes=base_config.num_classes, 
        pretrained=base_config.pretrained,
        activation=exp_config['activation']
    )
    model = model.to(base_config.device)
    
    # 计算模型参数量（只在第一个实验打印）
    total_params = sum(p.numel() for p in model.parameters())
    if exp_name == base_config.experiments[0]['name']:
        print(f"模型参数量: {total_params:,}")
    
    # 定义损失函数
    criterion = get_loss_function(
        exp_config['loss'], 
        class_weights=class_weights_tensor,
        device=base_config.device
    )
    
    # 定义优化器
    optimizer = get_optimizer(exp_config['optimizer'], model, exp_config['lr'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(f"损失函数: {exp_config['loss']}")
    print(f"优化器: {exp_config['optimizer']}")
    print(f"学习率调度: 每10轮降为原来的0.1倍")
    
    # 记录每个epoch的指标
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_val_preds = None
    best_val_targets = None
    
    # 训练循环
    print("\n" + "-" * 60)
    print("开始训练")
    print("-" * 60)
    
    for epoch in range(base_config.num_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, base_config.device
        )
        # 在验证集上评估
        val_loss, val_acc, val_preds, val_targets = evaluate(
            model, val_loader, criterion, base_config.device
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
            best_val_preds = val_preds
            best_val_targets = val_targets
            # 保存模型
            exp_save_path = os.path.join(base_config.save_dir, f'best_model_{exp_name.replace(" ", "_")}.pth')
            torch.save(model.state_dict(), exp_save_path)
        
        # 打印进度
        print(f"Epoch [{epoch+1:2d}/{base_config.num_epochs}] "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
              f"{'★' if val_acc == best_val_acc else ''}")
    
    print(f"\n{exp_name} 训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    # 测试集评估
    print("\n" + "-" * 60)
    print("测试集评估")
    print("-" * 60)
    
    exp_save_path = os.path.join(base_config.save_dir, f'best_model_{exp_name.replace(" ", "_")}.pth')
    model.load_state_dict(torch.load(exp_save_path))
    test_loss, test_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, base_config.device
    )
    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Accuracy: {test_acc:.4f}")
    
    return history, best_val_acc, test_acc, test_preds, test_targets

# ==================== 绘制对比曲线 ====================
def plot_comparison_curves(all_results, save_dir):
    """绘制所有实验的对比曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 准确率对比
    for exp_name, results in all_results.items():
        history = results['history']
        axes[0].plot(history['val_acc'], label=exp_name, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[0].set_title('不同配置的验证准确率对比', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 损失对比
    for exp_name, results in all_results.items():
        history = results['history']
        axes[1].plot(history['val_loss'], label=exp_name, linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('不同配置的验证损失对比', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_experiments_comparison.png'), dpi=150)
    plt.show()

# ==================== 生成对比报告 ====================
def generate_comparison_report(all_results, save_dir):
    """生成对比分析报告"""
    print("\n" + "=" * 80)
    print("实验对比分析报告")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("实验对比分析报告 - OCTMNIST分类任务")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 创建结果表格
    results_table = []
    for exp_name, results in all_results.items():
        results_table.append({
            '实验名称': exp_name,
            '最佳验证准确率': f"{results['best_val_acc']:.4f}",
            '测试准确率': f"{results['test_acc']:.4f}"
        })
    
    # 排序找出最佳
    best_exp = max(all_results.items(), key=lambda x: x[1]['test_acc'])
    
    report_lines.append("实验结果汇总:")
    report_lines.append("-" * 60)
    for item in results_table:
        report_lines.append(f"  {item['实验名称']:40s} | Val Acc: {item['最佳验证准确率']} | Test Acc: {item['测试准确率']}")
    
    report_lines.append("")
    report_lines.append(f"最佳实验配置: {best_exp[0]}")
    report_lines.append(f"最佳测试准确率: {best_exp[1]['test_acc']:.4f}")
    report_lines.append("")
    
    report_lines.append("分析结论:")
    report_lines.append("-" * 60)
    report_lines.append("1. 激活函数对比: 对比Baseline、LeakyReLU和ELU的表现")
    report_lines.append("2. 损失函数对比: 对比CrossEntropyLoss和FocalLoss的效果")
    report_lines.append("3. 优化器对比: 对比Adam、SGD和AdamW的收敛性能")
    report_lines.append("")
    report_lines.append("影响因素分析:")
    report_lines.append("- 激活函数影响模型的非线性和梯度流动")
    report_lines.append("- 损失函数影响模型对难易样本的关注度")
    report_lines.append("- 优化器影响模型的收敛速度和最终性能")
    
    # 保存报告
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印报告
    for line in report_lines:
        print(line)
    
    print(f"\n报告已保存至: {report_path}")
    
    # 保存CSV表格
    df = pd.DataFrame(results_table)
    df.to_csv(os.path.join(save_dir, 'comparison_results.csv'), index=False, encoding='utf-8-sig')
    print(f"结果表格已保存至: {os.path.join(save_dir, 'comparison_results.csv')}")

# ==================== 主实验流程 ====================
if __name__ == "__main__":
    # 存储所有实验结果
    all_results = {}
    
    # 运行所有对比实验
    for exp in config.experiments:
        exp_config = {
            'activation': exp['activation'],
            'loss': exp['loss'],
            'optimizer': exp['optimizer'],
            'lr': exp['lr']
        }
        
        history, best_val_acc, test_acc, test_preds, test_targets = run_single_experiment(
            exp_config=exp_config,
            exp_name=exp['name'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights_tensor=class_weights,
            base_config=config
        )
        
        # 保存实验结果
        all_results[exp['name']] = {
            'history': history,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'config': exp_config
        }
        
        # 单独绘制每个实验的训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(range(1, config.num_epochs+1), history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0].plot(range(1, config.num_epochs+1), history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'{exp["name"]} - 准确率曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(range(1, config.num_epochs+1), history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[1].plot(range(1, config.num_epochs+1), history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'{exp["name"]} - 损失曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.comparison_dir, f'{exp["name"].replace(" ", "_")}_curves.png'), dpi=150)
        plt.close()
        
        # 打印每个实验的分类报告
        print("\n" + "=" * 60)
        print(f"{exp['name']} - 分类评估报告")
        print("=" * 60)
        label_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        print("\n分类报告:")
        print(classification_report(test_targets, test_preds, target_names=label_names, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(test_targets, test_preds)
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.figure.colorbar(im, ax=ax2)
        ax2.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=label_names,
                yticklabels=label_names,
                xlabel='预测类别',
                ylabel='真实类别',
                title=f'{exp["name"]} - 混淆矩阵')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black",
                        fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.comparison_dir, f'{exp["name"].replace(" ", "_")}_confusion_matrix.png'), dpi=150)
        plt.close()
    
    # 绘制所有实验的对比曲线
    plot_comparison_curves(all_results, config.comparison_dir)
    
    # 生成对比报告
    generate_comparison_report(all_results, config.comparison_dir)
    
    print("\n" + "=" * 60)
    print("所有实验完成！")
    print(f"实验结果保存在: {config.comparison_dir}")
    print("=" * 60)