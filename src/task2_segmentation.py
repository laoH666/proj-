import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ==========================================
# Task 2.1: 数据集加载与可视化
# ==========================================
class REFUGEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        注意：请根据你解压后的实际情况修改 image_dir 和 mask_dir。
        REFUGE 的 mask 通常包含视盘和视杯的不同灰度值。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # 假设 mask 和 image 同名，但后缀可能不同，具体依数据集而定
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.bmp')) 
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 读取为灰度图
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask

def visualize_sample(image_tensor, mask_tensor):
    """展示原图与Mask，观察视盘与视杯的特点"""
    image = image_tensor.permute(1, 2, 0).numpy()
    mask = mask_tensor.squeeze().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Fundus Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Optic Disc & Cup Mask")
    plt.axis('off')
    plt.show()

    # ==========================================
# Task 2.2: 网络结构 (U-Net)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1): # out_channels=1 适合二分类(如仅分割视盘)
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        c1 = self.conv1(torch.cat([u1, d2], dim=1)) # 跳跃连接
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([u2, d1], dim=1))
        
        return self.out_conv(c2)
    
    # ==========================================
# Task 2.3: 损失函数与评估指标
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, targets):
        return self.w_ce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)

def calculate_metrics(logits, targets):
    """计算 IoU 和 Dice Score"""
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum().item()
    union = (preds + targets).sum().item() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / ((preds + targets).sum().item() + 1e-6)
    return iou, dice

# ==========================================
# Task 2.4: 训练与参数调优引擎
# ==========================================
def train_model(image_dir, mask_dir, batch_size=4, lr=1e-3, epochs=10, loss_type='combined'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = REFUGEDataset(image_dir, mask_dir, transform=transform, mask_transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型、损失函数与优化器
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if loss_type == 'ce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    else:
        criterion = CombinedLoss()

    # 3. 训练循环
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            iou, _ = calculate_metrics(outputs, masks)
            epoch_iou += iou
            
        avg_loss = epoch_loss / len(dataloader)
        avg_iou = epoch_iou / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")

    return model, loss_history

# ==========================================
# 结果可视化函数 (追加到代码文件中)
# ==========================================
# ==========================================
# 结果可视化函数 (加入自动保存功能)
# ==========================================
def show_results(losses, model, image_dir, mask_dir, device, save_dir="results"):
    # 如果不存在 results 文件夹，自动创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 获取当前时间，用来给图片命名，防止覆盖之前的实验记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 画出 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss', color='blue', marker='o')
    plt.title('Loss Curve over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 自动保存 Loss 曲线 (注意：savefig 必须写在 show 的前面)
    loss_path = os.path.join(save_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(loss_path, bbox_inches='tight', dpi=300) # dpi=300 保证图片超高清
    print(f"\n✅ Loss曲线已自动保存至: {loss_path}")
    
    plt.show(block=False) # block=False 让代码继续往下走，不需要手动关掉 Loss 图
    plt.pause(2) # 停顿2秒让你看一眼

    # 2. 模型预测可视化 (拿两张图测试)
    model.eval() 
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    test_dataset = REFUGEDataset(image_dir, mask_dir, transform=transform, mask_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True) 
    
    images, true_masks = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float() 

    images = images.cpu().permute(0, 2, 3, 1).numpy()
    true_masks = true_masks.cpu().squeeze().numpy()
    preds = preds.cpu().squeeze().numpy()

    # 画图对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(2):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("1. Original Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_masks[i], cmap='gray')
        axes[i, 1].set_title("2. Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds[i], cmap='gray')
        axes[i, 2].set_title("3. Model Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    
    # 自动保存预测对比图
    pred_path = os.path.join(save_dir, f"prediction_{timestamp}.png")
    plt.savefig(pred_path, bbox_inches='tight', dpi=300)
    print(f"✅ 预测对比图已自动保存至: {pred_path}\n")
    
    plt.show() # 最后一张图展示出来，等你手动关闭后程序正式结束

# ==========================================
# 执行入口 (替换掉你现在的这部分)
# ==========================================
if __name__ == "__main__":
    IMG_DIR = r"K:\桌面\数据统计\project\data\refuge\REFUGE\train\Images"
    MASK_DIR = r"K:\桌面\数据统计\project\data\refuge\REFUGE\train\gts"
    
    # 想要模型看出一点效果，建议 epochs 至少设为 5 或者 10。
    # 既然你有 GPU，跑 10 个 epoch 应该很快！
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")
    
    print("开始训练模型...")
    # 这里你可以改参数做对比实验 (对应大作业 2.4)
    model, losses = train_model(IMG_DIR, MASK_DIR, batch_size=4, lr=1e-3, epochs=10, loss_type='combined')
    
    print("训练结束！正在生成结果对比图...")
    show_results(losses, model, IMG_DIR, MASK_DIR, device)