# Build a Small-Dataset Pipeline

⭐ **Difficulty**：★★★☆☆

💎 **Reward**：+180 XP

🏅 **Current Rank**：Deep Learning Practitioner   **XP**：160 / 1200

## 上午（09:00--11:30）

### 官方教程

- 阅读 PyTorch Transfer Learning Tutorial 的数据加载部分

- 理解 ImageFolder 的目录约定

- 理解训练增强与验证预处理为何不同

### 数据检查

- 查看类别数量、样本数量与类别平衡

- 可视化增强前后的样本

## 下午（15:00--17:30）

### Coding Lab

- 为一个小型图像数据集建立 train / val DataLoader

- 使用 RandomResizedCrop、RandomHorizontalFlip 与 Normalize

- 编写反归一化可视化函数检查数据

## Challenge

当每类只有约 100 张图像时，随机划分一次验证集有什么风险？你会如何让结论更可靠？

## Today's Checklist

- [ ] 正确建立 ImageFolder 数据结构

- [ ] 区分训练与验证 transforms

- [ ] 完成数据分布与增强检查
