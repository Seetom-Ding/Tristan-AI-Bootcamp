# Data Augmentation and Generalization

⭐ **Difficulty**：★★★☆☆

💎 **Reward**：+130 XP

🏅 **Current Rank**：Deep Learning Apprentice   **XP**：220 / 800

## 上午（09:00--11:30）

### 网课

- CS231n：Data Augmentation 与模型泛化

- 理解训练集增强和验证集预处理的区别

### 官方教程

- 阅读 torchvision transforms 文档中的 RandomCrop、RandomHorizontalFlip、Normalize

- 理解为什么 Normalize 的统计量必须保持一致

## 下午（15:00--17:30）

### Coding Lab

- 建立无增强 Baseline

- 加入 RandomCrop 与 RandomHorizontalFlip

- 固定其他条件，记录两组 Train / Test Loss 与 Accuracy

## Challenge

为什么随机增强通常只用于训练集？如果验证集也随机增强，评估结果会受到什么影响？

## Today's Checklist

- [ ] 正确区分训练与测试 transforms

- [ ] 完成一组受控增强实验

- [ ] 写出基于结果的泛化分析
