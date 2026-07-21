# Training Stability and Learning Rate

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+170 XP

🏅 **Current Rank**：CNN Builder   **XP**：460 / 1000

## 上午（09:00--11:30）

### 网课

- CS231n：Weight Initialization、Learning Rate 与训练监控

- 理解过大或过小 Learning Rate 的曲线特征

### 官方教程

- 阅读 PyTorch Learning Rate Scheduler 的基本用法

- 理解保存最佳模型而非最后模型

## 下午（15:00--17:30）

### Coding Lab

- 为训练循环加入 StepLR 或 CosineAnnealingLR

- 每个 Epoch 记录 Learning Rate、Loss、Accuracy 与梯度范数

- 构造一次过大 Learning Rate 的失败实验并恢复训练

## Challenge

当 Loss 突然变成 NaN 时，你会先检查数据、前向输出、Loss、梯度还是参数？给出有理由的排查顺序。

## Today's Checklist

- [ ] 正确接入 Learning Rate Scheduler

- [ ] 复现并定位一次训练不稳定

- [ ] 保存并恢复最佳 Checkpoint
