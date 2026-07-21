# BatchNorm and Regularization

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+160 XP

🏅 **Current Rank**：CNN Builder   **XP**：300 / 1000

## 上午（09:00--11:30）

### 网课

- [CS231n：Training Neural Networks](https://cs231n.github.io/neural-networks-2/)

- 理解 BatchNorm 在训练与推理阶段的行为差异

- 比较 Weight Decay、Dropout 与 Data Augmentation

### 官方教程

- 阅读 nn.BatchNorm2d、model.train() 与 model.eval() 文档

## 下午（15:00--17:30）

### Coding Lab

- 在同一 CNN 中比较有无 BatchNorm

- 记录 Loss 曲线、梯度范数和 Test Accuracy

- 加入 Weight Decay，观察泛化变化

## Challenge

忘记在验证时调用 model.eval() 会怎样影响 BatchNorm？设计一个最小实验验证。

## Today's Checklist

- [ ] 能解释 BatchNorm 的两种模式

- [ ] 完成 BatchNorm 对照实验

- [ ] 完成 Weight Decay 泛化分析
