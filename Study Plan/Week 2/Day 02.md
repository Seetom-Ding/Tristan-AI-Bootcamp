# Build a CNN for CIFAR-10

⭐ **Difficulty**：★★★☆☆

💎 **Reward**：+120 XP

🏅 **Current Rank**：Deep Learning Apprentice   **XP**：100 / 800

## 上午（09:00--11:30）

### 官方教程

- [PyTorch：Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- 理解 Conv → ReLU → Pool 的基本模块

- 理解 Flatten 前后 Tensor 形状如何衔接全连接层

## 下午（15:00--17:30）

### Coding Lab

- 从零编写一个两层卷积的 CIFAR-10 分类器

- 在 forward 中为每次 shape 变化写注释

- 用一批数据完成前向传播并确认输出为 Batch×10

## Challenge

如果删除所有 Pooling 层，模型参数量、显存占用和感受野会怎样变化？先推理，再用代码检查其中一项。

## Today's Checklist

- [ ] 能独立编写 nn.Conv2d 网络

- [ ] 能正确连接卷积层与全连接层

- [ ] 完成一次无 shape error 的前向传播
