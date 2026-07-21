# 认识卷积与特征图

⭐ **Difficulty**：★★☆☆☆

💎 **Reward**：+100 XP

🏅 **Current Rank**：Deep Learning Apprentice   **XP**：0 / 800

## 上午（09:00--11:30）

### 网课

- [CS231n：Convolutional Networks](https://cs231n.github.io/convolutional-networks/)

- 理解局部感受野、权重共享、Channel 与 Feature Map

- 掌握 Kernel、Stride、Padding 对输出尺寸的影响

### 官方教程

- 阅读 PyTorch Training a Classifier 中的 CIFAR-10 数据与网络部分

- 认识 3×32×32 彩色图像的张量布局

## 下午（15:00--17:30）

### Coding Lab

- 使用 nn.Conv2d 对一批 CIFAR-10 形状的随机 Tensor 做前向传播

- 改变 Kernel、Stride、Padding，记录每层输入与输出 shape

- 编写一个函数，根据卷积参数计算输出空间尺寸

## Challenge

不运行代码，手算输入 32×32、Kernel=5、Stride=2、Padding=1 时的输出尺寸，再用 PyTorch 验证。

## Today's Checklist

- [ ] 能解释卷积为什么适合图像

- [ ] 能手算卷积输出尺寸

- [ ] 能追踪 NCHW Tensor 的 shape
