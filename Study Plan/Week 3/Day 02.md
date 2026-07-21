# Build a Residual Block

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+160 XP

🏅 **Current Rank**：CNN Builder   **XP**：140 / 1000

## 上午（09:00--11:30）

### 论文导读

- 理解 Residual Mapping：H(x)=F(x)+x

- 理解 Identity Shortcut 与 Projection Shortcut

- 分析 Channel 或空间尺寸变化时为什么不能直接相加

### 官方实现

- 阅读 torchvision ResNet BasicBlock 的结构，不复制代码

## 下午（15:00--17:30）

### Coding Lab

- 从零实现 BasicBlock

- 支持 Stride=1 的 Identity Shortcut

- 支持 Stride=2 或 Channel 改变时的 Projection Shortcut

- 为三组输入编写 shape 测试

## Challenge

为什么残差分支最后一层之后通常先与 Shortcut 相加，再使用激活函数？尝试画出数据流并解释。

## Today's Checklist

- [ ] 独立实现 BasicBlock

- [ ] 正确处理 shape 不匹配

- [ ] 通过三组前向测试
