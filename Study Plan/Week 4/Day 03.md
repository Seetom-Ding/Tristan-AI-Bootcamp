# Fixed Feature Extractor

⭐ **Difficulty**：★★★★☆

💎 **Reward**：+190 XP

🏅 **Current Rank**：Deep Learning Practitioner   **XP**：340 / 1200

## 上午（09:00--11:30）

### 官方教程

- 学习冻结 Backbone 参数

- 确认只有新分类头保持 requires_grad=True

- 理解冻结参数仍参与前向传播但不计算其梯度

## 下午（15:00--17:30）

### Coding Lab

- 将 ResNet18 作为 Fixed Feature Extractor

- 只训练新分类头

- 记录训练时间、Train Loss 与 Validation Accuracy

- 保存验证集最佳模型

## Challenge

冻结参数后，为什么前向传播仍然有计算成本？如何验证反向传播和优化器确实只处理分类头？

## Today's Checklist

- [ ] 正确冻结 Backbone

- [ ] 优化器只接收可训练参数

- [ ] 完成 Fixed Feature Extractor 基线
