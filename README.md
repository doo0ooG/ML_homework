**持续更新中...**

- hw1：简单的MLP处理回归任务
  - baseline使用全部特征进行很简单的训练
  - 可以通过特征工程，例如缺失值处理，异常值处理，标准化，特征筛选，对类型变量进行onehot编码等
- hw2：简单的MLP对已经处理好的音素进行分类
  - 一个frame对音素的分类效果比较差，采用“参考上下文”的办法拼接多个frame提升模型的判断能力
- hw3：图像分类
  - 图像增强来实现
  - 分别采用自定义cnn和resnet50进行对比
- hw4：注意力机制，
  - 仅使用transformer的encoder模块对语音序列处理，投射到矩阵中
- hw5：