《大模型期中作业》
语言：Python

快速上手开始训练：

conda create -n transformer 

conda activate transformer

pip install -r requirements.txt

python src/train.py

现在设置的是4head,可以更改base.yaml设置其他的头数或者更改其他的模型超参数，之后同样执行python src/train.py

消融实验（验证无位置编码的情况）：python src/ablation.py

如果测试实验效果，可以把src/train.py中最后一行main()注释掉，然后把之前注释的恢复，可以得到损失、准确率和困惑度的计算（生成文本的比较暂未调用）。
随机种子：seed=42

结构：

<img width="265" height="356" alt="image" src="https://github.com/user-attachments/assets/0214c3d3-0452-4b0c-8e7e-c7baa92a47ff" />

任务：实现前n-1个token预测后n-1个token的任务

数据预处理：首先加载Tiny Shakespeare文本数据，使用GPT-2分词器将原始文本转换为token序列，通过滑动窗口策略将长文本切分为固定长度的重叠片段以增加训练样本并保持上下文连贯性。随后将token序列转换为PyTorch张量，按8:1:1的比例随机分割为训练集、验证集和测试集，最后通过数据加载器进行批处理并动态填充序列至相同长度，同时构建输入-目标对实现前n-1个token预测后n-1个token的任务格式。

模型：Encoder of Transformer

训练过程：训练过程中交叉熵损失函数作为损失函数，使用AdamW优化器配合权重衰减正则化，实施梯度裁剪防止梯度爆炸，并通过早停机制在验证损失不再改善时保存最佳模型。

