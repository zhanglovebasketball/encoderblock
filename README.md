语言：Python
快速上手开始训练：
conda create -n transformer 
conda activate transformer
pip install -r requirements.txt
python src/train.py

现在设置的是4head,可以更改base.yaml设置其他的头数或者更改其他的模型超参数，之后同样执行python src/train.py
消融实验（验证无位置编码的情况）：python src/ablation.py
如果测试实验效果，可以把src/train.py中最后一行main()注释掉，然后把之前注释的恢复，可以得到损失、准确率和困惑度的计算。

结构：
<img width="265" height="356" alt="image" src="https://github.com/user-attachments/assets/0214c3d3-0452-4b0c-8e7e-c7baa92a47ff" />

