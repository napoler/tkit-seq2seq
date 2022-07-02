# tkitSeq2seq

简单的seq2seq模块

基于gru



安装解析配置文件

> pip install jsonargparse[signatures]

# 生成配置文件
Dump default configuration to have as reference

> python trainer.py fit --print_config > config/cpu_config.yaml

# Create config including only options to modify
> nano config.yaml

# 运行训练操作
Run training using created configuration


> python trainer.py fit --config config/cpu_config.yaml




运行示例

https://www.kaggle.com/terrychanorg/tkitseq2seq-notebook5c33cc1be2

核心训练模块

https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html