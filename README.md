# tkitSeq2seq

简单的seq2seq模块

基于gru

1. 使用双向BiGRU做编码
2. 编码层加入注意力操作

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

# 查看日志

> tensorboard --logdir lightning_logs



运行示例

https://www.kaggle.com/terrychanorg/tkitseq2seq-notebook5c33cc1be2

核心训练模块

https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html

数据构建

示例数据(加法计算模型，使用tokenizer目录里的小词典)
> python demo.py
https://github.com/napoler/BulidDataset
Seq2seq
https://github.com/napoler/BulidDataset/blob/main/buildDataSeq2seq.py