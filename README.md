# tkitSeq2seq

简单的seq2seq模块

基于gru







# 生成配置文件
Dump default configuration to have as reference

> python  trainer.py --print_config > default_config.yaml

# Create config including only options to modify
> nano config.yaml

# 运行训练操作
Run training using created configuration


> python trainer.py--config config.yaml


核心训练模块

https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html