# -*- coding: utf-8 -*-
import pickle
import torch,random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
# 自动停止
# https://pytorch-lightning.readthedocs.io/en/1.2.1/common/early_stopping.html
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.optim as optim
from tqdm.auto import tqdm
import torchmetrics

from .encoder import EncoderRNN

from  .decoder import DecoderRNN


class autoEncDec(pl.LightningModule):
    """
    继承自bertlm模型
    https://colab.research.google.com/drive/1-OEwiD9ouGjWrSFEWhgEnWiNvwwxlqd7#scrollTo=no6DwOqaE9Jw
    做预测
    
    https://github.com/lucidrains/performer-pytorch
    """
# class COCO(nn.Module):
    def __init__(
        self,learning_rate=3e-4,T_max=5,hidden_size=256,input_vocab_size=30522,output_vocab_size=21128,ignore_index=0,teacher_forcing_ratio=0.5,en_num_layers=2,de_num_layers=2,optimizer_name="AdamW",
        batch_size=2,trainfile="./data/train.pkt",valfile="./data/val.pkt",testfile="./data/test.pkt", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # SRC_SEQ_LEN=128
        # TGT_SEQ_LEN=128
        # DE_SEQ_LEN=128
        # EN_SEQ_LEN=128
        # self.hparams.hidden_size
        self.enc = EncoderRNN(input_size=self.hparams.input_vocab_size,hidden_size=self.hparams.hidden_size,num_layers=self.hparams.en_num_layers)
        self.dec = DecoderRNN(output_size=self.hparams.output_vocab_size,hidden_size=self.hparams.hidden_size,num_layers=self.hparams.de_num_layers)
        self.accuracy = torchmetrics.Accuracy(ignore_index=self.hparams.ignore_index)
#         self.encoder_hidden = self.enc.initHidden()

    def forward(self, x,y,x_attention_mask,y_attention_mask, **kwargs):

        x=x.permute(1,0)
        y=y.permute(1,0)
        teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
        loss_fc=torch.nn.CrossEntropyLoss(ignore_index=self.hparams.ignore_index)
        trg_len,batch_size=y.size()
        outputs = torch.zeros(trg_len,batch_size, self.hparams.output_vocab_size).to(self.device)
        dec_input = torch.zeros(1,batch_size).to(self.device).long()
#         dec_input = y[0, :]
        x_output, hidden=self.enc(x)
        loss=None
#         dec_input=x_output.long()
        for i in range(y.shape[0]):
#             print("dec_input",dec_input,dec_input.size())
            output, hidden=self.dec(dec_input,hidden)
#             print("output",output,output.size())
            
            outputs[i]=output
        
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            dec_input = y[i] if teacher_force else top1
            dec_input = dec_input.unsqueeze(0)

            if loss!=None:
                loss+=loss_fc(output,y[i])
            else:
                loss=loss_fc(output,y[i])

        loss=loss/y.shape[0]

        return loss,outputs.permute(1,0,2)
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x,x_attention_mask,y,y_attention_mask = batch
        loss,outputs  = self(x,y,x_attention_mask,y_attention_mask)
        self.log('train_loss',loss)
        return  loss
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x,x_attention_mask,y,y_attention_mask = batch
        loss,outputs  = self(x,y,x_attention_mask,y_attention_mask)
#         print("outputs",outputs.size())
        acc=self.accuracy(outputs.reshape(-1,self.hparams.output_vocab_size,), y.reshape(-1))
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x,x_attention_mask,y,y_attention_mask = batch
        loss,outputs  = self(x,y,x_attention_mask,y_attention_mask)
#         print("outputs",outputs.size())
        acc=self.accuracy(outputs.reshape(-1,self.hparams.output_vocab_size,), y.reshape(-1))
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics
        
    def train_dataloader(self):
        train=torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size),num_workers=2,pin_memory=True, shuffle=True)
    def val_dataloader(self):
        val=torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size),num_workers=2,pin_memory=True)
    def test_dataloader(self):
        val=torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size),num_workers=2,pin_memory=True)

    
    def configure_optimizers(self):
            """优化器 # 类似于余弦，但其周期是变化的，初始周期为T_0,而后周期会✖️T_mult。每个周期学习率由大变小； https://www.notion.so/62e72678923f4e8aa04b73dc3eefaf71"""
    #         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))

            #只优化部分
#             optimizer = torch.optim.AdamW(self.parameters(), lr=(self.hparams.learning_rate))
            optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
            #         使用自适应调整模型
            T_mult=2
            scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=self.hparams.T_max,T_mult=T_mult,eta_min=0 ,verbose=False)
    #         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119

            lr_scheduler={
    #            'optimizer': optimizer,
               'scheduler': scheduler,
#                 'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
                'interval': 'epoch', #epoch/step
                'frequency': 1,
                'name':"lr_scheduler",
                'monitor': 'train_loss', #监听数据变化
                'strict': True,
            }
    #         return [optimizer], [lr_scheduler]
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}