from torch import nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """[summary]
    编码层
    

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_size, hidden_size=256,num_layers=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.d=torch.nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size,dropout=0.2,num_layers=num_layers)

    def forward(self, input):
#         embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)

        output = self.d(embedded)
        output, hidden = self.gru(output)
        return output, hidden
