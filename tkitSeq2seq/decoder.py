import torch
from torch import nn


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size=256, num_layers=2):
        """[summary]
        
        解码层

        Args:
            hidden_size ([type]): [description]
            output_size ([type]): [description]
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.d = torch.nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.2, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data)
        # output=self.d(output)
        # # hidden=self.d(hidden)
        # output = F.tanh(output)
        # hidden = self.d(hidden)
        output, hidden = self.gru(output, hidden)
        #         output = self.softmax(self.out(output[0]))
        output = self.d(output)
        output = self.softmax(self.out(output.squeeze(0)))
        return output, hidden
