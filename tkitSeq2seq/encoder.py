import torch
import torch.nn.functional as F
from torch import nn


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


# https://www.freesion.com/article/99241355122/

class EncoderRNN(nn.Module):
    """[summary]
    编码层
    

    Args:
        nn ([type]): [description]
    """

    def __init__(self, input_size, hidden_size=256, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.en_num_layers=en_num_layers
        # self.en_hidden_size=en_hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.d = torch.nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.2, num_layers=num_layers, bidirectional=True)
        #  对双向的进行降维度
        self.out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        #         embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)

        output = self.d(embedded)
        output, hidden = self.gru(output)
        # print("rnn 编码结果：",output.shape, hidden.shape)
        # return output, hidden

        output = output.permute(1, 0, 2)  # 修改rnn输出的结构 output : [batch_size, len_seq, n_hidden]
        # 注意力操作
        attn_output, attention = self.attention_net(output, hidden)

        # print("attn_output",attn_output.shape)
        context = self.out(attn_output)
        output = self.out(output)
        # 输出  output context 注意力
        # 输出 已经是 B L  H
        return output, context, attention  # output hidden_state : [batch_size, len_seq, n_hidden] model : [batch_size, num_classes], attention : [batch_size, n_step]
        # else:
        #     return output

    def attention_net(self, lstm_output, final_state):
        """
        注意力

        Args:
            lstm_output:
            final_state:

        Returns:

        """
        hidden = final_state.view(-1, self.hidden_size * 2,
                                  self.num_layers)  # 双向操作  hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # print(lstm_output.size(),hidden.size())

        # bmm操作，由于使用了双向，需要后求均值
        attn_weights = torch.mean(torch.bmm(lstm_output, hidden), dim=2)  # [batch_size, seq_len]
        # print("attn_weights mean: ",attn_weights.shape)
        # attn_weights = attn_weights.squeeze(2)  # attn_weights : [batch_size, n_step]

        # print("attn_weights",attn_weights.shape)
        # 执行softmax
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden *
        # num_directions(=2), 1]

        # print("context ",lstm_output.transpose(1, 2).shape,soft_attn_weights.unsqueeze(2).shape)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).unsqueeze(2).squeeze(2)

        # print("context形状：",context.shape)
        return context, soft_attn_weights.cpu().data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]
