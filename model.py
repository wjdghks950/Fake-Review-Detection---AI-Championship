import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NumberPretrainerConfig(object):
    def __init__(
        self,
        input_size=4,
        hidden_size=768,
        output_size=1,
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


class NumberPretrainer(nn.Module):
    def __init__(self, config):
        super(NumberPretrainer, self).__init__()
        self.linear1 = nn.Linear(config.input_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.linear3 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # self.out = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, input_val, labels):
        out = F.relu(self.linear1(input_val))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(out, labels)
        return out


class AggregatorPretrainerConfig(object):
    def __init__(
        self,
        input_size=4,
        hidden_size=768,
        output_size=1,
        max_seq_len=50,
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len


class Aggregator(nn.Module):
    def __init__(self, config):
        super(Aggregator, self).__init__()
        self.layer1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.aggregate = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, input_rep, labels):
        hidden = F.relu(self.layer1(input_rep))
        out = self.aggregate(hidden)
        loss_weight = torch.tensor([0.3, 0.7])
        loss_fc = nn.CrossEntropyLoss()  # weight=loss_weight
        loss = loss_fc(out, labels)
        return out, loss, hidden
